"""ค้นคืนบริบทจากเอกสารของตัวละคร แล้วให้ Ollama สร้างคำตอบ"""
import hashlib
import json
import re
from math import sqrt
from pathlib import Path

import requests

import settings as cfg

BASE_DIR = Path(__file__).resolve().parent
VECTOR_DB_DIR = BASE_DIR / "vector_db"

CHAT_ENDPOINT = f"{cfg.OLLAMA_URL}/api/chat"
EMBED_ENDPOINT = f"{cfg.EMBED_URL}/api/embed"

# โมเดลที่มีโหมดคิดก่อนตอบ ต้องสั่งปิดและตัด <think> ออกจากคำตอบ
THINKING_MODELS = ("qwen3", "deepseek-r1", "qwq", "magistral", "phi4-reasoning", "gpt-oss")
THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
THINK_TAIL = re.compile(r"^.*?</think>", re.DOTALL | re.IGNORECASE)

DEFAULT_SYSTEM_PROMPT = """คุณคือ {name} จงสวมบทบาทตัวละครนี้ตามบุคลิกใน CONTEXT

กฎ:
- ตอบเป็นบทพูดของตัวละครเท่านั้น ห้ามหลุดบทบาท
- ข้อเท็จจริงทั้งหมดต้องมาจาก CONTEXT ห้ามแต่งข้อมูลขึ้นเอง
- ถ้าถูกถามเรื่องที่ไม่มีใน CONTEXT ให้ปฏิเสธตามบุคลิกของตัวละคร
- คำทักทายหรือคุยทั่วไป ตอบสั้น ๆ ได้เลย
- ตอบสั้น 1-2 ประโยค ห้ามแสดงขั้นตอนการคิด"""


# ---------------------------------------------------------------------------
# เอกสารและการแบ่งหน่วยข้อมูล
# ---------------------------------------------------------------------------

def load_npc_documents(npc_dir: Path) -> str:
    """รวมไฟล์ .txt และ .md ทั้งหมดในโฟลเดอร์ของตัวละคร"""
    if not npc_dir.is_dir():
        return ""
    parts = []
    for pattern in ("*.txt", "*.md"):
        for path in sorted(npc_dir.glob(pattern)):
            text = path.read_text(encoding="utf-8", errors="ignore").strip()
            if text:
                parts.append(text)
    return "\n\n".join(parts)


def chunk_document(text: str, max_chars: int = 600) -> list:
    """แบ่งเอกสารตามหัวข้อ [ชื่อหัวข้อ] หัวข้อที่ยาวเกินจะถูกแบ่งย่อยโดยยังติดชื่อหัวข้อไว้"""
    sections = []
    title, lines = "[ทั่วไป]", []

    def flush():
        body = "\n".join(lines).strip()
        if body:
            sections.append((title, body))

    for line in text.splitlines():
        s = line.strip()
        if s.startswith("[") and s.endswith("]"):
            flush()
            title, lines = s, []
        else:
            lines.append(line)
    flush()

    chunks = []
    for title, body in sections:
        if len(body) <= max_chars:
            chunks.append(f"{title}\n{body}")
            continue
        part, size = [], 0
        for ln in body.splitlines():
            if part and size + len(ln) > max_chars:
                chunks.append(title + "\n" + "\n".join(part))
                part, size = [], 0
            part.append(ln)
            size += len(ln)
        if part:
            chunks.append(title + "\n" + "\n".join(part))
    return chunks


def section_name(chunk: str):
    first = chunk.splitlines()[0].strip()
    if first.startswith("[") and first.endswith("]"):
        return first[1:-1]
    return None


def apply_locks(chunks: list, locked: dict) -> list:
    """แทนเนื้อหาของหัวข้อที่ยังล็อกด้วยข้อความแทน ก่อนส่งเข้าพร็อมต์

    การค้นคืนยังใช้เนื้อหาจริง แต่โมเดลจะได้เห็นแค่ข้อความแทน
    ข้อมูลที่ล็อกจึงไม่อยู่ในพร็อมต์เลย
    """
    if not locked:
        return chunks
    out, done = [], set()
    for chunk in chunks:
        name = section_name(chunk)
        if name in locked:
            if name not in done:          # หัวข้อเดียวอาจมีหลายก้อน ใส่ข้อความแทนครั้งเดียว
                done.add(name)
                out.append(f"[{name}]\n{locked[name]}")
        else:
            out.append(chunk)
    return out


# ---------------------------------------------------------------------------
# Embedding และดัชนีเวกเตอร์
# ---------------------------------------------------------------------------

def _embed_options() -> dict:
    if cfg.EMBED_ON_CPU:
        return {"num_gpu": 0, "num_thread": cfg.EMBED_NUM_THREAD}
    return {}


def embed_texts(texts: list) -> list:
    body = {
        "model": cfg.EMBED_MODEL,
        "input": texts,
        "options": _embed_options(),
        "keep_alive": cfg.EMBED_KEEP_ALIVE,
    }
    resp = requests.post(EMBED_ENDPOINT, json=body, timeout=cfg.EMBED_TIMEOUT)

    if resp.status_code == 404:
        # Ollama รุ่นเก่ามีแต่ /api/embeddings ซึ่งรับทีละข้อความ
        vectors = []
        for t in texts:
            r = requests.post(
                f"{cfg.EMBED_URL}/api/embeddings",
                json={"model": cfg.EMBED_MODEL, "prompt": t,
                      "options": _embed_options(), "keep_alive": cfg.EMBED_KEEP_ALIVE},
                timeout=cfg.EMBED_TIMEOUT,
            )
            r.raise_for_status()
            vec = r.json().get("embedding")
            if not vec:
                raise ValueError(f"ไม่ได้ embedding กลับมา (pull โมเดล {cfg.EMBED_MODEL} แล้วหรือยัง)")
            vectors.append(vec)
        return vectors

    resp.raise_for_status()
    vectors = resp.json().get("embeddings")
    if not vectors or len(vectors) != len(texts):
        raise ValueError("จำนวน embedding ที่ได้ไม่ตรงกับจำนวนข้อความ")
    return vectors


def cosine(a: list, b: list) -> float:
    if len(a) != len(b):
        raise ValueError(
            f"เวกเตอร์คนละมิติ ({len(a)} กับ {len(b)}) ลองลบไฟล์ในโฟลเดอร์ vector_db แล้วรันใหม่"
        )
    dot = sum(x * y for x, y in zip(a, b))
    na = sqrt(sum(x * x for x in a))
    nb = sqrt(sum(x * x for x in b))
    return dot / (na * nb) if na and nb else 0.0


def _read_cache(path: Path, doc_hash: str, dim: int):
    """คืนแคชถ้ายังใช้ได้ (เอกสารไม่เปลี่ยน และมิติตรงกับโมเดล embedding ตัวปัจจุบัน)"""
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        vectors = data.get("embeddings") or []
        if data.get("hash") == doc_hash and vectors and len(vectors[0]) == dim:
            return data
    except (ValueError, OSError):
        pass
    return None


def build_index(npc_id: str, text: str, dim: int) -> dict:
    """โหลดดัชนีจาก vector_db/ ถ้าเอกสารหรือโมเดล embedding เปลี่ยนจะสร้างใหม่ให้เอง"""
    doc_hash = hashlib.sha256((cfg.EMBED_MODEL + text).encode("utf-8")).hexdigest()
    path = VECTOR_DB_DIR / f"{npc_id}.json"

    index = _read_cache(path, doc_hash, dim)
    if index:
        return index

    chunks = chunk_document(text)
    index = {
        "hash": doc_hash,
        "embed_model": cfg.EMBED_MODEL,
        "chunks": chunks,
        "embeddings": embed_texts(chunks),
    }
    VECTOR_DB_DIR.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(index, ensure_ascii=False), encoding="utf-8")
    return index


def retrieve_context(npc_id: str, npc_dir: Path, question: str, k: int = 3,
                     pinned: list = None, locked: dict = None, last_question: str = ""):
    """เลือกหน่วยข้อมูลที่จะส่งเข้าพร็อมต์ คืน (context, info)

    pinned  หัวข้อที่แนบไปทุกคำถาม เช่น บุคลิกของตัวละคร
    locked  {ชื่อหัวข้อ: ข้อความแทน} ของหัวข้อที่ยังล็อกอยู่
    ถ้าค้นคืนไม่สำเร็จจะส่งเอกสารทั้งหมดแทน (หัวข้อที่ล็อกยังถูกแทนที่เหมือนเดิม)
    """
    pinned = pinned or []
    locked = locked or {}

    full_text = load_npc_documents(npc_dir)
    if not full_text:
        return "", {"rag": False, "reason": "no_documents"}

    def whole_document():
        return "\n\n".join(apply_locks(chunk_document(full_text), locked))

    try:
        # คำถามสั้นมาก ๆ เช่น "แล้วไงต่อ" ค้นคืนไม่ค่อยตรง เลยเอาคำถามก่อนหน้ามาช่วย
        query = question.strip()
        if last_question and len(query) < 25:
            query = f"{last_question.strip()} {query}"

        q_vec = embed_texts([query])[0]
        index = build_index(npc_id, full_text, dim=len(q_vec))
        chunks, vectors = index["chunks"], index["embeddings"]

        if len(chunks) <= k:
            return whole_document(), {"rag": False, "reason": "document_smaller_than_k",
                                      "locked": sorted(locked)}

        scores = [cosine(q_vec, v) for v in vectors]
        ranked = sorted(range(len(chunks)), key=lambda i: -scores[i])

        keep = set(ranked[:k])
        always = set(pinned) | set(locked)
        for i, chunk in enumerate(chunks):
            if section_name(chunk) in always:
                keep.add(i)

        picked = apply_locks([chunks[i] for i in sorted(keep)], locked)
        return "\n\n".join(picked), {
            "rag": True,
            "k": k,
            "chunks_total": len(chunks),
            "chunks_used": len(picked),
            "top_chunks": [chunks[i].splitlines()[0] for i in ranked[:k]],
            "top_scores": [round(scores[i], 4) for i in ranked[:k]],
            "locked": sorted(locked),
        }
    except Exception as e:
        return whole_document(), {"rag": False, "reason": f"{type(e).__name__}: {e}",
                                  "locked": sorted(locked)}


def check_hint(npc_id: str, npc_dir: Path, question: str, answer: str):
    """เทียบบทสนทนากับประโยคสำคัญใน hints.json (ถ้ามี) ให้เกมรู้ว่าผู้เล่นเข้าใกล้ประเด็นหรือยัง"""
    path = npc_dir / "hints.json"
    if not path.exists():
        return None
    try:
        hints = json.loads(path.read_text(encoding="utf-8"))
        statements = hints.get("statements") or []
        if not statements:
            return None
        threshold = float(hints.get("threshold", 0.62))

        talk_vec = embed_texts([f"{question}\n{answer}"])[0]

        doc_hash = hashlib.sha256((cfg.EMBED_MODEL + "\n".join(statements)).encode("utf-8")).hexdigest()
        cache_path = VECTOR_DB_DIR / f"{npc_id}_hints.json"
        cache = _read_cache(cache_path, doc_hash, len(talk_vec))
        if not cache:
            cache = {"hash": doc_hash, "embeddings": embed_texts(statements)}
            VECTOR_DB_DIR.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(json.dumps(cache), encoding="utf-8")

        scores = [cosine(talk_vec, v) for v in cache["embeddings"]]
        best = max(range(len(scores)), key=scores.__getitem__)
        return {
            "triggered": scores[best] >= threshold,
            "score": round(scores[best], 4),
            "threshold": threshold,
            "topic": statements[best][:60],
        }
    except Exception as e:
        return {"triggered": False, "error": f"{type(e).__name__}: {e}"}


# ---------------------------------------------------------------------------
# เรียกโมเดลภาษา
# ---------------------------------------------------------------------------

def is_thinking_model(model: str) -> bool:
    name = (model or "").lower()
    return any(tag in name for tag in THINKING_MODELS)


def strip_think(text: str) -> str:
    if not text or "think>" not in text.lower():
        return text
    text = THINK_BLOCK.sub("", text)
    if "</think>" in text.lower():
        text = THINK_TAIL.sub("", text)
    return text.replace("<think>", "").replace("</think>", "").strip()


def chat(model: str, messages: list, options: dict, keep_alive: str = None, think=None) -> dict:
    """เรียก /api/chat ของ Ollama หนึ่งครั้ง คืน JSON ที่ได้"""
    options = dict(options)
    if cfg.NUM_GPU is not None:
        options["num_gpu"] = cfg.NUM_GPU

    payload = {
        "model": model,
        "stream": False,
        "messages": messages,
        "options": options,
        "keep_alive": keep_alive or cfg.KEEP_ALIVE,
    }
    if think is None and is_thinking_model(model):
        think = False
    if think is not None:
        payload["think"] = bool(think)

    resp = requests.post(CHAT_ENDPOINT, json=payload, timeout=cfg.CHAT_TIMEOUT)
    if resp.status_code == 400 and "think" in payload:
        # Ollama บางรุ่นหรือบางโมเดลไม่รับคีย์ think
        del payload["think"]
        resp = requests.post(CHAT_ENDPOINT, json=payload, timeout=cfg.CHAT_TIMEOUT)
    resp.raise_for_status()
    return resp.json()


def build_messages(question: str, npc_name: str, context: str, history: list = None,
                   system_prompt=None, system_as_user: bool = False) -> list:
    if isinstance(system_prompt, (list, tuple)):
        system_prompt = "\n".join(system_prompt)
    if not (system_prompt or "").strip():
        system_prompt = DEFAULT_SYSTEM_PROMPT.format(name=npc_name)

    system_text = f"{system_prompt.strip()}\n\nCONTEXT:\n{context}\n"

    if system_as_user:
        # สำหรับโมเดลที่ chat template ไม่รองรับ role system
        messages = [
            {"role": "user", "content": system_text},
            {"role": "assistant", "content": "รับทราบ"},
        ]
    else:
        messages = [{"role": "system", "content": system_text}]

    for q, a in (history or []):
        messages.append({"role": "user", "content": q})
        messages.append({"role": "assistant", "content": a})
    messages.append({"role": "user", "content": question})
    return messages


def find_prompt_leak(text: str, markers) -> str:
    """โมเดลบางตัวพ่นคำสั่งระบบออกมาแทนคำตอบ ถ้าเจอคำใน markers ช่วงต้นคำตอบถือว่าหลุด"""
    head = (text or "")[:400]
    for m in markers:
        if m and m in head:
            return m
    return None


def find_repetition(text: str) -> str:
    """จับคำตอบที่วนประโยคเดิมซ้ำ ๆ"""
    flat = " ".join((text or "").split())
    if len(flat) < 60:
        return None
    head = flat[:24]
    if flat.count(head) >= 3:
        return f"'{head.strip()}' x{flat.count(head)}"
    for sep in ("|", "•", "·", " - "):
        parts = [p.strip() for p in flat.split(sep) if p.strip()]
        if len(parts) >= 4 and len(set(parts)) * 2 <= len(parts):
            return f"repeated parts split by '{sep.strip()}'"
    return None


def generate_answer(question: str, npc: dict, context: str, history: list = None,
                    model: str = None, max_tokens: int = None) -> dict:
    """สร้างคำตอบของตัวละคร ถ้าโมเดลล่มหรือตอบผิดรูปจะคืนข้อความสำรองพร้อม fallback=True"""
    model = model or npc.get("model") or cfg.CHAT_MODEL
    fallback_text = npc.get("fallback_answer") or cfg.FALLBACK_ANSWER_TEXT

    def fallback(reason):
        return {"answer": fallback_text, "fallback": True, "fallback_reason": reason, "model": model}

    messages = build_messages(
        question, npc.get("name", ""), context, history,
        system_prompt=npc.get("system_prompt"),
        system_as_user=npc.get("system_as_user", False),
    )
    options = {
        "temperature": float(npc.get("temperature", 0.2)),
        "num_predict": int(max_tokens or npc.get("max_tokens", 80)),
        "num_ctx": int(npc.get("num_ctx", 4096)),
        "repeat_penalty": float(npc.get("repeat_penalty", 1.15)),
    }

    try:
        data = chat(model, messages, options, npc.get("keep_alive"), npc.get("think"))
    except Exception as e:
        return fallback(f"{type(e).__name__}: {e}")

    answer = strip_think((data.get("message") or {}).get("content", "")).strip()
    if not answer:
        return fallback("empty_response")

    leak = find_prompt_leak(answer, ["CONTEXT:"] + list(npc.get("leak_markers") or []))
    if leak:
        return fallback(f"prompt_leak: '{leak}' (ลองตั้ง \"system_as_user\": true ใน meta.json)")

    loop = find_repetition(answer)
    if loop:
        return fallback(f"repetition: {loop}")

    for wrong, right in (npc.get("text_fixes") or {}).items():
        answer = answer.replace(wrong, right)

    return {
        "answer": answer.strip(),
        "fallback": False,
        "model": data.get("model", model),
        "prompt_tokens": data.get("prompt_eval_count"),
        "completion_tokens": data.get("eval_count"),
        "eval_duration": data.get("eval_duration"),
    }
