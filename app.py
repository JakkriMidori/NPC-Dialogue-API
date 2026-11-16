"""เซิร์ฟเวอร์บทสนทนาของตัวละคร (NPC) สำหรับเกม

    python app.py

เกมส่งคำถามมาที่ POST /ask แล้วรับคำตอบของตัวละครกลับเป็น JSON
ตัวละครแต่ละตัวอยู่ในโฟลเดอร์ documents/npc_<ชื่อ>/ ดูวิธีตั้งค่าใน README.md
"""
import json
import logging
import re
import threading
import time
from datetime import datetime
from pathlib import Path

import requests
from flask import Flask, jsonify, request
from flask_cors import CORS

import settings as cfg
from grader import grade_answer
from rag_chain import (
    apply_locks,
    check_hint,
    chunk_document,
    generate_answer,
    load_npc_documents,
    retrieve_context,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("npc_api")

app = Flask(__name__)
CORS(app)

DOCS_DIR = Path(__file__).resolve().parent / "documents"
MAX_QUESTION_CHARS = 4000
HISTORY_KEEP = 12            # จำนวนคู่ถาม-ตอบที่เก็บไว้ต่อหนึ่งตัวละคร

npcs = {}                    # npc_id -> ค่าจาก meta.json
history = {}                 # npc_id -> [(คำถาม, คำตอบ), ...]
unlocked = {}                # npc_id -> ชื่อหัวข้อที่ผู้เล่นปลดล็อกแล้ว

# คำที่บอกว่าผู้เล่นถามอะไรบางอย่าง ไม่ได้ส่งรหัสมาอย่างเดียว
QUESTION_WORDS = (
    "?", "ไหม", "มั้ย", "หรือเปล่า", "รึเปล่า", "หรือไม่",
    "อะไร", "ใคร", "ทำไม", "เมื่อไหร่", "เมื่อไร", "ที่ไหน",
    "กี่", "ยังไง", "อย่างไร", "เท่าไหร่", "เท่าไร",
    "บอก", "ขอดู", "ขอทราบ", "เล่า", "อธิบาย", "รายงาน", "แสดง",
)


# ---------------------------------------------------------------------------
# ตัวละคร
# ---------------------------------------------------------------------------

def load_npcs():
    """อ่าน meta.json ของทุกโฟลเดอร์ npc_* ใน documents/"""
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    found = {}
    for folder in sorted(DOCS_DIR.iterdir()):
        if not (folder.is_dir() and folder.name.lower().startswith("npc_")):
            continue
        meta = {}
        meta_path = folder / "meta.json"
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
            except ValueError as e:
                log.warning("อ่าน %s ไม่ได้: %s", meta_path, e)
        meta.setdefault("name", folder.name)
        found[folder.name] = meta
    npcs.clear()
    npcs.update(found)


def as_text(value) -> str:
    """ค่าใน meta.json เขียนเป็นข้อความเดียวหรือเป็นลิสต์ของบรรทัดก็ได้"""
    if isinstance(value, (list, tuple)):
        return "\n".join(str(v) for v in value)
    return str(value or "")


# ---------------------------------------------------------------------------
# การล็อกและปลดล็อกหัวข้อ
# ---------------------------------------------------------------------------

def normalize_code(text: str) -> str:
    # ผู้เล่นพิมพ์รหัสเว้นวรรคหรือใส่ขีดต่างกันได้ เทียบแบบตัดออกหมด
    return re.sub(r"[\s\-_.]", "", text or "").upper()


def try_unlock(npc_id: str, npc: dict, question: str) -> list:
    """ปลดล็อกหัวข้อที่คำถามนี้มีรหัสหรือคำสำคัญตรง คืนชื่อหัวข้อที่เพิ่งปลด"""
    opened = unlocked.setdefault(npc_id, set())
    typed = normalize_code(question)
    lowered = (question or "").lower()
    newly = []
    for name, rule in (npc.get("locked_sections") or {}).items():
        if name in opened:
            continue
        rule = rule or {}
        code = normalize_code(rule.get("code", ""))
        keywords = [k.lower() for k in (rule.get("reveal_keywords") or []) if k]
        if (code and code in typed) or any(k in lowered for k in keywords):
            opened.add(name)
            newly.append(name)
            log.info("UNLOCK %s | %s", npc_id, name)
    return newly


def locked_now(npc_id: str, npc: dict) -> dict:
    """หัวข้อที่ยังล็อกอยู่ -> ข้อความที่ส่งเข้าพร็อมต์แทนเนื้อหาจริง"""
    opened = unlocked.get(npc_id, set())
    return {
        name: as_text((rule or {}).get("locked_text"))
        for name, rule in (npc.get("locked_sections") or {}).items()
        if name not in opened
    }


def is_code_only(question: str, npc: dict) -> bool:
    """ผู้เล่นส่งรหัสมาเฉย ๆ โดยไม่ได้ถามอะไรต่อท้าย"""
    text = question or ""
    for rule in (npc.get("locked_sections") or {}).values():
        code = (rule or {}).get("code")
        if code:
            text = re.sub(re.escape(code), " ", text, flags=re.IGNORECASE)
            text = re.sub(re.escape(normalize_code(code)), " ", text, flags=re.IGNORECASE)
    return not any(w in text for w in QUESTION_WORDS)


def unlock_message(npc: dict, names: list) -> str:
    if as_text(npc.get("unlock_reply")).strip():
        return as_text(npc.get("unlock_reply")).strip()
    parts = []
    for name in names:
        msg = as_text((npc["locked_sections"][name] or {}).get("unlock_message")).strip()
        if msg and msg not in parts:
            parts.append(msg)
    return " ".join(parts)


# ---------------------------------------------------------------------------
# POST /ask
# ---------------------------------------------------------------------------

def get_npc_or_error(npc_id: str):
    if not npc_id:
        return None, (jsonify({"error": "Field 'npc_id' is required"}), 400)
    if npc_id not in npcs:
        return None, (jsonify({"error": f"Invalid NPC ID '{npc_id}'"}), 404)
    return npcs[npc_id], None


@app.route("/ask", methods=["POST"])
def ask():
    """รับ {"question": "...", "npc_id": "npc_example"}  คืน {"answer": "...", ...}

    ส่ง "model", "use_rag", "num_ctx" มาด้วยได้ถ้าต้องการแทนค่าใน meta.json เฉพาะคำขอนั้น
    """
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return jsonify({"error": "Request must be JSON"}), 415

    question = (data.get("question") or "").strip()
    npc_id = (data.get("npc_id") or "").strip()
    if not question:
        return jsonify({"error": "Field 'question' is required"}), 400
    if len(question) > MAX_QUESTION_CHARS:
        return jsonify({"error": "Question too long"}), 400
    npc, error = get_npc_or_error(npc_id)
    if error:
        return error

    try:
        return jsonify(answer_question(npc_id, npc, question, data))
    except Exception as e:
        # เกมต้องได้คำตอบกลับเสมอ ไม่อย่างนั้นหน้าสนทนาจะค้าง
        log.exception("/ask ล้มเหลว")
        return jsonify({
            "answer": npc.get("fallback_answer") or cfg.FALLBACK_ANSWER_TEXT,
            "npc_id": npc_id,
            "npc_name": npc.get("name"),
            "hint": None,
            "timestamp": datetime.now().isoformat(),
            "debug": {"fallback": True, "fallback_reason": f"{type(e).__name__}: {e}"},
        })


def answer_question(npc_id: str, npc: dict, question: str, overrides: dict) -> dict:
    started = time.perf_counter()
    npc_dir = DOCS_DIR / npc_id
    past = history.setdefault(npc_id, [])

    model = (overrides.get("model") or "").strip() or npc.get("model") or cfg.CHAT_MODEL
    use_rag = overrides["use_rag"] if isinstance(overrides.get("use_rag"), bool) \
        else npc.get("use_rag", True)
    run = dict(npc)
    if overrides.get("num_ctx"):
        run["num_ctx"] = max(2048, min(int(overrides["num_ctx"]), 32768))

    # ตรวจรหัสก่อนค้นคืน หัวข้อที่เพิ่งปลดจะได้อยู่ในคำตอบของรอบนี้เลย
    newly = try_unlock(npc_id, npc, question)
    locks = locked_now(npc_id, npc)
    if newly:
        # ประวัติเก่ามีแต่คำปฏิเสธ ถ้าปล่อยไว้โมเดลจะตอบปฏิเสธซ้ำทั้งที่ปลดล็อกแล้ว
        past.clear()

    t0 = time.perf_counter()
    if use_rag:
        # หัวข้อที่ตั้งล็อกไว้แนบไปทุกครั้งทั้งก่อนและหลังปลด
        # ก่อนปลดโมเดลจะได้รู้ว่ามีข้อมูลส่วนนี้แต่เปิดไม่ได้ หลังปลดก็ไม่ต้องรอให้ค้นคืนเจอเอง
        pinned = list(npc.get("always_sections") or []) + list(npc.get("locked_sections") or {})
        context, rag_info = retrieve_context(
            npc_id, npc_dir, question,
            k=int(npc.get("retrieval_k", 3)),
            pinned=pinned,
            locked=locks,
            last_question=past[-1][0] if past else "",
        )
    else:
        chunks = chunk_document(load_npc_documents(npc_dir))
        context = "\n\n".join(apply_locks(chunks, locks))
        rag_info = {"rag": False, "reason": "use_rag=false", "locked": sorted(locks)}
    retrieval_ms = (time.perf_counter() - t0) * 1000

    if not context.strip():
        raise FileNotFoundError(f"ไม่พบเอกสาร .txt หรือ .md ใน documents/{npc_id}")

    pairs = int(npc.get("history_pairs", 4))
    recent = past[-pairs:] if npc.get("use_history", True) and pairs > 0 else []

    t1 = time.perf_counter()
    result = generate_answer(question, run, context, recent, model=model)
    generate_ms = (time.perf_counter() - t1) * 1000

    answer = result["answer"] or cfg.NO_ANSWER_TEXT
    failed = result["fallback"]

    if newly and not failed:
        # ประโยคยืนยันมาจากระบบ เพราะพร็อมต์ของตัวละครมักห้ามพูดเรื่องรหัส
        note = unlock_message(npc, newly)
        if note:
            answer = note if is_code_only(question, npc) else f"{note}\n\n{answer}"

    hint = None if failed else check_hint(npc_id, npc_dir, question, answer)

    if not failed:               # ข้อความสำรองไม่เก็บลงประวัติ
        past.append((question, answer))
        del past[:-HISTORY_KEEP]

    log.info("ASK %s | %.0f ms | prompt=%s completion=%s%s",
             npc_id, generate_ms, result.get("prompt_tokens"), result.get("completion_tokens"),
             f" | fallback: {result.get('fallback_reason')}" if failed else "")

    return {
        "answer": answer,
        "npc_id": npc_id,
        "npc_name": npc.get("name"),
        "hint": hint,
        "unlocked": newly,
        "locked_sections": sorted(locked_now(npc_id, npc)),
        "model": result.get("model", model),
        "timestamp": datetime.now().isoformat(),
        "debug": {
            "fallback": failed,
            "fallback_reason": result.get("fallback_reason"),
            "use_rag": use_rag,
            "retrieval_ms": round(retrieval_ms, 1),
            "generate_ms": round(generate_ms, 1),
            "server_ms": round((time.perf_counter() - started) * 1000, 1),
            "prompt_tokens": result.get("prompt_tokens"),
            "completion_tokens": result.get("completion_tokens"),
            "num_ctx": int(run.get("num_ctx", 4096)),
            "rag": rag_info,
        },
    }


# ---------------------------------------------------------------------------
# endpoint อื่น ๆ
# ---------------------------------------------------------------------------

@app.route("/npcs", methods=["GET"])
def list_npcs():
    return jsonify({"npcs": [
        {
            "id": npc_id,
            "name": npc.get("name"),
            "model": npc.get("model") or cfg.CHAT_MODEL,
            "history_count": len(history.get(npc_id, [])),
            "locked_sections": sorted(locked_now(npc_id, npc)),
        }
        for npc_id, npc in npcs.items()
    ]})


@app.route("/reload_npcs", methods=["POST"])
def reload_npcs():
    """อ่าน meta.json ใหม่หลังแก้ไฟล์ ไม่ต้องปิดเซิร์ฟเวอร์"""
    load_npcs()
    return jsonify({"npcs": list(npcs)})


@app.route("/reset_conversation", methods=["POST"])
def reset_conversation():
    npc_id = ((request.get_json(silent=True) or {}).get("npc_id") or "").strip()
    _, error = get_npc_or_error(npc_id)
    if error:
        return error
    history[npc_id] = []
    return jsonify({"message": f"History for {npc_id} reset"})


@app.route("/unlock_state", methods=["GET"])
def unlock_state():
    return jsonify({
        npc_id: {
            "locked": sorted(locked_now(npc_id, npc)),
            "unlocked": sorted(unlocked.get(npc_id, set())),
        }
        for npc_id, npc in npcs.items()
    })


@app.route("/reset_unlocks", methods=["POST"])
def reset_unlocks():
    """ล็อกหัวข้อกลับ ใช้ตอนเริ่มเกมใหม่ ไม่ระบุ npc_id = ทุกตัวละคร"""
    npc_id = ((request.get_json(silent=True) or {}).get("npc_id") or "").strip()
    if npc_id:
        _, error = get_npc_or_error(npc_id)
        if error:
            return error
        unlocked[npc_id] = set()
    else:
        unlocked.clear()
    return jsonify({"message": "locked again", "npc_id": npc_id or "all"})


@app.route("/grade", methods=["POST"])
def grade():
    """ตรวจคำตอบของผู้เล่น ตามเกณฑ์ใน documents/<npc_id>/case_report.json

    รับ {"npc_id": "npc_example", "answers": {"<id ของข้อ>": "...", ...}}
    answers ส่งเป็นลิสต์เรียงตามลำดับข้อก็ได้
    """
    data = request.get_json(silent=True) or {}
    npc_id = (data.get("npc_id") or "").strip()
    npc, error = get_npc_or_error(npc_id)
    if error:
        return error

    path = DOCS_DIR / npc_id / "case_report.json"
    if not path.exists():
        return jsonify({"error": f"ไม่พบ documents/{npc_id}/case_report.json"}), 404
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except ValueError as e:
        return jsonify({"error": f"อ่าน case_report.json ไม่ได้: {e}"}), 400

    questions = report.get("questions") or []
    answers = data.get("answers")
    if isinstance(answers, list):
        answers = {q.get("id"): answers[i] for i, q in enumerate(questions) if i < len(answers)}
    if not isinstance(answers, dict):
        return jsonify({"error": "Field 'answers' must be an object or a list"}), 400

    results = []
    for q in questions:
        given = answers.get(q.get("id"), "")
        # ใช้ num_ctx เท่ากับตอนสนทนา ไม่อย่างนั้น Ollama จะโหลดโมเดลใหม่
        verdict = grade_answer(q, given, model=npc.get("model"),
                               num_ctx=int(npc.get("num_ctx", 4096)),
                               keep_alive=npc.get("keep_alive"))
        results.append({"id": q.get("id"), "question": q.get("text"), "answer": given, **verdict})

    score = sum(1 for r in results if r["correct"])
    pass_mark = int(report.get("pass_mark", len(questions)))
    log.info("GRADE %s | %d/%d", npc_id, score, len(questions))
    return jsonify({
        "npc_id": npc_id,
        "score": score,
        "total": len(questions),
        "pass_mark": pass_mark,
        "passed": score >= pass_mark,
        "results": results,
    })


@app.route("/health", methods=["GET"])
def health():
    try:
        requests.get(f"{cfg.OLLAMA_URL}/api/tags", timeout=3).raise_for_status()
        ollama_ok = True
    except requests.RequestException:
        ollama_ok = False
    return jsonify({
        "status": "healthy" if ollama_ok else "ollama_unreachable",
        "ollama_url": cfg.OLLAMA_URL,
        "embed_model": cfg.EMBED_MODEL,
        "npc_models": {k: v.get("model") or cfg.CHAT_MODEL for k, v in npcs.items()},
        "timestamp": datetime.now().isoformat(),
    })


# ---------------------------------------------------------------------------

def warmup():
    """สร้างดัชนีและโหลดโมเดลไว้ก่อน คำถามแรกจากเกมจะได้ไม่ช้า"""
    time.sleep(1)
    for npc_id, npc in npcs.items():
        t0 = time.perf_counter()
        context, info = retrieve_context(
            npc_id, DOCS_DIR / npc_id, "สวัสดี",
            k=int(npc.get("retrieval_k", 3)),
            pinned=npc.get("always_sections"),
            locked=locked_now(npc_id, npc),
        )
        if not info.get("rag") and info.get("reason") not in ("document_smaller_than_k",):
            log.warning("WARMUP %s | ค้นคืนไม่สำเร็จ: %s", npc_id, info.get("reason"))
        result = generate_answer("สวัสดี", npc, context, max_tokens=8)
        took = time.perf_counter() - t0
        if result["fallback"]:
            log.warning("WARMUP %s | โมเดล %s ยังใช้ไม่ได้: %s",
                        npc_id, result["model"], result.get("fallback_reason"))
        else:
            log.info("WARMUP %s | โมเดล %s พร้อม (%.1f วินาที)", npc_id, result["model"], took)
    log.info("WARMUP เสร็จ")


load_npcs()

if __name__ == "__main__":
    log.info("ตัวละคร: %s", ", ".join(npcs) or "(ยังไม่มี สร้างโฟลเดอร์ documents/npc_<ชื่อ>)")
    log.info("Ollama: %s | embedding: %s", cfg.OLLAMA_URL, cfg.EMBED_MODEL)
    if cfg.WARMUP and npcs:
        threading.Thread(target=warmup, daemon=True).start()
    app.run(host=cfg.HOST, port=cfg.PORT, debug=False, use_reloader=False)
