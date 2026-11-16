"""ตรวจคำตอบของผู้เล่นด้วยโมเดลภาษา

เกณฑ์ของแต่ละข้ออยู่ใน documents/<npc_id>/case_report.json
"""
import settings as cfg
from rag_chain import chat, strip_think

GRADER_PROMPT = (
    "You are a grader for a story game. You judge one answer at a time.\n"
    "You are not the player and you are not a character. You only grade.\n\n"
    "The question is not \"did the player phrase it well?\" but \"does the player know the fact that "
    "answers this question?\"\n\n"
    "Follow these steps in order:\n"
    "1. Take the REQUIRED POINTS one by one. Does the player's answer point to that fact?\n"
    "2. ACCEPTABLE VARIATIONS show the range that counts. They are examples, not a closed list — "
    "anything equivalent in meaning counts too.\n"
    "3. Every required point pointed to → CORRECT.\n"
    "4. A required point genuinely absent, or contradicted → WRONG. Name what is missing or wrong.\n\n"
    "Judge by agreement with the facts of the case and by what the answer means, never by keyword "
    "matching. The player chooses their own words, language, sentence shape and level of detail.\n"
    "A short answer that points straight at the right fact is CORRECT. Never require the player to "
    "spell out every detail the reference answer happens to contain.\n"
    "When the answer carries extra detail, check only whether that detail contradicts the case. "
    "Never mark an answer wrong merely for leaving something out that the reference mentions.\n"
    "Misspellings, casual speech, abbreviations and near-synonyms are fine as long as the meaning is clear.\n"
    "Invent no requirement that the criteria and the story do not state.\n"
    "The question ends with a parenthesis telling the player what to supply. That parenthesis is a "
    "promise: ask for exactly what it asks for and nothing more. If it asks for a name, a name alone "
    "is a complete answer. If it asks what was used, naming that thing alone is complete.\n"
    "When you are torn, favour the player.\n\n"
    "About ALWAYS WRONG: each entry applies only when its whole condition holds. "
    "If an entry says \"without mentioning X\" and the player did mention X, that entry "
    "does NOT apply. Never reject an answer just because it contains a word that appears "
    "in that list — read the condition.\n\n"
    "Judge meaning, not wording. Thai spelling mistakes, missing dates, missing codes, "
    "and very short phrasing are all fine.\n"
    "An answer that is extra as well as correct is still correct.\n\n"
    "Reply in exactly this format and nothing else:\n"
    "VERDICT: correct\n"
    "REASON: <one short sentence in Thai>\n"
    "or\n"
    "VERDICT: wrong\n"
    "REASON: <one short sentence in Thai saying which required point is missing>"
)


def _bullets(items) -> str:
    return "\n".join(f"- {x}" for x in items) if items else "- (ไม่มี)"


def grade_answer(question: dict, player_answer: str, model: str = None,
                 num_ctx: int = 4096, keep_alive: str = None) -> dict:
    """ตรวจทีละข้อ คืน {"correct": bool, "reason": str}"""
    answer = (player_answer or "").strip()
    if not answer:
        return {"correct": False, "reason": "ไม่ได้ตอบข้อนี้"}

    text = question.get("text", "")
    hint = (question.get("hint") or "").strip()
    if hint:
        text += f" ({hint})"

    user = (
        f"QUESTION\n{text}\n\n"
        f"REFERENCE ANSWER\n{question.get('answer', '')}\n\n"
        f"REQUIRED POINTS (all must be present)\n{_bullets(question.get('key_facts'))}\n\n"
        f"ALWAYS WRONG (each applies only when its full condition holds)\n{_bullets(question.get('reject'))}\n\n"
        f"ACCEPTABLE VARIATIONS (these still count as correct)\n{_bullets(question.get('accept'))}\n\n"
        f"PLAYER'S ANSWER\n{answer}"
    )
    messages = [{"role": "system", "content": GRADER_PROMPT},
                {"role": "user", "content": user}]
    # temperature 0 ให้คำตอบเดิมได้ผลตัดสินเดิมทุกครั้ง
    options = {"temperature": 0.0, "num_predict": 80, "num_ctx": int(num_ctx)}

    try:
        data = chat(model or cfg.CHAT_MODEL, messages, options, keep_alive)
        raw = strip_think((data.get("message") or {}).get("content", "")).strip()
    except Exception as e:
        return {"correct": False, "reason": f"ระบบตรวจคำตอบขัดข้อง ({type(e).__name__})"}

    verdict, reason = None, ""
    for line in raw.splitlines():
        s = line.strip()
        low = s.lower()
        if low.startswith("verdict:"):
            verdict = "correct" in low
        elif low.startswith("reason:"):
            reason = s.split(":", 1)[1].strip()

    if verdict is None:       # โมเดลไม่ตอบตามรูปแบบ ดูจากคำที่มีในคำตอบแทน
        low = raw.lower()
        verdict = ("correct" in low or "ถูก" in raw) and "wrong" not in low

    if not reason:
        reason = "ตรงกับคำตอบอ้างอิง" if verdict else "ยังไม่ตรงกับคำตอบอ้างอิง"
    return {"correct": bool(verdict), "reason": reason}
