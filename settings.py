"""ค่าตั้งของระบบ

แก้ค่าในไฟล์นี้ได้เลย หรือจะตั้งผ่าน environment variable ชื่อเดียวกันก็ได้
ค่าเฉพาะตัวละคร (โมเดล, system prompt, temperature ฯลฯ) อยู่ใน documents/<npc_id>/meta.json
"""
import os


def _flag(name, default):
    return os.getenv(name, default).strip().lower() not in ("0", "false", "no", "")


# ---- Ollama ----
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434").rstrip("/")
EMBED_URL = os.getenv("EMBED_URL", OLLAMA_URL).rstrip("/")   # แยกเครื่อง embedding ได้

# โมเดลตั้งต้น ใช้เมื่อ meta.json ของตัวละครไม่ได้ระบุ "model"
CHAT_MODEL = os.getenv("OLLAMA_MODEL", "scb10x/typhoon2.5-qwen3-4b:latest")
EMBED_MODEL = os.getenv("EMBED_MODEL", "bge-m3")

CHAT_TIMEOUT = int(os.getenv("OLLAMA_CHAT_TIMEOUT", "75"))    # วินาที ควรน้อยกว่า timeout ฝั่งเกม
EMBED_TIMEOUT = int(os.getenv("OLLAMA_EMBED_TIMEOUT", "120"))
KEEP_ALIVE = os.getenv("OLLAMA_KEEP_ALIVE", "30m")            # โมเดลค้างในหน่วยความจำนานแค่ไหน
EMBED_KEEP_ALIVE = os.getenv("EMBED_KEEP_ALIVE", "4h")

# รันโมเดล embedding บน CPU เพื่อเก็บ VRAM ไว้ให้โมเดลภาษา (และตัวเกม)
EMBED_ON_CPU = _flag("EMBED_ON_CPU", "1")
EMBED_NUM_THREAD = int(os.getenv("EMBED_NUM_THREAD", "8"))

# จำนวนชั้นของโมเดลภาษาที่วางบน GPU  เว้นว่าง = ให้ Ollama เลือกเอง, 99 = ทั้งหมด, 0 = CPU ล้วน
_num_gpu = os.getenv("OLLAMA_NUM_GPU", "").strip()
NUM_GPU = int(_num_gpu) if _num_gpu else None

# ---- เซิร์ฟเวอร์ ----
HOST = os.getenv("API_HOST", "0.0.0.0")
PORT = int(os.getenv("API_PORT", "5005"))

# โหลดโมเดลและสร้างดัชนีไว้ตั้งแต่เปิดเซิร์ฟเวอร์ คำถามแรกจะได้ไม่ช้า
WARMUP = _flag("API_WARMUP", "1")

# ---- ข้อความสำรอง ----
NO_ANSWER_TEXT = "ขออภัย ฉันไม่มีข้อมูลในเรื่องนี้"
FALLBACK_ANSWER_TEXT = "ขออภัย ระบบประมวลผลขัดข้องชั่วคราว กรุณาถามใหม่อีกครั้ง"
