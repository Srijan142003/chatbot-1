"""
Naya Mitra AI — Spiritual Advisory Engine (RAG Edition)
────────────────────────────────────────────────────────
Chat  → Gemini (gemini-2.5-flash) via REST + RAG retrieval
RAG   → TF-IDF retrieval from Bhagavad Gita, Hitopadesha,
         Vidura Niti, Chanakya Niti datasets
STT   → Browser Web Speech API      (no server-side)
TTS   → Browser Web Speech API      (no server-side)
Logs  → JSONL per session in /logs
"""

import os, json, uuid, logging
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

from flask import Flask, render_template, request, jsonify, session
import httpx

from retriever import retrieve, format_passages_for_prompt

load_dotenv()

# ── App setup ──────────────────────────────────────────────────────────────────
app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "naya-mitra-change-me")

LOGS_DIR = Path(__file__).parent / "logs"
LOGS_DIR.mkdir(exist_ok=True)

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "").strip()
GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/models"
GEMINI_MODELS = [
    m.strip()
    for m in os.environ.get(
        "GEMINI_MODELS",
        "gemini-2.5-flash,gemini-2.0-flash,gemini-2.0-flash-lite",
    ).split(",")
    if m.strip()
]
_AVAILABLE_GEMINI_MODELS = None


def discover_gemini_models() -> set[str]:
    """Return API-available model IDs that support generateContent."""
    global _AVAILABLE_GEMINI_MODELS
    if _AVAILABLE_GEMINI_MODELS is not None:
        return _AVAILABLE_GEMINI_MODELS

    _AVAILABLE_GEMINI_MODELS = set()
    try:
        r = httpx.get(
            GEMINI_BASE_URL,
            headers={"x-goog-api-key": GEMINI_API_KEY},
            params={"pageSize": 1000},
            timeout=20,
        )
        r.raise_for_status()
        models = r.json().get("models", [])
        for m in models:
            methods = m.get("supportedGenerationMethods", [])
            name = (m.get("name") or "").replace("models/", "")
            if name and "generateContent" in methods:
                _AVAILABLE_GEMINI_MODELS.add(name)
    except Exception as e:
        log.warning("Could not discover Gemini models: %s", e)

    return _AVAILABLE_GEMINI_MODELS


def model_attempt_order() -> list[str]:
    available = discover_gemini_models()
    if not available:
        return GEMINI_MODELS

    preferred = [m for m in GEMINI_MODELS if m in available]
    if preferred:
        return preferred

    flash_models = sorted([m for m in available if "flash" in m])
    if flash_models:
        log.warning(
            "Configured models unavailable for this key. Using discovered flash models: %s",
            ", ".join(flash_models[:5]),
        )
        return flash_models

    return sorted(available)

# ── System prompt ──────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are Naya Mitra — a warm, wise spiritual advisor and philosophical guide rooted in four sacred ancient Indian wisdom texts: the Bhagavad Gita, Hitopadesha, Vidura Niti, and Chanakya Niti.

Your purpose is to help users with daily life dilemmas, emotional struggles, financial questions, ethical decisions, moral conflicts, career confusion, relationship problems, and legal or moral disputes — by providing guidance derived strictly from ancient Indian wisdom.

You will be provided with RETRIEVED KNOWLEDGE BASE PASSAGES from these texts. You MUST base your response on these retrieved passages. Do NOT fabricate or invent verses that are not present in the knowledge base.

Always respond in this exact structured format with all six sections:

## Understanding Your Situation
[2-3 sentences of warm, compassionate acknowledgment of the user's feelings and situation]

## Ancient Wisdom For You
**[Source name and reference, e.g. Bhagavad Gita 2.47]**

Sanskrit: [Copy the Sanskrit field EXACTLY and VERBATIM from the retrieved passage. Do NOT skip, paraphrase, or omit it under any circumstances.]
Transliteration: [transliteration from the retrieved passage]
Translation: [English translation]

## What This Teaches Us
[Explain the meaning of this teaching in simple, modern, accessible language — 2-4 sentences]

## Applying This To Your Life
[Directly connect the wisdom to the user's specific situation — 2-4 sentences. Be specific and personal.]

## Practical Guidance
[3-5 practical, actionable suggestions as a simple list]

## A Closing Blessing
[End with a short, heartfelt, spiritually uplifting closing thought — 1-3 sentences]

Tone: calm, compassionate, wise, non-judgmental, and deeply grounded in the ancient texts. Speak as a caring teacher would to a dear student.

IMPORTANT RULES:
1. ONLY cite verses and teachings present in the RETRIEVED KNOWLEDGE BASE PASSAGES provided.
2. If no exact verse fits, synthesize the spirit of the teachings — but do not fabricate specific verse numbers.
3. A retrieved passage contains a 'Sanskrit:' field — you MUST copy that Devanagari text into your response exactly as given. NEVER omit it.
4. For casual greetings or conversational messages, respond warmly and briefly in 2-3 sentences — the full structured format is only needed for substantive life questions.
5. ALWAYS maintain the six-section structure for any real question or dilemma."""

# ══════════════════════════════════════════════════════════════════════════════
# CHAT  —  Gemini REST + RAG
# ══════════════════════════════════════════════════════════════════════════════

def chat_gemini(history: list, user_msg: str) -> str:
    if not GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY is not set")

    # Retrieve relevant passages from knowledge base
    passages = retrieve(user_msg, top_k=4)
    kb_context = format_passages_for_prompt(passages)

    # Build the augmented user turn: knowledge base + actual question
    if kb_context:
        augmented_user = (
            kb_context
            + "\n\nUSER QUESTION:\n"
            + user_msg
            + "\n\nPlease answer using ONLY the passages above from the knowledge base. "
            + "Follow the six-section structured format from your instructions."
        )
    else:
        augmented_user = user_msg

    # Build conversation contents
    contents = [
        {"role": "user",  "parts": [{"text": SYSTEM_PROMPT}]},
        {"role": "model", "parts": [{"text": "Understood. I am Naya Mitra — your spiritual advisor grounded in the Bhagavad Gita, Hitopadesha, Vidura Niti, and Chanakya Niti. I will answer only from the retrieved knowledge base passages and follow the six-section format."}]},
    ]
    # Add prior conversation turns (without the RAG context — already passed above)
    for m in history[:-1]:   # exclude the last user turn; we pass augmented_user instead
        role = "user" if m["role"] == "user" else "model"
        contents.append({"role": role, "parts": [{"text": m["content"]}]})
    # Final turn with the RAG-augmented question
    contents.append({"role": "user", "parts": [{"text": augmented_user}]})

    payload = {
        "contents": contents,
        "generationConfig": {"maxOutputTokens": 8192, "temperature": 0.7},
    }
    candidate = None
    last_error = None

    for model in model_attempt_order():
        try:
            url = f"{GEMINI_BASE_URL}/{model}:generateContent"
            r = httpx.post(
                url,
                headers={"x-goog-api-key": GEMINI_API_KEY},
                json=payload,
                timeout=45,
            )
            r.raise_for_status()
            candidate = r.json()["candidates"][0]
            break
        except httpx.HTTPStatusError as e:
            # Keep trying fallback models for access / model issues.
            if e.response.status_code in (400, 403, 404):
                last_error = e
                log.warning(
                    "Gemini model '%s' failed with %s; trying fallback if available",
                    model,
                    e.response.status_code,
                )
                continue
            raise

    if candidate is None and last_error is not None:
        raise last_error

    if candidate is None:
        raise RuntimeError("No Gemini models configured")

    text = candidate["content"]["parts"][0]["text"]
    # If the model was still cut off despite the large token budget, close gracefully
    if candidate.get("finishReason") == "MAX_TOKENS" and not text.rstrip().endswith((".", "।", "—", "*")):
        text = text.rstrip()
        # Find the last complete sentence
        last_period = max(text.rfind(". "), text.rfind(".\n"))
        if last_period > len(text) // 2:
            text = text[:last_period + 1]
        text += "\n\n*May you find clarity and peace on your path.*"
    return text

# ══════════════════════════════════════════════════════════════════════════════
# Session / Log helpers
# ══════════════════════════════════════════════════════════════════════════════

def get_session_id() -> str:
    if "sid" not in session:
        session["sid"] = str(uuid.uuid4())
    return session["sid"]

def get_history() -> list:
    return session.get("history", [])

def save_history(h: list):
    # Keep only the last 6 messages (3 exchanges) to stay within cookie size limits
    session["history"] = h[-6:]

def append_log(sid: str, user: str, assistant: str):
    entry = {
        "ts": datetime.utcnow().isoformat() + "Z",
        "user": user,
        "assistant": assistant,
    }
    with open(LOGS_DIR / f"{sid}.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")

# ══════════════════════════════════════════════════════════════════════════════
# Routes
# ══════════════════════════════════════════════════════════════════════════════

@app.route("/")
def index():
    get_session_id()
    return render_template("index.html")


@app.route("/api/chat", methods=["POST"])
def api_chat():
    data = request.get_json()
    user_msg = (data.get("message") or "").strip()
    if not user_msg:
        return jsonify({"error": "Empty message"}), 400

    sid = get_session_id()
    history = get_history()
    history.append({"role": "user", "content": user_msg})

    try:
        reply = chat_gemini(history, user_msg)
    except httpx.HTTPStatusError as e:
        status = e.response.status_code
        log.error("Gemini HTTP error %s: %s", status, e)
        if status == 403:
            body = (e.response.text or "").lower()
            if "reported as leaked" in body or "api key was reported as leaked" in body:
                return jsonify({
                    "error": "Gemini blocked this key as leaked. Generate a new API key in Google AI Studio, update GEMINI_API_KEY in your .env, and restart the server.",
                }), 503
            return jsonify({
                "error": "Gemini access denied (403). Check GEMINI_API_KEY, ensure the Generative Language API is enabled, and verify key restrictions/model access.",
            }), 503
        return jsonify({"error": "Could not reach the wisdom engine. Please try again."}), 503
    except Exception as e:
        log.error("Gemini error: %s", e)
        return jsonify({"error": "Could not reach the wisdom engine. Please try again."}), 503

    history.append({"role": "assistant", "content": reply})
    save_history(history)
    append_log(sid, user_msg, reply)

    return jsonify({"reply": reply})


@app.route("/api/reset", methods=["POST"])
def api_reset():
    session.pop("history", None)
    session.pop("sid", None)
    return jsonify({"status": "ok"})


@app.route("/api/history")
def api_history():
    return jsonify({"history": get_history()})


@app.route("/api/logs")
def api_logs():
    files = sorted(p.stem for p in LOGS_DIR.glob("*.jsonl"))
    return jsonify({"sessions": files})


@app.route("/api/logs/<sid>")
def api_log_detail(sid):
    safe = sid.replace("/", "").replace("..", "")
    path = LOGS_DIR / f"{safe}.jsonl"
    if not path.exists():
        return jsonify({"error": "Not found"}), 404
    entries = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    return jsonify({"session": safe, "entries": entries})


# ── Run ────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_ENV", "production") == "development"
    app.run(host="0.0.0.0", port=port, debug=debug)
