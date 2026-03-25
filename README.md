# Dharma Upadeshak — Flask Web App

## Architecture

| Purpose | Provider |
|---|---|
| 💬 Chat / Wisdom | **Gemini** `gemini-2.5-flash` (REST API) |
| 🎙️ Voice Input (STT) | **Browser** `SpeechRecognition` API — free, no key needed |
| 🔊 Voice Output (TTS) | **Browser** `SpeechSynthesis` API — free, no key needed |
| 💾 Chat Logs | Server-side JSONL in `/logs` |

---

## Quick Start

```bash
# 1. Create virtual environment
python -m venv venv
source venv/bin/activate       # Windows: venv\Scripts\activate

# 2. Install dependencies  (only 4 packages!)
pip install -r requirements.txt

# 3. Set your Gemini key
export GEMINI_API_KEY="your_api_key_here"

# Optional: model preference + fallback order
export GEMINI_MODELS="gemini-2.5-flash,gemini-2.0-flash,gemini-2.0-flash-lite"

# 4. Run
python app.py
# Open http://localhost:5000
```

## Production

```bash
gunicorn -c gunicorn.conf.py app:app
```

## Project Structure

```
chatbot-1/
├── app.py              # Flask backend — Gemini chat + session + logs
├── gunicorn.conf.py    # Production server config
├── requirements.txt    # flask, httpx, python-dotenv, gunicorn
├── .env                # Optional local env file (auto-loaded by python-dotenv)
├── logs/               # Auto-created — JSONL chat logs per session
├── templates/
│   └── index.html
└── static/js/
    └── app.js          # Chat UI, browser STT, browser TTS, download
```

## API Endpoints

| Method | Path | Description |
|---|---|---|
| GET  | `/` | Chat UI |
| POST | `/api/chat` | `{"message": "..."}` → `{"reply": "..."}` |
| POST | `/api/reset` | Clear session history |
| GET  | `/api/history` | Return session history |
| GET  | `/api/logs` | List saved session IDs |
| GET  | `/api/logs/<sid>` | Full JSONL log for a session |

## Voice Notes

- **STT**: Works in Chrome, Edge, and Safari. Click 🎙️, speak, it auto-sends.
- **TTS**: Reads every AI reply aloud. Uses `en-IN` locale for Indian English accent. Toggle with the 🔊 button.
- Both are entirely client-side — zero cost, zero API calls.
