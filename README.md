# Naya Mitra AI — Flask Web App

## Architecture

| Purpose | Provider |
|---|---|
| 💬 Chat / Wisdom | **Gemini** `gemini-1.5-flash` (REST API) |
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

# 3. Run — Gemini key is already in .env
python app.py
# Open http://localhost:5000
```

## Production

```bash
gunicorn -c gunicorn.conf.py app:app
```

## Project Structure

```
naya-mitra-v2/
├── app.py              # Flask backend — Gemini chat + session + logs
├── gunicorn.conf.py    # Production server config
├── requirements.txt    # flask, httpx, python-dotenv, gunicorn
├── .env                # GEMINI_API_KEY lives here
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
