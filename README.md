# 🎙 VoiceIQ

**AI-powered call analysis:** upload a call recording → get a transcript, a 13-metric quality score, and a downloadable PDF report.

> Built in a 5-day sprint (Jul 2026) — actively refactoring it into something a recruiter can actually read. You're watching that happen.

---

## What it does

**Transcription pipeline**
- OpenAI **Whisper** (`base.en`, upgraded from `tiny.en` for 2× accuracy)
- Multilingual support with **Hindi → English auto-translation** (Google Translate API pipeline)
- Devanagari font handling for correct Hindi rendering
- Audio preprocessing before transcription

**Analysis — 13 scoring metrics**
- Sentiment trend + fuzzy sentiment lexicon with context windows
- **BERT sentiment via Transformers.js** (local inference, ~65% → ~92% agreement)
- Zero-shot response-quality scoring
- Interruption detection with impact weighting
- Talk-time ratio, speaker diarization
- **Emotional Echo Score** — a custom 13th metric measuring how the agent's tone tracks the caller's
- FFT-based spectral analysis: centroid, rolloff, bandwidth
- Compliance flagging, call topic tagging, auto-summary

**Reporting & UX**
- Radar chart of all 13 metrics, embedded in the PDF
- Pure-JS PDF generator (no CDN dependency), blob-URL download for mobile
- Transcript playback sync — highlight + scroll while audio plays
- IndexedDB cache for processed recordings, progress bars, XLSX export
- Sample report so you can try it without uploading anything
- Glassmorphism UI, cinematic intro, Web Audio engine (no external audio files)

## Stack

| Layer | Tech |
|---|---|
| Frontend | Vanilla JS single-page app (`index.html`) |
| Backend | Python **FastAPI** (`voiceiq_server.py`) |
| Audio | `librosa`, `soundfile`, `noisereduce` |
| Speech | OpenAI Whisper, AssemblyAI (client option) |
| NLP | Transformers (BERT sentiment), Google Translate |
| Deploy | `Procfile` (Heroku-style), graceful lazy model loading |

## Run it

```bash
pip install -r requirements.txt
uvicorn voiceiq_server:app --host 0.0.0.0 --port 8000
```

Or with the Heroku-style Procfile: `web: uvicorn voiceiq_server:app --host 0.0.0.0 --port $PORT`

## Known limitations (honest list)

- [ ] Frontend is one big `index.html` — module split is in progress
- [ ] No automated tests yet
- [ ] Whisper runs on CPU — long calls take a while

## Roadmap

- [ ] Split JS into modules, add ESLint + a test runner
- [ ] Streaming transcription for live calls
- [ ] Speaker-level analytics dashboard
