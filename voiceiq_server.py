"""
VoiceIQ - Full Python Backend Server
Serves the frontend and provides API for transcription, speaker diarization,
emotion detection, and audio processing.
"""
import os, sys, json, uuid, time, shutil, warnings, io
warnings.filterwarnings('ignore')
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# ── Audio Processing ──
import librosa
import soundfile as sf
import numpy as np

# ── Whisper ──
import whisper

# ── HuggingFace Emotion ──
from transformers import pipeline

# ── Noise Reduction ──
import noisereduce as nr

app = FastAPI(title="VoiceIQ API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

TEMP = Path(__file__).parent / "_temp"
TEMP.mkdir(exist_ok=True)

# Lazy-loaded models
_whisper_model = None
_emotion_pipe = None

def get_whisper():
    global _whisper_model
    if _whisper_model is None:
        print("[VoiceIQ] Loading Whisper large model...")
        _whisper_model = whisper.load_model("large")
        print("[VoiceIQ] Whisper loaded.")
    return _whisper_model

def get_emotion():
    global _emotion_pipe
    if _emotion_pipe is None:
        print("[VoiceIQ] Loading emotion model...")
        _emotion_pipe = pipeline(
            "text-classification",
            model="j-hartmann/emotion-english-distilroberta-base",
            top_k=None
        )
        print("[VoiceIQ] Emotion model loaded.")
    return _emotion_pipe

def process_audio(file_path: Path) -> dict:
    """Full pipeline: noise reduction → transcription → speaker → emotion → scoring"""
    print(f"[VoiceIQ] Processing: {file_path.name}")

    # 1. Load & preprocess with librosa
    audio, sr = librosa.load(str(file_path), sr=16000, mono=True)
    print(f"[VoiceIQ] Audio: {len(audio)/sr:.1f}s at {sr}Hz")

    # 2. Noise reduction
    audio_clean = nr.reduce_noise(y=audio, sr=sr, prop_decrease=0.8)
    print("[VoiceIQ] Noise reduction done")

    # 3. Normalize volume
    peak = np.max(np.abs(audio_clean))
    if peak > 0:
        audio_clean = audio_clean / peak * 0.9

    # 4. Transcribe with Whisper (English only)
    model = get_whisper()
    result = model.transcribe(
        audio_clean,
        language="english",
        task="transcribe",
        verbose=False,
        word_timestamps=True
    )
    print(f"[VoiceIQ] Transcription done: {len(result.get('segments', []))} segments")

    # 5. Build utterances from whisper segments
    utterances = []
    for seg in result.get("segments", []):
        utterances.append({
            "speaker": "A",
            "text": seg["text"].strip(),
            "start": int(seg["start"] * 1000),
            "end": int(seg["end"] * 1000),
            "sentiment": "NEUTRAL",
            "confidence": seg.get("confidence", 0.9)
        })

    if not utterances:
        # Fallback: use full text
        utterances.append({
            "speaker": "A",
            "text": result["text"].strip(),
            "start": 0,
            "end": int(len(audio_clean) / sr * 1000),
            "sentiment": "NEUTRAL",
            "confidence": 0.8
        })

    # 6. Speaker diarization (simple heuristic based on gaps and pauses)
    utterances = assign_speakers(utterances)

    # 7. Emotion detection per utterance
    pipe = get_emotion()
    for utt in utterances:
        if utt["text"] and len(utt["text"]) > 3:
            try:
                emotions = pipe(utt["text"])[0]
                top = max(emotions, key=lambda e: e["score"])
                utt["sentiment"] = map_emotion_to_sentiment(top["label"])
                utt["emotion"] = top["label"]
                utt["emotion_score"] = round(top["score"], 3)
            except:
                utt["sentiment"] = "NEUTRAL"
                utt["emotion"] = "neutral"
                utt["emotion_score"] = 0.5
        else:
            utt["sentiment"] = "NEUTRAL"
            utt["emotion"] = "neutral"
            utt["emotion_score"] = 0.5

    print(f"[VoiceIQ] Emotion analysis done")

    # 8. Find negative moments
    negative_moments = find_negative_moments(utterances)

    # 9. Calculate scores
    scores = calculate_scores(utterances, negative_moments)

    # 10. Build call data
    total_ms = utterances[-1]["end"] if utterances else 60000
    agent_words = sum(len(u["text"].split()) for u in utterances if u["speaker"] == "A")
    cust_words = sum(len(u["text"].split()) for u in utterances if u["speaker"] == "B")

    call_data = {
        "duration": f"{total_ms//60000}:{(total_ms//1000)%60:02d}",
        "totalSecs": total_ms / 1000,
        "totalUtts": len(utterances),
        "agentUtts": sum(1 for u in utterances if u["speaker"] == "A"),
        "custUtts": sum(1 for u in utterances if u["speaker"] == "B"),
        "agentWords": agent_words,
        "custWords": cust_words,
        "totalWords": agent_words + cust_words,
        "startEmotion": utterances[0]["sentiment"] if utterances else "NEUTRAL",
        "endEmotion": utterances[-1]["sentiment"] if utterances else "NEUTRAL",
        "stress": "Low Stress" if scores["overall"] >= 70 else "Moderate" if scores["overall"] >= 55 else "High Stress",
        "outcome": "Positive Resolution" if scores["overall"] >= 70 else "Neutral" if scores["overall"] >= 55 else "Escalation Needed",
        "negCount": sum(1 for u in utterances if u["sentiment"] == "NEGATIVE"),
        "posCount": sum(1 for u in utterances if u["sentiment"] == "POSITIVE"),
        "avgResponseTime": calc_avg_response(utterances),
        "fillerWords": count_fillers(utterances)
    }

    return {
        "utterances": utterances,
        "negativeMoments": negative_moments,
        "scores": scores,
        "callData": call_data
    }

def assign_speakers(utterances):
    """Simple speaker diarization: alternate speakers based on gaps."""
    if not utterances:
        return utterances
    # First utterance is Agent
    utterances[0]["speaker"] = "A"
    last_speaker = "A"
    for i in range(1, len(utterances)):
        gap = utterances[i]["start"] - utterances[i-1]["end"]
        # If gap is short, same speaker; if long, swap
        if gap < 1500:
            utterances[i]["speaker"] = last_speaker
        else:
            last_speaker = "B" if last_speaker == "A" else "A"
            utterances[i]["speaker"] = last_speaker
    return utterances

def map_emotion_to_sentiment(emotion):
    positive = {"joy", "surprise"}
    negative = {"anger", "frustration", "disgust", "fear", "sadness", "sad"}
    if emotion.lower() in positive:
        return "POSITIVE"
    elif emotion.lower() in negative:
        return "NEGATIVE"
    return "NEUTRAL"

def find_negative_moments(utterances):
    moments = []
    for i, u in enumerate(utterances):
        if u["speaker"] == "B" and u["sentiment"] == "NEGATIVE":
            agent_reply = None
            for j in range(i + 1, len(utterances)):
                if utterances[j]["speaker"] == "A":
                    agent_reply = utterances[j]
                    break
            moments.append({
                "customerText": u["text"],
                "customerStart": u["start"],
                "customerEnd": u["end"],
                "agentText": agent_reply["text"] if agent_reply else "(No response)",
                "agentStart": agent_reply["start"] if agent_reply else u["end"],
                "agentEnd": agent_reply["end"] if agent_reply else u["end"],
                "responseQuality": rate_response(agent_reply["text"] if agent_reply else ""),
                "customerSentiment": "NEGATIVE",
                "customerEmotion": u.get("emotion", "anger")
            })
    return moments

def rate_response(text):
    if not text or len(text.strip()) < 3:
        return "FAILED"
    t = text.lower()
    score = 0
    empathy = ["sorry","apologize","understand","frustrat","appreciate","thank","help","resolve","fix","assist","concern","patience","empathize","regret","inconvenience"]
    action = ["let me","i will","going to","right away","happy to","glad to","definitely","immediately","as soon","personally","escalated","confirmed","i'll"]
    for w in empathy:
        if w in t:
            score += 2
    for w in action:
        if w in t:
            score += 1.8
    if len(text) > 25:
        score += 1
    if len(text) > 50:
        score += 0.5
    if score >= 7:
        return "EXCELLENT"
    if score >= 3.5:
        return "GOOD"
    if score >= 1:
        return "POOR"
    return "FAILED"

def calculate_scores(utterances, moments):
    if not utterances:
        return {"empathy":50,"tone":50,"resolution":50,"negativity":50,"overall":50,"grade":"C"}
    ec, et = 0, 0
    for i in range(len(utterances) - 1):
        if utterances[i]["speaker"] == "B" and utterances[i]["sentiment"] == "NEGATIVE":
            et += 1
            n = utterances[i + 1]
            if n["speaker"] == "A" and n["sentiment"] in ("POSITIVE", "NEUTRAL"):
                ec += 1
    empathy = (ec / et * 100) if et > 0 else 75
    agent_u = [u for u in utterances if u["speaker"] == "A"]
    pos_a = sum(1 for u in agent_u if u["sentiment"] == "POSITIVE")
    tone = (pos_a / len(agent_u) * 100) if agent_u else 50
    last_cust = None
    for u in reversed(utterances):
        if u["speaker"] == "B":
            last_cust = u
            break
    resolution = 100 if last_cust and last_cust["sentiment"] == "POSITIVE" else 60 if last_cust and last_cust["sentiment"] == "NEUTRAL" else 25
    good_h = sum(1 for m in moments if m["responseQuality"] in ("EXCELLENT","GOOD"))
    negativity = (good_h / len(moments) * 100) if moments else 100
    overall = round(empathy * 0.30 + tone * 0.25 + resolution * 0.25 + negativity * 0.20)
    grade = "A" if overall > 90 else "B" if overall > 80 else "C" if overall > 70 else "D" if overall > 55 else "F"
    return {"empathy":round(empathy),"tone":round(tone),"resolution":round(resolution),"negativity":round(negativity),"overall":overall,"grade":grade}

def calc_avg_response(utterances):
    total, count = 0, 0
    for i in range(len(utterances) - 1):
        if utterances[i]["speaker"] == "B" and utterances[i + 1]["speaker"] == "A":
            total += (utterances[i + 1]["start"] or 0) - (utterances[i]["end"] or 0)
            count += 1
    return round(total / count / 1000) if count > 0 else 2

def count_fillers(utterances):
    fw = ["um","uh","like","you know","actually","basically","literally","sort of","kind of","i mean"]
    count = 0
    for u in utterances:
        tl = u["text"].lower()
        for f in fw:
            import re
            count += len(re.findall(r'\b' + re.escape(f) + r'\b', tl))
    return count

# ── API Routes ──

@app.get("/api/status")
def status():
    return {"status": "running", "models": ["whisper-large", "emotion-distilroberta"]}

@app.post("/api/transcribe")
async def transcribe(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(400, "No file")
    ext = Path(file.filename).suffix.lower()
    if ext not in (".mp3", ".wav", ".m4a", ".ogg", ".flac"):
        raise HTTPException(400, "Unsupported format. Use MP3, WAV, or M4A")

    temp_path = TEMP / f"{uuid.uuid4().hex}{ext}"
    try:
        content = await file.read()
        temp_path.write_bytes(content)

        result = process_audio(temp_path)
        return JSONResponse(result)
    except Exception as e:
        print(f"[VoiceIQ] ERROR: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(500, str(e))
    finally:
        if temp_path.exists():
            temp_path.unlink()

@app.get("/", response_class=HTMLResponse)
def serve_frontend():
    html_path = Path(__file__).parent / "index.html"
    if html_path.exists():
        return HTMLResponse(html_path.read_text(encoding="utf-8"))
    return HTMLResponse("<h1>VoiceIQ</h1><p>index.html not found. Place it next to this server.</p>")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", sys.argv[1] if len(sys.argv) > 1 else 8080))
    print(f"\n{'='*50}")
    print(f"  VoiceIQ Server running on port {port}")
    print(f"{'='*50}\n")
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
