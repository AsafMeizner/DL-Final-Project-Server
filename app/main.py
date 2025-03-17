import os
import io
import json
import numpy as np
import librosa
import soundfile as sf
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import List
from PIL import Image

from app.audio_processing import compute_mel_spectrogram, SAMPLE_RATE, N_MELS, HOP_LENGTH, FMIN, FMAX, N_FFT
from app.onnx_models import (
    sentiment_session, sentiment_input_name, sentiment_output_name,
    genre_session, genre_input_name, genre_output_name
)

# Processing parameters
SEGMENT_DURATION = 10.0      # seconds per sentiment segment
TARGET_FRAMES = 200          # fixed number of time frames for sentiment analysis

GENRE_CHUNK_DURATION = 4.0   # seconds for genre analysis
GENRE_TARGET_SIZE = (150, 150)  # target image size for genre analysis

# Genre classes (adjust to match your training)
GENRE_CLASSES = ['blues', 'classical', 'country', 'disco', 'hiphop',
                 'jazz', 'metal', 'pop', 'reggae', 'rock']

app = FastAPI(
    title="Audio Analysis API",
    description="A FastAPI application that analyzes audio using ONNX models and librosa.",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models for responses
class SentimentSegment(BaseModel):
    timeSec: float
    valence: float
    arousal: float
    dominance: float

class AnalysisResult(BaseModel):
    segments: List[SentimentSegment]
    genre: str

@app.get("/")
def read_root():
    return {"message": "Welcome to the Audio Analysis API using librosa and ONNX"}

@app.get("/api-docs", response_class=HTMLResponse)
def api_docs():
    # Render a simple HTML page that shows the OpenAPI JSON
    openapi_json = app.openapi()
    html_content = f"""
    <html>
      <head>
        <title>API Documentation</title>
        <style>
          body {{ font-family: Arial, sans-serif; padding: 20px; }}
          pre {{ background-color: #f4f4f4; padding: 10px; border: 1px solid #ccc; }}
        </style>
      </head>
      <body>
        <h1>API Documentation</h1>
        <pre>{json.dumps(openapi_json, indent=2)}</pre>
      </body>
    </html>
    """
    return HTMLResponse(content=html_content)

def process_sentiment(audio: np.ndarray, sr: int) -> List[dict]:
    segments = []
    total_duration = len(audio) / sr
    num_segments = int(total_duration // SEGMENT_DURATION)
    num_segments = max(num_segments, 1)
    for i in range(num_segments):
        start_sec = i * SEGMENT_DURATION
        end_sec = start_sec + SEGMENT_DURATION
        start_sample = int(start_sec * sr)
        end_sample = int(end_sec * sr)
        audio_segment = audio[start_sample:end_sample]
        S_db = compute_mel_spectrogram(audio_segment, sr)
        # Ensure fixed time dimension
        if S_db.shape[1] < TARGET_FRAMES:
            pad_width = TARGET_FRAMES - S_db.shape[1]
            S_db_fixed = np.pad(S_db, ((0, 0), (0, pad_width)), mode='constant')
        else:
            S_db_fixed = S_db[:, :TARGET_FRAMES]
        S_db_fixed = S_db_fixed.T  # shape: (TARGET_FRAMES, N_MELS)
        input_tensor = np.expand_dims(np.expand_dims(S_db_fixed, -1), 0).astype(np.float32)
        if sentiment_session is None:
            raise HTTPException(status_code=500, detail="Sentiment model not loaded")
        pred = sentiment_session.run([sentiment_output_name], {sentiment_input_name: input_tensor})
        # Assume the model outputs an array of shape (1, 2): [valence, arousal]
        valence, arousal = float(pred[0][0][0]), float(pred[0][0][1])
        dominance = 0.5  # Dummy value; adjust as needed.
        segments.append({
            "timeSec": start_sec,
            "valence": valence,
            "arousal": arousal,
            "dominance": dominance
        })
    return segments

def process_genre(audio: np.ndarray, sr: int) -> str:
    total_duration = len(audio) / sr
    if total_duration < GENRE_CHUNK_DURATION:
        start_sec = 0.0
    else:
        start_sec = (total_duration - GENRE_CHUNK_DURATION) / 2.0
    end_sec = start_sec + GENRE_CHUNK_DURATION
    start_sample = int(start_sec * sr)
    end_sample = int(end_sec * sr)
    audio_chunk = audio[start_sample:end_sample]
    S_db = compute_mel_spectrogram(audio_chunk, sr)
    # Normalize and convert the spectrogram to an image for resizing
    S_norm = (S_db - np.min(S_db)) / (np.max(S_db) - np.min(S_db) + 1e-10) * 255
    S_norm = S_norm.astype(np.uint8)
    img = Image.fromarray(S_norm)
    img_resized = img.resize(GENRE_TARGET_SIZE, resample=Image.BILINEAR)
    S_resized = np.array(img_resized).astype(np.float32)
    input_tensor = np.expand_dims(np.expand_dims(S_resized, -1), 0)
    if genre_session is None:
        raise HTTPException(status_code=500, detail="Genre model not loaded")
    pred = genre_session.run([genre_output_name], {genre_input_name: input_tensor})
    pred_index = int(np.argmax(pred[0], axis=1)[0])
    return GENRE_CLASSES[pred_index]

@app.post("/analyze", response_model=AnalysisResult)
async def analyze_audio_endpoint(audio: UploadFile = File(...)):
    if audio.content_type.split("/")[0] != "audio":
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload an audio file.")
    try:
        contents = await audio.read()
        # Use librosa to load audio from bytes; force mono and sample rate
        audio_np, sr = librosa.load(io.BytesIO(contents), sr=SAMPLE_RATE, mono=True)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing audio: {e}")

    sentiment_segments = process_sentiment(audio_np, sr)
    genre_prediction = process_genre(audio_np, sr)
    return AnalysisResult(segments=sentiment_segments, genre=genre_prediction)
