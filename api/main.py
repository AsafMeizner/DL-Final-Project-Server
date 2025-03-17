import os
import io
import numpy as np
import librosa
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import onnxruntime as ort
from PIL import Image

# ---------------------------------------------------------------------
# Load ONNX Models
# ---------------------------------------------------------------------
SENTIMENT_MODEL_PATH = os.environ.get("SENTIMENT_MODEL_PATH", "public/assets/sentiment_model.onnx")
GENRE_MODEL_PATH = os.environ.get("GENRE_MODEL_PATH", "public/assets/genre_model.onnx")

try:
    sentiment_session = ort.InferenceSession(SENTIMENT_MODEL_PATH)
    sentiment_input_name = sentiment_session.get_inputs()[0].name
    sentiment_output_name = sentiment_session.get_outputs()[0].name
    print("Sentiment ONNX model loaded.")
except Exception as e:
    print("Error loading sentiment ONNX model:", e)
    sentiment_session = None

try:
    genre_session = ort.InferenceSession(GENRE_MODEL_PATH)
    genre_input_name = genre_session.get_inputs()[0].name
    genre_output_name = genre_session.get_outputs()[0].name
    print("Genre ONNX model loaded.")
except Exception as e:
    print("Error loading genre ONNX model:", e)
    genre_session = None

# List of genres used during training (adjust as necessary)
GENRE_CLASSES = ['blues', 'classical', 'country', 'disco', 'hiphop',
                 'jazz', 'metal', 'pop', 'reggae', 'rock']

# ---------------------------------------------------------------------
# Pydantic Models for API Responses
# ---------------------------------------------------------------------
class SentimentSegment(BaseModel):
    timeSec: float
    valence: float
    arousal: float
    dominance: float

class AnalysisResult(BaseModel):
    segments: List[SentimentSegment]
    genre: str

# ---------------------------------------------------------------------
# FastAPI App Setup
# ---------------------------------------------------------------------
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------
# Audio Processing Parameters
# ---------------------------------------------------------------------
# For sentiment analysis:
SAMPLE_RATE = 22050
SEGMENT_DURATION = 10.0   # seconds per segment
TARGET_FRAMES = 200       # fixed time dimension for mel-spectrogram
N_MELS = 128
HOP_LENGTH = 512
FMIN = 20
FMAX = 8000

# For genre classification:
GENRE_CHUNK_DURATION = 4.0  # seconds
GENRE_TARGET_SIZE = (150, 150)  # width, height

# ---------------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------------
def process_sentiment(audio: np.ndarray, sr: int) -> List[dict]:
    """
    Splits the audio into segments of SEGMENT_DURATION seconds,
    computes a mel-spectrogram for each segment (fixed to TARGET_FRAMES),
    and predicts sentiment using the ONNX sentiment model.
    Returns a list of dictionaries containing timeSec, valence, arousal, and a dummy dominance.
    """
    segments = []
    total_duration = librosa.get_duration(y=audio, sr=sr)
    num_segments = int(total_duration // SEGMENT_DURATION)
    num_segments = max(num_segments, 1)
    for i in range(num_segments):
        start_sec = i * SEGMENT_DURATION
        end_sec = start_sec + SEGMENT_DURATION
        start_sample = int(start_sec * sr)
        end_sample = int(end_sec * sr)
        audio_segment = audio[start_sample:end_sample]
        # Compute mel-spectrogram
        S = librosa.feature.melspectrogram(y=audio_segment, sr=sr, n_mels=N_MELS,
                                           hop_length=HOP_LENGTH, fmin=FMIN, fmax=FMAX)
        S_db = librosa.power_to_db(S, ref=np.max)
        # Ensure fixed time dimension
        T = S_db.shape[1]
        if T < TARGET_FRAMES:
            pad_width = TARGET_FRAMES - T
            S_db_fixed = np.pad(S_db, ((0, 0), (0, pad_width)), mode='constant')
        else:
            S_db_fixed = S_db[:, :TARGET_FRAMES]
        # Transpose so that shape becomes (TARGET_FRAMES, N_MELS)
        S_db_fixed = S_db_fixed.T
        # Expand dims to get shape (1, TARGET_FRAMES, N_MELS, 1)
        input_tensor = np.expand_dims(np.expand_dims(S_db_fixed, -1), 0).astype(np.float32)
        # Run inference with ONNX runtime
        pred = sentiment_session.run([sentiment_output_name], {sentiment_input_name: input_tensor})
        # Assume model output shape is (1,2): [valence, arousal]
        valence, arousal = float(pred[0][0][0]), float(pred[0][0][1])
        # Set a dummy dominance value (adjust if you have a model for it)
        dominance = 0.5
        segments.append({
            "timeSec": start_sec,
            "valence": valence,
            "arousal": arousal,
            "dominance": dominance
        })
    return segments

def process_genre(audio: np.ndarray, sr: int) -> str:
    """
    Extracts a GENRE_CHUNK_DURATION-second chunk from the middle of the audio,
    computes its mel-spectrogram, resizes it to GENRE_TARGET_SIZE,
    and predicts the genre using the ONNX genre model.
    Returns the predicted genre label.
    """
    total_duration = librosa.get_duration(y=audio, sr=sr)
    if total_duration < GENRE_CHUNK_DURATION:
        start_sec = 0.0
    else:
        start_sec = (total_duration - GENRE_CHUNK_DURATION) / 2.0
    end_sec = start_sec + GENRE_CHUNK_DURATION
    start_sample = int(start_sec * sr)
    end_sample = int(end_sec * sr)
    audio_chunk = audio[start_sample:end_sample]
    # Compute mel-spectrogram for the chunk
    S = librosa.feature.melspectrogram(y=audio_chunk, sr=sr, n_mels=N_MELS,
                                       hop_length=HOP_LENGTH, fmin=FMIN, fmax=FMAX)
    S_db = librosa.power_to_db(S, ref=np.max)
    # Use PIL to resize the spectrogram to GENRE_TARGET_SIZE.
    # Convert S_db to an image in "F" mode (32-bit floating point)
    img = Image.fromarray(S_db, mode="F")
    img_resized = img.resize(GENRE_TARGET_SIZE, resample=Image.BILINEAR)
    # Convert back to numpy array
    S_db_resized = np.array(img_resized)
    # Expand dims to get shape (150, 150, 1) then add batch dimension -> (1, 150, 150, 1)
    input_tensor = np.expand_dims(S_db_resized, axis=(0, -1)).astype(np.float32)
    # Run inference with ONNX runtime
    pred = genre_session.run([genre_output_name], {genre_input_name: input_tensor})
    # Assume output is a vector of class probabilities
    pred_index = int(np.argmax(pred[0], axis=1)[0])
    return GENRE_CLASSES[pred_index]

# ---------------------------------------------------------------------
# FastAPI Endpoints
# ---------------------------------------------------------------------
@app.get("/")
def read_root():
    return {"message": "Audio Analysis API"}

@app.post("/analyze", response_model=AnalysisResult)
async def analyze_audio_endpoint(audio: UploadFile = File(...)):
    # Check file type
    if audio.content_type.split("/")[0] != "audio":
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload an audio file.")
    try:
        contents = await audio.read()
        # Load audio from bytes using librosa
        audio_np, sr = librosa.load(io.BytesIO(contents), sr=SAMPLE_RATE, mono=True)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing audio: {e}")
    
    if sentiment_session is None or genre_session is None:
        raise HTTPException(status_code=500, detail="Model(s) not loaded properly.")
    
    # Process sentiment over segments
    sentiment_segments = process_sentiment(audio_np, sr)
    # Process genre prediction from a chunk
    genre_prediction = process_genre(audio_np, sr)
    
    return AnalysisResult(segments=sentiment_segments, genre=genre_prediction)
