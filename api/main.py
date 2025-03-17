import os
import io
import numpy as np
import scipy.signal
import soundfile as sf
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import onnxruntime as ort
from PIL import Image

# ---------------------------------------------------------------------
# ONNX Model Loading
# ---------------------------------------------------------------------
SENTIMENT_MODEL_PATH = os.environ.get("SENTIMENT_MODEL_PATH", "models/sentiment_model.onnx")
GENRE_MODEL_PATH = os.environ.get("GENRE_MODEL_PATH", "models/genre_model.onnx")

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
# For both sentiment and genre predictions
SAMPLE_RATE = 22050
N_MELS = 128
HOP_LENGTH = 512
FMIN = 20
FMAX = 8000
N_FFT = 2048

# For sentiment prediction
SEGMENT_DURATION = 10.0   # seconds per segment
TARGET_FRAMES = 200       # fixed time dimension for mel spectrogram

# For genre classification
GENRE_CHUNK_DURATION = 4.0  # seconds
GENRE_TARGET_SIZE = (150, 150)  # (width, height)

# ---------------------------------------------------------------------
# Lightweight Mel Spectrogram Utilities
# ---------------------------------------------------------------------
def hz_to_mel(f):
    return 2595 * np.log10(1 + f / 700.0)

def mel_to_hz(m):
    return 700 * (10**(m / 2595.0) - 1)

def mel_filter_bank(sr, n_fft, n_mels, fmin, fmax):
    # Frequency bins
    fft_freqs = np.linspace(0, sr / 2, n_fft // 2 + 1)
    mel_min = hz_to_mel(fmin)
    mel_max = hz_to_mel(fmax)
    mel_points = np.linspace(mel_min, mel_max, n_mels + 2)
    hz_points = mel_to_hz(mel_points)
    bin_indices = np.floor((n_fft + 1) * hz_points / sr).astype(int)
    fb = np.zeros((n_mels, n_fft // 2 + 1))
    for m in range(1, n_mels + 1):
        f_m_minus = bin_indices[m - 1]
        f_m = bin_indices[m]
        f_m_plus = bin_indices[m + 1]
        for k in range(f_m_minus, f_m):
            fb[m - 1, k] = (k - f_m_minus) / (f_m - f_m_minus + 1e-10)
        for k in range(f_m, f_m_plus):
            fb[m - 1, k] = (f_m_plus - k) / (f_m_plus - f_m + 1e-10)
    return fb

def compute_mel_spectrogram(audio, sr, n_mels=N_MELS, hop_length=HOP_LENGTH, fmin=FMIN, fmax=FMAX, n_fft=N_FFT):
    # Compute STFT using SciPy
    f, t, Zxx = scipy.signal.stft(audio, fs=sr, window='hann',
                                  nperseg=n_fft, noverlap=n_fft - hop_length, boundary=None)
    S = np.abs(Zxx)**2  # power spectrogram
    fb = mel_filter_bank(sr, n_fft, n_mels, fmin, fmax)
    mel_spec = np.dot(fb, S)
    mel_spec_db = 10 * np.log10(np.maximum(mel_spec, 1e-10))
    return mel_spec_db

# ---------------------------------------------------------------------
# Processing Functions
# ---------------------------------------------------------------------
def process_sentiment(audio: np.ndarray, sr: int) -> List[dict]:
    """
    Splits the audio into SEGMENT_DURATION-second segments,
    computes a mel spectrogram for each segment (fixed to TARGET_FRAMES),
    and predicts sentiment using the ONNX sentiment model.
    Returns a list of dictionaries with timeSec, valence, arousal, and a dummy dominance.
    """
    segments = []
    total_duration = len(audio) / sr
    num_segments = int(total_duration // SEGMENT_DURATION)
    if num_segments < 1:
        num_segments = 1

    for i in range(num_segments):
        start_sec = i * SEGMENT_DURATION
        end_sec = start_sec + SEGMENT_DURATION
        start_sample = int(start_sec * sr)
        end_sample = int(end_sec * sr)
        audio_segment = audio[start_sample:end_sample]
        S_db = compute_mel_spectrogram(audio_segment, sr)
        # Ensure fixed number of frames along time axis
        if S_db.shape[1] < TARGET_FRAMES:
            pad_width = TARGET_FRAMES - S_db.shape[1]
            S_db_fixed = np.pad(S_db, ((0, 0), (0, pad_width)), mode='constant')
        else:
            S_db_fixed = S_db[:, :TARGET_FRAMES]
        # Transpose to shape (TARGET_FRAMES, N_MELS)
        S_db_fixed = S_db_fixed.T
        # Expand dims for batch and channel: (1, TARGET_FRAMES, N_MELS, 1)
        input_tensor = np.expand_dims(np.expand_dims(S_db_fixed, -1), 0).astype(np.float32)
        pred = sentiment_session.run([sentiment_output_name], {sentiment_input_name: input_tensor})
        # Assume output shape is (1, 2): [valence, arousal]
        valence, arousal = float(pred[0][0][0]), float(pred[0][0][1])
        dominance = 0.5  # Dummy value; adjust if needed.
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
    computes its mel spectrogram, resizes it to GENRE_TARGET_SIZE,
    and predicts the genre using the ONNX genre model.
    Returns the predicted genre label.
    """
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
    # Convert spectrogram to an image using PIL for resizing
    # Normalize S_db to 0-255 for image conversion
    S_norm = (S_db - np.min(S_db)) / (np.max(S_db) - np.min(S_db) + 1e-10) * 255
    S_norm = S_norm.astype(np.uint8)
    img = Image.fromarray(S_norm)
    img_resized = img.resize(GENRE_TARGET_SIZE, resample=Image.BILINEAR)
    S_resized = np.array(img_resized).astype(np.float32)
    # Expand dimensions to get shape (1, 150, 150, 1)
    input_tensor = np.expand_dims(np.expand_dims(S_resized, -1), 0)
    pred = genre_session.run([genre_output_name], {genre_input_name: input_tensor})
    pred_index = int(np.argmax(pred[0], axis=1)[0])
    genre_label = GENRE_CLASSES[pred_index]
    return genre_label

# ---------------------------------------------------------------------
# FastAPI Endpoints
# ---------------------------------------------------------------------
@app.get("/")
def read_root():
    return {"message": "Audio Analysis API using ONNX and lightweight audio processing"}

@app.post("/analyze", response_model=AnalysisResult)
async def analyze_audio_endpoint(audio: UploadFile = File(...)):
    if audio.content_type.split("/")[0] != "audio":
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload an audio file.")
    try:
        contents = await audio.read()
        # Load audio with soundfile
        audio_np, sr = sf.read(io.BytesIO(contents))
        # Ensure mono: if audio is stereo, average the channels
        if len(audio_np.shape) > 1:
            audio_np = np.mean(audio_np, axis=1)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing audio: {e}")

    if sentiment_session is None or genre_session is None:
        raise HTTPException(status_code=500, detail="Model(s) not loaded properly.")

    sentiment_segments = process_sentiment(audio_np, sr)
    genre_prediction = process_genre(audio_np, sr)
    return AnalysisResult(segments=sentiment_segments, genre=genre_prediction)
