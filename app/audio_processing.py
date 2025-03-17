import librosa
import numpy as np

# Audio processing parameters
SAMPLE_RATE = 22050
N_MELS = 128
HOP_LENGTH = 512
FMIN = 20
FMAX = 8000
N_FFT = 2048

def compute_mel_spectrogram(audio, sr=SAMPLE_RATE, n_mels=N_MELS, hop_length=HOP_LENGTH, fmin=FMIN, fmax=FMAX, n_fft=N_FFT):
    """
    Computes a mel spectrogram from audio data using librosa.
    Returns a decibel-scaled spectrogram.
    """
    S = librosa.feature.melspectrogram(
        y=audio,
        sr=sr,
        n_mels=n_mels,
        hop_length=hop_length,
        n_fft=n_fft,
        fmin=fmin,
        fmax=fmax
    )
    return librosa.power_to_db(S, ref=np.max)
