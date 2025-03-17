import os
import onnxruntime as ort

# Set model paths (can also be overridden via environment variables)
SENTIMENT_MODEL_PATH = os.environ.get("SENTIMENT_MODEL_PATH", os.path.join("models", "sentiment_model.onnx"))
GENRE_MODEL_PATH = os.environ.get("GENRE_MODEL_PATH", os.path.join("models", "genre_model.onnx"))

try:
    sentiment_session = ort.InferenceSession(SENTIMENT_MODEL_PATH)
    sentiment_input_name = sentiment_session.get_inputs()[0].name
    sentiment_output_name = sentiment_session.get_outputs()[0].name
    print("Sentiment ONNX model loaded successfully.")
except Exception as e:
    print("Error loading sentiment ONNX model:", e)
    sentiment_session = None
    sentiment_input_name = None
    sentiment_output_name = None

try:
    genre_session = ort.InferenceSession(GENRE_MODEL_PATH)
    genre_input_name = genre_session.get_inputs()[0].name
    genre_output_name = genre_session.get_outputs()[0].name
    print("Genre ONNX model loaded successfully.")
except Exception as e:
    print("Error loading genre ONNX model:", e)
    genre_session = None
    genre_input_name = None
    genre_output_name = None
