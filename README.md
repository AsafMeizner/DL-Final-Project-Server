# Audio Analysis API

This FastAPI application analyzes audio files using ONNX models for sentiment analysis and genre classification. Audio processing is performed with the trusted `librosa` library.

## Project Structure

```
fastapi_project/  
├── app/  
│   ├── __init__.py  
│   ├── main.py               # Main FastAPI app with endpoints  
│   ├── audio_processing.py   # Librosa-based audio processing utilities  
│   └── onnx_models.py        # ONNX model loading for sentiment & genre  
├── models/  
│   ├── sentiment_model.onnx  # Your ONNX sentiment model  
│   └── genre_model.onnx      # Your ONNX genre model  
├── requirements.txt          # Python dependencies  
└── README.md                 # Project documentation
```
## Installation

1. Install dependencies:  
   Run `pip install -r requirements.txt`

2. Place your ONNX model files (`sentiment_model.onnx` and `genre_model.onnx`) in the `models/` directory.

## Running the Application

Run the FastAPI app with:  
   `uvicorn app.main:app --reload --host 0.0.0.0 --port 8000`

The API will be available at [http://localhost:8000](http://localhost:8000).

- **Welcome Endpoint:** GET `/`
- **Custom API Docs:** GET `/api-docs` (renders the OpenAPI JSON)
- **Interactive Docs:** GET `/docs`
- **ReDoc Docs:** GET `/redoc`

## API Endpoints

### POST /analyze

- **Description:** Upload an audio file (multipart/form-data) to receive sentiment and genre predictions.
- **Response:** JSON containing an array of sentiment segments (each with time, valence, arousal, and dominance) and the predicted genre.

Example using curl:  
   `curl -X POST "http://localhost:8000/analyze" -F "audio=@your_audio_file.wav"`

## License

MIT License
