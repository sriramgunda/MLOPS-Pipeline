"""
FastAPI Inference Service for Cats vs Dogs Classification
Provides REST API endpoints for health check and image classification
"""
import logging
import time
import os
from io import BytesIO
import base64
import json

from fastapi import FastAPI, Request, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel
import numpy as np

# Deep Learning
import tensorflow as tf
from PIL import Image

# Prometheus metrics
from prometheus_client import Counter, Histogram, generate_latest

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# ========================
# Metrics Configuration
# ========================
REQUEST_COUNT = Counter(
    "api_requests_total",
    "Total API requests",
    ["method", "endpoint", "status"]
)

REQUEST_LATENCY = Histogram(
    "api_request_latency_seconds",
    "Request latency in seconds",
    ["endpoint"]
)

PREDICTION_CONFIDENCE = Histogram(
    "prediction_confidence",
    "Prediction confidence scores",
    ["class_label"]
)

# ========================
# Load Model
# ========================
MODEL_PATH = os.path.join(os.path.dirname(__file__), "mobilenet_v2.keras")
logger.info(f"Loading model from {MODEL_PATH}")

try:
    model = tf.keras.models.load_model(MODEL_PATH)
    logger.info("Model loaded successfully")
    MODEL_LOADED = True
except Exception as e:
    logger.error(f"Failed to load model: {str(e)}")
    MODEL_LOADED = False

# ========================
# FastAPI App Setup
# ========================
app = FastAPI(
    title="Cats vs Dogs Classification API",
    description="Binary image classification API for pet adoption platform",
    version="1.0.0"
)

# ========================
# Request/Response Models
# ========================
class PredictionResponse(BaseModel):
    prediction: str  # "cat" or "dog"
    confidence: float
    class_probabilities: dict
    processing_time_ms: float


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    timestamp: str


# ========================
# Middleware
# ========================
@app.middleware("http")
async def log_requests_and_metrics(request: Request, call_next):
    """Log all requests and track metrics"""
    start_time = time.time()
    
    try:
        response = await call_next(request)
        duration = time.time() - start_time
        
        # Log request
        logger.info(
            f"{request.method} {request.url.path} "
            f"status_code={response.status_code} "
            f"duration={duration:.3f}s"
        )
        
        # Track metrics
        REQUEST_COUNT.labels(
            method=request.method,
            endpoint=request.url.path,
            status=response.status_code
        ).inc()
        
        REQUEST_LATENCY.labels(endpoint=request.url.path).observe(duration)
        
        return response
    
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        REQUEST_COUNT.labels(
            method=request.method,
            endpoint=request.url.path,
            status=500
        ).inc()
        raise


# ========================
# Utility Functions
# ========================
def preprocess_image(image: Image.Image) -> np.ndarray:
    """
    Preprocess image to model input format
    - Resize to 224x224
    - Normalize to [0, 1]
    
    Args:
        image: PIL Image object
    
    Returns:
        Preprocessed numpy array
    """
    # Resize
    image = image.resize((224, 224))
    
    # Convert to RGB if needed
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Convert to array and normalize
    img_array = np.array(image, dtype=np.float32) / 255.0
    
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array


def make_prediction(img_array: np.ndarray) -> tuple:
    """
    Make prediction using loaded model
    
    Args:
        img_array: Preprocessed image array
    
    Returns:
        Tuple of (predicted_class, confidence)
    """
    prediction = model.predict(img_array, verbose=0)
    confidence = float(prediction[0][0])
    
    # 0 = cat, 1 = dog
    predicted_class = "dog" if confidence > 0.5 else "cat"
    pet_confidence = confidence if confidence > 0.5 else (1 - confidence)
    
    return predicted_class, pet_confidence, {
        "cat": float(1 - confidence),
        "dog": float(confidence)
    }


# ========================
# API Endpoints
# ========================
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint
    Returns service status and model availability
    """
    from datetime import datetime
    return {
        "status": "healthy" if MODEL_LOADED else "degraded",
        "model_loaded": MODEL_LOADED,
        "timestamp": datetime.utcnow().isoformat()
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    """
    Prediction endpoint
    
    Args:
        file: Image file (JPG, PNG, etc.)
    
    Returns:
        Prediction with class probabilities
    """
    if not MODEL_LOADED:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please try again later."
        )
    
    start_time = time.time()
    
    try:
        # Read and validate image
        contents = await file.read()
        
        try:
            image = Image.open(BytesIO(contents))
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid image file: {str(e)}"
            )
        
        # Preprocess
        img_array = preprocess_image(image)
        
        # Predict
        predicted_class, confidence, probabilities = make_prediction(img_array)
        
        processing_time = (time.time() - start_time) * 1000  # Convert to ms
        
        # Log prediction
        logger.info(
            f"Prediction: {predicted_class} "
            f"(confidence: {confidence:.4f}, time: {processing_time:.2f}ms)"
        )
        
        # Track confidence metric
        PREDICTION_CONFIDENCE.labels(class_label=predicted_class).observe(confidence)
        
        return {
            "prediction": predicted_class,
            "confidence": round(confidence, 4),
            "class_probabilities": probabilities,
            "processing_time_ms": round(processing_time, 2)
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )


@app.post("/predict-base64")
async def predict_base64(data: dict):
    """
    Prediction endpoint accepting base64-encoded image
    
    Request body:
    {
        "image_base64": "<base64-encoded image data>"
    }
    """
    if not MODEL_LOADED:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please try again later."
        )
    
    try:
        # Decode base64
        image_data = base64.b64decode(data.get("image_base64", ""))
        image = Image.open(BytesIO(image_data))
        
        # Preprocess and predict
        img_array = preprocess_image(image)
        predicted_class, confidence, probabilities = make_prediction(img_array)
        
        return {
            "prediction": predicted_class,
            "confidence": round(confidence, 4),
            "class_probabilities": probabilities
        }
    
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid base64 image: {str(e)}"
        )


@app.get("/metrics")
async def metrics():
    """
    Prometheus metrics endpoint
    """
    return Response(
        content=generate_latest(),
        media_type="text/plain"
    )


@app.get("/")
async def root():
    """
    Root endpoint - API information
    """
    return {
        "service": "Cats vs Dogs Classification API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "predict": "/predict (POST with image file)",
            "predict_base64": "/predict-base64 (POST with base64 image)",
            "metrics": "/metrics",
            "docs": "/docs"
        },
        "model_status": "loaded" if MODEL_LOADED else "not_loaded"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")