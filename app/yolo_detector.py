"""
YOLOv8 Object Detection Server

FastAPI-based server for object detection using YOLOv8.
Designed for Blue Iris IP camera integration.
"""

from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import uvicorn
import torch
import os
import logging
from datetime import datetime
from pathlib import Path
import tempfile
from typing import Optional

from detector import YOLODetector
from config import Config
from utils import setup_logging, get_gpu_info

# Initialize configuration
config = Config()

# Setup logging
setup_logging(config.LOG_LEVEL)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="YOLOv8 Object Detection Server",
    description="Production-ready YOLOv8 detection server for Blue Iris integration",
    version="1.0.0"
)

# CORS middleware for web UI
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global detector instance
detector: Optional[YOLODetector] = None
server_start_time = datetime.now()


@app.on_event("startup")
async def startup_event():
    """Initialize detector on server startup."""
    global detector

    logger.info("=" * 80)
    logger.info("YOLOv8 Object Detection Server Starting")
    logger.info("=" * 80)

    # Log GPU information
    gpu_info = get_gpu_info()
    if gpu_info['available']:
        logger.info(f"GPU: {gpu_info['name']}")
        logger.info(f"GPU Memory: {gpu_info['memory_total_gb']:.2f} GB")
        logger.info(f"CUDA Version: {gpu_info['cuda_version']}")
    else:
        logger.warning("GPU not available - using CPU")

    # Initialize detector
    try:
        model_path = config.MODEL_PATH
        logger.info(f"Loading model: {model_path}")

        detector = YOLODetector(
            model_path=str(model_path),
            conf_threshold=config.CONFIDENCE_THRESHOLD
        )

        logger.info("Model loaded successfully")
        logger.info(f"API Server: http://0.0.0.0:{config.API_PORT}")
        logger.info(f"Web UI: http://0.0.0.0:{config.API_PORT}/")
        logger.info("=" * 80)

    except Exception as e:
        logger.error(f"Failed to initialize detector: {e}", exc_info=True)
        raise


@app.post("/detect")
async def detect_objects(
    image: UploadFile = File(...),
    confidence: float = Form(default=None),
    imgsz: int = Form(default=None),
    iou: float = Form(default=None),
    max_det: int = Form(default=None)
):
    """
    Primary detection endpoint for Blue Iris integration.

    Args:
        image: Image file (JPEG/PNG)
        confidence: Optional confidence threshold override (0.0-1.0)
        imgsz: Image size for inference (640=fast, 1280=better for small objects)
        iou: IoU threshold for NMS (0.45 default, lower=more detections)
        max_det: Maximum detections per image (300 default)

    Returns:
        JSON response with detections matching Blue Iris format
    """
    if detector is None:
        raise HTTPException(status_code=503, detail="Detector not initialized")

    # Log the content type for debugging
    logger.info(f"Received image content_type: {image.content_type}")

    # Validate image format - be lenient for Blue Iris compatibility
    # Blue Iris might not send proper content-type headers
    if image.content_type and not image.content_type.startswith('image/'):
        if not image.content_type.startswith('application/octet-stream'):
            logger.warning(f"Unexpected content type: {image.content_type}, attempting anyway")
            # Don't reject - try to process it anyway

    try:
        # Save uploaded image to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
            contents = await image.read()
            tmp_file.write(contents)
            tmp_path = tmp_file.name

        # Use provided parameters or defaults from config
        conf_threshold = confidence if confidence is not None else config.CONFIDENCE_THRESHOLD
        img_size = imgsz if imgsz is not None else config.DEFAULT_IMGSZ
        iou_threshold = iou if iou is not None else config.DEFAULT_IOU
        max_detections = max_det if max_det is not None else config.DEFAULT_MAX_DET

        # Run detection with advanced parameters
        start_time = datetime.now()
        result = detector.detect(
            tmp_path,
            conf=conf_threshold,
            imgsz=img_size,
            iou=iou_threshold,
            max_det=max_detections
        )
        process_time = (datetime.now() - start_time).total_seconds() * 1000

        # Clean up temporary file
        os.unlink(tmp_path)

        # Format response for Blue Iris
        response = {
            "success": True,
            "predictions": result["detections"],
            "count": len(result["detections"]),
            "inference_ms": result["inference_ms"],
            "process_ms": round(process_time, 1),
            "model": config.MODEL_NAME,
            "image_size": list(result["image_size"]),
            "timestamp": datetime.now().isoformat()
        }

        logger.info(
            f"Detection complete: {len(result['detections'])} objects, "
            f"{result['inference_ms']:.1f}ms inference, "
            f"{process_time:.1f}ms total"
        )

        return JSONResponse(content=response)

    except Exception as e:
        logger.error(f"Detection failed: {e}", exc_info=True)

        # Clean up temp file if it exists
        if 'tmp_path' in locals() and os.path.exists(tmp_path):
            os.unlink(tmp_path)

        raise HTTPException(
            status_code=500,
            detail={
                "success": False,
                "error": "Detection failed",
                "message": str(e),
                "code": "DETECTION_ERROR"
            }
        )


@app.get("/v1/status/ping")
@app.get("/v1/server/status/ping")
async def codeproject_ping():
    """
    CodeProject.AI compatible ping endpoint.
    Blue Iris uses this to check if the server is alive.

    Returns:
        JSON response matching CodeProject.AI format
    """
    return {
        "success": True,
        "message": "YOLOv8 Server Online",
        "version": "1.0.0"
    }


@app.get("/v1/status/version")
@app.get("/v1/server/status/version")
async def codeproject_version():
    """
    CodeProject.AI compatible version endpoint.

    Returns:
        Version information
    """
    return {
        "success": True,
        "message": "1.0.0",
        "version": {
            "major": 1,
            "minor": 0,
            "patch": 0,
            "prerelease": None,
            "build": "custom-yolov8"
        }
    }


@app.get("/health")
async def health_check():
    """
    Health check endpoint.

    Returns:
        Server health status including GPU information
    """
    gpu_info = get_gpu_info()

    uptime_seconds = (datetime.now() - server_start_time).total_seconds()

    health = {
        "status": "healthy" if detector is not None else "initializing",
        "gpu_available": gpu_info['available'],
        "model_loaded": detector is not None,
        "uptime_seconds": round(uptime_seconds, 1),
        "version": "1.0.0"
    }

    # Add GPU details if available
    if gpu_info['available']:
        health.update({
            "gpu_name": gpu_info['name'],
            "gpu_memory_total_mb": gpu_info['memory_total_mb'],
            "gpu_memory_used_mb": gpu_info['memory_used_mb'],
            "cuda_version": gpu_info['cuda_version']
        })

    return health


@app.get("/models/info")
async def model_info():
    """
    Get model information.

    Returns:
        Model metadata and statistics
    """
    if detector is None:
        raise HTTPException(status_code=503, detail="Detector not initialized")

    model_info_dict = detector.get_model_info()
    stats = detector.get_stats()

    return {
        **model_info_dict,
        **stats,
        "input_size": 640,
        "batch_size": 1,
        "half_precision": False
    }


@app.get("/classes")
async def list_classes():
    """
    List all available detection classes.

    Returns:
        All 600 classes from Open Images V7 dataset
    """
    if detector is None:
        raise HTTPException(status_code=503, detail="Detector not initialized")

    class_names = detector.get_class_names()

    # Categorize common security-relevant classes
    categories = {
        "people": [],
        "vehicles": [],
        "animals": []
    }

    # Common people-related classes
    people_keywords = ['person', 'man', 'woman', 'boy', 'girl', 'human']
    vehicle_keywords = ['car', 'truck', 'van', 'bus', 'motorcycle', 'bicycle', 'vehicle']
    animal_keywords = ['dog', 'cat', 'horse', 'bird', 'deer', 'bear', 'animal']

    for cls_id, cls_name in class_names.items():
        cls_lower = cls_name.lower()

        if any(keyword in cls_lower for keyword in people_keywords):
            categories['people'].append(cls_name)
        elif any(keyword in cls_lower for keyword in vehicle_keywords):
            categories['vehicles'].append(cls_name)
        elif any(keyword in cls_lower for keyword in animal_keywords):
            categories['animals'].append(cls_name)

    return {
        "total": len(class_names),
        "classes": class_names,
        "categories": categories
    }


@app.get("/ui")
@app.get("/web")
async def web_ui():
    """
    Redirects to the root / endpoint, which serves the main Web UI.
    """
    return RedirectResponse(url="/")


@app.post("/v1/vision/custom/list")
@app.get("/v1/vision/custom/list")
async def list_custom_models():
    """
    CodeProject.AI endpoint for listing custom models.
    Blue Iris checks this to see what models are available.

    Returns:
        List of available models (we only have YOLOv8x-OIV7)
    """
    return {
        "success": True,
        "models": [
            {
                "name": "yolov8x-oiv7",
                "displayName": "YOLOv8x Open Images V7",
                "description": "YOLOv8 Extra Large model trained on Open Images V7 (600 classes)",
                "status": "active",
                "loaded": True
            }
        ]
    }


@app.post("/v1/vision/custom/{model_name}")
async def custom_model_detection(
    model_name: str,
    image: UploadFile = File(None),
    min_confidence: float = Form(default=None)
):
    """
    CodeProject.AI custom model endpoint.
    Blue Iris sends requests to /v1/vision/custom/MODEL_NAME

    This is a wrapper that calls the main detect_objects function.
    """
    logger.info(f"🟢 Received POST on /v1/vision/custom/{model_name} endpoint")
    logger.info(f"Image received: {image is not None}, filename: {image.filename if image else 'None'}")
    logger.info(f"Min confidence: {min_confidence}")

    if image is None:
        logger.error("No image file received in request!")
        raise HTTPException(status_code=400, detail="No image file provided")

    # Call the main detect_objects function
    return await detect_objects(
        image=image,
        confidence=min_confidence,
        imgsz=None,
        iou=None,
        max_det=None
    )


@app.post("/v1/vision/detection")
async def deepstack_compatible_detection(
    image: UploadFile = File(...),
    min_confidence: float = Form(default=None)
):
    """
    DeepStack/CodeProject.AI compatible endpoint.
    Blue Iris uses this endpoint format: /v1/vision/detection

    This is a clean wrapper that calls the main detect_objects function.
    """
    logger.info("🔵 Received POST on /v1/vision/detection endpoint")

    # Call the main detect_objects function with min_confidence mapped to confidence
    return await detect_objects(
        image=image,
        confidence=min_confidence,
        imgsz=None,
        iou=None,
        max_det=None
    )


@app.post("/alert")
async def detect_alert(
    image: UploadFile = File(...),
    confidence: float = Form(default=None),
    imgsz: int = Form(default=None),
    iou: float = Form(default=None),
    max_det: int = Form(default=None)
):
    """
    Simple /alert endpoint for Blue Iris "On alert" web hook.
    This bypasses all CodeProject.AI / DeepStack complexity.
    Just a clean, simple endpoint for receiving alerts.
    """
    logger.info("🔔 Received POST on /alert endpoint")
    return await detect_objects(image, confidence, imgsz, iou, max_det)


@app.post("/")
async def detect_root(
    image: UploadFile = File(...),
    confidence: float = Form(default=None),
    imgsz: int = Form(default=None),
    iou: float = Form(default=None),
    max_det: int = Form(default=None)
):
    """
    Root detection endpoint for Blue Iris compatibility.
    Blue Iris sends POST requests to IP:PORT/ so we handle it here.

    This is just a wrapper that calls the main detect_objects function.
    """
    return await detect_objects(image, confidence, imgsz, iou, max_det)


@app.get("/", response_class=HTMLResponse)
async def root_info():
    """
    Root GET endpoint - serves the Web UI dashboard.
    This impersonates the CodeProject.AI server, which serves a dashboard at its root.
    Blue Iris expects this.
    """
    logger.info("GET / (root) - Serving Web UI Dashboard")
    html_path = Path(__file__).parent / "web_ui.html"

    if html_path.exists():
        with open(html_path, 'r') as f:
            return HTMLResponse(content=f.read())
    else:
        logger.error("web_ui.html not found!")
        return HTMLResponse(
            content="<h1>Error: web_ui.html not found.</h1>",
            status_code=500
        )


if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=config.API_PORT,
        log_level=config.LOG_LEVEL.lower()
    )
