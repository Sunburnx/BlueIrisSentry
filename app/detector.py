"""
YOLOv8 Object Detection Module

This module provides the core object detection functionality using YOLOv8.
It handles model loading, GPU initialization, and inference.
"""

from ultralytics import YOLO
import torch
import time
import logging
from typing import Dict, List, Optional, Tuple
from pathlib import Path

logger = logging.getLogger(__name__)


class YOLODetector:
    """
    YOLOv8 object detector with GPU acceleration.

    Attributes:
        model: Loaded YOLO model instance
        device: Device to run inference on (cuda/cpu)
        conf_threshold: Default confidence threshold for detections
        stats: Dictionary containing detection statistics
    """

    def __init__(
        self,
        model_path: str,
        conf_threshold: float = 0.25,
        device: Optional[str] = None
    ):
        """
        Initialize the YOLO detector.

        Args:
            model_path: Path to the YOLO model weights file
            conf_threshold: Minimum confidence threshold for detections (0.0-1.0)
            device: Device to use ('cuda', 'cpu', or None for auto-detect)
        """
        logger.info(f"Initializing YOLODetector with model: {model_path}")

        # Determine device
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device

        logger.info(f"Using device: {self.device}")

        if self.device == 'cuda':
            logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
            logger.info(f"CUDA Version: {torch.version.cuda}")
            logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")

        # Load model
        try:
            self.model = YOLO(model_path)
            self.model.to(self.device)
            logger.info(f"Model loaded successfully on {self.device}")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

        self.conf_threshold = conf_threshold
        self.model_path = model_path

        # Statistics
        self.stats = {
            'total_detections': 0,
            'total_inferences': 0,
            'total_inference_time_ms': 0.0,
            'warmup_complete': False
        }

        # Warmup
        self._warmup()

    def _warmup(self) -> None:
        """
        Perform warmup inference to initialize CUDA kernels.
        This significantly improves first real inference time.
        """
        logger.info("Performing model warmup...")
        try:
            # Create dummy input matching expected size
            dummy_img = torch.zeros((1, 3, 640, 640)).to(self.device)

            # Run warmup inference
            start_time = time.time()
            _ = self.model(dummy_img, verbose=False, conf=self.conf_threshold)
            warmup_time = (time.time() - start_time) * 1000

            self.stats['warmup_complete'] = True
            logger.info(f"Warmup complete in {warmup_time:.1f}ms")
        except Exception as e:
            logger.warning(f"Warmup failed: {e}")

    def detect(
        self,
        image_path: str,
        conf: Optional[float] = None,
        iou: float = 0.45,
        imgsz: int = 640,
        max_det: int = 300,
        agnostic_nms: bool = False
    ) -> Dict:
        """
        Detect objects in an image.

        Args:
            image_path: Path to the image file
            conf: Confidence threshold (overrides default if provided)
            iou: IoU threshold for NMS (Non-Maximum Suppression)
            imgsz: Image size for inference (640, 1280, etc.)
            max_det: Maximum number of detections per image
            agnostic_nms: Class-agnostic NMS (can help find overlapping objects)

        Returns:
            Dictionary containing:
                - detections: List of detected objects with bounding boxes
                - inference_ms: Inference time in milliseconds
                - image_size: Tuple of (width, height)
        """
        conf_threshold = conf if conf is not None else self.conf_threshold

        logger.debug(
            f"Running detection on {image_path} with confidence {conf_threshold}, "
            f"iou={iou}, imgsz={imgsz}, max_det={max_det}"
        )

        try:
            # Run inference with advanced parameters
            start_time = time.time()
            results = self.model(
                image_path,
                conf=conf_threshold,
                iou=iou,
                imgsz=imgsz,
                max_det=max_det,
                agnostic_nms=agnostic_nms,
                verbose=False,
                device=self.device
            )
            inference_time = (time.time() - start_time) * 1000

            # Parse results
            detections = []
            image_size = (0, 0)

            for r in results:
                # Get image dimensions
                if hasattr(r, 'orig_shape'):
                    image_size = (r.orig_shape[1], r.orig_shape[0])  # (width, height)

                boxes = r.boxes
                for box in boxes:
                    # Extract box coordinates
                    x1, y1, x2, y2 = box.xyxy[0].tolist()

                    # Extract confidence and class
                    conf_score = float(box.conf[0])
                    cls = int(box.cls[0])
                    label = r.names[cls]

                    detections.append({
                        "label": label,
                        "confidence": round(conf_score, 2),
                        "x_min": int(x1),
                        "y_min": int(y1),
                        "x_max": int(x2),
                        "y_max": int(y2)
                    })

            # Update statistics
            self.stats['total_inferences'] += 1
            self.stats['total_detections'] += len(detections)
            self.stats['total_inference_time_ms'] += inference_time

            logger.debug(
                f"Detected {len(detections)} objects in {inference_time:.1f}ms"
            )

            return {
                "detections": detections,
                "inference_ms": round(inference_time, 1),
                "image_size": image_size
            }

        except Exception as e:
            logger.error(f"Detection failed: {e}", exc_info=True)
            raise

    def get_class_names(self) -> Dict[int, str]:
        """
        Get mapping of class IDs to class names.

        Returns:
            Dictionary mapping class ID to class name
        """
        if hasattr(self.model, 'names'):
            return self.model.names
        return {}

    def get_stats(self) -> Dict:
        """
        Get detection statistics.

        Returns:
            Dictionary containing detection statistics
        """
        avg_inference_ms = 0.0
        if self.stats['total_inferences'] > 0:
            avg_inference_ms = (
                self.stats['total_inference_time_ms'] /
                self.stats['total_inferences']
            )

        return {
            **self.stats,
            'avg_inference_ms': round(avg_inference_ms, 1)
        }

    def get_model_info(self) -> Dict:
        """
        Get model information.

        Returns:
            Dictionary containing model metadata
        """
        model_size_mb = 0.0
        model_path = Path(self.model_path)
        if model_path.exists():
            model_size_mb = model_path.stat().st_size / (1024 * 1024)

        num_classes = len(self.get_class_names())

        return {
            'model_name': model_path.stem,
            'model_path': str(model_path),
            'model_size_mb': round(model_size_mb, 1),
            'num_classes': num_classes,
            'device': self.device,
            'conf_threshold': self.conf_threshold
        }
