"""
Configuration Management Module

Handles environment variables and configuration settings.
"""

import os
from pathlib import Path
from typing import Optional


class Config:
    """
    Application configuration from environment variables.
    """

    def __init__(self):
        """Initialize configuration from environment variables."""

        # Model Configuration
        self.MODEL_NAME: str = os.getenv('MODEL_NAME', 'yolov8x-oiv7.pt')
        self.MODELS_DIR: Path = Path('/app/models')
        self.MODEL_PATH: Path = self.MODELS_DIR / self.MODEL_NAME

        # Confidence Configuration
        self.CONFIDENCE_THRESHOLD: float = float(
            os.getenv('CONFIDENCE_THRESHOLD', '0.25')
        )

        # Detection Parameters (optimized for small/distant objects)
        self.DEFAULT_IMGSZ: int = int(os.getenv('DEFAULT_IMGSZ', '640'))
        self.DEFAULT_IOU: float = float(os.getenv('DEFAULT_IOU', '0.45'))
        self.DEFAULT_MAX_DET: int = int(os.getenv('DEFAULT_MAX_DET', '300'))

        # Server Configuration
        self.API_PORT: int = int(os.getenv('API_PORT', '9080'))
        self.WEB_UI_PORT: int = int(os.getenv('WEB_UI_PORT', '9081'))
        self.LOG_LEVEL: str = os.getenv('LOG_LEVEL', 'INFO').upper()

        # Performance Configuration
        self.MAX_QUEUE_SIZE: int = int(os.getenv('MAX_QUEUE_SIZE', '10'))
        self.BATCH_SIZE: int = int(os.getenv('BATCH_SIZE', '1'))

        # Debug Configuration
        self.SAVE_DEBUG_IMAGES: bool = (
            os.getenv('SAVE_DEBUG_IMAGES', 'false').lower() == 'true'
        )

        # Directory paths
        self.OUTPUT_DIR: Path = Path('/app/output')
        self.LOGS_DIR: Path = Path('/app/logs')
        self.CONFIG_DIR: Path = Path('/app/config')

        # Ensure directories exist
        self._create_directories()

    def _create_directories(self) -> None:
        """Create necessary directories if they don't exist."""
        directories = [
            self.MODELS_DIR,
            self.OUTPUT_DIR,
            self.LOGS_DIR,
            self.CONFIG_DIR
        ]

        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)

    def __repr__(self) -> str:
        """String representation of configuration."""
        return (
            f"Config("
            f"model={self.MODEL_NAME}, "
            f"port={self.API_PORT}, "
            f"conf_threshold={self.CONFIDENCE_THRESHOLD}, "
            f"log_level={self.LOG_LEVEL}"
            f")"
        )
