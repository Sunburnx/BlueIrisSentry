"""
Utility Functions Module

Helper functions for logging, GPU information, and other utilities.
"""

import logging
import sys
from typing import Dict
import torch


def setup_logging(log_level: str = "INFO") -> None:
    """
    Configure logging for the application.

    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    """
    # Convert string to logging level
    level = getattr(logging, log_level.upper(), logging.INFO)

    # Configure root logger
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('/app/logs/app.log')
        ]
    )

    # Set ultralytics logger to WARNING to reduce noise
    logging.getLogger('ultralytics').setLevel(logging.WARNING)


def get_gpu_info() -> Dict:
    """
    Get GPU information.

    Returns:
        Dictionary containing GPU information:
            - available: Whether GPU is available
            - name: GPU name
            - memory_total_mb: Total GPU memory in MB
            - memory_used_mb: Used GPU memory in MB
            - memory_total_gb: Total GPU memory in GB
            - cuda_version: CUDA version
    """
    info = {
        'available': False,
        'name': 'N/A',
        'memory_total_mb': 0,
        'memory_used_mb': 0,
        'memory_total_gb': 0.0,
        'cuda_version': 'N/A'
    }

    if torch.cuda.is_available():
        info['available'] = True
        info['name'] = torch.cuda.get_device_name(0)

        # Get memory info
        memory_total = torch.cuda.get_device_properties(0).total_memory
        info['memory_total_mb'] = int(memory_total / (1024 * 1024))
        info['memory_total_gb'] = memory_total / (1024 ** 3)

        # Get used memory
        memory_allocated = torch.cuda.memory_allocated(0)
        info['memory_used_mb'] = int(memory_allocated / (1024 * 1024))

        # CUDA version
        if torch.version.cuda:
            info['cuda_version'] = torch.version.cuda

    return info


def format_bytes(bytes_value: int) -> str:
    """
    Format bytes to human-readable string.

    Args:
        bytes_value: Number of bytes

    Returns:
        Formatted string (e.g., "1.5 GB")
    """
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes_value < 1024.0:
            return f"{bytes_value:.2f} {unit}"
        bytes_value /= 1024.0
    return f"{bytes_value:.2f} PB"


def validate_confidence(conf: float) -> bool:
    """
    Validate confidence threshold value.

    Args:
        conf: Confidence threshold

    Returns:
        True if valid, False otherwise
    """
    return 0.0 <= conf <= 1.0
