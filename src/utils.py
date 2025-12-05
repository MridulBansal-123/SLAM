"""
Utility functions for SLAM depth estimation.

This module contains helper functions used across the application.
"""

import sys
import logging
from typing import Optional

import torch


def setup_logging(
    level: int = logging.INFO,
    format_string: Optional[str] = None
) -> None:
    """
    Configure application logging.
    
    Args:
        level: Logging level (default: INFO)
        format_string: Custom format string for log messages
    """
    if format_string is None:
        format_string = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    logging.basicConfig(
        level=level,
        format=format_string,
        handlers=[logging.StreamHandler(sys.stdout)]
    )


def get_device_info() -> dict:
    """
    Get information about available compute devices.
    
    Returns:
        Dictionary with device information
    """
    info = {
        "cuda_available": torch.cuda.is_available(),
        "device": "cpu",
        "device_name": "CPU",
        "cuda_version": None,
        "device_count": 0
    }
    
    if torch.cuda.is_available():
        info["device"] = "cuda"
        info["device_name"] = torch.cuda.get_device_name(0)
        info["cuda_version"] = torch.version.cuda
        info["device_count"] = torch.cuda.device_count()
    
    return info


def print_device_info() -> None:
    """Print device information to console."""
    info = get_device_info()
    print(f"CUDA Available: {info['cuda_available']}")
    print(f"Device: {info['device_name']}")
    if info['cuda_available']:
        print(f"CUDA Version: {info['cuda_version']}")
        print(f"GPU Count: {info['device_count']}")


def format_file_size(size_bytes: int) -> str:
    """
    Format file size in human-readable format.
    
    Args:
        size_bytes: Size in bytes
        
    Returns:
        Formatted size string
    """
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} TB"


def format_duration(seconds: float) -> str:
    """
    Format duration in human-readable format.
    
    Args:
        seconds: Duration in seconds
        
    Returns:
        Formatted duration string
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    
    minutes = int(seconds // 60)
    secs = seconds % 60
    
    if minutes < 60:
        return f"{minutes}m {secs:.0f}s"
    
    hours = minutes // 60
    mins = minutes % 60
    return f"{hours}h {mins}m {secs:.0f}s"
