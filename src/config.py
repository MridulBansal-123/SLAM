"""
Configuration module for SLAM Depth Estimation.

This module contains all configuration constants and settings
used throughout the application.
"""

import os
from typing import Dict, List, Optional, Tuple


class Config:
    """Application configuration settings."""
    
    def __init__(self):
        # Paths
        self.ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.MODEL_DIR = os.path.dirname(self.ROOT_DIR)  # Parent of SLAM folder
        self.MODEL_FILENAME = "resnet152_depth_model.pth"
        
        # Model settings
        self.INPUT_SIZE: Tuple[int, int] = (640, 480)  # (width, height)
        self.DEPTH_SCALE: float = 10.0  # Maximum depth in meters
        
        # Image normalization (ImageNet standards)
        self.NORMALIZE_MEAN: Tuple[float, float, float] = (0.485, 0.456, 0.406)
        self.NORMALIZE_STD: Tuple[float, float, float] = (0.229, 0.224, 0.225)
        
        # Available colormaps for depth visualization
        self.COLORMAPS: List[str] = [
            "magma", "plasma", "inferno", "viridis", "gray", "gray_r", "jet"
        ]
        self.DEFAULT_COLORMAP: str = "magma"
        
        # Resolution presets for live streaming
        self.RESOLUTION_PRESETS: Dict[str, Optional[Tuple[int, int]]] = {
            "360p (640x360)": (640, 360),
            "480p (854x480)": (854, 480),
            "720p (1280x720)": (1280, 720),
            "Original": None
        }
        self.DEFAULT_RESOLUTION: str = "360p (640x360)"
        
        # Video settings
        self.VIDEO_CODEC: str = "mp4v"
        self.SUPPORTED_VIDEO_FORMATS: List[str] = ["mp4", "avi", "mov", "mkv"]
        
        # Streamlit page configuration
        self.PAGE_TITLE: str = "Depth Estimation - ResNet152"
        self.PAGE_ICON: str = "🔭"
        self.PAGE_LAYOUT: str = "wide"
    
    @property
    def model_path(self) -> str:
        """Get the full path to the model file."""
        return os.path.join(self.MODEL_DIR, self.MODEL_FILENAME)


# Create a default configuration instance
config = Config()
