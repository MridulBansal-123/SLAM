"""
SLAM - Monocular Depth Estimation
=================================

A deep learning application for real-time monocular depth estimation
using ResNet-152 encoder-decoder architecture.

Modules:
    - model: Neural network architecture definitions
    - inference: Depth prediction and image processing
    - video: Video processing utilities
    - config: Configuration and constants
"""

from src.config import Config
from src.model import ResNetDepthModel, UpSample
from src.inference import DepthEstimator

__version__ = "1.0.0"
__author__ = "Mridul Bansal"
__all__ = ["Config", "ResNetDepthModel", "UpSample", "DepthEstimator"]
