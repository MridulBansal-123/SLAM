"""
Inference module for depth estimation.

This module handles model loading, image preprocessing, and depth prediction.
"""

import logging
from typing import Optional, Tuple

import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.cm as cm

from .config import config
from .model import ResNetDepthModel

# Configure logging
logger = logging.getLogger(__name__)


class DepthEstimator:
    """
    Depth estimation inference class.
    
    Handles model loading, preprocessing, and prediction for
    monocular depth estimation.
    
    Attributes:
        model: The loaded ResNetDepthModel
        device: torch.device for computation (cuda or cpu)
        transform: Image preprocessing transforms
    """
    
    def __init__(
        self, 
        model_path: Optional[str] = None,
        device: Optional[torch.device] = None
    ):
        """
        Initialize the depth estimator.
        
        Args:
            model_path: Path to the model weights file.
                       If None, uses the default from config.
            device: Computation device. If None, auto-detects GPU/CPU.
        """
        self.model_path = model_path or config.model_path
        self.device = device or self._get_device()
        self.model: Optional[ResNetDepthModel] = None
        
        # Set up preprocessing transforms
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=list(config.NORMALIZE_MEAN),
                std=list(config.NORMALIZE_STD)
            )
        ])
        
        logger.info(f"DepthEstimator initialized with device: {self.device}")
    
    @staticmethod
    def _get_device() -> torch.device:
        """Detect and return the best available device."""
        if torch.cuda.is_available():
            device = torch.device("cuda")
            logger.info(f"CUDA available: {torch.cuda.get_device_name(0)}")
        else:
            device = torch.device("cpu")
            logger.info("CUDA not available, using CPU")
        return device
    
    def load_model(self) -> bool:
        """
        Load the model weights.
        
        Returns:
            True if model loaded successfully, False otherwise.
        """
        try:
            self.model = ResNetDepthModel()
            state_dict = torch.load(
                self.model_path, 
                map_location=self.device,
                weights_only=True
            )
            self.model.load_state_dict(state_dict)
            self.model = self.model.to(self.device)
            self.model.eval()
            
            logger.info(f"Model loaded successfully from {self.model_path}")
            return True
            
        except FileNotFoundError:
            logger.error(f"Model file not found: {self.model_path}")
            return False
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return False
    
    def preprocess(
        self, 
        image: Image.Image
    ) -> Tuple[torch.Tensor, Tuple[int, int]]:
        """
        Preprocess an image for model input.
        
        Args:
            image: PIL Image to preprocess
            
        Returns:
            Tuple of (input_tensor, original_size)
        """
        original_size = image.size
        
        # Resize to model input size
        resized_image = image.resize(config.INPUT_SIZE)
        
        # Apply transforms
        input_tensor = self.transform(resized_image).unsqueeze(0)
        
        return input_tensor, original_size
    
    @torch.no_grad()
    def predict(
        self, 
        input_tensor: torch.Tensor, 
        original_size: Tuple[int, int]
    ) -> np.ndarray:
        """
        Run depth prediction on preprocessed input.
        
        Args:
            input_tensor: Preprocessed image tensor
            original_size: Original image size (width, height)
            
        Returns:
            Depth map as numpy array resized to original dimensions
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        input_tensor = input_tensor.to(self.device)
        depth_output = self.model(input_tensor)
        
        # Convert to numpy and resize to original dimensions
        depth_map = depth_output.squeeze().cpu().numpy()
        depth_map = cv2.resize(depth_map, original_size)
        
        return depth_map
    
    def estimate_depth(self, image: Image.Image) -> np.ndarray:
        """
        Convenience method for full depth estimation pipeline.
        
        Args:
            image: Input PIL Image
            
        Returns:
            Depth map as numpy array
        """
        input_tensor, original_size = self.preprocess(image)
        return self.predict(input_tensor, original_size)


def colorize_depth(
    depth_map: np.ndarray, 
    colormap: str = "magma"
) -> np.ndarray:
    """
    Apply colormap to depth map for visualization.
    
    Args:
        depth_map: Depth map as numpy array
        colormap: Matplotlib colormap name
        
    Returns:
        Colored depth map as RGB numpy array (uint8)
    """
    # Normalize to 0-1 range
    depth_normalized = (depth_map - depth_map.min()) / \
                       (depth_map.max() - depth_map.min() + 1e-8)
    
    # Apply colormap
    cmap = cm.get_cmap(colormap)
    depth_colored = cmap(depth_normalized)
    
    # Convert to uint8 RGB
    depth_rgb = (depth_colored[:, :, :3] * 255).astype(np.uint8)
    
    return depth_rgb


def colorize_depth_bgr(
    depth_map: np.ndarray, 
    colormap: str = "magma"
) -> np.ndarray:
    """
    Apply colormap to depth map for OpenCV (BGR format).
    
    Args:
        depth_map: Depth map as numpy array
        colormap: Matplotlib colormap name
        
    Returns:
        Colored depth map as BGR numpy array (uint8)
    """
    depth_rgb = colorize_depth(depth_map, colormap)
    return cv2.cvtColor(depth_rgb, cv2.COLOR_RGB2BGR)
