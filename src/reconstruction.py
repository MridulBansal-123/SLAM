"""
3D Reconstruction module for depth-based point cloud generation.

This module implements:
    - Similarity-based depth denoising filter (Section III of the paper)
    - Point cloud generation from RGB-D images
    - Interactive 3D visualization using Plotly
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
from PIL import Image


class SimilarityBasedFilter(nn.Module):
    """
    Implements the Similarity-Based Filter from Section III of the paper.
    
    This filter refines depth maps by computing surface normals and
    applying content-aware smoothing based on depth similarity.
    """
    
    def __init__(self, kernel_size: int = 5):
        """
        Initialize the filter.
        
        Args:
            kernel_size: Size of the neighborhood kernel (default: 5)
        """
        super().__init__()
        self.k = kernel_size
        self.pad = kernel_size // 2

    def compute_normals(self, depth: torch.Tensor) -> torch.Tensor:
        """
        Estimate surface normals from depth gradients.
        
        Args:
            depth: Depth tensor of shape (B, 1, H, W)
            
        Returns:
            Normal vectors of shape (B, 3, H, W)
        """
        # N = (-dz/dx, -dz/dy, 1) normalized
        dz_dy, dz_dx = torch.gradient(depth, dim=[2, 3])
        normal = torch.cat([-dz_dx, -dz_dy, torch.ones_like(depth)], dim=1)
        normal = F.normalize(normal, dim=1)
        return normal

    def forward(self, depth_map: torch.Tensor) -> torch.Tensor:
        """
        Apply similarity-based filtering to refine depth map.
        
        Args:
            depth_map: Input depth tensor of shape (B, 1, H, W)
            
        Returns:
            Refined depth tensor of shape (B, 1, H, W)
        """
        b, c, h, w = depth_map.shape

        # 1. Compute Normals (n_s)
        normals = self.compute_normals(depth_map)  # (B, 3, H, W)

        # 2. Extract Neighborhoods using Unfold
        normals_unfold = F.unfold(normals, kernel_size=self.k, padding=self.pad)
        normals_unfold = normals_unfold.view(b, 3, self.k * self.k, h, w)
        
        # Center normal index
        center_idx = (self.k * self.k) // 2
        normal_center = normals_unfold[:, :, center_idx:center_idx + 1, :, :]

        # 3. Compute Difference for Covariance estimation
        diff = normals_unfold - normal_center  # (B, 3, K^2, H, W)
        covariance_trace = torch.mean(torch.sum(diff ** 2, dim=1), dim=1)  # (B, H, W)

        # 4. Compute Similarity Weights based on depth
        depth_unfold = F.unfold(depth_map, kernel_size=self.k, padding=self.pad)
        depth_unfold = depth_unfold.view(b, 1, self.k * self.k, h, w)
        depth_center = depth_unfold[:, :, center_idx:center_idx + 1, :, :]

        # Gaussian weight based on Depth Difference (Content Similarity)
        depth_diff = (depth_unfold - depth_center) ** 2
        sigma_depth = 0.1  # Threshold from paper
        weights = torch.exp(-depth_diff / (2 * sigma_depth ** 2))

        weighted_sum = torch.sum(depth_unfold * weights, dim=2)
        weight_total = torch.sum(weights, dim=2)

        depth_refined = weighted_sum / (weight_total + 1e-6)
        return depth_refined


class PointCloudGenerator:
    """
    Generates 3D point clouds from RGB images and depth maps.
    
    Uses camera intrinsics to back-project depth values into 3D space.
    """
    
    def __init__(
        self, 
        image_size: Tuple[int, int] = (224, 224),
        focal_length: float = 200.0,
        device: Optional[torch.device] = None
    ):
        """
        Initialize the point cloud generator.
        
        Args:
            image_size: Size of the input images (height, width)
            focal_length: Camera focal length in pixels
            device: Computation device (cuda/cpu)
        """
        self.image_size = image_size
        self.fx = focal_length
        self.fy = focal_length
        self.cx = image_size[1] / 2  # Principal point x
        self.cy = image_size[0] / 2  # Principal point y
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.filter_module = SimilarityBasedFilter().to(self.device)
        
    def refine_depth(self, depth_map: np.ndarray) -> np.ndarray:
        """
        Apply similarity-based filter to refine depth map.
        
        Args:
            depth_map: Raw depth map of shape (H, W)
            
        Returns:
            Refined depth map of shape (H, W)
        """
        # Convert to tensor
        depth_tensor = torch.from_numpy(depth_map).float().unsqueeze(0).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            refined = self.filter_module(depth_tensor)
        
        return refined.squeeze().cpu().numpy()
    
    def generate_point_cloud(
        self, 
        rgb_image: np.ndarray, 
        depth_map: np.ndarray,
        apply_filter: bool = True,
        downsample_factor: int = 2
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate 3D point cloud from RGB image and depth map.
        
        Args:
            rgb_image: RGB image of shape (H, W, 3) with values 0-255
            depth_map: Depth map of shape (H, W) in meters
            apply_filter: Whether to apply denoising filter
            downsample_factor: Factor to reduce point cloud density
            
        Returns:
            Tuple of (points, colors) where:
                - points: (N, 3) array of 3D coordinates
                - colors: (N, 3) array of RGB colors (0-1 range)
        """
        h, w = depth_map.shape
        
        # Apply denoising filter if requested
        if apply_filter:
            depth_map = self.refine_depth(depth_map)
        
        # Create pixel coordinate grids
        u = np.arange(0, w, downsample_factor)
        v = np.arange(0, h, downsample_factor)
        u, v = np.meshgrid(u, v)
        
        # Sample depth and color at grid points
        depth_sampled = depth_map[::downsample_factor, ::downsample_factor]
        rgb_sampled = rgb_image[::downsample_factor, ::downsample_factor]
        
        # Filter out invalid depths (zero or very large)
        valid_mask = (depth_sampled > 0.1) & (depth_sampled < 10.0)
        
        # Back-project to 3D using pinhole camera model
        # X = (u - cx) * Z / fx
        # Y = (v - cy) * Z / fy
        # Z = depth
        z = depth_sampled[valid_mask]
        x = (u[valid_mask] - self.cx) * z / self.fx
        y = (v[valid_mask] - self.cy) * z / self.fy
        
        # Stack into points array
        points = np.stack([x, -y, -z], axis=-1)  # Flip Y and Z for visualization
        
        # Get colors (normalize to 0-1)
        colors = rgb_sampled[valid_mask].astype(np.float32) / 255.0
        
        return points, colors


def create_plotly_pointcloud(
    points: np.ndarray, 
    colors: np.ndarray,
    title: str = "3D Reconstruction",
    point_size: int = 2
) -> dict:
    """
    Create Plotly figure data for point cloud visualization.
    
    Args:
        points: (N, 3) array of 3D coordinates
        colors: (N, 3) array of RGB colors (0-1 range)
        title: Figure title
        point_size: Size of points in visualization
        
    Returns:
        Dictionary with Plotly figure configuration
    """
    # Convert colors to plotly format (rgb strings)
    rgb_strings = [
        f'rgb({int(c[0]*255)},{int(c[1]*255)},{int(c[2]*255)})' 
        for c in colors
    ]
    
    trace = {
        'type': 'scatter3d',
        'x': points[:, 0].tolist(),
        'y': points[:, 1].tolist(),
        'z': points[:, 2].tolist(),
        'mode': 'markers',
        'marker': {
            'size': point_size,
            'color': rgb_strings,
            'opacity': 1.0
        },
        'hoverinfo': 'skip'
    }
    
    layout = {
        'title': {
            'text': title,
            'x': 0.5,
            'xanchor': 'center'
        },
        'scene': {
            'aspectmode': 'data',
            'xaxis': {'visible': False, 'showgrid': False},
            'yaxis': {'visible': False, 'showgrid': False},
            'zaxis': {'visible': False, 'showgrid': False},
            'bgcolor': 'rgb(20, 20, 20)'
        },
        'paper_bgcolor': 'rgb(20, 20, 20)',
        'margin': {'l': 0, 'r': 0, 't': 40, 'b': 0},
        'showlegend': False
    }
    
    return {'data': [trace], 'layout': layout}


def reconstruct_3d_from_image(
    image: Image.Image,
    depth_map: np.ndarray,
    target_size: Tuple[int, int] = (224, 224),
    apply_filter: bool = True,
    downsample_factor: int = 2
) -> Tuple[np.ndarray, np.ndarray]:
    """
    High-level function to create 3D reconstruction from image and depth.
    
    Args:
        image: PIL Image (RGB)
        depth_map: Depth map as numpy array
        target_size: Size to resize inputs to
        apply_filter: Whether to apply denoising filter
        downsample_factor: Point cloud density reduction factor
        
    Returns:
        Tuple of (points, colors) for point cloud
    """
    # Resize image to target size
    image_resized = image.resize(target_size, Image.Resampling.BILINEAR)
    rgb_array = np.array(image_resized)
    
    # Resize depth map if needed
    if depth_map.shape != target_size[::-1]:
        import cv2
        depth_resized = cv2.resize(depth_map, target_size)
    else:
        depth_resized = depth_map
    
    # Generate point cloud
    generator = PointCloudGenerator(image_size=target_size)
    points, colors = generator.generate_point_cloud(
        rgb_array, 
        depth_resized,
        apply_filter=apply_filter,
        downsample_factor=downsample_factor
    )
    
    return points, colors
