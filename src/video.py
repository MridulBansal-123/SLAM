"""
Video processing utilities for depth estimation.

This module provides functions for processing video files
with depth estimation.
"""

import os
import tempfile
from typing import Callable, Optional, Tuple

import cv2
import numpy as np
from PIL import Image

from .config import config
from .inference import DepthEstimator, colorize_depth_bgr


class VideoProcessor:
    """
    Video processing class for depth estimation.
    
    Handles reading, processing, and writing video files with
    frame-by-frame depth estimation.
    """
    
    def __init__(self, depth_estimator: DepthEstimator):
        """
        Initialize the video processor.
        
        Args:
            depth_estimator: Initialized DepthEstimator instance
        """
        self.estimator = depth_estimator
    
    @staticmethod
    def get_video_info(video_path: str) -> dict:
        """
        Get video metadata.
        
        Args:
            video_path: Path to video file
            
        Returns:
            Dictionary with video properties
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            return {}
        
        info = {
            "fps": int(cap.get(cv2.CAP_PROP_FPS)),
            "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "total_frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            "duration": 0.0
        }
        
        if info["fps"] > 0:
            info["duration"] = info["total_frames"] / info["fps"]
        
        cap.release()
        return info
    
    def process_video(
        self,
        video_path: str,
        colormap: str = "magma",
        progress_callback: Optional[Callable[[float], None]] = None,
        status_callback: Optional[Callable[[str], None]] = None
    ) -> Tuple[Optional[str], Optional[str]]:
        """
        Process video and generate depth estimation for each frame.
        
        Args:
            video_path: Path to input video
            colormap: Colormap for depth visualization
            progress_callback: Optional callback for progress updates (0-1)
            status_callback: Optional callback for status messages
            
        Returns:
            Tuple of (output_path, error_message)
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            return None, "Failed to open video file"
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Create temporary output file
        temp_output = tempfile.NamedTemporaryFile(
            delete=False, 
            suffix='.mp4'
        )
        temp_output_path = temp_output.name
        temp_output.close()
        
        # Initialize video writer
        fourcc = cv2.VideoWriter_fourcc(*config.VIDEO_CODEC)
        out = cv2.VideoWriter(
            temp_output_path, fourcc, fps, 
            (frame_width, frame_height)
        )
        
        # Process each frame
        frame_count = 0
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Convert BGR to RGB and then to PIL Image
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(frame_rgb)
                
                # Predict depth
                depth_map = self.estimator.estimate_depth(pil_image)
                
                # Create depth visualization frame
                depth_frame = colorize_depth_bgr(depth_map, colormap)
                
                # Resize to match original video dimensions
                depth_frame_resized = cv2.resize(
                    depth_frame, 
                    (frame_width, frame_height)
                )
                
                # Write frame
                out.write(depth_frame_resized)
                
                frame_count += 1
                
                # Update callbacks
                if progress_callback and total_frames > 0:
                    progress_callback(frame_count / total_frames)
                if status_callback:
                    status_callback(f"Processing frame {frame_count}/{total_frames}")
                    
        finally:
            cap.release()
            out.release()
        
        return temp_output_path, None
    
    def process_video_side_by_side(
        self,
        video_path: str,
        colormap: str = "magma",
        progress_callback: Optional[Callable[[float], None]] = None,
        status_callback: Optional[Callable[[str], None]] = None
    ) -> Tuple[Optional[str], Optional[str]]:
        """
        Process video with side-by-side comparison (original + depth).
        
        Args:
            video_path: Path to input video
            colormap: Colormap for depth visualization
            progress_callback: Optional callback for progress updates (0-1)
            status_callback: Optional callback for status messages
            
        Returns:
            Tuple of (output_path, error_message)
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            return None, "Failed to open video file"
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Create temporary output file
        temp_output = tempfile.NamedTemporaryFile(
            delete=False, 
            suffix='.mp4'
        )
        temp_output_path = temp_output.name
        temp_output.close()
        
        # Initialize video writer (double width for side-by-side)
        fourcc = cv2.VideoWriter_fourcc(*config.VIDEO_CODEC)
        out = cv2.VideoWriter(
            temp_output_path, fourcc, fps, 
            (frame_width * 2, frame_height)
        )
        
        # Process each frame
        frame_count = 0
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Convert BGR to RGB and then to PIL Image
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(frame_rgb)
                
                # Predict depth
                depth_map = self.estimator.estimate_depth(pil_image)
                
                # Create depth visualization frame
                depth_frame = colorize_depth_bgr(depth_map, colormap)
                
                # Resize to match original video dimensions
                depth_frame_resized = cv2.resize(
                    depth_frame, 
                    (frame_width, frame_height)
                )
                
                # Create side-by-side frame
                combined_frame = np.hstack([frame, depth_frame_resized])
                
                # Write frame
                out.write(combined_frame)
                
                frame_count += 1
                
                # Update callbacks
                if progress_callback and total_frames > 0:
                    progress_callback(frame_count / total_frames)
                if status_callback:
                    status_callback(f"Processing frame {frame_count}/{total_frames}")
                    
        finally:
            cap.release()
            out.release()
        
        return temp_output_path, None
