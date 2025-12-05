"""
Unit tests for the depth estimation model.
"""

import sys
import os
import unittest

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from PIL import Image

from src.model import ResNetDepthModel, UpSample, create_model
from src.config import config


class TestUpSample(unittest.TestCase):
    """Tests for the UpSample module."""
    
    def test_forward_shape(self):
        """Test that UpSample produces correct output shape."""
        upsample = UpSample(skip_input=512, output_features=256)
        
        # Create dummy tensors
        x = torch.randn(1, 256, 8, 8)
        skip = torch.randn(1, 256, 16, 16)
        
        output = upsample(x, skip)
        
        self.assertEqual(output.shape, (1, 256, 16, 16))


class TestResNetDepthModel(unittest.TestCase):
    """Tests for the ResNetDepthModel."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.model = create_model(pretrained_backbone=False)
        self.model.eval()
    
    def test_forward_shape(self):
        """Test that model produces correct output shape."""
        # Create dummy input (batch_size=1, channels=3, height=480, width=640)
        x = torch.randn(1, 3, 480, 640)
        
        with torch.no_grad():
            output = self.model(x)
        
        # Output should have 1 channel (depth)
        self.assertEqual(output.shape[0], 1)  # batch size
        self.assertEqual(output.shape[1], 1)  # depth channel
    
    def test_output_range(self):
        """Test that output values are in expected range (0-10m)."""
        x = torch.randn(1, 3, 480, 640)
        
        with torch.no_grad():
            output = self.model(x)
        
        self.assertTrue(output.min() >= 0)
        self.assertTrue(output.max() <= 10.0)
    
    def test_parameter_count(self):
        """Test that model has expected number of parameters."""
        num_params = self.model.get_num_parameters()
        
        # ResNet-152 has ~60M parameters, plus decoder
        self.assertGreater(num_params, 50_000_000)


class TestConfig(unittest.TestCase):
    """Tests for configuration."""
    
    def test_input_size(self):
        """Test that input size is valid."""
        self.assertEqual(len(config.INPUT_SIZE), 2)
        self.assertGreater(config.INPUT_SIZE[0], 0)
        self.assertGreater(config.INPUT_SIZE[1], 0)
    
    def test_colormaps(self):
        """Test that colormaps are defined."""
        self.assertGreater(len(config.COLORMAPS), 0)
        self.assertIn(config.DEFAULT_COLORMAP, config.COLORMAPS)
    
    def test_resolution_presets(self):
        """Test that resolution presets are valid."""
        self.assertGreater(len(config.RESOLUTION_PRESETS), 0)
        self.assertIn(config.DEFAULT_RESOLUTION, config.RESOLUTION_PRESETS)


if __name__ == '__main__':
    unittest.main()
