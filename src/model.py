"""
Neural Network Model Definitions for Depth Estimation.

This module contains the ResNet-152 based encoder-decoder architecture
for monocular depth estimation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class UpSample(nn.Module):
    """
    Upsampling block with skip connections for the decoder.
    
    Uses bilinear interpolation followed by two convolutional layers
    with LeakyReLU activation. Concatenates features from the encoder
    via skip connections.
    
    Args:
        skip_input: Number of input channels (upsampled + skip connection)
        output_features: Number of output channels
    """
    
    def __init__(self, skip_input: int, output_features: int):
        super(UpSample, self).__init__()
        self.convA = nn.Conv2d(
            skip_input, output_features, 
            kernel_size=3, stride=1, padding=1
        )
        self.leakyreluA = nn.LeakyReLU(0.2)
        self.convB = nn.Conv2d(
            output_features, output_features, 
            kernel_size=3, stride=1, padding=1
        )
        self.leakyreluB = nn.LeakyReLU(0.2)

    def forward(self, x: torch.Tensor, concat_with: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with skip connection.
        
        Args:
            x: Input tensor from previous decoder layer
            concat_with: Skip connection tensor from encoder
            
        Returns:
            Upsampled and processed tensor
        """
        up_x = F.interpolate(
            x, 
            size=[concat_with.size(2), concat_with.size(3)], 
            mode='bilinear', 
            align_corners=True
        )
        concatenated = torch.cat([up_x, concat_with], dim=1)
        return self.leakyreluB(
            self.convB(
                self.leakyreluA(
                    self.convA(concatenated)
                )
            )
        )


class ResNetDepthModel(nn.Module):
    """
    ResNet-152 based depth estimation model.
    
    Architecture:
        - Encoder: Pre-trained ResNet-152 backbone split into layers
          for skip connections
        - Decoder: Custom upsampling blocks with skip connections
        - Output: Single channel depth map scaled to 0-10 meters
    
    The model uses an encoder-decoder architecture with skip connections
    similar to U-Net, but with ResNet-152 as the encoder backbone.
    
    Feature dimensions at each encoder stage:
        - Layer 0: 64 channels
        - Layer 1: 256 channels  
        - Layer 2: 512 channels
        - Layer 3: 1024 channels
        - Layer 4: 2048 channels
    """
    
    def __init__(self, pretrained_backbone: bool = False):
        """
        Initialize the depth estimation model.
        
        Args:
            pretrained_backbone: Whether to use ImageNet pretrained weights
                                for ResNet-152 encoder (default: False)
        """
        super(ResNetDepthModel, self).__init__()

        # ENCODER: ResNet-152 backbone
        weights = models.ResNet152_Weights.IMAGENET1K_V1 if pretrained_backbone else None
        original_model = models.resnet152(weights=weights)

        # Split ResNet into layers for skip connections
        self.encoder_layer0 = nn.Sequential(
            original_model.conv1, 
            original_model.bn1, 
            original_model.relu
        )  # Output: 64 channels
        
        self.encoder_layer1 = nn.Sequential(
            original_model.maxpool, 
            original_model.layer1
        )  # Output: 256 channels
        
        self.encoder_layer2 = original_model.layer2  # Output: 512 channels
        self.encoder_layer3 = original_model.layer3  # Output: 1024 channels
        self.encoder_layer4 = original_model.layer4  # Output: 2048 channels

        # DECODER: Upsampling blocks with skip connections
        self.up_block1 = UpSample(
            skip_input=2048 + 1024, 
            output_features=1024
        )
        self.up_block2 = UpSample(
            skip_input=1024 + 512, 
            output_features=512
        )
        self.up_block3 = UpSample(
            skip_input=512 + 256, 
            output_features=256
        )
        self.up_block4 = UpSample(
            skip_input=256 + 64, 
            output_features=128
        )

        # Final output layer (1 channel for depth)
        self.final_conv = nn.Conv2d(
            128, 1, 
            kernel_size=3, stride=1, padding=1
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x: Input RGB image tensor of shape (B, 3, H, W)
            
        Returns:
            Depth map tensor of shape (B, 1, H', W') scaled to 0-10 meters
        """
        # Encoder pass (save features for skip connections)
        x0 = self.encoder_layer0(x)   # 64 channels
        x1 = self.encoder_layer1(x0)  # 256 channels
        x2 = self.encoder_layer2(x1)  # 512 channels
        x3 = self.encoder_layer3(x2)  # 1024 channels
        x4 = self.encoder_layer4(x3)  # 2048 channels

        # Decoder pass with skip connections
        d1 = self.up_block1(x4, x3)  # 1024 channels
        d2 = self.up_block2(d1, x2)  # 512 channels
        d3 = self.up_block3(d2, x1)  # 256 channels
        d4 = self.up_block4(d3, x0)  # 128 channels

        # Final depth prediction (scaled to 0-10 meters)
        return torch.sigmoid(self.final_conv(d4)) * 10.0
    
    def get_num_parameters(self) -> int:
        """Get the total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_model(pretrained_backbone: bool = False) -> ResNetDepthModel:
    """
    Factory function to create a depth estimation model.
    
    Args:
        pretrained_backbone: Whether to use ImageNet pretrained weights
        
    Returns:
        Initialized ResNetDepthModel instance
    """
    return ResNetDepthModel(pretrained_backbone=pretrained_backbone)
