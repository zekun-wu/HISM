"""
Spatial Encoder Implementation

Uses ResNet50 to encode UI image and highlight mask into a spatial feature vector.
"""

import torch
import torch.nn as nn
import torchvision.models as models


class SpatialEncoder(nn.Module):
    """
    Spatial Encoder using ResNet50.
    
    Encodes UI image and highlight mask into a spatial feature vector.
    Takes UI [batch, 3, H, W] and mask [batch, 1, H, W], concatenates to 4 channels,
    and passes through ResNet50 backbone to output [batch, 2048] feature vector.
    """
    
    def __init__(
        self,
        freeze_backbone: bool = True,
        unfreeze_last_n_layers: int = 20,
        pretrained: bool = True,
    ):
        """
        Initialize Spatial Encoder.
        
        Args:
            freeze_backbone: If True, freeze all ResNet layers initially
            unfreeze_last_n_layers: Number of last layers to unfreeze for fine-tuning
            pretrained: If True, load ImageNet pretrained weights
        """
        super(SpatialEncoder, self).__init__()
        
        # Load ResNet50 with ImageNet weights (without final fc layer)
        resnet = models.resnet50(weights='IMAGENET1K_V2' if pretrained else None)
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])
        
        # Modify first conv layer to accept 4 channels (RGB + mask)
        self._modify_first_conv()
        
        # Global average pooling
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Freeze/unfreeze layers
        if freeze_backbone:
            self._freeze_layers(unfreeze_last_n_layers)
    
    def _modify_first_conv(self):
        """Modify first conv layer to accept 4 channels (UI 3ch + mask 1ch)."""
        old_conv = self.backbone[0]
        
        # Create new conv layer with 4 input channels
        new_conv = nn.Conv2d(
            in_channels=4,
            out_channels=old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=old_conv.bias is not None
        )
        
        # Copy pretrained weights for RGB channels (first 3 channels)
        # Initialize 4th channel (mask) with kaiming_normal
        with torch.no_grad():
            new_conv.weight[:, :3] = old_conv.weight
            nn.init.kaiming_normal_(new_conv.weight[:, 3:], mode='fan_out', nonlinearity='relu')
            if new_conv.bias is not None:
                new_conv.bias = old_conv.bias
        
        # Replace the first conv layer
        self.backbone[0] = new_conv
    
    def _freeze_layers(self, unfreeze_last_n: int):
        """
        Freeze all layers, then unfreeze the last N layers.
        
        Args:
            unfreeze_last_n: Number of last layers to unfreeze for fine-tuning
        """
        # Freeze all parameters
        for param in self.backbone.parameters():
            param.requires_grad = False
        
        # Unfreeze the last N layers
        layers = list(self.backbone.children())
        layers_to_unfreeze = min(unfreeze_last_n, len(layers))
        
        for module in layers[-layers_to_unfreeze:]:
            for param in module.parameters():
                param.requires_grad = True
    
    def forward(self, ui_image: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through spatial encoder.
        
        Args:
            ui_image: UI image tensor [batch, 3, H, W]
            mask: Highlight mask tensor [batch, 1, H, W]
        
        Returns:
            Spatial feature vector [batch, 2048]
        """
        # Concatenate UI (3 channels) + mask (1 channel) = 4 channels
        x = torch.cat([ui_image, mask], dim=1)  # [batch, 4, H, W]
        
        # Pass through ResNet50 backbone
        x = self.backbone(x)  # [batch, 2048, H', W']
        
        # Global average pooling
        x = self.avgpool(x)  # [batch, 2048, 1, 1]
        x = x.view(x.size(0), -1)  # [batch, 2048]
        
        return x
