"""
NS Predictor (Fusion Module) Implementation

Concatenates spatial and temporal features, then predicts normalized saliency
through a multi-layer perceptron. Automatically infers input dimensions.
"""

import torch
import torch.nn as nn


class NSPredictor(nn.Module):
    """
    NS Predictor that fuses spatial and temporal features to predict normalized saliency.
    
    Architecture:
    - Concatenates spatial and temporal features (dimensions inferred automatically)
    - Passes through MLP: Linear(512) → ReLU → Dropout → Linear(128) → ReLU → Linear(1) → Sigmoid
    - Outputs: [batch, 1] normalized saliency values in [0, 1] range
    """
    
    def __init__(
        self,
        hidden_dims: list = [512, 128],
        dropout: float = 0.2,
    ):
        """
        Initialize NS Predictor.
        
        Args:
            hidden_dims: List of hidden layer dimensions [512, 128]
            dropout: Dropout rate (default 0.2)
        """
        super(NSPredictor, self).__init__()
        
        self.hidden_dims = hidden_dims
        self.dropout = dropout
        self.mlp = None  # Will be built on first forward pass
        self._input_dim = None  # Track input dimension
    
    def _build_mlp(self, input_dim: int):
        """Build MLP layers based on input dimension."""
        layers = []
        
        # First hidden layer: Linear + ReLU + Dropout
        layers.append(nn.Linear(input_dim, self.hidden_dims[0]))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(self.dropout))
        
        # Second hidden layer: Linear + ReLU
        if len(self.hidden_dims) > 1:
            layers.append(nn.Linear(self.hidden_dims[0], self.hidden_dims[1]))
            layers.append(nn.ReLU())
        
        # Output layer: Linear + Sigmoid
        output_dim = self.hidden_dims[-1] if len(self.hidden_dims) > 0 else self.hidden_dims[0]
        layers.append(nn.Linear(output_dim, 1))
        layers.append(nn.Sigmoid())
        
        self.mlp = nn.Sequential(*layers)
        self._input_dim = input_dim
    
    def forward(
        self,
        spatial_features: torch.Tensor,
        temporal_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass through NS Predictor.
        
        Args:
            spatial_features: Spatial feature vector [batch, spatial_dim] 
                             (dimension inferred automatically)
            temporal_features: Temporal feature vector [batch, temporal_dim]
                              (dimension inferred automatically)
        
        Returns:
            Normalized saliency predictions [batch, 1] in [0, 1] range
        """
        # Concatenate spatial and temporal features
        fused_features = torch.cat([spatial_features, temporal_features], dim=1)
        
        # Infer input dimension from concatenated features
        input_dim = fused_features.size(1)
        
        # Build MLP on first forward pass if not already built
        if self.mlp is None or self._input_dim != input_dim:
            self._build_mlp(input_dim)
        
        # Pass through MLP
        predictions = self.mlp(fused_features)  # [batch, 1]
        
        return predictions
