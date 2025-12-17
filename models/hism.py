"""
Unified HISM Model

Combines spatial encoder, temporal encoder, and fusion module to predict
normalized saliency from UI images, highlight masks, and temporal sequences.
"""

import torch
import torch.nn as nn
import yaml
from typing import Dict, Optional

from .spatial import SpatialEncoder
from .temporal import TemporalEncoder
from .fusion import NSPredictor


class HISMModel(nn.Module):
    """
    Unified HISM Model combining spatial, temporal, and fusion components.
    
    Architecture:
    1. Spatial Encoder: Encodes UI image + mask → spatial features
    2. Temporal Encoder: Encodes temporal sequence (v_t, c_t) → temporal features
    3. Fusion Module: Concatenates features and predicts NS(e,t)
    """
    
    def __init__(
        self,
        # Spatial encoder config
        spatial_encoder_config: Dict,
        # Temporal encoder config
        temporal_encoder_config: Dict,
        # Fusion config
        fusion_config: Dict,
    ):
        """
        Initialize HISM Model.
        
        Args:
            spatial_encoder_config: Configuration dict for SpatialEncoder
            temporal_encoder_config: Configuration dict for TemporalEncoder
            fusion_config: Configuration dict for NSPredictor
        """
        super(HISMModel, self).__init__()
        
        # Determine input dimension for temporal encoder
        use_task_vector = temporal_encoder_config.get('use_task_vector', True)
        input_dim = 2 if use_task_vector else 1
        
        # Create spatial encoder
        self.spatial_encoder = SpatialEncoder(
            freeze_backbone=spatial_encoder_config.get('freeze_backbone', True),
            unfreeze_last_n_layers=spatial_encoder_config.get('unfreeze_last_n_layers', 20),
            pretrained=spatial_encoder_config.get('pretrained', True),
        )
        
        # Create temporal encoder
        encoder_type = temporal_encoder_config.get('type', 'transformer')
        self.temporal_encoder = TemporalEncoder(
            encoder_type=encoder_type,
            input_dim=input_dim,
            embed_dim=temporal_encoder_config.get('embed_dim', 64),
            num_heads=temporal_encoder_config.get('num_heads', 8),
            num_layers=temporal_encoder_config.get('num_layers', 2),
            ff_dim=temporal_encoder_config.get('ff_dim', 128),
            dropout=temporal_encoder_config.get('dropout', 0.1),
            max_seq_len=temporal_encoder_config.get('max_seq_len', 100),
            lstm_hidden_sizes=temporal_encoder_config.get('lstm_hidden_sizes', [256, 64, 8]),
            lstm_dropout=temporal_encoder_config.get('lstm_dropout', 0.2),
        )
        
        # Create fusion module
        self.fusion = NSPredictor(
            hidden_dims=fusion_config.get('hidden_dims', [512, 128]),
            dropout=fusion_config.get('dropout', 0.2),
        )
    
    def forward(
        self,
        spatial_ui: torch.Tensor,
        spatial_mask: torch.Tensor,
        temporal_features: torch.Tensor,
        sequence_lengths: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass through HISM model.
        
        Args:
            spatial_ui: UI image tensor [batch, 3, H, W]
            spatial_mask: Highlight mask tensor [batch, 1, H, W]
            temporal_features: Temporal sequence [batch, seq_len, input_dim]
                              - input_dim=1: [v_t] only
                              - input_dim=2: [v_t, c_t]
            sequence_lengths: Optional [batch] tensor with actual sequence lengths
                            (for handling variable-length sequences)
        
        Returns:
            Normalized saliency predictions [batch, 1] in [0, 1] range
        """
        # Encode spatial features
        spatial_features = self.spatial_encoder(spatial_ui, spatial_mask)  # [batch, 2048]
        
        # Encode temporal features
        temporal_features_vec = self.temporal_encoder(
            temporal_features, 
            sequence_lengths=sequence_lengths
        )  # [batch, temporal_dim]
        
        # Fuse and predict
        predictions = self.fusion(spatial_features, temporal_features_vec)  # [batch, 1]
        
        return predictions


def load_model_config(config_path: str = "configs/model.yaml") -> Dict:
    """
    Load model configuration from YAML file.
    
    Args:
        config_path: Path to model configuration YAML file
    
    Returns:
        Configuration dictionary
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def create_hism_model(
    model_config_path: str = "configs/model.yaml",
    dataset_config_path: str = "configs/dataset.yaml",
) -> HISMModel:
    """
    Create HISM model from configuration files.
    
    Args:
        model_config_path: Path to model configuration YAML file
        dataset_config_path: Path to dataset configuration YAML file
                          (needed to check use_task_vector setting)
    
    Returns:
        Initialized HISMModel instance
    """
    import yaml
    
    # Load model config
    model_config = load_model_config(model_config_path)
    
    # Load dataset config to check use_task_vector
    with open(dataset_config_path, 'r') as f:
        dataset_config = yaml.safe_load(f)
    
    # Ensure temporal encoder config has use_task_vector from dataset config
    if 'temporal_encoder' in model_config:
        model_config['temporal_encoder']['use_task_vector'] = dataset_config.get(
            'use_task_vector', 
            model_config['temporal_encoder'].get('use_task_vector', True)
        )
    
    # Create model
    model = HISMModel(
        spatial_encoder_config=model_config.get('spatial_encoder', {}),
        temporal_encoder_config=model_config.get('temporal_encoder', {}),
        fusion_config=model_config.get('fusion', {}),
    )
    
    return model
