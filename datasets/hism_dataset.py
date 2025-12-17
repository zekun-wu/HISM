"""
HISM Dataset Implementation

Loads trial data from data/trials/ directories with temporal sequences,
static UI images, and highlight masks.
"""

import os
import yaml
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from typing import Dict, List, Optional, Tuple


def load_split_file(splits_dir: str, split: str) -> List[str]:
    """
    Load trial IDs from a split file.
    
    Args:
        splits_dir: Directory containing split files
        split: Split name ('train', 'val', 'test')
    
    Returns:
        List of trial IDs (strings)
    """
    split_file = os.path.join(splits_dir, f"{split}.txt")
    if not os.path.exists(split_file):
        raise FileNotFoundError(f"Split file not found: {split_file}")
    
    trial_ids = []
    with open(split_file, 'r') as f:
        for line in f:
            line = line.strip()
            # Skip empty lines and comments
            if line and not line.startswith('#'):
                trial_ids.append(line)
    
    return trial_ids


class HISMDataset(Dataset):
    """
    HISM Dataset for loading trial data.
    
    Each trial contains:
    - Temporal sequence: CSV with normalized_saliency_smoothed, Highlight, Task
    - Spatial context: UI image (global.png) and highlight mask (mask.png)
    """
    
    def __init__(
        self,
        data_root: str,
        splits_dir: str,
        split: str,
        csv_filename: str = "saliency_summary.csv",
        ui_image: str = "global.png",
        mask_image: str = "mask.png",
        target_column: str = "normalized_saliency_smoothed",
        highlight_column: str = "Highlight",
        task_column: str = "Task",
        image_size: Tuple[int, int] = (224, 224),
        normalize_images: bool = True,
        imagenet_mean: List[float] = [0.485, 0.456, 0.406],
        imagenet_std: List[float] = [0.229, 0.224, 0.225],
        use_task_vector: bool = True,
    ):
        """
        Initialize HISM Dataset.
        
        Args:
            data_root: Root directory containing trial folders
            splits_dir: Directory containing split files (train.txt, val.txt, test.txt)
            split: Split name ('train', 'val', 'test')
            csv_filename: Name of CSV file in each trial directory
            ui_image: Name of UI image file
            mask_image: Name of mask image file
            target_column: Column name for target NS(e,t)
            highlight_column: Column name for highlight vector v_t
            task_column: Column name for task feature c_t
            image_size: Target image size (height, width)
            normalize_images: Whether to normalize UI images
            imagenet_mean: Mean for ImageNet normalization
            imagenet_std: Std for ImageNet normalization
            use_task_vector: If True, use both v_t and c_t. If False, use only v_t.
        """
        self.data_root = data_root
        self.splits_dir = splits_dir
        self.split = split
        self.csv_filename = csv_filename
        self.ui_image = ui_image
        self.mask_image = mask_image
        self.target_column = target_column
        self.highlight_column = highlight_column
        self.task_column = task_column
        self.image_size = image_size
        self.normalize_images = normalize_images
        self.use_task_vector = use_task_vector
        
        # Load trial IDs for this split
        self.trial_ids = load_split_file(splits_dir, split)
        
        # Validate that all trials exist
        self._validate_trials()
        
        # Setup image transforms
        self._setup_transforms(imagenet_mean, imagenet_std)
    
    def _validate_trials(self):
        """Validate that all trial directories exist and have required files."""
        missing_trials = []
        missing_files = []
        
        for trial_id in self.trial_ids:
            trial_dir = os.path.join(self.data_root, trial_id)
            if not os.path.exists(trial_dir):
                missing_trials.append(trial_id)
                continue
            
            # Check for required files
            csv_path = os.path.join(trial_dir, self.csv_filename)
            ui_path = os.path.join(trial_dir, self.ui_image)
            mask_path = os.path.join(trial_dir, self.mask_image)
            
            if not os.path.exists(csv_path):
                missing_files.append(f"{trial_id}/{self.csv_filename}")
            if not os.path.exists(ui_path):
                missing_files.append(f"{trial_id}/{self.ui_image}")
            if not os.path.exists(mask_path):
                missing_files.append(f"{trial_id}/{self.mask_image}")
        
        if missing_trials:
            raise ValueError(f"Missing trial directories: {missing_trials}")
        if missing_files:
            raise ValueError(f"Missing required files: {missing_files}")
    
    def _setup_transforms(self, imagenet_mean: List[float], imagenet_std: List[float]):
        """Setup image transformation pipelines."""
        # UI image transforms: resize, convert to tensor, normalize
        ui_transforms = [
            transforms.Resize(self.image_size),
            transforms.ToTensor(),
        ]
        if self.normalize_images:
            ui_transforms.append(transforms.Normalize(mean=imagenet_mean, std=imagenet_std))
        self.ui_transform = transforms.Compose(ui_transforms)
        
        # Mask transforms: resize, convert to grayscale, binarize, convert to tensor
        mask_transforms = [
            transforms.Resize(self.image_size),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
        ]
        self.mask_transform = transforms.Compose(mask_transforms)
    
    def _load_csv(self, trial_dir: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Load CSV file and extract temporal features and targets.
        
        Args:
            trial_dir: Directory containing the trial data
        
        Returns:
            Tuple of (temporal_features, targets)
            - temporal_features: [seq_len, 1] or [seq_len, 2] tensor
              - If use_task_vector=True: [v_t, c_t] per timestep → [seq_len, 2]
              - If use_task_vector=False: [v_t] per timestep → [seq_len, 1]
            - targets: [seq_len] tensor with NS(e,t) values
        """
        csv_path = os.path.join(trial_dir, self.csv_filename)
        df = pd.read_csv(csv_path)
        
        # Extract columns
        if self.target_column not in df.columns:
            raise ValueError(f"Target column '{self.target_column}' not found in CSV")
        if self.highlight_column not in df.columns:
            raise ValueError(f"Highlight column '{self.highlight_column}' not found in CSV")
        if self.task_column not in df.columns:
            raise ValueError(f"Task column '{self.task_column}' not found in CSV")
        
        targets = df[self.target_column].values.astype(np.float32)
        v_t = df[self.highlight_column].values.astype(np.float32)
        c_t = df[self.task_column].values.astype(np.float32)
        
        # Conditionally stack temporal features based on use_task_vector
        if self.use_task_vector:
            # Stack [v_t, c_t] per timestep
            temporal_features = np.stack([v_t, c_t], axis=1)  # [seq_len, 2]
        else:
            # Use only v_t
            temporal_features = v_t[:, np.newaxis]  # [seq_len, 1]
        
        # Convert to tensors
        temporal_features = torch.from_numpy(temporal_features)
        targets = torch.from_numpy(targets)
        
        return temporal_features, targets
    
    def _load_images(self, trial_dir: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Load UI image and mask image.
        
        Args:
            trial_dir: Directory containing the trial data
        
        Returns:
            Tuple of (ui_image, mask)
            - ui_image: [3, H, W] tensor (normalized RGB)
            - mask: [1, H, W] tensor (binary mask)
        """
        ui_path = os.path.join(trial_dir, self.ui_image)
        mask_path = os.path.join(trial_dir, self.mask_image)
        
        # Load UI image (ensure RGB)
        ui_img = Image.open(ui_path).convert('RGB')
        ui_tensor = self.ui_transform(ui_img)  # [3, H, W]
        
        # Load mask image
        mask_img = Image.open(mask_path)
        mask_tensor = self.mask_transform(mask_img)  # [1, H, W]
        
        # Binarize mask (threshold at 0.5)
        mask_tensor = (mask_tensor > 0.5).float()
        
        return ui_tensor, mask_tensor
    
    def __len__(self) -> int:
        """Return number of trials in this split."""
        return len(self.trial_ids)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single trial's data.
        
        Args:
            idx: Index of trial in split
        
        Returns:
            Dictionary containing:
            - 'trial_id': str
            - 'temporal_features': [seq_len, 1] or [seq_len, 2] tensor
              (shape depends on use_task_vector)
            - 'spatial_ui': [3, H, W] tensor
            - 'spatial_mask': [1, H, W] tensor
            - 'targets': [seq_len] tensor
            - 'sequence_length': int
        """
        trial_id = self.trial_ids[idx]
        trial_dir = os.path.join(self.data_root, trial_id)
        
        # Load temporal data
        temporal_features, targets = self._load_csv(trial_dir)
        seq_len = temporal_features.shape[0]
        
        # Load spatial data
        ui_tensor, mask_tensor = self._load_images(trial_dir)
        
        return {
            'trial_id': trial_id,
            'temporal_features': temporal_features,  # [seq_len, 2]
            'spatial_ui': ui_tensor,  # [3, H, W]
            'spatial_mask': mask_tensor,  # [1, H, W]
            'targets': targets,  # [seq_len]
            'sequence_length': seq_len,
        }


def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """
    Custom collate function to handle variable-length sequences.
    
    Pads temporal sequences to the maximum length in the batch.
    
    Args:
        batch: List of samples from dataset 
    
    Returns:
        Batched dictionary with padded sequences
    """
    # Get maximum sequence length in batch
    max_seq_len = max(item['sequence_length'] for item in batch)
    batch_size = len(batch)
    
    # Get spatial dimensions (should be same for all)
    spatial_ui_shape = batch[0]['spatial_ui'].shape  # [3, H, W]
    spatial_mask_shape = batch[0]['spatial_mask'].shape  # [1, H, W]
    temporal_feature_dim = batch[0]['temporal_features'].shape[1]  # 2
    
    # Initialize batched tensors
    batched_temporal_features = torch.zeros(batch_size, max_seq_len, temporal_feature_dim)
    batched_targets = torch.zeros(batch_size, max_seq_len)
    batched_spatial_ui = torch.zeros(batch_size, *spatial_ui_shape)
    batched_spatial_mask = torch.zeros(batch_size, *spatial_mask_shape)
    sequence_lengths = torch.zeros(batch_size, dtype=torch.long)
    trial_ids = []
    
    for i, item in enumerate(batch):
        seq_len = item['sequence_length']
        
        # Pad temporal features and targets
        batched_temporal_features[i, :seq_len, :] = item['temporal_features']
        batched_targets[i, :seq_len] = item['targets']
        
        # Spatial features (no padding needed)
        batched_spatial_ui[i] = item['spatial_ui']
        batched_spatial_mask[i] = item['spatial_mask']
        
        sequence_lengths[i] = seq_len
        trial_ids.append(item['trial_id'])
    
    return {
        'trial_id': trial_ids,
        'temporal_features': batched_temporal_features,  # [batch_size, max_seq_len, 2]
        'spatial_ui': batched_spatial_ui,  # [batch_size, 3, H, W]
        'spatial_mask': batched_spatial_mask,  # [batch_size, 1, H, W]
        'targets': batched_targets,  # [batch_size, max_seq_len]
        'sequence_lengths': sequence_lengths,  # [batch_size]
    }


def load_config(config_path: str) -> dict:
    """Load YAML configuration file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def get_dataloader(
    split: str,
    config_path: str = "configs/dataset.yaml",
    batch_size: int = 1,
    shuffle: bool = False,
    num_workers: int = 0,
    pin_memory: bool = False,
) -> DataLoader:
    """
    Factory function to create DataLoader for a given split.
    
    Args:
        split: Split name ('train', 'val', 'test')
        config_path: Path to dataset configuration YAML file
        batch_size: Batch size
        shuffle: Whether to shuffle data (typically True for train, False for val/test)
        num_workers: Number of worker processes for data loading
        pin_memory: Whether to pin memory for faster GPU transfer
    
    Returns:
        DataLoader instance
    """
    # Load configuration
    config = load_config(config_path)
    
    # Create dataset
    dataset = HISMDataset(
        data_root=config['data_root'],
        splits_dir=config['splits_dir'],
        split=split,
        csv_filename=config['csv_filename'],
        ui_image=config['ui_image'],
        mask_image=config['mask_image'],
        target_column=config['target_column'],
        highlight_column=config['highlight_column'],
        task_column=config['task_column'],
        image_size=tuple(config['image_size']),
        normalize_images=config['normalize_images'],
        imagenet_mean=config.get('imagenet_mean', [0.485, 0.456, 0.406]),
        imagenet_std=config.get('imagenet_std', [0.229, 0.224, 0.225]),
        use_task_vector=config.get('use_task_vector', True),
    )
    
    # Create DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
    )
    
    return dataloader
