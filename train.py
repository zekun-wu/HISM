"""
HISM Model Training Script

Trains the HISM model with Adam optimizer, MSE loss, and callbacks
(ModelCheckpoint, EarlyStopping, ReduceLROnPlateau).

Configuration Control:
----------------------
All training configurations are controlled via YAML files:

1. Temporal Encoder Selection (configs/model.yaml):
   - Set `temporal_encoder.type` to 'transformer' or 'lstm'
   - Example:
     temporal_encoder:
       type: transformer  # Change to 'lstm' for LSTM encoder

2. Task Vector Usage (configs/dataset.yaml):
   - Set `use_task_vector` to true or false
   - true: Uses both highlight vector (v_t) and task vector (c_t)
   - false: Uses only highlight vector (v_t)
   - Example:
     use_task_vector: true  # Change to false to disable task vector
   
   Note: This setting must match in both dataset.yaml and model.yaml

3. Training Hyperparameters (configs/training.yaml):
   - learning_rate: 0.0001 (1e-4)
   - batch_size: 32
   - max_epochs: 1000
   - early_stopping patience: 20
   - reduce_lr_on_plateau: factor=0.8, patience=5

Checkpoint Naming:
- Format: best_model_{encoder}_{task_status}.pth
- Examples: 
  * best_model_tran_task.pth (transformer with task vector)
  * best_model_tran_notask.pth (transformer without task vector)
  * best_model_lstm_task.pth (LSTM with task vector)
  * best_model_lstm_notask.pth (LSTM without task vector)

Simply edit the YAML files before running this script.
"""

import os
import yaml
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np

from models import create_hism_model, load_model_config
from datasets import get_dataloader, load_config


class EarlyStopping:
    """Early stopping callback to stop training when validation loss stops improving."""
    
    def __init__(self, patience: int = 20, min_delta: float = 0.0, mode: str = 'min'):
        """
        Args:
            patience: Number of epochs to wait before stopping
            min_delta: Minimum change to qualify as an improvement
            mode: 'min' or 'max' - whether to minimize or maximize the monitored metric
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        
    def __call__(self, score: float) -> bool:
        """
        Check if training should stop.
        
        Args:
            score: Current value of the monitored metric
            
        Returns:
            True if training should stop, False otherwise
        """
        if self.best_score is None:
            self.best_score = score
        elif self._is_better(score, self.best_score):
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        
        return self.early_stop
    
    def _is_better(self, current: float, best: float) -> bool:
        """Check if current score is better than best score."""
        if self.mode == 'min':
            return current < (best - self.min_delta)
        else:
            return current > (best + self.min_delta)


class ReduceLROnPlateau:
    """Reduce learning rate when a metric has stopped improving."""
    
    def __init__(self, optimizer: torch.optim.Optimizer, factor: float = 0.8, 
                 patience: int = 5, min_lr: float = 1e-4, mode: str = 'min'):
        """
        Args:
            optimizer: PyTorch optimizer
            factor: Factor by which to reduce learning rate
            patience: Number of epochs to wait before reducing LR
            min_lr: Minimum learning rate
            mode: 'min' or 'max' - whether to minimize or maximize the monitored metric
        """
        self.optimizer = optimizer
        self.factor = factor
        self.patience = patience
        self.min_lr = min_lr
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.current_lr = None
        
    def __call__(self, score: float) -> bool:
        """
        Check if learning rate should be reduced.
        
        Args:
            score: Current value of the monitored metric
            
        Returns:
            True if LR was reduced, False otherwise
        """
        if self.best_score is None:
            self.best_score = score
        elif self._is_better(score, self.best_score):
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self._reduce_lr()
                self.counter = 0
                return True
        
        return False
    
    def _is_better(self, current: float, best: float) -> bool:
        """Check if current score is better than best score."""
        if self.mode == 'min':
            return current < best
        else:
            return current > best
    
    def _reduce_lr(self):
        """Reduce learning rate for all parameter groups."""
        for param_group in self.optimizer.param_groups:
            old_lr = param_group['lr']
            new_lr = max(old_lr * self.factor, self.min_lr)
            param_group['lr'] = new_lr
            self.current_lr = new_lr
            print(f"  Learning rate reduced: {old_lr:.6f} -> {new_lr:.6f}")


class ModelCheckpoint:
    """Save model checkpoint when monitored metric improves."""
    
    def __init__(self, filepath: str, mode: str = 'min', save_best_only: bool = True):
        """
        Args:
            filepath: Path to save checkpoint
            mode: 'min' or 'max' - whether to minimize or maximize the monitored metric
            save_best_only: If True, only save when metric improves
        """
        self.filepath = filepath
        self.mode = mode
        self.save_best_only = save_best_only
        self.best_score = None
        
    def __call__(self, model: nn.Module, optimizer: torch.optim.Optimizer, 
                 epoch: int, score: float, **kwargs) -> bool:
        """
        Save checkpoint if metric improved.
        
        Args:
            model: Model to save
            optimizer: Optimizer to save
            epoch: Current epoch
            score: Current value of the monitored metric
            **kwargs: Additional values to save in checkpoint
            
        Returns:
            True if checkpoint was saved, False otherwise
        """
        if self.best_score is None or self._is_better(score, self.best_score):
            self.best_score = score
            
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(self.filepath), exist_ok=True)
            
            # Save checkpoint
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_score': score,
                **kwargs
            }
            torch.save(checkpoint, self.filepath)
            print(f"  ✓ Checkpoint saved: {self.filepath} (val_loss improved: {score:.6f})")
            return True
        
        # Not saving - val_loss did not improve
        return False
    
    def _is_better(self, current: float, best: float) -> bool:
        """Check if current score is better than best score."""
        if self.mode == 'min':
            return current < best
        else:
            return current > best


def train_epoch(model: nn.Module, dataloader: DataLoader, optimizer: torch.optim.Optimizer,
                criterion: nn.Module, device: torch.device) -> Dict[str, float]:
    """
    Train model for one epoch.
    
    Args:
        model: Model to train
        dataloader: Training dataloader
        optimizer: Optimizer
        criterion: Loss function
        device: Device to run on
        
    Returns:
        Dictionary with training metrics
    """
    model.train()
    total_loss = 0.0
    total_mse = 0.0
    num_batches = 0
    
    for batch in dataloader:
        # Move data to device
        spatial_ui = batch['spatial_ui'].to(device)
        spatial_mask = batch['spatial_mask'].to(device)
        temporal_features = batch['temporal_features'].to(device)
        sequence_lengths = batch['sequence_lengths'].to(device)
        targets = batch['targets'].to(device)  # [batch, seq_len]
        
        # Forward pass
        optimizer.zero_grad()
        predictions = model(
            spatial_ui=spatial_ui,
            spatial_mask=spatial_mask,
            temporal_features=temporal_features,
            sequence_lengths=sequence_lengths,
        )  # [batch, 1]
        
        # Average targets to match prediction shape [batch, 1]
        # Only average over valid sequence positions (not padding)
        # Create mask for valid positions
        max_len = targets.shape[1]
        mask = torch.arange(max_len, device=device).unsqueeze(0) < sequence_lengths.unsqueeze(1)
        mask = mask.float()  # [batch, seq_len]
        
        # Masked average: sum valid targets / count valid positions
        target_avg = (targets * mask).sum(dim=1, keepdim=True) / sequence_lengths.unsqueeze(1).float()  # [batch, 1]
        
        # Compute loss
        loss = criterion(predictions, target_avg)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Compute MSE for metrics
        mse = nn.functional.mse_loss(predictions, target_avg, reduction='mean')
        
        total_loss += loss.item()
        total_mse += mse.item()
        num_batches += 1
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    avg_mse = total_mse / num_batches if num_batches > 0 else 0.0
    
    return {
        'loss': avg_loss,
        'mse': avg_mse
    }


def validate(model: nn.Module, dataloader: DataLoader, criterion: nn.Module,
             device: torch.device) -> Dict[str, float]:
    """
    Validate model.
    
    Args:
        model: Model to validate
        dataloader: Validation dataloader
        criterion: Loss function
        device: Device to run on
        
    Returns:
        Dictionary with validation metrics
    """
    model.eval()
    total_loss = 0.0
    total_mse = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for batch in dataloader:
            # Move data to device
            spatial_ui = batch['spatial_ui'].to(device)
            spatial_mask = batch['spatial_mask'].to(device)
            temporal_features = batch['temporal_features'].to(device)
            sequence_lengths = batch['sequence_lengths'].to(device)
            targets = batch['targets'].to(device)  # [batch, seq_len]
            
            # Forward pass
            predictions = model(
                spatial_ui=spatial_ui,
                spatial_mask=spatial_mask,
                temporal_features=temporal_features,
                sequence_lengths=sequence_lengths,
            )  # [batch, 1]
            
            # Average targets to match prediction shape [batch, 1]
            # Create mask for valid positions
            max_len = targets.shape[1]
            mask = torch.arange(max_len, device=device).unsqueeze(0) < sequence_lengths.unsqueeze(1)
            mask = mask.float()  # [batch, seq_len]
            
            # Masked average: sum valid targets / count valid positions
            target_avg = (targets * mask).sum(dim=1, keepdim=True) / sequence_lengths.unsqueeze(1).float()  # [batch, 1]
            
            # Compute loss
            loss = criterion(predictions, target_avg)
            
            # Compute MSE for metrics
            mse = nn.functional.mse_loss(predictions, target_avg, reduction='mean')
            
            total_loss += loss.item()
            total_mse += mse.item()
            num_batches += 1
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    avg_mse = total_mse / num_batches if num_batches > 0 else 0.0
    
    return {
        'loss': avg_loss,
        'mse': avg_mse
    }


def save_checkpoint(model: nn.Module, optimizer: torch.optim.Optimizer, epoch: int,
                    val_loss: float, checkpoint_path: str, **kwargs):
    """
    Save model checkpoint.
    
    Args:
        model: Model to save
        optimizer: Optimizer to save
        epoch: Current epoch
        val_loss: Current validation loss
        checkpoint_path: Path to save checkpoint
        **kwargs: Additional values to save
    """
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_loss': val_loss,
        **kwargs
    }
    
    torch.save(checkpoint, checkpoint_path)
    print(f"  Checkpoint saved: {checkpoint_path}")


def load_training_config(config_path: str = "configs/training.yaml") -> Dict:
    """Load training configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def get_model_type(model_config: Dict) -> str:
    """
    Get model type string for checkpoint naming.
    
    Returns format: {encoder_type}_{task_vector_status}
    Examples: tran_task, tran_notask, lstm_task, lstm_notask
    """
    encoder_type = model_config.get('temporal_encoder', {}).get('type', 'transformer')
    use_task_vector = model_config.get('temporal_encoder', {}).get('use_task_vector', True)
    
    # Get encoder abbreviation
    if encoder_type == 'transformer':
        encoder_abbr = 'tran'
    elif encoder_type == 'lstm':
        encoder_abbr = 'lstm'
    else:
        encoder_abbr = 'unknown'
    
    # Add task vector status
    task_status = 'task' if use_task_vector else 'notask'
    
    return f"{encoder_abbr}_{task_status}"


def main():
    """Main training function."""
    print("=" * 60)
    print("HISM Model Training")
    print("=" * 60)
    
    # Load configurations
    print("\n1. Loading configurations...")
    training_config = load_training_config("configs/training.yaml")
    model_config = load_model_config("configs/model.yaml")
    
    print(f"   ✓ Training config loaded")
    print(f"   ✓ Model config loaded")
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n2. Device: {device}")
    if torch.cuda.is_available():
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
    
    # Create dataloaders
    print("\n3. Creating dataloaders...")
    train_loader = get_dataloader(
        split='train',
        config_path="configs/dataset.yaml",
        batch_size=training_config['batch_size'],
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = get_dataloader(
        split='val',
        config_path="configs/dataset.yaml",
        batch_size=training_config['batch_size'],
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )
    print(f"   ✓ Train dataloader: {len(train_loader.dataset)} samples")
    print(f"   ✓ Val dataloader: {len(val_loader.dataset)} samples")
    
    # Create model
    print("\n4. Creating model...")
    model = create_hism_model(
        model_config_path="configs/model.yaml",
        dataset_config_path="configs/dataset.yaml",
    )
    model = model.to(device)
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   ✓ Model created")
    print(f"   - Total parameters: {num_params:,}")
    print(f"   - Trainable parameters: {trainable_params:,}")
    
    # Initialize optimizer
    print("\n5. Initializing optimizer and loss...")
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=training_config['learning_rate']
    )
    criterion = nn.MSELoss()
    print(f"   ✓ Optimizer: {training_config['optimizer']} (lr={training_config['learning_rate']})")
    print(f"   ✓ Loss: {training_config['loss']}")
    
    # Setup callbacks
    print("\n6. Setting up callbacks...")
    model_type = get_model_type(model_config)
    checkpoint_dir = training_config.get('checkpoint_dir', 'checkpoints')
    checkpoint_path = os.path.join(
        checkpoint_dir,
        f"best_model_{model_type}.pth"
    )
    
    early_stopping = EarlyStopping(
        patience=training_config['early_stopping']['patience'],
        mode=training_config['early_stopping']['mode']
    )
    
    reduce_lr = ReduceLROnPlateau(
        optimizer=optimizer,
        factor=training_config['reduce_lr_on_plateau']['factor'],
        patience=training_config['reduce_lr_on_plateau']['patience'],
        min_lr=training_config['reduce_lr_on_plateau']['min_lr'],
        mode=training_config['reduce_lr_on_plateau']['mode']
    )
    
    model_checkpoint = ModelCheckpoint(
        filepath=checkpoint_path,
        mode=training_config['model_checkpoint']['mode'],
        save_best_only=training_config['model_checkpoint']['save_best_only']
    )
    
    print(f"   ✓ EarlyStopping: patience={early_stopping.patience}")
    print(f"   ✓ ReduceLROnPlateau: factor={reduce_lr.factor}, patience={reduce_lr.patience}")
    print(f"   ✓ ModelCheckpoint: {checkpoint_path}")
    
    # Training loop
    print("\n7. Starting training...")
    print("=" * 60)
    
    max_epochs = training_config['max_epochs']
    history = {
        'train_loss': [],
        'train_mse': [],
        'val_loss': [],
        'val_mse': [],
        'learning_rate': []
    }
    
    for epoch in range(1, max_epochs + 1):
        # Train
        train_metrics = train_epoch(model, train_loader, optimizer, criterion, device)
        
        # Validate
        val_metrics = validate(model, val_loader, criterion, device)
        
        # Update history
        history['train_loss'].append(train_metrics['loss'])
        history['train_mse'].append(train_metrics['mse'])
        history['val_loss'].append(val_metrics['loss'])
        history['val_mse'].append(val_metrics['mse'])
        history['learning_rate'].append(optimizer.param_groups[0]['lr'])
        
        # Log epoch results
        print(f"Epoch {epoch}/{max_epochs}")
        print(f"  Train Loss: {train_metrics['loss']:.6f}, Train MSE: {train_metrics['mse']:.6f}")
        print(f"  Val Loss: {val_metrics['loss']:.6f}, Val MSE: {val_metrics['mse']:.6f}")
        print(f"  LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Check callbacks
        # ModelCheckpoint
        model_checkpoint(model, optimizer, epoch, val_metrics['loss'])
        
        # ReduceLROnPlateau
        reduce_lr(val_metrics['loss'])
        
        # EarlyStopping
        if early_stopping(val_metrics['loss']):
            print(f"\n  Early stopping triggered after {epoch} epochs")
            print(f"  Best val_loss: {early_stopping.best_score:.6f}")
            break
        
        print()
    
    # Save training history
    if training_config.get('save_history', True):
        log_dir = training_config.get('log_dir', 'logs')
        os.makedirs(log_dir, exist_ok=True)
        history_path = os.path.join(log_dir, 'training_history.json')
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=2)
        print(f"Training history saved to: {history_path}")
    
    print("=" * 60)
    print("Training completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
