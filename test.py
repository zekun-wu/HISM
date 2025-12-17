"""
HISM Model Testing Script

Evaluates trained HISM model on test dataset and computes MSE and MAE metrics.
"""

import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from models import create_hism_model, load_model_config
from datasets import get_dataloader, load_config


def load_checkpoint(checkpoint_path: str, model: torch.nn.Module, device: torch.device):
    """
    Load trained model checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint file
        model: Model to load weights into
        device: Device to load checkpoint on
        
    Returns:
        Checkpoint dictionary with metadata
    """
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Initialize MLP if it hasn't been built yet (NSPredictor builds dynamically)
    # We need to do a dummy forward pass to build the MLP before loading weights
    # Create minimal dummy inputs to initialize the MLP
    batch_size = 1
    spatial_ui_dummy = torch.zeros(batch_size, 3, 224, 224, device=device)
    spatial_mask_dummy = torch.zeros(batch_size, 1, 224, 224, device=device)
    temporal_features_dummy = torch.zeros(batch_size, 10, 2, device=device)  # seq_len=10, input_dim=2
    sequence_lengths_dummy = torch.tensor([10], device=device)
    
    with torch.no_grad():
        _ = model(
            spatial_ui=spatial_ui_dummy,
            spatial_mask=spatial_mask_dummy,
            temporal_features=temporal_features_dummy,
            sequence_lengths=sequence_lengths_dummy,
        )
    
    # Now load the checkpoint
    model.load_state_dict(checkpoint['model_state_dict'])
    
    return checkpoint


def test_model(model: torch.nn.Module, dataloader: DataLoader, device: torch.device):
    """
    Test model on dataset and compute MSE and MAE.
    
    Args:
        model: Model to test
        dataloader: Test dataloader
        device: Device to run on
        
    Returns:
        Tuple of (mse, mae) metrics
    """
    model.eval()
    total_mse = 0.0
    total_mae = 0.0
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
            
            # Compute metrics
            mse = F.mse_loss(predictions, target_avg, reduction='mean')
            mae = F.l1_loss(predictions, target_avg, reduction='mean')
            
            total_mse += mse.item()
            total_mae += mae.item()
            num_batches += 1
    
    avg_mse = total_mse / num_batches if num_batches > 0 else 0.0
    avg_mae = total_mae / num_batches if num_batches > 0 else 0.0
    
    return avg_mse, avg_mae


def get_model_type(model_config: dict) -> str:
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
    """Main testing function."""
    print("=" * 60)
    print("HISM Model Testing")
    print("=" * 60)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    
    # Load configurations
    print("\n1. Loading configurations...")
    model_config = load_model_config("configs/model.yaml")
    dataset_config = load_config("configs/dataset.yaml")
    print(f"   ✓ Model config loaded")
    print(f"   ✓ Dataset config loaded")
    
    # Create model
    print("\n2. Creating model...")
    model = create_hism_model(
        model_config_path="configs/model.yaml",
        dataset_config_path="configs/dataset.yaml",
    )
    model = model.to(device)
    print(f"   ✓ Model created")
    
    # Auto-detect checkpoint path
    print("\n3. Loading checkpoint...")
    model_type = get_model_type(model_config)
    checkpoint_path = f"checkpoints/best_model_{model_type}.pth"
    
    if not os.path.exists(checkpoint_path):
        print(f"   ✗ Checkpoint not found: {checkpoint_path}")
        print(f"   Available checkpoints in checkpoints/:")
        if os.path.exists("checkpoints"):
            for f in os.listdir("checkpoints"):
                if f.endswith('.pth'):
                    print(f"     - {f}")
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    checkpoint = load_checkpoint(checkpoint_path, model, device)
    print(f"   ✓ Checkpoint loaded: {checkpoint_path}")
    if 'epoch' in checkpoint:
        print(f"     - Trained for {checkpoint['epoch']} epochs")
    if 'best_score' in checkpoint:
        print(f"     - Best validation loss: {checkpoint['best_score']:.6f}")
    
    # Create test dataloader
    print("\n4. Creating test dataloader...")
    test_loader = get_dataloader(
        split='test',
        config_path="configs/dataset.yaml",
        batch_size=32,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )
    print(f"   ✓ Test set: {len(test_loader.dataset)} samples")
    
    # Run test
    print("\n5. Running inference on test set...")
    test_mse, test_mae = test_model(model, test_loader, device)
    print(f"   ✓ Inference completed")
    
    # Print results
    print("\n" + "=" * 60)
    print("Test Results:")
    print("=" * 60)
    print(f"  MSE: {test_mse:.6f}")
    print(f"  MAE: {test_mae:.6f}")
    print("=" * 60)
    print("Testing completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
