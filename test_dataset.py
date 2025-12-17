"""
Test script for HISM Dataset

This script tests the dataset loading functionality to ensure:
1. Single samples load correctly
2. DataLoader batches work properly
3. Tensor shapes are correct
4. Variable-length sequences are handled
5. All required fields are present
"""

import torch
from datasets import HISMDataset, get_dataloader, load_config


def test_single_sample():
    """Test loading a single sample from the dataset."""
    print("=" * 60)
    print("Test 1: Loading Single Sample")
    print("=" * 60)
    
    config = load_config("configs/dataset.yaml")
    
    # Create dataset for train split
    dataset = HISMDataset(
        data_root=config['data_root'],
        splits_dir=config['splits_dir'],
        split='train',
        csv_filename=config['csv_filename'],
        ui_image=config['ui_image'],
        mask_image=config['mask_image'],
        target_column=config['target_column'],
        highlight_column=config['highlight_column'],
        task_column=config['task_column'],
        image_size=tuple(config['image_size']),
        normalize_images=config['normalize_images'],
    )
    
    print(f"Dataset size: {len(dataset)} trials")
    
    # Load first sample
    sample = dataset[0]
    
    # Check all required fields
    required_fields = ['trial_id', 'temporal_features', 'spatial_ui', 
                       'spatial_mask', 'targets', 'sequence_length']
    for field in required_fields:
        assert field in sample, f"Missing field: {field}"
        print(f"✓ Field '{field}' present")
    
    # Check tensor shapes
    print("\nTensor Shapes:")
    print(f"  trial_id: {type(sample['trial_id'])} = '{sample['trial_id']}'")
    print(f"  temporal_features: {sample['temporal_features'].shape}")
    print(f"  spatial_ui: {sample['spatial_ui'].shape}")
    print(f"  spatial_mask: {sample['spatial_mask'].shape}")
    print(f"  targets: {sample['targets'].shape}")
    print(f"  sequence_length: {sample['sequence_length']}")
    
    # Validate shapes
    seq_len = sample['sequence_length']
    assert sample['temporal_features'].shape == (seq_len, 2), \
        f"Expected temporal_features shape ({seq_len}, 2), got {sample['temporal_features'].shape}"
    assert sample['spatial_ui'].shape == (3, 224, 224), \
        f"Expected spatial_ui shape (3, 224, 224), got {sample['spatial_ui'].shape}"
    assert sample['spatial_mask'].shape == (1, 224, 224), \
        f"Expected spatial_mask shape (1, 224, 224), got {sample['spatial_mask'].shape}"
    assert sample['targets'].shape == (seq_len,), \
        f"Expected targets shape ({seq_len},), got {sample['targets'].shape}"
    
    print("\n✓ All shape validations passed")
    
    # Check data types
    print("\nData Types:")
    print(f"  temporal_features: {sample['temporal_features'].dtype}")
    print(f"  spatial_ui: {sample['spatial_ui'].dtype}")
    print(f"  spatial_mask: {sample['spatial_mask'].dtype}")
    print(f"  targets: {sample['targets'].dtype}")
    
    assert sample['temporal_features'].dtype == torch.float32
    assert sample['spatial_ui'].dtype == torch.float32
    assert sample['spatial_mask'].dtype == torch.float32
    assert sample['targets'].dtype == torch.float32
    
    print("\n✓ All data type validations passed")
    
    # Check value ranges
    print("\nValue Ranges:")
    print(f"  temporal_features (v_t): min={sample['temporal_features'][:, 0].min():.2f}, "
          f"max={sample['temporal_features'][:, 0].max():.2f}")
    print(f"  temporal_features (c_t): min={sample['temporal_features'][:, 1].min():.2f}, "
          f"max={sample['temporal_features'][:, 1].max():.2f}")
    print(f"  targets: min={sample['targets'].min():.2f}, max={sample['targets'].max():.2f}")
    print(f"  spatial_mask: min={sample['spatial_mask'].min():.0f}, "
          f"max={sample['spatial_mask'].max():.0f} (should be 0 or 1)")
    
    # Validate mask is binary
    assert (sample['spatial_mask'] == 0.0).logical_or(sample['spatial_mask'] == 1.0).all(), \
        "Mask should be binary (0 or 1)"
    
    print("\n✓ All value range validations passed")
    print("\n✅ Test 1 PASSED\n")


def test_dataloader_batch():
    """Test DataLoader with batching."""
    print("=" * 60)
    print("Test 2: DataLoader Batching")
    print("=" * 60)
    
    # Create DataLoader with small batch size
    batch_size = 3
    dataloader = get_dataloader(
        split='train',
        config_path="configs/dataset.yaml",
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )
    
    print(f"DataLoader created with batch_size={batch_size}")
    
    # Get first batch
    batch = next(iter(dataloader))
    
    # Check batch structure
    print("\nBatch Structure:")
    print(f"  trial_id: {type(batch['trial_id'])} (list of {len(batch['trial_id'])})")
    print(f"  temporal_features: {batch['temporal_features'].shape}")
    print(f"  spatial_ui: {batch['spatial_ui'].shape}")
    print(f"  spatial_mask: {batch['spatial_mask'].shape}")
    print(f"  targets: {batch['targets'].shape}")
    print(f"  sequence_lengths: {batch['sequence_lengths'].shape}")
    
    # Validate batch shapes
    assert batch['temporal_features'].shape[0] == batch_size, \
        f"Batch size mismatch: expected {batch_size}, got {batch['temporal_features'].shape[0]}"
    assert batch['spatial_ui'].shape[0] == batch_size
    assert batch['spatial_mask'].shape[0] == batch_size
    assert batch['targets'].shape[0] == batch_size
    assert batch['sequence_lengths'].shape[0] == batch_size
    
    print("\n✓ Batch size validations passed")
    
    # Check variable-length sequence handling
    max_seq_len = batch['temporal_features'].shape[1]
    sequence_lengths = batch['sequence_lengths']
    
    print(f"\nVariable-Length Sequence Handling:")
    print(f"  Max sequence length in batch: {max_seq_len}")
    print(f"  Individual sequence lengths: {sequence_lengths.tolist()}")
    
    # Verify padding
    for i, seq_len in enumerate(sequence_lengths):
        # Check that padding is zeros
        if seq_len < max_seq_len:
            padded_temporal = batch['temporal_features'][i, seq_len:, :]
            padded_targets = batch['targets'][i, seq_len:]
            
            assert (padded_temporal == 0).all(), \
                f"Padding should be zeros for sample {i}"
            assert (padded_targets == 0).all(), \
                f"Target padding should be zeros for sample {i}"
            
            print(f"  ✓ Sample {i}: padded from {seq_len} to {max_seq_len}")
    
    print("\n✓ Variable-length sequence handling validated")
    print("\n✅ Test 2 PASSED\n")


def test_multiple_splits():
    """Test that all splits can be loaded."""
    print("=" * 60)
    print("Test 3: Multiple Splits")
    print("=" * 60)
    
    splits = ['train', 'val', 'test']
    
    for split in splits:
        print(f"\nTesting {split} split...")
        try:
            dataloader = get_dataloader(
                split=split,
                config_path="configs/dataset.yaml",
                batch_size=1,
                shuffle=False,
            )
            sample = next(iter(dataloader))
            print(f"  ✓ {split} split loaded successfully")
            print(f"    Batch size: {len(sample['trial_id'])}")
            print(f"    Sequence length: {sample['sequence_lengths'][0].item()}")
        except Exception as e:
            print(f"  ✗ {split} split failed: {e}")
            raise
    
    print("\n✅ Test 3 PASSED\n")


def test_edge_cases():
    """Test edge cases."""
    print("=" * 60)
    print("Test 4: Edge Cases")
    print("=" * 60)
    
    # Test with batch_size=1 (no padding needed if all same length)
    print("\nTesting batch_size=1...")
    dataloader = get_dataloader('train', batch_size=1, shuffle=False)
    batch = next(iter(dataloader))
    assert batch['temporal_features'].shape[0] == 1
    print("  ✓ batch_size=1 works")
    
    # Test accessing different indices
    print("\nTesting different sample indices...")
    dataset = HISMDataset(
        data_root="data/trials",
        splits_dir="data/splits",
        split='train',
    )
    
    # Test first, middle, last
    indices_to_test = [0, len(dataset) // 2, len(dataset) - 1]
    for idx in indices_to_test:
        sample = dataset[idx]
        assert 'trial_id' in sample
        assert sample['sequence_length'] > 0
        print(f"  ✓ Sample {idx} loaded successfully")
    
    print("\n✅ Test 4 PASSED\n")


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("HISM Dataset Test Suite")
    print("=" * 60 + "\n")
    
    try:
        test_single_sample()
        test_dataloader_batch()
        test_multiple_splits()
        test_edge_cases()
        
        print("=" * 60)
        print("🎉 ALL TESTS PASSED!")
        print("=" * 60)
        
    except Exception as e:
        print("\n" + "=" * 60)
        print(f"❌ TEST FAILED: {e}")
        print("=" * 60)
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()

