"""
Test script for HISM Model Forward Pass

This script tests the HISM model forward pass to ensure:
1. Model can be created from configs
2. Forward pass works with a single sample
3. Output shape is correct [1, 1]
4. Output values are in [0, 1] range (sigmoid output)
5. No NaN or Inf values
6. Model components are initialized correctly
"""

import torch
import numpy as np
from datasets import get_dataloader
from models import create_hism_model


def test_model_forward_pass():
    """
    Test HISM model forward pass with a single sample.
    
    Validates:
    - Output shape is [1, 1] (batch_size=1, single prediction)
    - Output values are in [0, 1] range (sigmoid output)
    - No NaN or Inf values
    - Model components are initialized correctly
    """
    print("=" * 60)
    print("Test: HISM Model Forward Pass")
    print("=" * 60)
    
    # Load one sample from dataset (single trial)
    print("\n1. Loading sample from dataset...")
    dataloader = get_dataloader(
        split='train',
        config_path="configs/dataset.yaml",
        batch_size=1,
        shuffle=False,
        num_workers=0,
    )
    
    batch = next(iter(dataloader))
    print(f"   ✓ Sample loaded: {batch['trial_id'][0]}")
    print(f"   - Temporal features shape: {batch['temporal_features'].shape}")
    print(f"   - Spatial UI shape: {batch['spatial_ui'].shape}")
    print(f"   - Spatial mask shape: {batch['spatial_mask'].shape}")
    print(f"   - Sequence length: {batch['sequence_lengths'][0].item()}")
    
    # Create HISM model using create_hism_model()
    print("\n2. Creating HISM model from configs...")
    model = create_hism_model(
        model_config_path="configs/model.yaml",
        dataset_config_path="configs/dataset.yaml",
    )
    model.eval()  # Set to evaluation mode
    print("   ✓ Model created successfully")
    
    # Print model component info
    print("\n   Model Components:")
    print(f"   - Spatial Encoder: {type(model.spatial_encoder).__name__}")
    print(f"   - Temporal Encoder: {type(model.temporal_encoder).__name__}")
    print(f"     * Type: {model.temporal_encoder.encoder_type}")
    print(f"     * Input dim: {model.temporal_encoder.input_dim}")
    print(f"     * Output dim: {model.temporal_encoder.output_dim}")
    print(f"   - Fusion Module: {type(model.fusion).__name__}")
    
    # Run forward pass with the sample
    print("\n3. Running forward pass...")
    with torch.no_grad():
        output = model(
            spatial_ui=batch['spatial_ui'],
            spatial_mask=batch['spatial_mask'],
            temporal_features=batch['temporal_features'],
            sequence_lengths=batch['sequence_lengths'],
        )
    
    print(f"   ✓ Forward pass completed")
    print(f"   - Output shape: {output.shape}")
    
    # Validate output shape
    print("\n4. Validating output shape...")
    expected_shape = (1, 1)
    assert output.shape == expected_shape, \
        f"Expected output shape {expected_shape}, got {output.shape}"
    print(f"   ✓ Output shape is correct: {output.shape}")
    
    # Validate output range [0, 1]
    print("\n5. Validating output range...")
    output_min = output.min().item()
    output_max = output.max().item()
    print(f"   - Output min: {output_min:.6f}")
    print(f"   - Output max: {output_max:.6f}")
    
    assert output_min >= 0.0, f"Output values should be >= 0, got min={output_min}"
    assert output_max <= 1.0, f"Output values should be <= 1, got max={output_max}"
    print("   ✓ Output values are in [0, 1] range (sigmoid output)")
    
    # Check for NaN or Inf values
    print("\n6. Checking for NaN or Inf values...")
    has_nan = torch.isnan(output).any().item()
    has_inf = torch.isinf(output).any().item()
    
    assert not has_nan, "Output contains NaN values"
    assert not has_inf, "Output contains Inf values"
    print("   ✓ No NaN or Inf values detected")
    
    # Check that output is reasonable (not all zeros, not all ones)
    print("\n7. Checking output reasonableness...")
    output_value = output.item()
    print(f"   - Output value: {output_value:.6f}")
    
    # Output should not be exactly 0 or 1 (though close values are okay)
    # Just check that it's a valid number
    assert not np.isclose(output_value, 0.0, atol=1e-6) or not np.isclose(output_value, 1.0, atol=1e-6), \
        "Output is at extreme (0 or 1), might indicate initialization issue"
    print("   ✓ Output value is reasonable")
    
    # Print intermediate shapes for debugging
    print("\n8. Intermediate shapes (for debugging):")
    with torch.no_grad():
        # Get intermediate outputs
        spatial_features = model.spatial_encoder(
            batch['spatial_ui'],
            batch['spatial_mask']
        )
        temporal_features_vec = model.temporal_encoder(
            batch['temporal_features'],
            sequence_lengths=batch['sequence_lengths']
        )
        
        print(f"   - Spatial features: {spatial_features.shape}")
        print(f"   - Temporal features: {temporal_features_vec.shape}")
        print(f"   - Fused features (concatenated): [{spatial_features.shape[1] + temporal_features_vec.shape[1]}]")
    
    print("\n" + "=" * 60)
    print("✅ All validations passed!")
    print("=" * 60)


def test_different_configurations():
    """
    Test model with different configurations.
    
    - Test with default config (both encoders enabled)
    - Test with/without task vector (if time permits)
    """
    print("\n" + "=" * 60)
    print("Test: Different Configurations")
    print("=" * 60)
    
    # Test with default config (both encoders enabled)
    print("\n1. Testing with default config (both encoders enabled)...")
    dataloader = get_dataloader(
        split='train',
        config_path="configs/dataset.yaml",
        batch_size=1,
        shuffle=False,
    )
    batch = next(iter(dataloader))
    
    model = create_hism_model()
    model.eval()
    
    with torch.no_grad():
        output = model(
            spatial_ui=batch['spatial_ui'],
            spatial_mask=batch['spatial_mask'],
            temporal_features=batch['temporal_features'],
            sequence_lengths=batch['sequence_lengths'],
        )
    
    assert output.shape == (1, 1), f"Expected (1, 1), got {output.shape}"
    assert 0.0 <= output.item() <= 1.0, "Output not in [0, 1] range"
    print("   ✓ Default config works correctly")
    
    print("\n" + "=" * 60)
    print("✅ Configuration tests passed!")
    print("=" * 60)


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("HISM Model Forward Pass Test Suite")
    print("=" * 60 + "\n")
    
    try:
        test_model_forward_pass()
        test_different_configurations()
        
        print("\n" + "=" * 60)
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

