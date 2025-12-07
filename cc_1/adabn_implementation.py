# Adaptive Batch Normalization (AdaBN) Implementation
# Add this cell to the notebook for optional domain adaptation

import copy

def apply_adaptive_batch_norm(model, adaptation_loader, device):
    """
    Apply Adaptive Batch Normalization (AdaBN) for domain adaptation.
    
    This recalculates BN statistics on test/target domain data
    while keeping all learned weights frozen. This helps the model
    adapt to distribution shifts between training and test data.
    
    Reference: 
    - Li et al. "Revisiting Batch Normalization For Practical Domain Adaptation" (2016)
    
    Args:
        model: Trained model with BatchNorm layers
        adaptation_loader: DataLoader with target domain data (test set)
        device: torch device (e.g., 'cuda', 'mps', or 'cpu')
        
    Returns:
        Model with updated BN statistics adapted to test domain
    """
    # Create a copy to avoid modifying the original model
    adapted_model = copy.deepcopy(model)
    adapted_model.train()  # Set to train mode to update BN stats
    
    # Freeze all parameters except BN running stats
    for param in adapted_model.parameters():
        param.requires_grad = False
    
    # Reset and configure BN layers for adaptation
    for module in adapted_model.modules():
        if isinstance(module, nn.BatchNorm2d) or isinstance(module, nn.BatchNorm1d):
            # Reset BN statistics
            module.reset_running_stats()
            # Use cumulative moving average (no momentum)
            module.momentum = None
            # Keep BN in training mode to update stats
            module.training = True
    
    # Run forward passes to accumulate BN statistics on test data
    print("Adapting Batch Normalization statistics to test domain...")
    print(f"Processing {len(adaptation_loader)} batches...")
    
    with torch.no_grad():  # Don't update weights, only BN stats
        for batch_idx, batch_data in enumerate(tqdm(adaptation_loader, desc='AdaBN adaptation')):
            # Handle different dataset return formats
            if isinstance(batch_data, (tuple, list)):
                data = batch_data[0]
            else:
                data = batch_data
                
            data = data.to(device)
            
            # Forward pass to update BN running statistics
            _ = adapted_model(data)
            
            # Synchronize for MPS/CUDA
            if device.type == 'mps':
                torch.mps.synchronize()
            elif device.type == 'cuda':
                torch.cuda.synchronize()
    
    # Set back to eval mode for inference
    adapted_model.eval()
    
    print("✓ Batch Normalization adaptation complete!")
    print(f"  Model now adapted to test domain distribution")
    
    return adapted_model


# Example Usage 1: Basic AdaBN adaptation
def apply_adabn_to_model(model, test_loader, device):
    """
    Wrapper function to apply AdaBN with the test loader.
    
    Usage:
        adapted_model = apply_adabn_to_model(model, test_loader, device)
    """
    return apply_adaptive_batch_norm(model, test_loader, device)


# Example Usage 2: Complete inference pipeline with AdaBN
def generate_predictions_with_adabn(model, test_loader, device, apply_tta=False):
    """
    Generate predictions using a model adapted with AdaBN.
    
    Args:
        model: Trained model
        test_loader: DataLoader for test data
        device: torch device
        apply_tta: Whether to also apply test-time augmentation
        
    Returns:
        predictions: List of predicted class indices
        filenames: List of test filenames
    """
    # Step 1: Apply AdaBN
    print("\n" + "="*70)
    print("GENERATING PREDICTIONS WITH ADAPTIVE BATCH NORMALIZATION (AdaBN)")
    print("="*70)
    
    adapted_model = apply_adaptive_batch_norm(model, test_loader, device)
    
    # Step 2: Generate predictions
    print("\nGenerating predictions with adapted model...")
    adapted_model.eval()
    predictions = []
    filenames = []
    
    with torch.no_grad():
        for batch_data in tqdm(test_loader, desc='Inference'):
            # Handle different dataset return formats
            if isinstance(batch_data, (tuple, list)):
                if len(batch_data) == 2:
                    data, batch_filenames = batch_data
                else:
                    data = batch_data[0]
                    batch_filenames = None
            else:
                data = batch_data
                batch_filenames = None
            
            data = data.to(device)
            
            if apply_tta and 'apply_test_time_adaptation_augmentations' in dir():
                # Use TTA if available
                output = apply_test_time_adaptation_augmentations(
                    adapted_model, data, device, n_augmentations=8
                )
            else:
                # Standard inference
                output = adapted_model(data)
            
            _, predicted = torch.max(output, 1)
            predictions.extend(predicted.cpu().numpy())
            
            if batch_filenames is not None:
                filenames.extend(batch_filenames)
    
    print(f"✓ Generated {len(predictions)} predictions with AdaBN")
    
    if filenames:
        return predictions, filenames
    else:
        return predictions


# Example Usage 3: Compare baseline vs AdaBN predictions
def compare_baseline_vs_adabn(model, test_loader, device):
    """
    Generate predictions with and without AdaBN for comparison.
    
    Returns:
        baseline_predictions: Predictions without AdaBN
        adabn_predictions: Predictions with AdaBN
        difference_count: Number of predictions that changed
    """
    print("\n" + "="*70)
    print("COMPARING BASELINE vs AdaBN PREDICTIONS")
    print("="*70)
    
    # Baseline predictions (no adaptation)
    print("\n1. Generating baseline predictions...")
    model.eval()
    baseline_predictions = []
    
    with torch.no_grad():
        for batch_data in tqdm(test_loader, desc='Baseline'):
            data = batch_data[0] if isinstance(batch_data, (tuple, list)) else batch_data
            data = data.to(device)
            output = model(data)
            _, predicted = torch.max(output, 1)
            baseline_predictions.extend(predicted.cpu().numpy())
    
    # AdaBN predictions
    print("\n2. Generating AdaBN predictions...")
    adabn_predictions = generate_predictions_with_adabn(model, test_loader, device)
    if isinstance(adabn_predictions, tuple):
        adabn_predictions = adabn_predictions[0]
    
    # Compare
    baseline_arr = np.array(baseline_predictions)
    adabn_arr = np.array(adabn_predictions)
    differences = (baseline_arr != adabn_arr).sum()
    
    print(f"\n" + "="*70)
    print(f"COMPARISON RESULTS")
    print("="*70)
    print(f"Total predictions: {len(baseline_predictions)}")
    print(f"Predictions changed by AdaBN: {differences} ({differences/len(baseline_predictions)*100:.2f}%)")
    
    if differences > 0:
        print(f"\nAdaBN modified predictions for {differences} samples")
        print(f"This suggests domain adaptation is working!")
    else:
        print(f"\nNo predictions changed - domain shift may be minimal")
    
    return baseline_predictions, adabn_predictions, differences


# ============================================================================
# WHEN TO USE AdaBN
# ============================================================================
"""
Use AdaBN when:
1. ✓ You have a trained model with BatchNorm layers
2. ✓ There is a distribution shift between train and test data
3. ✓ You have access to unlabeled test data
4. ✓ Model weights are frozen (no fine-tuning needed)

AdaBN is particularly effective for:
- Domain shift (Level-1C → Level-2A in Sentinel-2)
- Different sensors or acquisition conditions
- Seasonal or temporal distribution changes

Expected improvements:
- Typically 2-5% accuracy improvement on shifted test data
- Most effective when BN statistics differ significantly between domains
"""

# ============================================================================
# USAGE INSTRUCTIONS
# ============================================================================
"""
# Option 1: Simple usage
adapted_model = apply_adabn_to_model(model, test_loader, device)

# Then use adapted_model for predictions
predictions = []
adapted_model.eval()
with torch.no_grad():
    for data, _ in test_loader:
        data = data.to(device)
        output = adapted_model(data)
        _, predicted = torch.max(output, 1)
        predictions.extend(predicted.cpu().numpy())


# Option 2: Complete pipeline with TTA
predictions, filenames = generate_predictions_with_adabn(
    model, test_loader, device, apply_tta=True
)


# Option 3: Compare baseline vs AdaBN
baseline_preds, adabn_preds, n_changes = compare_baseline_vs_adabn(
    model, test_loader, device
)
"""
