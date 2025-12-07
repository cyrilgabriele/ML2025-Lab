# Band Reordering & Domain Shift Fix Report

## Summary

This report documents the analysis and fixes for correct band alignment between Sentinel-2 Level-1C (training) and Level-2A (test) data.

---

## ✅ Step 1: Correct Band Order, Drop B10, Move B8A

### Status: **⚠️ INCORRECT → FIXED**

### Problem

The code was **partially correct** but had a critical flaw:

**Current Implementation:**
- Training: `f.read([1,2,3,4,5,6,7,8,9,11,12,13])` → Order: `[B1-B9, B11, B12, B8A]`
- Test: Loads `.npy` files as-is → Order: `[B1-B8, B8A, B9, B11, B12]`

**Issue:** B8A is at **different positions**:
- Training: B8A is at index 11 (last position)
- Test: B8A is at index 8 (after B8, before B9)

This causes **channel misalignment** - the model trained on B8A features at position 11 will receive B11 values at that position during testing!

### Solution Implemented

Added proper band reordering functions to the notebook (inserted after line 271):

```python
# ============================================================================
# BAND ORDER CORRECTION: Critical for domain shift handling
# ============================================================================
# Training data (Level-1C): 13 bands in order B1-B9, B10, B11, B12, B8A
# Test data (Level-2A): 12 bands in order B1-B8, B8A, B9, B11, B12 (no B10)
# We must reorder training data to match test order

TRAIN_ORDER = ["B1","B2","B3","B4","B5","B6","B7","B8","B9","B10","B11","B12","B8A"]
TEST_ORDER = ["B1","B2","B3","B4","B5","B6","B7","B8","B8A","B9","B11","B12"]

def reorder_bands(img, src_order, dst_order):
    """
    Reorder spectral bands from source order to destination order.
    
    Args:
        img: numpy array of shape (H, W, C) where C is number of bands
        src_order: list of band names in source order
        dst_order: list of band names in destination order
    
    Returns:
        Reordered image with shape (H, W, len(dst_order))
    """
    idx = [src_order.index(b) for b in dst_order if b in src_order]
    return img[:, :, idx]
```

### Updated Dataset Classes

**EuroSATDataset.__getitem__():**
```python
if file_path.endswith('.npy'):
    # Test data: already in TEST_ORDER
    img = np.load(file_path)
else:
    # Training data: load ALL bands then reorder
    with rio.open(file_path, "r") as f:
        img = f.read()  # Read all 13 bands
        img = reshape_as_image(img)
        # Reorder: drops B10, moves B8A from position 12 to position 8
        img = reorder_bands(img, TRAIN_ORDER, TEST_ORDER)
```

**ImprovedEuroSATDataset.__getitem__():**
```python
if file_path.endswith('.npy'):
    img = np.load(file_path)
else:
    with rio.open(file_path, "r") as f:
        img = f.read()
        img = reshape_as_image(img)
        img = reorder_bands(img, TRAIN_ORDER, TEST_ORDER)
```

---

## ✅ Step 2: Scaling and Normalization

### Status: **✅ ALREADY CORRECT**

### Current Implementation

The code already implements proper normalization:

1. **Float32 conversion with per-band z-score normalization:**
```python
def normalize_for_model(band_data):
    band_data = band_data.astype(np.float32)
    band_means = np.mean(band_data, axis=(0, 1), keepdims=True)
    band_stds = np.std(band_data, axis=(0, 1), keepdims=True)
    band_stds = np.where(band_stds == 0, 1.0, band_stds)
    normalized = (band_data - band_means) / band_stds
    return normalized
```

2. **Training statistics computed for 12 bands:**
```python
# Lines 2341-2508: ImprovedEuroSATDataset
train_stats_12band = {
    'means': [...],  # 12 values
    'stds': [...],   # 12 values
    'mins': [...],
    'maxs': [...]
}
```

3. **Applied consistently in both train and test:**
- Both `EuroSATDataset` and `ImprovedEuroSATDataset` use the same normalization
- Z-score normalization: `(x - mean) / std`
- Clipping to [-10, 10] to handle outliers

### ✅ Verification
- [x] Converts to float32
- [x] Applies per-band mean/std normalization
- [x] Same normalization for train/val/test
- [x] Handles 12 channels correctly

---

## ✅ Step 3: Model Input Channels

### Status: **✅ ALREADY CORRECT**

### Current Implementation

Both model classes correctly use **12 input channels:**

**EuroSATClassifier (line 642):**
```python
def __init__(self, num_classes=NUM_CLASSES, input_channels=12):
    super(EuroSATClassifier, self).__init__()
    self.initial_conv = nn.Sequential(
        nn.Conv2d(input_channels, 64, kernel_size=3, padding=1),
        ...
    )
```

**ImprovedEuroSATClassifier (line 2187):**
```python
def __init__(self, num_classes=10, input_channels=12, dropout=0.3):
    super(ImprovedEuroSATClassifier, self).__init__()
    self.conv1 = nn.Sequential(
        nn.Conv2d(input_channels, 64, kernel_size=3, padding=1),
        ...
    )
```

**Model Instantiation (line 751, 2306):**
```python
model = EuroSATClassifier(num_classes=NUM_CLASSES, input_channels=12)
improved_model = ImprovedEuroSATClassifier(num_classes=NUM_CLASSES, input_channels=12)
```

### ✅ Verification
- [x] Models defined with `input_channels=12`
- [x] First conv layer: `nn.Conv2d(12, 64, ...)`
- [x] No pretrained weights from 13-channel models (training from scratch)

**Note:** Since training from scratch, no weight mapping needed.

---

## ✅ Step 4: Resampling Consistency

### Status: **⚠️ NOT EXPLICITLY IMPLEMENTED**

### Current Situation

The code does **not** explicitly handle spatial resampling of 20m and 60m bands to 10m resolution.

**EuroSAT Data Format:**
- The EuroSAT dataset (`.tif` files) is **pre-processed** - all bands are already resampled to 64x64 pixels
- Test data (`.npy` files) are also provided at a consistent spatial resolution

**Band Resolutions in Sentinel-2:**
- **10m bands:** B2, B3, B4, B8
- **20m bands:** B5, B6, B7, B8A, B11, B12
- **60m bands:** B1, B9

### Analysis

**For EuroSAT training data:**
- All GeoTiff files are already resampled to 64x64 pixels by the dataset creators
- The `rasterio.read()` call returns data at the native resolution of the file
- Since EuroSAT standardized all bands to the same spatial resolution, **no additional resampling is needed**

**For test data:**
- `.npy` files contain arrays of consistent shape (64, 64, 12)
- All bands have already been resampled to the same resolution

### Recommendation

✅ **No action needed** - Both training and test data are pre-resampled by dataset providers.

If you were working with **raw Sentinel-2 L1C/L2A tiles**, you would need:

```python
def resample_bands_to_10m(img_dict):
    """
    Resample 20m and 60m bands to 10m using bilinear interpolation.
    
    Args:
        img_dict: Dictionary with band names as keys, arrays as values
    
    Returns:
        Dictionary with all bands resampled to 10m resolution
    """
    from scipy.ndimage import zoom
    
    target_shape = img_dict['B2'].shape  # B2 is 10m reference
    resampled = {}
    
    for band, data in img_dict.items():
        if data.shape != target_shape:
            # Calculate zoom factors
            zoom_factors = (target_shape[0] / data.shape[0],
                          target_shape[1] / data.shape[1])
            # Bilinear interpolation (order=1)
            resampled[band] = zoom(data, zoom_factors, order=1)
        else:
            resampled[band] = data
    
    return resampled
```

But this is **not needed for EuroSAT**, which is pre-processed.

---

## ✅ Step 5: Optional Domain Adaptation

### Status: **✅ PARTIALLY IMPLEMENTED** (Statistical normalization exists, AdaBN missing)

### Current Implementation

The code includes **statistical domain adaptation** but **not Adaptive Batch Normalization (AdaBN)**.

**Statistical Normalization (Lines 2307-2330):**
```python
def normalize_test_to_train_statistics(test_data, train_stats):
    """
    Normalize test data using training data statistics (z-score normalization).
    """
    # Apply z-score normalization: 
    # (x - test_mean) / test_std * train_std + train_mean
    test_mean = np.mean(test_data, axis=(0, 1))
    test_std = np.std(test_data, axis=(0, 1)) + 1e-8
    test_data_normalized = (test_data - test_mean) / test_std
    
    if 'train_std' in train_stats and 'train_mean' in train_stats:
        test_data_normalized = test_data_normalized * train_stats['train_std'] + train_stats['train_mean']
    
    return test_data_normalized
```

**EuroSATDatasetWithAdaptation (Lines 1391-1563):**
- Applies domain-shift correction using training statistics
- Used in inference pipeline (lines 1569-1688)

### Missing: Adaptive Batch Normalization (AdaBN)

**What is AdaBN?**
AdaBN recalculates BatchNorm statistics (running mean/var) on the **test domain** while keeping trained weights frozen. This helps the model adapt to test distribution shifts.

### Recommended Addition

Add this function to enable AdaBN:

```python
def apply_adaptive_batch_norm(model, adaptation_loader, device):
    """
    Apply Adaptive Batch Normalization (AdaBN) for domain adaptation.
    
    This recalculates BN statistics on test/target domain data
    while keeping all learned weights frozen.
    
    Args:
        model: Trained model with BatchNorm layers
        adaptation_loader: DataLoader with target domain data (test set)
        device: torch device
        
    Returns:
        Model with updated BN statistics
    """
    import copy
    
    # Create a copy to avoid modifying original
    adapted_model = copy.deepcopy(model)
    adapted_model.train()  # Set to train mode to update BN stats
    
    # Freeze all parameters except BN running stats
    for module in adapted_model.modules():
        if isinstance(module, nn.BatchNorm2d) or isinstance(module, nn.BatchNorm1d):
            # Reset BN statistics
            module.reset_running_stats()
            module.momentum = None  # Use cumulative moving average
    
    # Run forward passes to accumulate BN statistics on test data
    print("Adapting Batch Normalization statistics to test domain...")
    with torch.no_grad():  # Don't update weights
        for data, _ in tqdm(adaptation_loader, desc='AdaBN adaptation'):
            data = data.to(device)
            _ = adapted_model(data)
    
    adapted_model.eval()  # Set back to eval mode
    return adapted_model

# Usage before inference:
# adapted_model = apply_adaptive_batch_norm(model, test_loader, device)
# # Then use adapted_model for predictions
```

### How to Use

**Option 1: Apply before final inference**
```python
# After loading best model checkpoint
checkpoint = torch.load('best_model.pth')
model.load_state_dict(checkpoint['model_state_dict'])

# Apply AdaBN using test data
adapted_model = apply_adaptive_batch_norm(model, test_loader, CONFIG['device'])

# Generate predictions with adapted model
test_predictions = []
adapted_model.eval()
with torch.no_grad():
    for data, filenames in tqdm(test_loader):
        data = data.to(CONFIG['device'])
        output = adapted_model(data)
        _, predicted = torch.max(output, 1)
        test_predictions.extend(predicted.cpu().numpy())
```

**Option 2: Combine with existing domain adaptation**
```python
# Use both statistical normalization AND AdaBN
test_dataset_adapted = EuroSATDatasetWithAdaptation(
    test_samples,
    transform=val_transforms,
    apply_adaptation=True,
    train_stats=shift_stats,
    adaptation_method='statistics'
)
test_loader_adapted = DataLoader(test_dataset_adapted, batch_size=32, shuffle=False)

# Apply AdaBN on statistically normalized data
adapted_model = apply_adaptive_batch_norm(model, test_loader_adapted, device)

# Generate predictions
# ... (use adapted_model)
```

---

## Summary of Fixes

### ❌ **CRITICAL FIX REQUIRED:**

1. **Step 1 - Band Reordering:** 
   - **Status:** Fixed ✅
   - **Action:** Added `reorder_bands()` function and updated both dataset classes
   - **Impact:** HIGH - Prevents channel misalignment between train/test

### ✅ **ALREADY CORRECT:**

2. **Step 2 - Normalization:**
   - **Status:** Correct ✅
   - **Verification:** Per-band z-score normalization applied consistently

3. **Step 3 - Model Input:**
   - **Status:** Correct ✅
   - **Verification:** All models use 12 input channels

4. **Step 4 - Resampling:**
   - **Status:** Not applicable ✅
   - **Reason:** EuroSAT data is pre-resampled

### ⚠️ **OPTIONAL ENHANCEMENT:**

5. **Step 5 - Domain Adaptation:**
   - **Status:** Partial ✅ (statistical normalization exists)
   - **Missing:** Adaptive Batch Normalization (AdaBN)
   - **Impact:** MEDIUM - Could improve test accuracy by 2-5%
   - **Action:** Provided implementation above (optional)

---

## Testing & Validation

After applying the band reordering fix, verify with:

```python
# Test band alignment
print("Testing band order alignment...")

# Load one training sample
train_sample_path = train_samples[0]
with rio.open(train_sample_path, "r") as f:
    train_img = f.read()
    train_img = reshape_as_image(train_img)
    train_img_reordered = reorder_bands(train_img, TRAIN_ORDER, TEST_ORDER)

# Load one test sample
test_sample_path = test_samples[0]
test_img = np.load(test_sample_path)

print(f"Training image shape (after reordering): {train_img_reordered.shape}")
print(f"Test image shape: {test_img.shape}")
print(f"Match: {train_img_reordered.shape == test_img.shape}")

# Verify B8A position
print(f"\nBand order verification:")
print(f"TRAIN_ORDER: {TRAIN_ORDER}")
print(f"TEST_ORDER: {TEST_ORDER}")
print(f"B8A in TEST_ORDER at index: {TEST_ORDER.index('B8A')}")  # Should be 8

# Check dataset outputs
train_dataset_item, _ = train_dataset[0]
test_dataset_item, _ = test_dataset[0]
print(f"\nDataset output shapes:")
print(f"Training: {train_dataset_item.shape}")  # Should be (12, 64, 64)
print(f"Test: {test_dataset_item.shape}")       # Should be (12, 64, 64)
```

Expected output:
```
Training image shape (after reordering): (64, 64, 12)
Test image shape: (64, 64, 12)
Match: True

Band order verification:
TRAIN_ORDER: ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B9', 'B10', 'B11', 'B12', 'B8A']
TEST_ORDER: ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B9', 'B11', 'B12']
B8A in TEST_ORDER at index: 8

Dataset output shapes:
Training: torch.Size([12, 64, 64])
Test: torch.Size([12, 64, 64])
```

---

## Next Steps

1. **Immediate:** Re-run the training with the fixed band reordering
2. **Validation:** Check that training/validation accuracy improves (properly aligned features)
3. **Optional:** Implement AdaBN for additional test performance boost
4. **Final:** Generate predictions and compare with baseline

The band reordering fix is **critical** and must be applied before any further training or inference.
