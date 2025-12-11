# EuroSAT Classification - Pretrained Model Implementation

## Overview

This notebook (`cc_01_pretrained.ipynb`) implements a **transfer learning approach** for satellite image classification using a pretrained ResNet backbone. It predicts EuroSAT land cover classes for Sentinel-2 multispectral imagery.

**Course**: 7,854,1.00 MCS Machine Learning, University of St.Gallen (HSG)

## AI Usage Diclaimer
For the Coding GitHub Co-Pilot was used as a help.

---

## Key Concepts

### Transfer Learning
Instead of training a model from scratch, we leverage **ImageNet-pretrained weights** from a ResNet model trained on millions of RGB images. This approach:
- Requires fewer training samples
- Converges faster (typically 20 epochs vs. 100+)
- Achieves better generalization with limited data
- Reduces computational requirements

### Spectral Band Processing
Sentinel-2 satellite data has 11-12 spectral bands compared to 3 RGB channels in natural images:
- **B1-B8**: Standard bands (blue, green, red, NIR, etc.)
- **B8A**: Narrow NIR band
- **B11, B12**: Shortwave infrared bands
- **B9, B10**: Removed for this task (water vapor, cirrus detection)

We standardize all data to **11 bands** to ensure consistent input shapes.

---

## Code Structure

### Section 1: Import Libraries
```python
# Core libraries
import os, glob, numpy as np, pandas as pd

# Raster I/O
import rasterio as rio

# Deep Learning
import torch, torch.nn as nn
import torchvision.models as models

# Utilities
from tqdm.auto import tqdm  # Progress bars
from sklearn.metrics import classification_report
```

**Key imports**:
- `rasterio`: Reads GeoTIFF satellite data
- `torch`: PyTorch framework for deep learning
- `torchvision.models`: Pretrained models like ResNet18

**Reproducibility**: Random seeds are set at the top to ensure consistent results across runs.

---

### Section 2: Configuration

The `CONFIG` dictionary centralizes all hyperparameters:

```python
CONFIG = {
    'batch_size': 32,              # Samples per batch
    'num_epochs': 20,              # Training epochs (early stopping may reduce this)
    'learning_rate': 1e-5,         # Low LR for transfer learning
    'weight_decay': 1e-4,          # L2 regularization
    'validation_split': 0.2,       # 20% validation data
    'freeze_backbone': True,       # Train only classifier head
    'pretrained_model': 'resnet18' # Model variant
}
```

**Configuration Functions**:
- `get_device()`: Automatically selects best available device (MPS/CUDA/CPU)
- `find_eurosat_dir()`: Locates training data automatically
- `find_testset_dir()`: Locates test data automatically

**Classes Definition**:
```python
CLASSES = ["AnnualCrop", "Forest", "HerbaceousVegetation", ...]
CLASS_TO_IDX = {"AnnualCrop": 0, "Forest": 1, ...}
IDX_TO_CLASS = {0: "AnnualCrop", 1: "Forest", ...}
```

---

### Section 3: Band Reordering Functions

**Problem**: Training and test data have different band orderings.

**Solution**: Create consistent 11-band format

```python
TRAIN_ORDER = ["B1","B2","B3","B4","B5","B6","B7","B8","B9","B10","B11","B12","B8A"]  # 13 bands
TEST_ORDER  = ["B1","B2","B3","B4","B5","B6","B7","B8","B8A","B11","B12"]             # 11 bands

def reorder_bands(img, src_order, dst_order):
    """Reorder bands from source to destination order."""
    idx = [src_order.index(b) for b in dst_order if b in src_order]
    return img[:, :, idx]  # Returns (Height, Width, 11_bands)
```

**Key Points**:
- Training data: 13 bands → reorder → 11 bands (removes B9, B10)
- Test data: 12 bands → remove B9 → 11 bands
- Ensures consistent input shape: (H, W, 11)

---

### Section 4: Custom Dataset Class

```python
class ImprovedEuroSATDataset(Dataset):
    """PyTorch Dataset for EuroSAT satellite imagery."""
    
    def __init__(self, samples, labels=None, transform=None, 
                 normalize=True, train_stats=None):
        self.samples = samples        # File paths
        self.labels = labels          # Ground truth labels
        self.train_stats = train_stats # Normalization statistics
    
    def __getitem__(self, idx):
        # Load image (TIF for training, NPY for test)
        if file.endswith('.npy'):
            img = np.load(file)           # Test: (H, W, 12)
            img = np.delete(img, 9, 2)    # Remove B9 → (H, W, 11)
        else:
            img = rasterio.open(file).read()  # Training: (H, W, 13)
            img = reorder_bands(img, ...)     # Reorder → (H, W, 11)
        
        # Normalization: Z-score using training statistics
        if self.train_stats:
            for band_idx in range(11):
                mean = self.train_stats['means'][band_idx]
                std = self.train_stats['stds'][band_idx]
                img[:,:,band_idx] = (img[:,:,band_idx] - mean) / std
        
        # Convert to tensor: (C, H, W) format for PyTorch
        img = torch.from_numpy(img.transpose(2, 0, 1))
        
        return img, label  # or (img, filename) for test data
```

**Key Features**:
- Handles both TIF (training) and NPY (test) formats
- Automatic band alignment
- Per-band normalization with training statistics
- Error handling and validation

---

### Section 5: Model Architecture

```python
class PretrainedEuroSATClassifier(nn.Module):
    """Adapted ResNet for 11-band satellite imagery."""
    
    def __init__(self, pretrained=True, freeze_backbone=True):
        super().__init__()
        
        # Load pretrained ResNet18 (trained on ImageNet RGB)
        self.backbone = models.resnet18(pretrained=True)
        
        # Adapt first layer: RGB (3) → 11-band input
        original_conv1 = self.backbone.conv1
        self.backbone.conv1 = nn.Conv2d(11, 64, kernel_size=7, stride=2, padding=3)
        
        # Initialize weights intelligently
        # Repeat RGB weights across 11 channels and scale
        new_weights = original_conv1.weight.repeat(1, 4, 1, 1)[:, :11, :, :]
        new_weights = new_weights / (11/3)  # Normalize activation magnitude
        self.backbone.conv1.weight.data = new_weights
        
        # Option 1: Freeze backbone (train only classifier)
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
            # Allow new conv1 layer to adapt
            for param in self.backbone.conv1.parameters():
                param.requires_grad = True
        
        # Replace ImageNet classifier with EuroSAT classifier
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 10)  # 10 EuroSAT classes
        )
    
    def forward(self, x):
        return self.backbone(x)
```

**Architecture Details**:

| Component | Details |
|-----------|---------|
| **Input** | (Batch, 11, 64, 64) - 11 spectral bands |
| **Backbone** | ResNet18 pretrained on ImageNet |
| **First Layer** | Conv2d(11→64 channels, kernel=7×7) |
| **Intermediate** | 4 residual blocks with skip connections |
| **Global Pool** | Average pooling (512 features) |
| **Head** | MLP: 512→256→10 (EuroSAT classes) |

**Design Choices**:
- `freeze_backbone=True`: Train only classifier head (faster, less overfitting)
- `freeze_backbone=False`: Fine-tune entire network with low learning rates
- Weight initialization preserves pretrained ImageNet features

---

### Section 6-7: Data Loading & Statistics

```python
# Load training data from organized directory structure
# EuroSAT_MS/
#   ├── AnnualCrop/
#   ├── Forest/
#   ├── Highway/
#   └── ... (10 classes total)

# Load test data
# testset/testset/
#   ├── test_0.npy
#   ├── test_1.npy
#   └── ... (1026 samples)

# Compute training statistics for normalization
# Sample 200 training images and calculate:
train_stats_11band = {
    'means': [mean_B1, mean_B2, ..., mean_B12],
    'stds': [std_B1, std_B2, ..., std_B12]
}
```

**Why Compute Statistics?**
- Z-score normalization uses training data statistics
- Ensures test data uses same scaling as training
- Reduces domain shift between train/test

---

### Section 8-9: Data Augmentation & DataLoaders

```python
# Training augmentations
train_transforms = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),    # Flip left-right
    transforms.RandomVerticalFlip(p=0.5),      # Flip up-down
    transforms.RandomRotation(45)               # Rotate ±45°
])

# No augmentation for validation/test
val_transforms = None
```

**Data Pipeline**:
1. **Train/Val Split**: 80/20 stratified split (preserves class distribution)
2. **DataLoaders**: Batch data with shuffling for training
3. **Batch Size**: 32 samples per batch

```python
train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True,           # Randomize order
    num_workers=0,          # Load on main thread (safe for all OS)
    pin_memory=True         # Speed up GPU transfer
)
```

---

### Section 10: Model Initialization

```python
# Create model with 11 input channels
model = PretrainedEuroSATClassifier(
    num_classes=10,
    input_channels=11,
    model_name='resnet18',
    pretrained=True,
    freeze_backbone=True
).to(device)

# Loss function with label smoothing
# Prevents overconfident predictions
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

# Optimizer: AdamW with weight decay (L2 regularization)
optimizer = optim.AdamW(
    model.parameters(),
    lr=1e-5,           # Very low LR for transfer learning
    weight_decay=1e-4  # L2 regularization
)

# Learning rate scheduler
scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer,
    T_0=10,      # Restart every 10 epochs
    T_mult=2,    # Double restart interval
    eta_min=1e-6 # Minimum learning rate
)
```

**Why These Choices?**
- **Low Learning Rate**: Pretrained weights shouldn't change much
- **AdamW**: Better than Adam, includes weight decay
- **Cosine Annealing**: Smooth learning rate decay with periodic restarts
- **Label Smoothing**: Regularization to prevent overconfidence

---

### Section 11: Training Functions

```python
def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()  # Enable dropout, batch norm training
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        
        # Forward pass
        output = model(data)
        loss = criterion(output, target)
        
        # Backward pass
        optimizer.zero_grad()      # Clear old gradients
        loss.backward()            # Compute gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Prevent explosion
        optimizer.step()           # Update weights
    
    return epoch_loss, epoch_accuracy

def validate_epoch(model, val_loader, criterion, device):
    """Validate without gradient updates."""
    model.eval()  # Disable dropout, batch norm training
    
    with torch.no_grad():  # Don't compute gradients
        for data, target in val_loader:
            output = model(data)
            loss = criterion(output, target)
    
    return epoch_loss, epoch_accuracy
```

**Key Details**:
- `model.train()`: Enables dropout and batch norm training mode
- `model.eval()`: Disables dropout, uses running batch statistics
- `torch.no_grad()`: Skips gradient computation for validation (faster)
- `clip_grad_norm_`: Prevents exploding gradients (common in RNNs, useful here too)

---

### Section 12: Training Loop

```python
for epoch in range(num_epochs):
    # Train and validate
    train_loss, train_acc = train_epoch(...)
    val_loss, val_acc = validate_epoch(...)
    
    # Update learning rate
    scheduler.step()
    
    # Early stopping: save if validation accuracy improves
    if val_acc > best_val_acc:
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'history': history,
            'train_stats': train_stats_11band,
            'config': CONFIG
        }, 'best_model_pretrained.pth')
        best_val_acc = val_acc
        patience_counter = 0
    else:
        patience_counter += 1
    
    # Stop if no improvement for 10 epochs
    if patience_counter >= 10:
        break
```

**Early Stopping Strategy**:
- Monitors validation accuracy
- Saves model whenever validation improves
- Stops if no improvement for 10 consecutive epochs
- Prevents overfitting

---

### Section 13-16: Evaluation & Prediction

```python
# Load best model
checkpoint = torch.load('best_model_pretrained.pth')
model.load_state_dict(checkpoint['model_state_dict'])

# Generate predictions on test set
model.eval()
test_predictions = []

with torch.no_grad():
    for batch_data in test_loader:
        output = model(batch_data)
        _, predicted = torch.max(output, 1)  # Get class with highest probability
        test_predictions.extend(predicted.cpu().numpy())

# Create submission file
submission_df = pd.DataFrame({
    'test_id': [0, 1, 2, ...],
    'label': ['Forest', 'Highway', 'River', ...]
})
submission_df.to_csv('predictions_pretrained.csv', index=False)
```

**Output Analysis**:
- **Confusion Matrix**: Shows which classes are confused with each other
- **Classification Report**: Precision, recall, F1-score per class
- **Prediction Distribution**: Ensures reasonable class balance in predictions

---

## Configuration Guide

### Key Hyperparameters

| Parameter | Default | Effect |
|-----------|---------|--------|
| `batch_size` | 32 | Larger = faster but more memory |
| `num_epochs` | 20 | Stopped early if validation plateaus |
| `learning_rate` | 1e-5 | Lower for transfer learning |
| `weight_decay` | 1e-4 | L2 regularization strength |
| `freeze_backbone` | True | Train only classifier (faster) |
| `pretrained_model` | 'resnet18' | resnet18/34/50 (larger = slower but better) |

### Adjusting Configuration

**For slower hardware (CPU):**
```python
CONFIG['batch_size'] = 8          # Reduce memory
CONFIG['freeze_backbone'] = True  # Skip backbone training
```

**For GPU with more memory:**
```python
CONFIG['batch_size'] = 64
CONFIG['freeze_backbone'] = False  # Fine-tune entire model
```

**For better accuracy:**
```python
CONFIG['pretrained_model'] = 'resnet50'  # Larger model
CONFIG['learning_rate'] = 2e-5           # Slightly higher
CONFIG['num_epochs'] = 30                # More training
```

---

## Output Files

| File | Description |
|------|-------------|
| `best_model_pretrained.pth` | Best model checkpoint (can be loaded later) |
| `predictions_pretrained.csv` | Test predictions (main submission file) |
| `training_history_pretrained.png` | Loss and accuracy curves |
| `confusion_matrix_pretrained.png` | Validation confusion matrix |

---

## Troubleshooting

### "EuroSAT_MS directory not found"
- Ensure you're in the `cc_1/` directory
- Or move EuroSAT_MS to current directory

### "testset directory not found"
- Check `testset/testset/` exists with `.npy` files
- Run from the notebook's directory

### Out of Memory (OOM) Error
```python
CONFIG['batch_size'] = 16  # or 8
CONFIG['num_workers'] = 0  # Don't parallelize data loading
```

### Training is very slow
- Use MPS (Mac) or CUDA (GPU) if available
- Reduce batch size (counter-intuitive, but less memory pressure)
- Set `freeze_backbone=True`

### Different results on re-run
- This is normal (GPU operations aren't deterministic)
- For reproducibility: `PYTHONHASHSEED=0 jupyter notebook`

---

## Performance Notes

### Expected Training Time
- **CPU**: ~5-10 min/epoch
- **GPU (CUDA)**: ~1-2 min/epoch
- **GPU (MPS - Apple Silicon)**: ~2-3 min/epoch

### Expected Accuracy
- **Validation Accuracy**: 85-92% (depends on training)
- **Best Model Selection**: Uses validation accuracy after ~20 epochs with early stopping

---

## References

- **ResNet**: He et al., "Deep Residual Learning for Image Recognition" (2015)
- **Transfer Learning**: Yosinski et al., "How transferable are features in deep neural networks?" (2014)
- **EuroSAT**: Helber et al., "EuroSAT: A Novel Dataset and Deep Learning Benchmark for Land Use and Land Cover Classification" (2019)
- **PyTorch**: https://pytorch.org
- **Rasterio**: https://rasterio.readthedocs.io

---

## Quick Start

1. **Set configuration** (Section 2) - adjust if needed
2. **Run cells 1-12** in order - sets up model and data
3. **Run cell 12** - trains model (20 epochs, stops early if validation plateaus)
4. **Run cells 13-16** - generates predictions and creates submission file
5. **Submit** `predictions_pretrained.csv`

That's it! Good luck with your assignment! 🎓
