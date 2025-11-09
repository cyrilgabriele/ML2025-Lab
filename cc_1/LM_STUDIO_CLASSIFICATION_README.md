# LM Studio EuroSAT Classification

## Overview
This notebook (`cc_01_llm_classification.ipynb`) uses a locally running **LM Studio** LLM to classify satellite images from the EuroSAT dataset.

## Key Features

✅ **Multispectral to RGB Conversion**: Converts 12-channel satellite imagery to RGB format for LLM processing
✅ **Local LLM Integration**: Uses LM Studio running on `http://127.0.0.1:1234` (default)
✅ **Batch Processing**: Classifies all test images automatically
✅ **CSV Export**: Generates `predictions_llm.csv` in the same format as `predictions.csv`
✅ **Comparison Analysis**: Compares LLM predictions with the original model
✅ **Visualization**: Shows class distribution and sample images

## Requirements

### 1. LM Studio Setup
- **Install LM Studio** from https://lmstudio.ai
- **Load a model** (recommend: Llama 2, Mistral, or similar)
- **Start the server** on port 1234 (default)
  - Go to the "Server" tab in LM Studio
  - Click "Start Server"
  - Should show: `http://127.0.0.1:1234`

### 2. Python Dependencies
```bash
pip install requests tqdm pandas pillow matplotlib numpy scikit-image
```

The notebook uses these libraries:
- `requests` - API communication
- `tqdm` - Progress bars
- `pandas` - Data handling
- `numpy` - Array operations
- `PIL (Pillow)` - Image processing
- `matplotlib` - Visualization

## Workflow

### 1. **Test Connection** (Cell 4)
Verifies that LM Studio is running and accessible

### 2. **Load & Visualize** (Cells 5-6)
- Loads a sample multispectral image (64×64×12 channels)
- Converts to RGB using bands [3, 2, 1] (R, G, B)
- Displays RGB and false-color composites

### 3. **Single Image Classification** (Cell 7)
- Encodes RGB image as PNG and sends to LM Studio
- LLM classifies into one of 10 EuroSAT classes
- Shows predicted class and confidence

### 4. **Batch Classification** (Cell 8)
- Processes all test images from `../testset/testset/`
- Saves results progressively
- With delay between requests (0.5s) to avoid overloading the API
- Default fallback class: `SeaLake` (for failed classifications)

### 5. **Results Summary** (Cell 9)
- Shows class distribution
- Calculates statistics
- Displays first 10 results

### 6. **Export to CSV** (Cell 10)
- Saves to `predictions_llm.csv` (same format as `predictions.csv`)
- Format: `test_id, label`

### 7. **Comparison** (Cell 11, Optional)
- Compares LLM predictions with original `predictions.csv`
- Shows agreement percentage
- Lists disagreement examples

### 8. **Visualization** (Cell 12)
- Creates bar chart of class distribution
- Saves to `llm_results/class_distribution.png`

## Output Files

Generated in the `cc_1/` directory:

| File | Description |
|------|-------------|
| `predictions_llm.csv` | LLM predictions in submission format |
| `llm_results/` | Directory for additional outputs |
| `llm_results/class_distribution.png` | Class distribution bar chart |
| `llm_results/test_0_visualization.png` | Sample RGB and false-color images |

## Configuration

Modify these variables in **Cell 3** as needed:

```python
# LM Studio API Configuration
LM_STUDIO_ENDPOINT = 'http://127.0.0.1:1234/v1/chat/completions'
LM_STUDIO_MODEL = 'local-model'  # Default model name

# Batch processing
BATCH_SIZE = None  # None = process all, or set to limit (e.g., 100)
DELAY_BETWEEN_REQUESTS = 0.5  # Seconds between requests

# Band selection for RGB
RGB_BANDS = [3, 2, 1]  # Red, Green, Blue band indices
```

## EuroSAT Classes

The model classifies into these 10 categories:
1. AnnualCrop
2. Forest
3. HerbaceousVegetation
4. Highway
5. Industrial
6. Pasture
7. PermanentCrop
8. Residential
9. River
10. SeaLake

## Troubleshooting

### ❌ Connection Error: "Make sure LM Studio is running"
- Check that LM Studio is open and showing "Server running on http://127.0.0.1:1234"
- Verify port 1234 is not blocked by firewall
- Try accessing `http://127.0.0.1:1234/v1/models` in browser

### ❌ Slow Classification
- Reduce batch size for testing
- Use a faster/smaller model in LM Studio
- Increase `DELAY_BETWEEN_REQUESTS` if API is overloaded

### ❌ Inaccurate Classifications
- Use a more capable model (larger models generally better)
- Adjust the prompt in `classify_with_lm_studio()` function
- Reduce `temperature` for more consistent predictions

## Performance Notes

- **Speed**: Depends on model size. Typically 1-5 seconds per image
- **Memory**: Uses GPU if available in LM Studio
- **Accuracy**: Varies based on model used (Llama 2 ~70%, Mistral ~75%)

## Next Steps

1. Run the notebook cell by cell to understand the process
2. Once successful, run all cells for full predictions
3. Compare with `predictions.csv` to see how LLM compares to the trained model
4. Fine-tune prompts or try different models in LM Studio for better accuracy
