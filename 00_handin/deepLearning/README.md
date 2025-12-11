# Running `CodeChallenge01_B10_Foundation_Model.ipynb`

## Dependencies
- Create/activate the `eurosat-ml` conda env exactly as for the ML baseline: `conda create -n eurosat-ml --file 00_handin/requirements_cyril.txt && conda activate eurosat-ml`.
- (Optional) register the env for Jupyter with `python -m ipykernel install --user --name eurosat-ml` and start Jupyter from the repo root.

## Data, models, and expected paths
Set these environment variables **before** launching Jupyter, or keep the defaults shown below:
- `EUROSAT_DATA_ROOT` (default `/Users/cyrilgabriele/Documents/School/00_Courses/02_ML/03_project/EuroSAT_MS`): extracted EuroSAT tiles (`<class>/*.tif`). If the folder is missing but `EUROSAT_DATA_ZIP` (default `.../EuroSAT_MS.zip`) exists, the notebook will unzip it automatically.
- `EUROSAT_KAGGLE_TEST_DIR` (default `.../kaggle_data/testset/testset`): Kaggle `test_*.npy` tiles used for inference.
- `EUROSAT_MODELS_DIR` / `EUROSAT_B10_MODEL_DIR` (defaults `<repo>/artifacts/models` and `<repo>/artifacts/models/cirrus_cnn`): location to cache the trained CirrusCNN weights.
- `EUROSAT_OUTPUT_DIR` (default `<repo>/outputs`) and `EUROSAT_PLOTS_DIR` (default `<repo>/artifacts/plots`): destinations for submission CSVs and diagnostic figures.
- Classifier weights: torchvision automatically downloads the ImageNet-1K `convnext_base` weights the first time the notebook calls `models.convnext_base(weights=...)`. Ensure outbound network access is allowed once, or pre-download the weights into the local torchvision cache if running offline later.

## Execution
1. `cd /path/to/ML2025-Lab && conda activate eurosat-ml`.
2. Launch Jupyter (`jupyter lab 00_handin/deepLearning/CodeChallenge01_B10_Foundation_Model.ipynb`) from the repo root so relative paths resolve.
3. Run all cells in order. The notebook will:
   - Load/extract EuroSAT data and Kaggle `.npy` tiles.
   - Train the CirrusCNN B10 reconstruction network (weights cached under `EUROSAT_B10_MODEL_DIR`).
   - Load a torchvision `convnext_base` foundation model (ImageNet-1K weights) and fine-tune its classifier head on synthesized B10 inputs.
   - Perform Kaggle inference and write `submission_with_cirrus_<timestamp>.csv` inside `EUROSAT_OUTPUT_DIR`.
4. Inspect `artifacts/plots/` for training diagnostics and `outputs/` for the submission plus any auxiliary logs.

## Approach summary
- **Band bookkeeping & preprocessing:** EuroSAT tiles expose 13 Sentinel-2 bands (`B1…B12` + `B8A`), whereas Kaggle `.npy` files only supply 12 bands (missing `B10` and reshuffled). The notebook records both layouts, computes index maps, and always works in a canonical 12-band ordering so the missing channel can be synthesized deterministically. All rasters are robustly normalized (2nd/98th percentile stretch) before feeding neural networks to avoid sensor-specific scale drift.
- **CirrusCNN training (B10 reconstruction):** Every GeoTIFF is sliced into random patches (default 64 per tile, 64×64 pixels). Each patch is split into inputs (all bands except `B10`) and the ground-truth target (`B10`). The lightweight convolutional auto-regressor (1×1/3×3 conv stack) is optimized with SmoothL1 loss, Adam, and early stopping. Validation metrics plus MAE/RMSE are logged, and the best checkpoint (state dict + metadata) is saved to `EUROSAT_B10_MODEL_DIR`.
- **Synthesizing consistent 13-band tensors:** Helpers `synthesize_cirrus` and `pad_to_13_bands` use the trained CirrusCNN to infer `B10` for any 12-band tile (EuroSAT or Kaggle), insert the prediction at the correct index, and return a channel-first tensor. This ensures that both the fine-tuning stage and the final inference stage consume inputs with identical band semantics despite the Kaggle gap.
- **Classifier loading and fine-tuning:** Torchvision’s `convnext_base` backbone (ImageNet-1K weights) is instantiated, its feature extractor is frozen, and only the classifier head is replaced. After inserting CirrusCNN-generated `B10`, the notebook builds stratified train/val/test folds, converts tiles into RGB tensors via the foundation transforms, and fine-tunes the head with AdamW, gradient clipping, patience-based early stopping, and full metric tracking (accuracy, PR/ROC curves, confusion matrix). The resulting weights plus history are saved for reproducibility.
- **Kaggle inference pipeline:** Kaggle `.npy` tiles are read via a custom `Dataset` that reorders/pads/normalizes bands and feeds them through the ConvNeXt preprocessing stack. A `DataLoader` streams batches through the fine-tuned classifier on the chosen device (`cuda`, `mps`, or CPU). Predictions are mapped back to class names, sorted by `test_id`, and exported as `submission_with_cirrus_<timestamp>.csv` in `EUROSAT_OUTPUT_DIR`.
