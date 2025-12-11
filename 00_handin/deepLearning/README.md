# Running `CodeChallenge01_B10_Reconstruction_Inference_local.ipynb`

## Dependencies
- Create/activate the `eurosat-ml` conda env exactly as for the ML baseline: `conda create -n eurosat-ml --file 00_handin/requirements_cyril.txt && conda activate eurosat-ml`.
- (Optional) register the env for Jupyter with `python -m ipykernel install --user --name eurosat-ml` and start Jupyter from the repo root.

## Data, models, and expected paths
Set these environment variables **before** launching Jupyter, or keep the defaults shown below:
- `EUROSAT_DATA_ROOT` (default `/Users/cyrilgabriele/Documents/School/00_Courses/02_ML/03_project/EuroSAT_MS`): extracted EuroSAT tiles (`<class>/*.tif`). If the folder is missing but `EUROSAT_DATA_ZIP` (default `.../EuroSAT_MS.zip`) exists, the notebook will unzip it automatically.
- `EUROSAT_KAGGLE_TEST_DIR` (default `.../kaggle_data/testset/testset`): Kaggle `test_*.npy` tiles used for inference.
- `EUROSAT_MODELS_DIR` / `EUROSAT_B10_MODEL_DIR` (defaults `<repo>/artifacts/models` and `<repo>/artifacts/models/cirrus_cnn`): location to cache the trained CirrusCNN weights.
- `EUROSAT_OUTPUT_DIR` (default `<repo>/outputs`) and `EUROSAT_PLOTS_DIR` (default `<repo>/artifacts/plots`): destinations for submission CSVs and diagnostic figures.
- Classifier weights: the notebook first looks for a local copy of the Hugging Face repo `Rhodham96/EuroSatCNN` inside any of
  `/Users/cyrilgabriele/Documents/School/00_Courses/02_ML/03_project/Rhodham96-EuroSatCNN`, `<repo>/artifacts/models/Rhodham96-EuroSatCNN`, `./local_models/Rhodham96-EuroSatCNN`, `./models/Rhodham96-EuroSatCNN`, or `~/models/Rhodham96-EuroSatCNN`. If none exist it falls back to downloading from the HF Hub (requires network + optional token configured via `huggingface-cli login`). Ensure `model_def.py` and `pytorch_model.bin` are present if you rely on the local option.

## Execution
1. `cd /path/to/ML2025-Lab && conda activate eurosat-ml`.
2. Launch Jupyter (`jupyter lab 00_handin/deepLearning/CodeChallenge01_B10_Reconstruction_Inference_local.ipynb`) from the repo root so relative paths resolve.
3. Run all cells in order. The notebook will:
   - Load/extract EuroSAT data and Kaggle `.npy` tiles.
   - Train the CirrusCNN B10 reconstruction network (weights cached under `EUROSAT_B10_MODEL_DIR`).
   - Load the EuroSat classifier (local weights preferred) and fine-tune it on synthesized B10 inputs.
   - Perform Kaggle inference and write `submission_with_cirrus_<timestamp>.csv` inside `EUROSAT_OUTPUT_DIR`.
4. Inspect `artifacts/plots/` for training diagnostics and `outputs/` for the submission plus any auxiliary logs.
