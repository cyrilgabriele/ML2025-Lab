# Running `CodeChallenge01_ML_Cyril*.ipynb`

## 1. Dependencies
1. From the repo root run `conda create -n eurosat-ml --file 00_handin/requirements_cyril.txt` and then `conda activate eurosat-ml`.
2. (Optional but convenient) register the environment for Jupyter with `python -m ipykernel install --user --name eurosat-ml`.

## 2. Data expected on disk/Drive
- **EuroSAT_MS training set**: keep the official archive as `EuroSAT_MS.zip`. The Colab notebook unzips it to `/content/EuroSAT_MS`; the local notebook expects it extracted to a folder named `EuroSAT_MS`.
- **Kaggle test set (`testset/testset/test_*.npy`)**: keep it inside `kaggle_data.zip` for Colab; extract it locally to `.../kaggle_data/testset/testset/`.

## 3. `CodeChallenge01_ML_Cyril.ipynb` (Colab + Google Drive)
1. Upload `EuroSAT_MS.zip` and `kaggle_data.zip` to Drive (default path assumed in the notebook: `MyDrive/ML_HSG/`).
2. Open the notebook in Google Colab, run the first `!pip install rasterio` cell, then execute cells sequentially.
3. If your Drive path differs, edit the constants `ZIP_PATH` (EuroSAT archive) and `KAGGLE_ZIP` (Kaggle test archive) before running the unzip cells.
4. The notebook mounts Drive, extracts both archives under `/content`, trains the baseline model, builds predictions, and writes `submission.csv` in `/content` as well as copying it to `MyDrive/ML_HSG/kaggle_submissions/`.

## 4. `CodeChallenge01_ML_Cyril_local.ipynb` (macOS/local run)
1. Extract `EuroSAT_MS.zip` so that the directory structure `EuroSAT_MS/<class_name>/*.tif` exists.
2. Extract the Kaggle test archive so that the `.npy` files live under `kaggle_data/testset/testset/`.
3. Either keep the defaults (`EUROSAT_DATA_BASE=/Users/cyrilgabriele/Documents/School/00_Courses/02_ML/03_project`) or set your own paths **before** starting Jupyter:
   - `EUROSAT_DATA_ROOT` → folder containing the extracted EuroSAT tiles.
   - `EUROSAT_KAGGLE_TEST_DIR` → folder containing the Kaggle `test_*.npy` files.
   - `EUROSAT_OUTPUT_DIR` (optional) → destination for plots/submission; defaults to `<repo>/cc_1/outputs`.
4. Launch Jupyter (`jupyter lab 00_handin/ml/CodeChallenge01_ML_Cyril_local.ipynb`) inside the `eurosat-ml` environment and run all cells. The notebook verifies the paths, trains, and writes `submission_ml_baseline.csv` plus plots under `EUROSAT_OUTPUT_DIR`.

## 5. Approach summary (applies to both notebooks)
- **Data ingestion & normalization:** Each notebook scans the EuroSAT class folders, records `(path, label)` pairs, and loads tiles via `rasterio`. Every band is robustly scaled to `[0,1]` using the 2nd/98th percentiles to damp outliers, and NDVI (`(B8-B4)/(B8+B4)`) is computed for vegetation-specific context.
- **Band alignment:** Training tiles contain 13 bands (`B1…B12` + `B8A`), but the Kaggle test set provides only 12 bands (missing `B10` and shuffling the order). Both notebooks document the canonical training/test band orders, compute index maps, reorder Kaggle arrays into the canonical layout, and insert a zero-filled placeholder band at the `B10` index so all downstream feature extractors receive tensors with identical semantics.
- **Feature engineering:** Two complementary feature families are constructed for every tile:
  1. **Statistical descriptors** – per-band mean, standard deviation, and percentile triplets plus global NDVI statistics, yielding an interpretable vector that summarizes spectral distributions.
  2. **Downsampled spatial descriptors** – each band is strided down to roughly `16×16`, flattened, and later compressed with PCA to retain coarse texture/shape cues while controlling dimensionality.
- **Dataset splitting:** The scripts build stratified train/val/test splits (80/20 train-test, then 80/20 within the train fold for validation) and slice both feature matrices plus labels/paths with the same indices to maintain alignment.
- **Model selection:** Multiple scikit-learn pipelines are defined per feature family (Logistic Regression, RBF SVM, Random Forest). Each pipeline includes the relevant preprocessing (scalers, PCA) and is evaluated on the validation split via the shared `eval_pipeline` helper. The best-performing pipeline is retained, and its name determines whether the statistical or spatial features will be used for inference.
- **Evaluation & reporting:** The winning model is applied to the held-out test split, printing classification metrics and rendering confusion matrices. Plots and summary artifacts are saved to the configured output directory.
- **Kaggle inference:** Kaggle `.npy` tiles are normalized, reordered, padded to 13 bands, and transformed using the feature extractor that matches the selected model. Predictions are generated, mapped to class names, and written as submission CSVs (`submission.csv` in Colab, `submission_ml_baseline.csv` locally) with optional Drive copies for Colab.
