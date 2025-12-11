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
