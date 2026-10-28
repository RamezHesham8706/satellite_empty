# Land Type Classification with Sentinel-2 Satellite Imagery

Project 6 for the Digital Egypt Pioneers Initiative (DEPI), AI & Data Science
Track, Round 2.

This repository contains data exploration notebooks and utilities for
land-cover classification with the EuroSAT dataset. It includes RGB and
multispectral dataset download support, reproducible class-stratified PyTorch
data loaders, and CNN/ResNet-18 model definitions with training and evaluation
entry points.

> **Project status:** EuroSAT download, exploration, training, and evaluation
> utilities are available. The Sentinel-2 AOI definitions are configuration
> only; automated Sentinel-2 scene search/download is not implemented here.

## Contents

| Path | Description |
| --- | --- |
| `src/aoi_config.py` | Example Egypt areas of interest, Sentinel-2 band groups, land-cover labels, and scene-query preferences |
| `src/download_eurosat.py` | Download and extract the EuroSAT RGB or multispectral archive |
| `src/dataset.py` | EuroSAT multispectral loader, reproducible train/validation/test splits |
| `src/model.py` | `SmallCNN` and `resnet18` model builders |
| `notebooks/initial_inspection.ipynb` | Initial dataset inspection notebook |
| `notebooks/01_data_exploration_eurosat_ms.ipynb` | Multispectral exploration, spectral signatures, and optional NDVI analysis |
| `notebooks/data_collection.ipynb` | Notebook for the EuroSAT download and project workflow |
| `outputs/` | Example exported tables and charts from dataset exploration |
| `reports/figures/` | Example report figures |
| `tests/` | Focused tests for the dataset loader, downloader, and model builder |

## Requirements

- Python 3.10 or newer
- Packages listed in [`requirements.txt`](requirements.txt)
- Jupyter Notebook or JupyterLab to run the notebooks

The multispectral exploration notebook may also require `rasterio` to read the
GeoTIFF images. Install it in your environment if you run the notebook cells
that inspect multispectral image pixels.

## Setup

Create and activate a virtual environment, then install the project
dependencies.

**Windows PowerShell**

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install notebook rasterio
```

**macOS or Linux**

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install notebook rasterio
```

`notebook` and `rasterio` are for interactive notebook work and are not listed
as core application dependencies in `requirements.txt`.

## Download EuroSAT

The downloader supports the RGB and multispectral archives. By default, it
downloads and extracts the RGB archive into `data/raw/EuroSAT_RGB`:

```bash
python src/download_eurosat.py
```

Download the 13-band multispectral version instead:

```bash
python src/download_eurosat.py --kind ms
```

The older `--ms` flag is also supported:

```bash
python src/download_eurosat.py --ms
```

To choose a different extraction directory or provide a mirror URL:

```bash
python src/download_eurosat.py --kind rgb --output_dir ./data/custom/EuroSAT_RGB
python src/download_eurosat.py --kind ms --download_url https://example.com/EuroSAT_MS.zip
```

The default archives are hosted on
[Zenodo record 7711810](https://zenodo.org/records/7711810). The downloader
stores each ZIP beside its extraction directory. Downloaded data is not
included in Git; the `.gitignore` excludes raw, interim, and processed data.

### Expected multispectral dataset layout

`src/dataset.py` reads the 13-band GeoTIFFs using `rasterio` and NumPy image
files directly. It accepts a dataset root or a containing directory and
searches nested directories for the class folders:

```text
data/raw/EuroSAT_MS/
  EuroSAT_MS/
    AnnualCrop/
      image_1.tif
    Forest/
      image_2.tif
    ...
```

If the archive extracts with an extra nested directory, `resolve_data_dir()`
finds the directory containing the class folders. The GeoTIFF band sequence is
preserved as Sentinel-2 B01, B02, B03, B04, B05, B06, B07, B08, B8A, B09,
B10, B11, and B12. Raw reflectance values are divided by 10000 when loaded
into the dataset; values already in the 0–1 range are left unchanged.

## Explore the data

Start Jupyter from the repository root:

```bash
jupyter notebook
```

Then open a notebook under `notebooks/`. The multispectral exploration covers
dataset structure and class counts, sample images, band metadata, per-class
spectral signatures, variability, and optional NDVI. The notebooks read local
dataset files and save selected results under `outputs/` and
`reports/figures/`.

EuroSAT RGB contains 27,000 labeled images across 10 classes. EuroSAT
multispectral contains the corresponding 13-band image data; the multispectral
notebook uses it to analyze class-level spectral responses. NDVI is calculated
from the red (`B04`) and near-infrared (`B08`) bands when the notebook's input
data supports that calculation.

## Python utilities

Create train, validation, and test loaders from the multispectral dataset:

```python
from src.dataset import get_loaders

train_loader, val_loader, test_loader, class_names = get_loaders(
    data_dir="data/raw/EuroSAT_MS",
    batch_size=64,
    seed=42,
    splits_path="data/processed/splits.csv",
)
```

The split is stratified by class (70% train, 15% validation, 15% test).
`splits.csv` is created on the first run and reused on later runs, so evaluation
uses the saved sample assignments instead of generating new splits. Paths in
the CSV are relative to the dataset root. Training images receive random
horizontal and vertical flips; validation and test images use no augmentation.
The loader returns the class names in the project's fixed class order.

Build one of the available models for the desired number of classes:

```python
from src.model import build_model

model = build_model("cnn", num_classes=len(class_names))
# Or use a ResNet-18 model:
model = build_model("resnet18", num_classes=len(class_names))
```

The ResNet-18 builder attempts to load ImageNet weights and falls back to
random initialization if the weights cannot be loaded.

Train the default 13-band CNN and save its best checkpoint:

```powershell
python -m src.train --data-dir data/raw/EuroSAT_MS
```

The training command creates `data/processed/splits.csv` on its first run and
reuses it afterward. Evaluate the saved checkpoint with the same split:

```powershell
python -m src.evaluate
```

## Tests

Run the focused project tests from the repository root:

```bash
python -m pytest
```

## Outputs

The repository includes example exploration outputs, such as spectral
signature and NDVI tables and plots. They are illustrative results already
stored in the repository; re-running the notebooks may replace or regenerate
them from the local dataset.

## Data and citation

EuroSAT is a land-use and land-cover dataset derived from Sentinel-2 satellite
imagery. See the
[EuroSAT project page](https://github.com/phelber/EuroSAT) and the
[Zenodo dataset record](https://zenodo.org/records/7711810) for dataset
information and citation details. Review and follow the dataset's terms of use
when downloading or redistributing it.
