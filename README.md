# Hyperspectral Glioblastoma Classification

This project uses hyperspectral histology images to classify glioblastoma tissue samples as **Tumor** or **Non-Tumor** using both classical machine learning models and convolutional neural networks (CNNs).

The project compares:
- Full-spectrum CNN models using all 826 spectral bands
- Reduced-band CNN models using ANOVA and L1-based feature selection

The goal is to determine whether hyperspectral imaging can effectively classify tumor tissue and whether spectral dimensionality reduction can preserve performance while reducing model complexity.

***

## Project Goal

Glioblastoma (GBM) is the most aggressive malignant brain tumor in adults. Accurate tumor identification is critical for treatment planning and surgical precision.

This project explores whether hyperspectral imaging can improve tumor classification by identifying spectral differences between tumor and non-tumor tissue that may not be visible using standard imaging methods.

***

## Dataset

The dataset contains hyperspectral image patches extracted from glioblastoma histology samples.

Each hyperspectral patch contains:
- Spatial information (32 × 32 pixels)
- Spectral information (826 bands)

**Classes:**
- Tumor
- Non-Tumor

**Training data:** Augmented training patches  
**Testing data:** Held-out test dataset

***

## Repository Structure

```text
capstone-hyperspectral-gbm/
│
├── README.md
├── requirements.txt
│
├── all_bands_cnn.py
├── partial_bands_cnn.py
│
├── train_augmented/
│   ├── manifest.csv
│   └── *.npz
│
├── test/
│   ├── manifest.csv
│   └── *.npz
│
└── results/
    ├── top100_bands_anova.csv
    ├── l1_selected_bands.csv
    └── model outputs
```

***

## Models

### 1. All Bands CNN

The all-bands CNN uses the complete hyperspectral input with all 826 spectral bands.

**Characteristics:**
- Full spectral representation
- Highest spectral information retention
- Largest computational cost

***

### 2. Partial Bands CNN

The partial-bands CNN uses reduced spectral-band subsets selected through:
- ANOVA feature ranking
- L1-regularized Logistic Regression

**Characteristics:**
- Reduced dimensionality
- Lower spectral input size
- Evaluates importance of selected spectral regions

***

## Data Format

Each `.npz` file stores hyperspectral image patches with the following expected shape:

```
(num_patches, height, width, bands)
```

**Example:**
```
(100, 32, 32, 826)
```

Each dataset folder (`train_augmented/`, `test/`) contains:
- `.npz` hyperspectral patch files
- `manifest.csv`

**`manifest.csv` fields may include:**

| Field | Description |
|-------|-------------|
| `file_name` | Name of the `.npz` file |
| `sample_id` | Unique sample identifier |
| `patient_id` | Patient identifier |
| `label` | Class label (Tumor / Non-Tumor) |
| `label_number` | Numeric class label |
| `augmentation_type` | Type of augmentation applied |
| `num_patches` | Number of patches in the file |

***

## Training and Evaluation

The CNN models use the following training configuration:

- Grouped train/validation split by `sample_id` to prevent data leakage
- Patch-based training with data augmentation
- Adam optimizer
- Early stopping
- Learning rate scheduling
- Threshold tuning

> The grouped split prevents augmented versions of the same sample from appearing in both training and validation sets, reducing validation leakage.

Final testing is performed on a completely held-out test dataset.

***

## Output Files

Model outputs are saved to the `results/` folder or model-specific results subfolders.

| File | Description |
|------|-------------|
| `*.pt` | Best model weights (e.g., `all_bands_cnn_best.pt`) |
| `training_curves.png` | Training and validation loss/accuracy curves |
| `test_confusion_matrix.png` | Confusion matrix on the test set |
| `all_bands_cnn_results.csv` | CSV metric summary for all-bands model |
| `partial_bands_cnn_results_anova.csv` | CSV metric summary for partial-bands model |
| `*.json` | JSON result summaries |
| `*.log` | Run log files |

***

## Evaluation Metrics

| Metric | Description |
|--------|-------------|
| **Accuracy** | Percentage of total predictions that are correct |
| **Precision** | Of all predicted tumor cases, how many were actually tumor — high precision means fewer false positives |
| **Recall (Sensitivity)** | Of all actual tumor cases, how many were correctly identified — high recall means fewer missed tumors |
| **F1 Score** | Harmonic mean of precision and recall; useful when both false positives and false negatives matter |
| **ROC-AUC** | Measures how well the model separates tumor and non-tumor classes across all thresholds — higher values indicate stronger class separation |

***

## Development Environment

This project was developed and tested using:

- **OS:** Windows
- **Editor:** VS Code
- **Shell:** PowerShell
- **Language:** Python
- **Framework:** PyTorch
- **GPU Acceleration:** PyTorch DirectML (AMD GPU)

> If DirectML is unavailable, the code automatically falls back to CPU execution.

***

## Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/nateLep/capstone-hyperspectral-gbm.git
cd capstone-hyperspectral-gbm
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Add the Dataset

> **Note:** The full `.npz` hyperspectral patch dataset is **not included** in this repository due to GitHub file size limitations.

Place your dataset files in the following structure before running any scripts:

```text
capstone-hyperspectral-gbm/
│
├── train_augmented/
│   ├── manifest.csv
│   └── *.npz
│
└── test/
    ├── manifest.csv
    └── *.npz
```

***

## Running the Project

### Run the Full-Spectrum CNN

```bash
python all_bands_cnn.py
```

Trains and evaluates the CNN using all 826 spectral bands.

### Run the Partial-Bands CNN

```bash
python partial_bands_cnn.py
```

Trains and evaluates the CNN using a reduced spectral band subset. Band-selection mode is controlled by modifying the following line inside `partial_bands_cnn.py`:

```python
USE_BANDS = "anova"  # Change to "l1" or "anova"
```

**Supported band files:**
- `top100_bands_anova.csv`
- `l1_selected_bands.csv`

> **Note:** Training times may vary significantly depending on hardware, GPU availability, DirectML support, dataset size, and the number of spectral bands used. Full-spectrum CNN models may require several hours of training on local hardware.

***

## Key Findings

Key observations from experimentation:

- Classical ML models performed strongly on spectral data
- CNNs benefited from grouped train/validation splitting
- Lower learning rates improved training stability
- Aggressive feature reduction reduced CNN performance
- ANOVA-100 preserved more spectral information than ANOVA-50
- ROC-AUC improved after optimization changes

***

## Limitations

Current limitations include:

- Small dataset size with limited patient diversity
- Long training times on local hardware
- Patch-level classification instead of sample-level aggregation
- Potential patch-label noise
- Reduced-band methods may discard useful spectral information

***

## Future Improvements

Possible future improvements include:

- Sample-level prediction aggregation
- Larger and more diverse patient datasets
- Additional band-selection techniques
- 3D CNN architectures
- Transformer-based hyperspectral models
- Improved GPU acceleration support
- External dataset validation
- Class-weighted loss functions
- Threshold optimization for improved recall