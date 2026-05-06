# Sleep Stage Classification

A machine-learning pipeline that classifies polysomnography (PSG) recordings into five sleep stages using EEG, EOG, and EMG signals.

## Overview

Sleep staging is typically done manually by experts who read overnight PSG recordings. This project automates that process by:

1. Loading raw PSG recordings (EDF format) and their corresponding hypnogram annotations.
2. Segmenting each recording into 30-second epochs.
3. Extracting time-domain statistical features (mean, std, min, max) for every channel in every epoch.
4. Training an XGBoost classifier to predict one of five sleep stages per epoch.
5. Evaluating the model with cross-validation and on a held-out test subject.

An experimental CNN + Transformer hybrid deep-learning model (`SleepHybridNet`) is also implemented in `final.ipynb`.

## Sleep Stages

| Label | Stage | Description |
|-------|-------|-------------|
| 0 | W | Wakefulness |
| 1 | N1 | Light sleep (NREM 1) |
| 2 | N2 | Intermediate sleep (NREM 2) |
| 3 | N3 | Deep / slow-wave sleep (NREM 3 & 4 combined) |
| 4 | REM | Rapid Eye Movement sleep |

## Dataset

The project uses recordings from the [PhysioNet Sleep-EDF Database](https://physionet.org/content/sleep-edfx/1.0.0/). Each subject contributes two EDF files:

- `SC4xxxE0-PSG.edf` / `ST7xxxJ0-PSG.edf` — raw multi-channel PSG signal
- `SC4xxxEC-Hypnogram.edf` / `ST7xxxJP-Hypnogram.edf` — sleep stage annotations

**Channels used:**
- `EEG Fpz-Cz` and `EEG Pz-Oz` — primary EEG signals
- `EOG horizontal` — electrooculogram (useful for detecting REM)
- `EMG submental` — electromyogram (useful for detecting REM)

Excluded channels: `Resp oro-nasal`, `Temp rectal`, `Event marker`.

## Repository Structure

```
final_proj_data/
├── final.ipynb    # Full pipeline notebook (recommended entry point)
├── trial1.py      # Standalone script — 5 training subjects, XGBoost
├── trial2.py      # Standalone script — 4 training subjects, XGBoost + mlxtend feature selection
└── README.md
```

### File Descriptions

| File | Description |
|------|-------------|
| `final.ipynb` | Complete, well-organized notebook. Includes data loading, EDA (stage distribution, transition matrix, correlation heatmap, EEG visualization), XGBoost training with subject-aware GroupKFold cross-validation, SHAP feature importance, hypnogram plots, confusion matrix, and the `SleepHybridNet` deep-learning model. |
| `trial1.py` | Early iteration script. Trains XGBoost on 5 SC subjects, runs 5-fold stratified cross-validation, evaluates on one held-out test subject, and plots the predicted hypnogram and confusion matrix. |
| `trial2.py` | Refinement of `trial1.py`. Uses 4 SC training subjects and adds `mlxtend` sequential feature selection. Includes additional EEG/PSD visualizations. |

## Getting Started

### Prerequisites

Install the required Python packages:

```bash
pip install mne yasa xgboost lightgbm catboost scikit-learn pandas numpy matplotlib seaborn shap tqdm mlxtend torch
```

### Data Setup

1. Download the Sleep-EDF recordings from PhysioNet.
2. Place all PSG and Hypnogram EDF files in a single local folder.
3. Update the `project_destination` variable at the top of the script / notebook to point to that folder:

```python
# trial1.py / trial2.py
project_destination = '/path/to/your/edf/files'
```

### Running the Notebook

```bash
jupyter notebook final.ipynb
```

### Running the Scripts

```bash
python trial1.py
python trial2.py
```

## Model Details

### Feature Extraction

For every 30-second epoch and every channel (EEG Fpz-Cz, EEG Pz-Oz, EOG horizontal, EMG submental), four statistics are computed:

- Mean amplitude
- Standard deviation
- Minimum amplitude
- Maximum amplitude

This yields 16 features per epoch (4 channels × 4 statistics).

### XGBoost Classifier

- **Algorithm:** XGBoost multi-class classification (`mlogloss` objective)
- **Validation:** Stratified 5-fold cross-validation; held-out test subject
- **Explainability:** SHAP summary plots for feature importance

### SleepHybridNet (experimental)

A PyTorch model that processes the raw epoch waveform directly:

1. **CNN block** — 1-D convolutions extract local temporal features.
2. **Transformer encoder** — self-attention captures long-range dependencies within the epoch.
3. **Fully-connected head** — maps representations to the 5-class output.

Training uses subject-aware `GroupKFold` to prevent data leakage across subjects.

## Outputs

- **Hypnogram** — visual timeline of predicted sleep stages across the night
- **Confusion matrix** — per-stage classification performance
- **SHAP plot** — feature importance ranked by mean absolute SHAP value
- **EEG / PSD plots** — time-domain and frequency-domain signal visualizations