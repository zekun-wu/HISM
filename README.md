# HISM: Highlight-Informed Saliency Model

Implementation of **HISM (Highlight-Informed Saliency Model)** for predicting temporal normalized saliency of highlighted UI elements in monitoring tasks.

**Paper:** Wu, Z. & Feit, A. M. (2024). *Understanding and Predicting Temporal Visual Attention Influenced by Dynamic Highlights in Monitoring Tasks*. IEEE Transactions on Human-Machine Systems.

## Overview

HISM predicts **normalized saliency NS(e, t)** of a highlighted UI element over time, modeling when, how much, and for how long attention is attracted by visual highlights. This is a temporal regression problem that combines spatial context (UI layout and highlight geometry) with temporal dynamics (highlight state and task context).

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

### Data Structure

Each trial should be organized as:
```
data/trials/{trial_id}/
├── saliency_summary.csv    # Temporal data: NS, Highlight, Task
├── global.png              # UI screenshot
└── mask.png                # Highlight mask
```

The CSV should contain columns: `normalized_saliency_smoothed`, `Highlight`, `Task`.

### Training

```bash
python train.py
```

Training configuration is controlled via `configs/training.yaml`. Model architecture (Transformer/LSTM, with/without task vector) is configured in `configs/model.yaml`.

### Testing

```bash
python test.py
```

Automatically loads the checkpoint matching your model configuration and evaluates on the test set.

## Model Architecture

HISM consists of three components:

1. **Spatial Encoder**: ResNet50-based encoder that processes UI image and highlight mask to capture spatial context
2. **Temporal Encoder**: LSTM or Transformer encoder that processes temporal sequences (highlight vector vₜ and optional task vector cₜ)
3. **Fusion Module**: MLP that concatenates spatial and temporal features to predict NS(e, t)

## Configuration

All settings are controlled via YAML files:

- **`configs/model.yaml`**: Temporal encoder type (transformer/lstm), task vector usage, architecture hyperparameters
- **`configs/dataset.yaml`**: Data paths, CSV column names, image preprocessing
- **`configs/training.yaml`**: Optimizer, learning rate, batch size, callbacks

## Citation

If you use this code, please cite:

```bibtex
@article{wu2024hism,
  title={Understanding and Predicting Temporal Visual Attention Influenced by Dynamic Highlights in Monitoring Tasks},
  author={Wu, Zekun and Feit, Aaron M.},
  journal={IEEE Transactions on Human-Machine Systems},
  year={2024}
}
```
