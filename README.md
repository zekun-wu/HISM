# HISM: Highlight-Informed Saliency Model

This repository implements **HISM (Highlight-Informed Saliency Model)**, a temporal attention prediction model proposed in:

> Wu, Z. & Feit, A. M.  
> *Understanding and Predicting Temporal Visual Attention Influenced by Dynamic Highlights in Monitoring Tasks*  
> IEEE Transactions on Human-Machine Systems

The goal of HISM is to predict **temporal normalized saliency (NS)** of a highlighted UI element, modeling *when*, *how much*, and *for how long* attention is attracted by a visual highlight.

---

## 1. Task Definition

HISM addresses the following task:

> Given the temporal dynamics of a visual highlight and task context, predict the **normalized saliency NS(e, t)** of a highlighted UI element over time.

This is a **temporal regression problem**, not pixel-level saliency prediction.

---

## 2. Data Assumptions

This repository assumes that **all saliency-related quantities are precomputed**.

### What is already computed (offline):
- Normalized saliency **NS(e, t)**
- Highlight vector **vₜ**
- Task vector **cₜ**

The model **does not** consume:
- Raw gaze points
- Fixations
- Saliency maps

---

## 3. Data Organization

### One folder per trial

Each trial is stored in its own directory:

```
data/trials/0001/
├── data.csv
├── ui.png
└── highlight_mask.png
```

### `data.csv`

Each CSV contains a time series with one row per timestep:

| column | description |
|------|-------------|
| `t`  | time index (implicit ordering) |
| `ns` | normalized saliency target |
| `v`  | highlight vector (−1, 0, 1) |
| `c_*` | task state features |

Example:

```csv
t,ns,v,c_1,c_2,c_3
0,0.02,-1,0.8,0.1,0.0
1,0.31,1,0.7,0.1,0.0
2,0.48,1,0.6,0.2,0.1
```

### Static spatial context

Each trial also includes:

- **ui.png**: a static screenshot of the user interface
- **highlight_mask.png**: a binary mask indicating the highlighted region

The mask is spatially aligned with the UI image.

Because highlight position and shape are fixed within a trial, spatial context is treated as static, not time-varying.

### Train / validation / test splits

Splits are defined by listing trial IDs:

```
data/splits/train.txt
data/splits/val.txt
data/splits/test.txt
```

This ensures no temporal leakage across trials.

---

## 4. Dataset (HISMDataset)

The dataset operates at the trial level.

For each trial, it loads:

- The full temporal sequence (vₜ, cₜ, NS)
- The static UI image
- The static highlight mask

The dataset returns tensors suitable for sequence modeling.

---

## 5. Model Architecture

HISM consists of three components:

### 5.1 Spatial Encoder

- **Input**: ui.png + highlight_mask.png
- **Implementation**: ResNet50 (from torchvision)
- **Output**: a single spatial feature vector per trial

The spatial encoder captures layout and highlight geometry.

### 5.2 Temporal Encoder

A single configurable temporal encoder is used.

**Supported modes:**
- LSTM
- Transformer Encoder

The choice is controlled via configuration, not separate files.

**Inputs:**
- Highlight vector vₜ
- Task vector cₜ (optional)

**Output:**
- A temporal feature representation

### 5.3 Fusion & Prediction

- Spatial and temporal features are concatenated
- A multi-layer perceptron predicts NS(e, t)
- The model is trained using MSE loss.

---

## 6. Configuration

All experiments are controlled via YAML files:

- **dataset.yaml**: paths, splits, CSV schema
- **training.yaml**: optimizer, learning rate, epochs
- **model.yaml**: LSTM vs Transformer, hidden sizes

This enables clean ablation studies without code duplication.

---

## 7. Training & Evaluation

### Training
```bash
python train.py
```

### Testing
```bash
python test.py
```

### Visualization
```bash
python visualize.py
```

Visualization scripts reproduce NS-over-time curves similar to those reported in the paper.

---

## 8. Design Philosophy

This repository is intentionally:

- **HISM-only** (no pixel-level saliency baselines)
- **Data-explicit** (NS is a first-class target)
- **Trial-centric** (no frame-level leakage)
- **Minimal** (no unnecessary abstraction layers)

The structure mirrors the conceptual separation in the paper:

- NS is defined and measured first
- NS is predicted afterward

---

## 9. Extensibility

The design supports future extensions:

- Additional temporal encoders (GRU, BiLSTM)
- Alternative highlight representations
- Multi-element NS prediction
- Reintroduction of pixel-level saliency baselines

---

## 10. Citation

If you use this code, please cite the original paper.

