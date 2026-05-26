# Robust Active Contour Models for Noisy and Low-Contrast Image Segmentation

**CS 269 Project — Effie Wu, Lyla Liu, Jaehoon Choi, Md Aeinul Islam**

---

## Overview

This project implements and evaluates four variants of the classical *snake* (parametric active contour) model, progressively introducing spatial and temporal robustness strategies:

| # | Method | Key idea |
|---|--------|----------|
| 1 | **Classical** | Baseline — fixed global β, same force/solver as Adaptive |
| 2 | **Adaptive** | Spatially adaptive β_i modulated by local gradient magnitude |
| 3 | **FullAdaptive** | Spatially adaptive β_i **and** γ_i |
| 4 | **RL-Adaptive** | Adaptive β_i + RL-learned temporal scheduling of (γ, σ) |

---

## Project Structure

```
269_project/
├── requirements.txt
│
├── src/                           # Core algorithm implementations
│   ├── snake.py                   # Classical snake (skimage wrapper, legacy)
│   ├── adaptive_snake.py          # Method 1 & 2: Classical (fixed β) and Adaptive (spatial β)
│   ├── full_adaptive_snake.py     # Method 3: FullAdaptive (spatial β + γ)
│   ├── energy.py                  # Shared energy + force field utilities
│   └── evaluation.py             # IoU, Hausdorff, mean contour distance
│
├── extensions/
│   ├── rl_snake_env.py            # Shared env utilities (_circular_snake, features)
│   └── adaptive_rl/               # Method 4: RL-Adaptive
│       ├── env.py                 # SnakeCtrlEnv — DQN environment (5 γ/σ presets)
│       └── train.py               # DQN training script (stable-baselines3)
│
├── experiments/
│   ├── synthetic_images.py        # Generate 5 synthetic test cases
│   ├── adaptive_results.py        # Main comparison: all 4 methods, synth + real data
│   └── rl_vs_adaptive.py          # Focused RL vs Adaptive comparison
│
├── data/
│   ├── Fluo-N2DL-HeLa/           # Fluorescence microscopy (Cell Tracking Challenge)
│   └── ultrasound-nerve-segmentation/  # Ultrasound nerve segmentation (Kaggle)
│
└── results/
    └── adaptive_results/          # Output figures and tables (tracked in git)
        ├── synthetic_visual.png
        ├── synthetic_metrics.png
        ├── synthetic_table.txt
        ├── fluo_visual.png / fluo_metrics.png / fluo_table.txt
        └── ultrasound_visual.png / ultrasound_metrics.png / ultrasound_table.txt
```

---

## Implementation Details

### Energy Formulation

The snake minimizes a total energy:

```
E_total = α · E_elastic  +  β · E_smooth  +  E_image
```

- **E_elastic** = Σ |v_i − v_{i−1}|²  — resists stretching (controls spacing)
- **E_smooth**  = Σ |v_i − 2v_{i−1} + v_{i−2}|²  — resists bending (controls curvature)
- **E_image**   = −|∇(G_σ * I)|²  — attracts contour toward edges

### Method 1 — Classical Snake (`src/adaptive_snake.py`, fixed β)

Uses the same implicit-time-step solver and edge force as Adaptive, but with
β fixed uniformly across all contour points (`beta_min = beta_max = 0.1`).
This provides a fair baseline that isolates the effect of spatial β adaptation.

**Key parameters:**
- `alpha` (0.015): elasticity
- `beta` (0.1): fixed bending stiffness, uniform everywhere
- `gamma` (5.0): time-step regularizer
- `sigma` (8.0): Gaussian smoothing for external energy
- `n_iter` (5000): optimization steps

### Method 2 — Adaptive Snake (`src/adaptive_snake.py`)

Replaces the fixed β with a **per-point** value β_i driven by local gradient magnitude:

```
β_i = β_min + (β_max − β_min) · exp(−k · |∇I(v_i)|)
```

- Near strong edges → |∇I| large → β_i → β_min → contour deforms freely
- In flat/noisy regions → |∇I| small → β_i → β_max → strong regularization

Each iteration solves the implicit linear system:

```
(A(β) + γI) · v^{t+1}  =  γ · v^t  +  F_ext(v^t)
```

The stiffness matrix A(β) is rebuilt every `update_every` iterations as the
contour moves to new positions.

**Key parameters:**
- `beta_min` (0.005): minimum bending stiffness (at strong edges)
- `beta_max` (0.3): maximum bending stiffness (in flat regions)
- `k` (5.0): sensitivity of β to gradient magnitude
- `gamma` (5.0): time-step regularizer
- `sigma` (8.0): Gaussian smoothing for external energy
- `update_every` (20): how often to recompute the adaptive stiffness matrix
- `n_iter` (5000): optimization steps

### Method 3 — FullAdaptive Snake (`src/full_adaptive_snake.py`)

Extends Adaptive by also making the damping parameter **γ per-point adaptive**:

```
γ_i = γ_min + (γ_max − γ_min) · exp(−k_γ · |∇I(v_i)|)
```

- Near strong edges → γ_i → γ_min → more aggressive motion
- In flat regions → γ_i → γ_max → more conservative, stable steps

**Key parameters (in addition to Adaptive):**
- `gamma_min` (2.0): minimum damping (at strong edges)
- `gamma_max` (10.0): maximum damping (in flat regions)
- `k_gamma` (5.0): sensitivity of γ to gradient magnitude

### Method 4 — RL-Adaptive Snake (`extensions/adaptive_rl/`)

Keeps the spatial β_i adaptation from Adaptive, and adds a DQN agent that
**temporally schedules** the global (γ, σ) pair every 50 iterations by
choosing among 5 presets:

| Preset | γ | σ | Strategy |
|--------|---|---|----------|
| 1 | 1 | 2 | Very aggressive, fine detail |
| 2 | 2 | 3 | Aggressive |
| 3 | 5 | 5 | Balanced |
| 4 | 10 | 6 | Conservative |
| 5 | 20 | 8 | Very conservative, coarse |

The agent observes gradient features and contour statistics at 32 probe points,
and learns when to apply coarse vs. fine parameter settings to maximize IoU.
Training uses stable-baselines3 DQN on procedurally generated synthetic images.

### Evaluation Metrics (`src/evaluation.py`)

| Metric | Formula | Direction |
|--------|---------|-----------|
| **IoU** | |Pred ∩ GT| / |Pred ∪ GT| | Higher is better ↑ |
| **Hausdorff Distance** | max(max_i min_j d(p_i, q_j), max_j min_i d(q_j, p_i)) | Lower is better ↓ |
| **Mean Contour Distance** | mean_i min_j d(p_i, q_j) | Lower is better ↓ |

---

## Setup

```bash
# 1. Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate        # macOS/Linux
# venv\Scripts\activate         # Windows

# 2. Install dependencies
pip install -r requirements.txt
```

---

## How to Run

### Main comparison (all 4 methods, synthetic + real data)

```bash
python experiments/adaptive_results.py
python experiments/adaptive_results.py --n-real 20       # more real images
python experiments/adaptive_results.py --skip-synth      # real data only
python experiments/adaptive_results.py --skip-real       # synthetic only
```

Saves to `results/adaptive_results/`:
- `synthetic_visual.png` / `synthetic_metrics.png` / `synthetic_table.txt`
- `fluo_visual.png` / `fluo_metrics.png` / `fluo_table.txt`
- `ultrasound_visual.png` / `ultrasound_metrics.png` / `ultrasound_table.txt`

### Train the RL agent

```bash
python extensions/adaptive_rl/train.py
```

### Run Adaptive snake on a custom image

```python
from src.adaptive_snake import AdaptiveSnake
import numpy as np

model = AdaptiveSnake(alpha=0.015, beta_min=0.005, beta_max=0.3,
                      k=5.0, gamma=5.0, sigma=8.0, n_iter=5000,
                      update_every=20, wedge=1.0)

# initial circular contour
cy, cx = image.shape[0]//2, image.shape[1]//2
theta = np.linspace(0, 2*np.pi, 120, endpoint=False)
init = np.column_stack([cy + 80*np.sin(theta), cx + 80*np.cos(theta)])

snake, history = model.fit(image, init)
```

---

## Troubleshooting

**Snake doesn't converge / stays as a circle:**
- Increase `wedge` (edge force strength) or decrease `alpha`/`beta`
- Check that the image is normalized to [0, 1] before passing to the model

**Snake collapses to a point:**
- `alpha` or `beta` too large relative to external forces — reduce them
- Reduce `gamma` so steps are larger

**Import errors:**
- Make sure you run scripts from the project root: `cd 269_project && python experiments/adaptive_results.py`

---

## Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| numpy | ≥1.24 | Array operations |
| scipy | ≥1.10 | Matrix inversion, interpolation |
| scikit-image | ≥0.21 | `active_contour` baseline, contour utilities |
| matplotlib | ≥3.7 | Visualization |
| pillow | ≥10.0 | Image I/O |
