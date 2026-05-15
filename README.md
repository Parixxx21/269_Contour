# Robust Active Contour Models for Noisy and Low-Contrast Image Segmentation

**CS 269 Project — Effie Wu, Lyla Liu, Jaehoon Choi, Md Aeinul Islam**

---

## Overview

This project implements and evaluates four variants of the classical *snake* (parametric active contour) model, progressively introducing two robustness strategies:

| # | Method | Key idea |
|---|--------|----------|
| 1 | **Classical** | Baseline — `skimage.segmentation.active_contour` |
| 2 | **Adaptive** | Spatially adaptive β(s) modulated by local gradient magnitude |
| 3 | **Multiscale** | Coarse-to-fine Gaussian pyramid initialization |
| 4 | **Combined** | Adaptive weighting **+** coarse-to-fine (proposed framework) |

---

## Project Structure

```
269_project/
├── main.py                        # Quick demo: one test case, 4 methods
├── requirements.txt
│
├── src/                           # Core algorithm implementations
│   ├── snake.py                   # Method 1: Classical (wraps skimage)
│   ├── adaptive_snake.py          # Method 2: Adaptive energy weighting
│   ├── multiscale_snake.py        # Method 3: Coarse-to-fine
│   ├── combined_snake.py          # Method 4: Combined (proposed)
│   ├── energy.py                  # Shared energy + force field utilities
│   └── evaluation.py             # IoU, Hausdorff, mean contour distance
│
├── utils/
│   ├── visualization.py           # All plotting functions
│   └── image_utils.py            # Load images, add noise, reduce contrast
│
├── experiments/
│   ├── synthetic_images.py        # Generate 5 synthetic test cases
│   └── run_experiments.py        # Full experiment suite + summary table
│
├── data/                          # Place your own images here
└── results/                       # Output figures and summary.txt
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

### Method 1 — Classical Snake (`src/snake.py`)

Uses `skimage.segmentation.active_contour` directly as the solver. The image is
pre-smoothed with a Gaussian (σ=2) before computing the external energy. Fixed
scalar weights α and β apply uniformly across all contour points.

**Key parameters:**
- `alpha` (0.015): elasticity — higher → snake resists stretching
- `beta` (0.1): bending — higher → snake stays smoother
- `gamma` (0.001): gradient-descent step size — smaller → more stable
- `sigma` (2.0): Gaussian smoothing for external energy
- `n_iter` (2500): number of optimization steps

### Method 2 — Adaptive Snake (`src/adaptive_snake.py`)

Replaces the fixed β with a **position-dependent** value β_i:

```
β_i = β_max · exp(−k · |∇I(v_i)|_normalized)
```

- Near strong edges → |∇I| large → β_i small → contour deforms freely
- In flat/noisy regions → |∇I| small → β_i ≈ β_max → strong regularization

The stiffness matrix A is rebuilt every `update_every` iterations as the
contour moves to new positions. The linear system is solved directly via
matrix inversion (implicit time stepping):

```
(A(β) + γI) · v^{t+1}  =  γ · v^t  +  F_ext(v^t)
```

**Key parameters:**
- `beta_min` (0.005): minimum bending stiffness (at strong edges)
- `beta_max` (0.3): maximum bending stiffness (in flat regions)
- `k` (5.0): sensitivity of β to gradient magnitude — higher → sharper transition
- `gamma` (100.0): time-step regularizer — higher → smaller steps, more stable
- `update_every` (20): how often to recompute the adaptive matrix

### Method 3 — Multiscale Snake (`src/multiscale_snake.py`)

Builds a **Gaussian pyramid** with `n_levels` images, σ going from `sigma_coarse`
down to 0.5. The snake runs on each level, using the previous level's output
as the warm-start for the next level:

```
Level 0 (σ=8): coarse, convex energy → escape local minima
Level 1 (σ=4): medium detail
Level 2 (σ=0.5): original-scale detail → precise boundary
```

Uses `skimage.active_contour` at each level with `n_iter / n_levels` iterations.

**Key parameters:**
- `sigma_coarse` (8.0): smoothing at the coarsest level
- `n_levels` (3): number of pyramid levels
- Total iterations are split evenly across levels

### Method 4 — Combined Snake (`src/combined_snake.py`)

Inherits from `AdaptiveSnake` and wraps it in the pyramid loop of
`MultiscaleSnake`. At each pyramid level:

1. External forces are computed on the **level-smoothed** image
2. Adaptive β weights are evaluated on the **original** image (preserving fine edge information)
3. The implicit system is solved with the adaptive stiffness matrix

This combination addresses both failure modes:
- Multi-scale prevents trapping in noise-induced local minima early on
- Adaptive weighting provides the right balance of flexibility vs. regularity at fine scale

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

### Step 1 — Quick demo (start here)

```bash
cd /Users/effiewu0628/Desktop/quarter3/269_project
python main.py --case noisy_disk --show
```

This runs all 4 methods on the noisy disk test case and saves figures to `results/`.

Available cases: `clean_disk`, `noisy_disk`, `low_contrast_ellipse`, `noisy_lc_ellipse`, `noisy_star`

### Step 2 — Full experiment suite

```bash
python experiments/run_experiments.py
```

Runs all 4 methods on all 5 test cases. Saves to `results/`:
- `<case>_comparison.png` — side-by-side method outputs
- `<case>_overlay.png` — all methods overlaid
- `<case>_metrics.png` — bar chart of IoU / Hausdorff / MCD
- `<case>_convergence.png` — snake evolution snapshots
- `summary.txt` — full numeric results table

### Step 3 — Use your own image

```python
from utils.image_utils import load_grayscale
from src.combined_snake import CombinedSnake
import numpy as np

image = load_grayscale("data/your_image.png", target_shape=(256, 256))

# Define initial contour (circle around center)
cy, cx = image.shape[0]//2, image.shape[1]//2
theta = np.linspace(0, 2*np.pi, 100, endpoint=False)
init = np.column_stack([cy + 80*np.sin(theta), cx + 80*np.cos(theta)])

model = CombinedSnake(alpha=0.015, beta_min=0.005, beta_max=0.3,
                      k=5.0, gamma=100.0, sigma=2.0, n_iter=500,
                      sigma_coarse=8.0, n_levels=3)
snake, history = model.fit(image, init)
```

---

## What To Do Next (Project Roadmap)

### Phase 1 — Get the code running ✅
- [ ] `pip install -r requirements.txt`
- [ ] Run `python main.py --case clean_disk --show` to verify setup
- [ ] Run `python main.py --case noisy_disk --show` and check the 4 results visually

### Phase 2 — Understand the results
- [ ] Run the full suite: `python experiments/run_experiments.py`
- [ ] Open `results/summary.txt` and compare IoU scores across methods
- [ ] Look at `*_comparison.png` for cases where classical snake fails but combined succeeds
- [ ] Key question: on which test cases does adaptive or multiscale alone do well? When do you need both?

### Phase 3 — Tune hyperparameters
- [ ] Experiment with `k` in AdaptiveSnake (try 2, 5, 10) — how does it affect noisy cases?
- [ ] Try `sigma_coarse` ∈ {4, 8, 16} — how much blurring is needed at coarse level?
- [ ] Try `n_levels` ∈ {2, 3, 4} — more levels vs. more iterations per level?
- [ ] Document your findings in the project report

### Phase 4 — Real-world images
- [ ] Find 2–3 medical or natural images with weak/noisy boundaries
- [ ] Place them in `data/` and run with `load_grayscale`
- [ ] Compare qualitative results to your synthetic experiments

### Phase 5 — Analysis and report
- [ ] Create convergence plots showing energy vs. iteration for each method
- [ ] Show failure cases — when does combined still fail?
- [ ] Compare IoU/Hausdorff in a table across noise levels (σ = 0.05, 0.10, 0.15, 0.20)
- [ ] Discuss the tradeoff: adaptive is slower (matrix recomputation) — is it worth it?

---

## Troubleshooting

**Snake doesn't converge / stays as a circle:**
- Increase `wedge` (edge force strength) or decrease `alpha`/`beta`
- Try a smaller initialization radius so the snake starts closer to the object
- For Adaptive/Combined: reduce `gamma` so steps are larger

**Snake collapses to a point:**
- `alpha` or `beta` too large relative to external forces — reduce them
- Add a balloon force: `wline=-0.5` (negative = outward pressure)

**Import errors:**
- Make sure you run scripts from the project root: `cd 269_project && python main.py`

---

## Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| numpy | ≥1.24 | Array operations |
| scipy | ≥1.10 | Matrix inversion, interpolation |
| scikit-image | ≥0.21 | `active_contour` baseline, contour utilities |
| matplotlib | ≥3.7 | Visualization |
| pillow | ≥10.0 | Image I/O |
