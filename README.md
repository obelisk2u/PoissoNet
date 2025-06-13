# PoissoNet

**Fast neural pressure solver with physics‑informed training**

PoissoNet is a lightweight research project that generates synthetic CFD
data, trains a deep U‑Net surrogate to solve the pressure Poisson
equation, and evaluates the model with physically consistent metrics.

---

## 1  Project structure

```
PoissoNet/
├── configs/
│   └── sim_config.yaml           # domain size, fluid params, output paths
├── scripts/
│   ├── generate_data_mp.py       # multiprocess data generator
│   ├── compress_data.py          # merge individual .npz → single archive
│   ├── train_model.py            # physics‑informed U‑Net training loop
│   ├── predict_and_plot.py       # inference + qualitative plot
├── utils/
|   ├── laplacian.py
|   └── compress_data.py          # sparse 5‑pt Laplacian builder
└── checkpoints/                  # saved weights
```

---

## 2  Synthetic‑data pipeline

1. **Configure** the domain in `configs/sim_config.yaml`
   (grid, viscosity, inflow velocity, number of samples).
2. **Generate** compressed training data
   ```bash
   python -m scripts.generate_data_mp --workers 8
   python -m scripts.compress_data
   ```
   _Each raw `.npz` contains `rhs`, `mask`, `pressure`, `u_star`, `v_star`
   on a user‑defined `Nx × Ny` mesh._

![Example Sample](plots/sample_example.png)

---

## 3  Model

| Component       | Details                                       |
| --------------- | --------------------------------------------- |
| Backbone        | 5‑level U‑Net (64 → 1024 channels)            |
| Activations     | GELU + GroupNorm                              |
| Loss            | `MSE(p)` + `λ‖∇·u_new‖²` (divergence penalty) |
| Solver prior    | Central‑difference ∇ / divergence operators   |
| Mixed precision | Enabled via `torch.cuda.amp`                  |

The network predicts a pressure field that, when back‑projected onto the
intermediate velocity **u\***, yields a divergence‑free flow.

---

## 4  Training

```bash
# optional: create conda env
conda create -n poissonet python=3.10 pytorch torchvision torchaudio                  -c pytorch -c nvidia
conda activate poissonet

pip install -r requirements.txt      # numpy, scipy, scikit‑image, tqdm,
                                     # matplotlib, pyyaml

# single‑GPU training (L40S / 3090 etc.)
python -m scripts.train_model
```

Default hyper‑parameters:

```yaml
BATCH   = 32
EPOCHS  = 40
LR      = 2e‑3         # AdamW + cosine decay
LAMBDA  = 0.1          # divergence weight
```

A full 30‑epoch run on an NVIDIA L40S finishes in ≈ 10 minutes.

---

## 5  Qualitative evaluation

```bash
python -m scripts.predict_and_plot
```

This script selects four random validation samples and writes
`pred_vs_true.png`, comparing ground‑truth pressure (top row) to the
network’s prediction (bottom row).

![Example Predictions](plots/pred_vs_true2.png)

---

## 6  Reproducibility checklist

- All random seeds are taken from `--seed` in the YAML or CLI.
- Data and model checkpoints include the grid resolution (`Nx`, `Ny`).
- Exact commit hash is logged to `logs/` at runtime.
