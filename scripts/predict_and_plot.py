import yaml, torch, random
import numpy as np
import matplotlib.pyplot as plt
from scripts.train_model import UNet, PoissoDataset      # import classes you already have

# ── 1. set up dataset & model ────────────────────────────────────────────────
DATA_FILE = "data/poisso_train_full.npz"
CKPT      = "checkpoints/poissonet.pt"
CFG_FILE  = "configs/sim_config.yaml"

full_ds = PoissoDataset(DATA_FILE, CFG_FILE)
N       = len(full_ds)
val_len = int(0.1 * N)                       # same split rule used in training
val_ids = list(range(N - val_len, N))        # last 10 % = validation indices

model   = UNet(in_ch=2, base=64)             # same hyper-params as training
model.load_state_dict(torch.load(CKPT, map_location="cpu"))
model.eval()

# ── 2. grab 4 random validation indices ─────────────────────────────────────
rand_ids = random.sample(val_ids, k=4)

# ── 3. run inference ────────────────────────────────────────────────────────
with torch.no_grad():
    preds, trues = [], []
    for idx in rand_ids:
        x, y, _ = full_ds[idx]                    # x: [2,H,W]  y: [1,H,W]

        # ── forward pass ─────────────────────────────────────────────
        p_pred = model(x.unsqueeze(0))            # [1,1,H,W]
        p_pred = p_pred.squeeze().cpu().numpy()   # [H,W]   ← fixed
        preds.append(p_pred)

        trues.append(y.squeeze().cpu().numpy())   # already [H,W]

# ── 4. plotting helper ──────────────────────────────────────────────────────
def add_subplot(ax, img, title):
    im = ax.imshow(img, cmap="RdBu_r")
    ax.set_title(title, fontsize=9)
    ax.axis("off")
    return im

fig, axs = plt.subplots(2, 4, figsize=(12, 6), constrained_layout=True)
for col in range(4):
    add_subplot(axs[0, col], trues[col], f"True #{rand_ids[col]}")
    add_subplot(axs[1, col], preds[col], f"Pred #{rand_ids[col]}")

# single shared colorbar
cbar = fig.colorbar(axs[0, 0].images[0], ax=axs.ravel().tolist(),
                    fraction=0.02, pad=0.02, shrink=0.9)
cbar.set_label("Pressure")

fig.suptitle("Poissonet – Pressure field predictions vs. ground truth",
             fontweight="bold")
fig.savefig("pred_vs_true.png", dpi=300)
print("✓ Saved pred_vs_true.png")
