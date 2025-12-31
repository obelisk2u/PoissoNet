#!/usr/bin/env python
"""
train_model_vit.py – ViT baseline for PoissoNet with divergence-penalty loss
---------------------------------------------------------------------------
  • input  (2×200×200): rhs , mask
  • target (1×200×200): pressure
  • extra  (2×200×200): u_star , v_star  (for divergence loss)

Loss = MSE(pressure_centered) + λ · ||∇·u_new||²
where u_new = u_star − (dt/ρ) ∇p_pred
"""

import os, yaml, numpy as np, torch, torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm


# ───────────────────────────────────────────────────────────
# 1. Dataset
# ───────────────────────────────────────────────────────────
class PoissoDataset(Dataset):
    def __init__(self, data_path, cfg_path="configs/sim_config.yaml"):
        z = np.load(data_path)
        SCALE = 1000
        self.rhs      = torch.from_numpy(z["rhs"]).float() / SCALE        # [N,H,W]
        self.mask     = torch.from_numpy(z["mask"]).float()               # [N,H,W], 1=fluid, 0=solid
        self.pressure = torch.from_numpy(z["pressure"]).float() / SCALE   # [N,H,W]
        self.u_star   = torch.from_numpy(z["u_star"]).float() / SCALE
        self.v_star   = torch.from_numpy(z["v_star"]).float() / SCALE

        cfg = yaml.safe_load(open(cfg_path))
        Nx, Ny = cfg["domain"]["Nx"], cfg["domain"]["Ny"]
        Lx, Ly = cfg["domain"]["Lx"], cfg["domain"]["Ly"]
        self.dx = Lx / (Nx - 1)
        self.dy = Ly / (Ny - 1)
        self.dt = cfg["solver"]["dt"]
        self.rho = cfg["physics"]["density"]

    def __len__(self): return len(self.rhs)

    def __getitem__(self, i):
        x = torch.stack([self.rhs[i], self.mask[i]])     # [2,H,W]
        y = self.pressure[i][None]                      # [1,H,W]
        extra = {"u_star": self.u_star[i], "v_star": self.v_star[i]}
        return x, y, extra


# ───────────────────────────────────────────────────────────
# 2. Finite-difference helpers
# ───────────────────────────────────────────────────────────
def central_diff(t, dim, h):
    """central finite diff with replicate padding"""
    if dim == -1:  # x
        t = F.pad(t, (1, 1, 0, 0), mode="replicate")
        return (t[..., 2:] - t[..., :-2]) / (2 * h)
    else:          # y
        t = F.pad(t, (0, 0, 1, 1), mode="replicate")
        return (t[..., 2:, :] - t[..., :-2, :]) / (2 * h)

def divergence(u, v, dx, dy):
    dudx = central_diff(u, -1, dx)
    dvdy = central_diff(v, -2, dy)
    return dudx + dvdy

def remove_mean_over_fluid(p, mask, eps=1e-8):
    """
    p:    [B,1,H,W]
    mask: [B,1,H,W] (1=fluid, 0=solid)
    Removes the spatial mean pressure over fluid cells (pressure gauge fix).
    """
    w = mask
    denom = w.sum(dim=(2,3), keepdim=True).clamp_min(eps)
    mean = (p * w).sum(dim=(2,3), keepdim=True) / denom
    return p - mean


# ───────────────────────────────────────────────────────────
# 3. ViT baseline (encoder-only, patchify -> transformer -> unpatchify)
# ───────────────────────────────────────────────────────────
class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ln2 = nn.LayerNorm(d_model)
        hidden = int(d_model * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        # x: [B, N, D]
        h = self.ln1(x)
        attn_out, _ = self.attn(h, h, h, need_weights=False)
        x = x + attn_out
        x = x + self.mlp(self.ln2(x))
        return x


class ViTPoisson(nn.Module):
    def __init__(
        self,
        in_ch=2,
        out_ch=1,
        grid_size=200,
        patch_size=20,     # must divide grid_size (200 -> ok)
        d_model=256,
        depth=6,
        n_heads=8,
        mlp_ratio=4.0,
        dropout=0.0,
    ):
        super().__init__()
        assert grid_size % patch_size == 0, "patch_size must divide grid_size"
        self.grid_size = grid_size
        self.patch_size = patch_size
        self.out_ch = out_ch

        self.nh = grid_size // patch_size
        self.nw = grid_size // patch_size
        self.n_tokens = self.nh * self.nw

        # Patch embedding
        self.patch_embed = nn.Conv2d(in_ch, d_model, kernel_size=patch_size, stride=patch_size)

        # Positional embedding (learned)
        self.pos = nn.Parameter(torch.zeros(1, self.n_tokens, d_model))
        nn.init.trunc_normal_(self.pos, std=0.02)

        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, mlp_ratio=mlp_ratio, dropout=dropout)
            for _ in range(depth)
        ])
        self.ln_f = nn.LayerNorm(d_model)

        # Decode tokens back to patches
        self.to_patch = nn.Linear(d_model, out_ch * patch_size * patch_size)

        # Optional light refinement (keeps it “baseline” but helps artifacts)
        self.refine = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
        )

    def forward(self, x):
        # x: [B,2,H,W] with H=W=200
        B, C, H, W = x.shape
        assert H == self.grid_size and W == self.grid_size, "unexpected grid size"

        t = self.patch_embed(x)             # [B, D, nh, nw]
        t = t.flatten(2).transpose(1, 2)    # [B, N, D]
        t = self.drop(t + self.pos)

        for blk in self.blocks:
            t = blk(t)
        t = self.ln_f(t)                    # [B, N, D]

        p = self.to_patch(t)                # [B, N, out_ch*P*P]
        P = self.patch_size
        p = p.view(B, self.nh, self.nw, self.out_ch, P, P)            # [B, nh, nw, C, P, P]
        p = p.permute(0, 3, 1, 4, 2, 5).contiguous()                  # [B, C, nh, P, nw, P]
        p = p.view(B, self.out_ch, self.grid_size, self.grid_size)    # [B, C, H, W]
        p = self.refine(p)
        return p


# ───────────────────────────────────────────────────────────
# 4. Training configuration (Mac-friendly defaults)
# ───────────────────────────────────────────────────────────
def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

DEVICE = get_device()
torch.set_num_threads(8)

DATA_FILE = "data/poisso_train_full.npz"
CKPT      = "checkpoints/poissonet_vit.pt"
os.makedirs("checkpoints", exist_ok=True)
os.makedirs("logs", exist_ok=True)

# For M4 / MPS, start conservative; bump after it’s stable.
BATCH   = 4
EPOCHS  = 30
LR      = 2e-4
LAMBDA  = 0.8

GRID    = 200
PATCH   = 20   # 200/20=10 -> 100 tokens


# ───────────────────────────────────────────────────────────
# 5. Training script
# ───────────────────────────────────────────────────────────
def main():
    full_ds = PoissoDataset(DATA_FILE)
    N = len(full_ds)
    val_len = int(0.1 * N)
    train_ds, val_ds = random_split(full_ds, [N - val_len, val_len])

    # On MPS, num_workers>0 can sometimes be slower/finicky depending on setup.
    num_workers = 0 if DEVICE.type == "mps" else 4
    pin = torch.cuda.is_available()

    train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True, num_workers=num_workers, pin_memory=pin)
    val_loader   = DataLoader(val_ds, batch_size=BATCH, shuffle=False, num_workers=num_workers, pin_memory=pin)

    model = ViTPoisson(
        in_ch=2,
        out_ch=1,
        grid_size=GRID,
        patch_size=PATCH,
        d_model=256,
        depth=6,
        n_heads=8,
        mlp_ratio=4.0,
        dropout=0.0,
    ).to(DEVICE)

    opt   = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)

    mse_loss = nn.MSELoss()
    best_val = float("inf")

    # AMP: use on CUDA; MPS autocast can work but isn’t always a win. Start off.
    use_amp = (DEVICE.type == "cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    dx, dy = full_ds.dx, full_ds.dy
    dt, rho = full_ds.dt, full_ds.rho

    for epoch in range(1, EPOCHS + 1):
        model.train()
        run_train = 0.0

        for x, y, extra in tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}"):
            x = x.to(DEVICE)
            y = y.to(DEVICE)
            u_star = extra["u_star"].to(DEVICE)
            v_star = extra["v_star"].to(DEVICE)

            opt.zero_grad(set_to_none=True)

            if use_amp:
                with torch.cuda.amp.autocast():
                    p_pred = model(x)

                    # pressure gauge fix for MSE (mask channel is x[:,1])
                    mask = x[:, 1:2]
                    p_pred_c = remove_mean_over_fluid(p_pred, mask)
                    y_c      = remove_mean_over_fluid(y, mask)
                    mse = mse_loss(p_pred_c, y_c)

                    dpdx = central_diff(p_pred, -1, dx)
                    dpdy = central_diff(p_pred, -2, dy)
                    u_new = u_star - (dt / rho) * dpdx
                    v_new = v_star - (dt / rho) * dpdy
                    div = divergence(u_new, v_new, dx, dy)
                    div_pen = (div ** 2).mean()

                    loss = mse + LAMBDA * div_pen

                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
            else:
                p_pred = model(x)

                mask = x[:, 1:2]
                p_pred_c = remove_mean_over_fluid(p_pred, mask)
                y_c      = remove_mean_over_fluid(y, mask)
                mse = mse_loss(p_pred_c, y_c)

                dpdx = central_diff(p_pred, -1, dx)
                dpdy = central_diff(p_pred, -2, dy)
                u_new = u_star - (dt / rho) * dpdx
                v_new = v_star - (dt / rho) * dpdy
                div = divergence(u_new, v_new, dx, dy)
                div_pen = (div ** 2).mean()

                loss = mse + LAMBDA * div_pen

                loss.backward()
                opt.step()

            run_train += loss.item() * x.size(0)

        train_loss = run_train / len(train_ds)

        # ----- validation -----
        model.eval()
        run_val = 0.0
        with torch.no_grad():
            for x, y, extra in val_loader:
                x = x.to(DEVICE)
                y = y.to(DEVICE)
                u_star = extra["u_star"].to(DEVICE)
                v_star = extra["v_star"].to(DEVICE)

                p_pred = model(x)

                mask = x[:, 1:2]
                p_pred_c = remove_mean_over_fluid(p_pred, mask)
                y_c      = remove_mean_over_fluid(y, mask)
                mse = mse_loss(p_pred_c, y_c)

                dpdx = central_diff(p_pred, -1, dx)
                dpdy = central_diff(p_pred, -2, dy)
                u_new = u_star - (dt / rho) * dpdx
                v_new = v_star - (dt / rho) * dpdy
                div = divergence(u_new, v_new, dx, dy)
                div_pen = (div ** 2).mean()

                loss = mse + LAMBDA * div_pen
                run_val += loss.item() * x.size(0)

        val_loss = run_val / len(val_ds)
        sched.step()

        print(f"Epoch {epoch:02d} | lr {sched.get_last_lr()[0]:.2e} | train {train_loss:.4e} | val {val_loss:.4e}")

        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), CKPT)
            print(f"Saved checkpoint -> {CKPT}")

    print(f"Training done. Best val loss: {best_val:.4e}")


if __name__ == "__main__":
    main()