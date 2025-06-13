#!/usr/bin/env python
"""
train_model.py  – PoissoNet with divergence-penalty loss
--------------------------------------------------------
  • input  (2×200×200)  : rhs , mask
  • target (1×200×200)  : pressure
  • extra  (2×200×200)  : u_star , v_star  (for divergence loss)

Loss  =  MSE(pressure)  +  λ · ‖∇·u_new‖²
where  u_new = u_star − (dt/ρ) ∇p_pred
"""

import os, yaml, numpy as np, torch, torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm

# ───────────────────────────────────────────────────────────
# 1.  Dataset
# ───────────────────────────────────────────────────────────
class PoissoDataset(Dataset):
    def __init__(self, data_path, cfg_path="configs/sim_config.yaml"):
        z = np.load(data_path)
        SCALE = 1000
        self.rhs      = torch.from_numpy(z["rhs"]).float() / SCALE        # [N,H,W]
        self.mask     = torch.from_numpy(z["mask"]).float()
        self.pressure = torch.from_numpy(z["pressure"]).float() /SCALE
        self.u_star   = torch.from_numpy(z["u_star"]).float() /SCALE
        self.v_star   = torch.from_numpy(z["v_star"]).float() /SCALE

        # grid spacing constants for finite differences
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
        extra = {
            "u_star": self.u_star[i],
            "v_star": self.v_star[i],
        }
        return x, y, extra


# ───────────────────────────────────────────────────────────
# 2.  Mini-U-Net
# ───────────────────────────────────────────────────────────
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True),
        )
    def forward(self, x): return self.net(x)

class UNet(nn.Module):
    def __init__(self, in_ch=2, base=64):
        super().__init__()
        self.enc1 = DoubleConv(in_ch,      base)
        self.enc2 = DoubleConv(base,       base*2)
        self.enc3 = DoubleConv(base*2,     base*4)
        self.pool = nn.MaxPool2d(2)

        self.up2  = nn.ConvTranspose2d(base*4, base*2, 2, 2)
        self.dec2 = DoubleConv(base*4, base*2)
        self.up1  = nn.ConvTranspose2d(base*2, base,   2, 2)
        self.dec1 = DoubleConv(base*2, base)

        self.out  = nn.Conv2d(base, 1, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))

        d2 = self.up2(e3)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))
        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))
        return self.out(d1)


# ───────────────────────────────────────────────────────────
# 3.  Divergence helper
# ───────────────────────────────────────────────────────────
def central_diff(t, dim, h):
    """central finite diff with replicate padding"""
    pad = (0,0,0,0)
    if dim == -1:          # x
        t = F.pad(t, (1,1,0,0), mode='replicate')
        return (t[..., 2:] - t[..., :-2]) / (2*h)
    else:                  # y
        t = F.pad(t, (0,0,1,1), mode='replicate')
        return (t[..., 2:, :] - t[..., :-2, :]) / (2*h)


def divergence(u, v, dx, dy):
    dudx = central_diff(u, -1, dx)      # ∂u/∂x
    dvdy = central_diff(v, -2, dy)      # ∂v/∂y
    return dudx + dvdy


# ───────────────────────────────────────────────────────────
# 4.  Training configuration
# ───────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_num_threads(8)                # stay inside PE allocation

DATA_FILE = "data/poisso_train_full.npz"
CKPT      = "checkpoints/poissonet.pt"
os.makedirs("checkpoints", exist_ok=True)
os.makedirs("logs", exist_ok=True)

BATCH   = 8
EPOCHS  = 30
LR      = 1e-3
LAMBDA  = 0.5          # weight for divergence penalty

# ───────────────────────────────────────────────────────────
# 5.  Training script
# ───────────────────────────────────────────────────────────
def main():
    # dataset & loaders
    full_ds = PoissoDataset(DATA_FILE)
    N = len(full_ds)
    val_len = int(0.1 * N)
    train_ds, val_ds = random_split(full_ds, [N - val_len, val_len])

    train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True,
                              num_workers=4, pin_memory=torch.cuda.is_available())
    val_loader   = DataLoader(val_ds,   batch_size=BATCH, shuffle=False,
                              num_workers=4, pin_memory=torch.cuda.is_available())

    # model, optimiser, scheduler
    model = UNet().to(DEVICE)
    opt   = torch.optim.Adam(model.parameters(), lr=LR)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=4, gamma=0.2)

    mse_loss = nn.MSELoss()
    best_val = float('inf')

    for epoch in range(1, EPOCHS+1):
        # ----- training -----
        model.train()
        run_train = 0
        for x, y, extra in tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}"):
            x, y = x.to(DEVICE), y.to(DEVICE)
            u_star = extra["u_star"].to(DEVICE)
            v_star = extra["v_star"].to(DEVICE)

            opt.zero_grad()
            p_pred = model(x)

            # MSE term
            mse = mse_loss(p_pred, y)

            # divergence penalty
            dx, dy = full_ds.dx, full_ds.dy
            dt, rho = full_ds.dt, full_ds.rho
            dpdx = central_diff(p_pred, -1, dx)
            dpdy = central_diff(p_pred, -2, dy)
            u_new = u_star - (dt / rho) * dpdx
            v_new = v_star - (dt / rho) * dpdy
            div   = divergence(u_new, v_new, dx, dy)
            div_pen = (div**2).mean()

            loss = mse + LAMBDA * div_pen
            loss.backward()
            opt.step()

            run_train += loss.item() * x.size(0)

        train_loss = run_train / len(train_ds)

        # ----- validation -----
        model.eval()
        run_val = 0
        with torch.no_grad():
            for x, y, extra in val_loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                u_star = extra["u_star"].to(DEVICE)
                v_star = extra["v_star"].to(DEVICE)

                p_pred = model(x)

                mse = mse_loss(p_pred, y)

                dx, dy = full_ds.dx, full_ds.dy
                dt, rho = full_ds.dt, full_ds.rho
                dpdx = central_diff(p_pred, -1, dx)
                dpdy = central_diff(p_pred, -2, dy)
                u_new = u_star - (dt / rho) * dpdx
                v_new = v_star - (dt / rho) * dpdy
                div   = divergence(u_new, v_new, dx, dy)
                div_pen = (div**2).mean()

                run_val += (mse + LAMBDA * div_pen).item() * x.size(0)

        val_loss = run_val / len(val_ds)
        sched.step()

        print(f"Epoch {epoch:02d} | lr {sched.get_last_lr()[0]:.2e} | "
              f"train {train_loss:.4e} | val {val_loss:.4e}")

        # checkpoint
        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), CKPT)
            print(f"📝  Saved checkpoint → {CKPT}")

    print(f"Training done. Best val loss: {best_val:.4e}")


if __name__ == "__main__":
    main()
