"""
train_model.py
--------------
Train PoissoNet on the combined dataset  data/poisso_train_5k.npz
"""

import os, numpy as np, torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm

# ───────────────── Dataset ─────────────────
class PoissoDataset(Dataset):
    def __init__(self, npz_path):
        data = np.load(npz_path)
        self.rhs      = data["rhs"]      # (N, 200, 200)
        self.mask     = data["mask"]
        self.pressure = data["pressure"]

    def __len__(self): return len(self.rhs)

    def __getitem__(self, idx):
        x = np.stack([self.rhs[idx], self.mask[idx]])      # [2, H, W]
        y = self.pressure[idx][None]                       # [1, H, W]
        return torch.from_numpy(x).float(), torch.from_numpy(y).float()


# ───────────────── Small U-Net ─────────────────
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True),
        )
    def forward(self, x): return self.block(x)

class UNet(nn.Module):
    def __init__(self, in_ch=2, base=32):
        super().__init__()
        self.enc1 = DoubleConv(in_ch,      base)
        self.enc2 = DoubleConv(base,       base*2)
        self.enc3 = DoubleConv(base*2,     base*4)
        self.pool = nn.MaxPool2d(2)

        self.up2  = nn.ConvTranspose2d(base*4, base*2, 2, stride=2)
        self.dec2 = DoubleConv(base*4, base*2)
        self.up1  = nn.ConvTranspose2d(base*2, base,   2, stride=2)
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


# ───────────────── Training setup ─────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH  = 16
EPOCHS = 10
LR     = 1e-3
DATA   = "data/poisso_train_5k.npz"
CKPT   = "checkpoints/poissonet.pt"

def main():
    os.makedirs("checkpoints", exist_ok=True)
    ds_full = PoissoDataset(DATA)
    n_total = len(ds_full)
    n_val   = int(0.1 * n_total)
    n_train = n_total - n_val
    train_ds, val_ds = random_split(ds_full, [n_train, n_val])

    train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH)

    model = UNet().to(DEVICE)
    opt   = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.MSELoss()

    best_val = float("inf")
    for epoch in range(1, EPOCHS + 1):
        # --- training ---
        model.train()
        running = 0
        for x, y in tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}"):
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            y_hat = model(x)
            loss  = loss_fn(y_hat, y)
            loss.backward()
            opt.step()
            running += loss.item() * x.size(0)
        train_loss = running / n_train

        # --- validation ---
        model.eval()
        running = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                y_hat = model(x)
                running += loss_fn(y_hat, y).item() * x.size(0)
        val_loss = running / n_val

        print(f"Epoch {epoch:02d} | train {train_loss:.4e} | val {val_loss:.4e}")

        # save best
        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), CKPT)
            print(f"📝  Saved checkpoint → {CKPT}")

    print(f"Training done. Best val loss: {best_val:.4e}")

if __name__ == "__main__":
    main()
