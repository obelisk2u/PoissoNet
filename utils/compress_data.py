"""
compress_data.py
----------------
Merge all per-sample .npz files in  data/raw/  into a single compressed archive
(data/poisso_train_full.npz) that now includes:
  • rhs       – (N, 200, 200)  float32
  • mask      – (N, 200, 200)
  • pressure  – (N, 200, 200)
  • u_star    – (N, 200, 200)
  • v_star    – (N, 200, 200)
"""

import glob
import os
import numpy as np
from tqdm import tqdm

# ───────────── hard-coded paths ─────────────
IN_DIR   = "data/raw"                         # where the individual .npz files live
OUT_FILE = "data/poisso_train_full.npz"       # combined archive
PATTERN  = "*.npz"                            # file name pattern
# ────────────────────────────────────────────


def main():
    files = sorted(glob.glob(os.path.join(IN_DIR, PATTERN)))
    if not files:
        raise SystemExit(f"[compress_data] No .npz files found in {IN_DIR}")

    print(f"[compress_data] Found {len(files)} samples → stacking")

    # Peek at first file to get shapes
    probe = np.load(files[0])
    Ny, Nx = probe["rhs"].shape
    N      = len(files)

    # Pre-allocate big arrays
    rhs_all      = np.empty((N, Ny, Nx), dtype=np.float32)
    mask_all     = np.empty((N, Ny, Nx), dtype=np.float32)
    pressure_all = np.empty((N, Ny, Nx), dtype=np.float32)
    u_star_all   = np.empty((N, Ny, Nx), dtype=np.float32)
    v_star_all   = np.empty((N, Ny, Nx), dtype=np.float32)

    # Load each file
    for i, f in enumerate(tqdm(files, desc="Loading")):
        z = np.load(f)
        rhs_all[i]      = z["rhs"]
        mask_all[i]     = z["mask"]
        pressure_all[i] = z["pressure"]
        u_star_all[i]   = z["u_star"]
        v_star_all[i]   = z["v_star"]

    # Ensure output folder exists
    os.makedirs(os.path.dirname(OUT_FILE), exist_ok=True)

    print("[compress_data] Saving compressed archive …")
    np.savez_compressed(
        OUT_FILE,
        rhs=rhs_all,
        mask=mask_all,
        pressure=pressure_all,
        u_star=u_star_all,
        v_star=v_star_all,
    )

    size_mb = os.path.getsize(OUT_FILE) / (1024 ** 2)
    print(f"[compress_data] ✓ Saved {OUT_FILE}  ({size_mb:.1f} MB)")
    print(f"[compress_data] Shapes → rhs {rhs_all.shape} | mask {mask_all.shape} | "
          f"pressure {pressure_all.shape} | u_star {u_star_all.shape} | v_star {v_star_all.shape}")


if __name__ == "__main__":
    main()
