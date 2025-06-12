#!/usr/bin/env python
"""
compress_data.py
----------------
Hard-coded merger: stacks all *.npz files found in  data/raw/
into one compressed archive  data/poisso_train_5k.npz
"""

import glob, os, sys
import numpy as np
from tqdm import tqdm

# ───────────── hard-coded paths ─────────────
IN_DIR   = "data/raw"                   # folder with individual samples
OUT_FILE = "data/poisso_train_5k.npz"   # combined archive
PATTERN  = "*.npz"                      # file pattern
# ────────────────────────────────────────────


def main():
    files = sorted(glob.glob(os.path.join(IN_DIR, PATTERN)))
    if not files:
        sys.exit(f"No files found in {IN_DIR}")

    print(f"Found {len(files)} samples in {IN_DIR} → stacking …")

    # probe first file for array shape
    probe = np.load(files[0])
    Ny, Nx = probe["rhs"].shape
    N      = len(files)

    rhs_all      = np.empty((N, Ny, Nx), dtype=np.float32)
    mask_all     = np.empty((N, Ny, Nx), dtype=np.float32)
    pressure_all = np.empty((N, Ny, Nx), dtype=np.float32)

    for i, f in enumerate(tqdm(files, desc="Loading")):
        z = np.load(f)
        rhs_all[i]      = z["rhs"]
        mask_all[i]     = z["mask"]
        pressure_all[i] = z["pressure"]

    # ensure destination folder exists
    os.makedirs(os.path.dirname(OUT_FILE), exist_ok=True)

    print("Saving compressed archive …")
    np.savez_compressed(OUT_FILE,
                        rhs=rhs_all,
                        mask=mask_all,
                        pressure=pressure_all)

    size_mb = os.path.getsize(OUT_FILE) / (1024 ** 2)
    print(f"✓ Saved {OUT_FILE}  ({size_mb:.1f} MB)")
    print(f"rhs      : {rhs_all.shape}")
    print(f"mask     : {mask_all.shape}")
    print(f"pressure : {pressure_all.shape}")


if __name__ == "__main__":
    main()