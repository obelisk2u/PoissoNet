"""
generate_data_mp.py  – parallel data generator (saves u_star & v_star)

Each sample NPZ contains:
  rhs      – divergence field            (Ny, Nx)
  mask     – geometry mask (1=fluid)     (Ny, Nx)
  pressure – Poisson solution            (Ny, Nx)
  u_star   – intermediate x-velocity     (Ny, Nx)
  v_star   – intermediate y-velocity     (Ny, Nx)
  python scripts.generate_data_mp --workers 8
"""

import os, yaml, argparse, numpy as np
from tqdm.contrib.concurrent import process_map
from functools import partial
from scipy.sparse.linalg import spsolve
from utils.laplacian import build_laplacian
from skimage.draw import polygon2mask, disk
from skimage.morphology import binary_dilation, disk as morph_disk
from scipy.ndimage import gaussian_filter

# ── helpers: config & grid ────────────────────────────────────────────
def load_config(path="configs/sim_config.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)

def create_grid(cfg):
    Nx, Ny = cfg["domain"]["Nx"], cfg["domain"]["Ny"]
    Lx, Ly = cfg["domain"]["Lx"], cfg["domain"]["Ly"]
    dx, dy = Lx / (Nx - 1), Ly / (Ny - 1)
    x = np.linspace(0, Lx, Nx)
    y = np.linspace(0, Ly, Ny)
    X, Y = np.meshgrid(x, y)
    return X, Y, dx, dy, Nx, Ny

# ── obstacle mask generator ──────────────────────────────────────────
def generate_fun_mask(X, Y, max_obstacles=3, seed=None):
    if seed is not None:
        np.random.seed(seed)
    Ny, Nx = X.shape
    def create():
        mask = np.ones((Ny, Nx), dtype=np.float32)

        def add_circle():
            r = np.random.uniform(0.05, 0.15)
            cx, cy = np.random.uniform(r, 1 - r, 2)
            rr, cc = disk((cy * Ny, cx * Nx), r * Nx)
            mask[rr.clip(0, Ny - 1), cc.clip(0, Nx - 1)] = 0.0

        def add_polygon():
            n = np.random.randint(3, 8)
            ang = np.sort(np.random.rand(n) * 2 * np.pi)
            rad = np.random.uniform(0.05, 0.15, n)
            cx, cy = np.random.uniform(0.2, 0.8, 2)
            xp = cx + rad * np.cos(ang)
            yp = cy + rad * np.sin(ang)
            poly = polygon2mask((Ny, Nx),
                                np.stack([yp * Ny, xp * Nx], -1))
            mask[poly] = 0.0

        def add_blob():
            blur = gaussian_filter(np.random.rand(Ny, Nx), sigma=6)
            blob = (blur > 0.9)
            blob = binary_dilation(blob, morph_disk(2))
            mask[blob] = 0.0

        for _ in range(np.random.randint(1, max_obstacles + 1)):
            np.random.choice([add_circle, add_polygon, add_blob])()
        return mask

    for _ in range(20):
        m = create()
        if 0.3 <= m.mean() <= 0.99:
            return m
    return np.ones_like(X, dtype=np.float32)

# ── per-sample worker function ───────────────────────────────────────
def generate_one(seed, cfg, X, Y, dx, dy, out_dir):
    # Build tiny Laplacian locally (fast, avoids pickling large CSR matrix)
    L = build_laplacian(cfg["domain"]["Nx"], cfg["domain"]["Ny"], dx, dy)

    Nx, Ny = cfg["domain"]["Nx"], cfg["domain"]["Ny"]
    dt, rho, U = cfg["solver"]["dt"], cfg["physics"]["density"], cfg["boundary"]["inflow_velocity"]

    mask = generate_fun_mask(X, Y, seed=seed)

    # u*, v*
    u_star = np.ones((Ny, Nx), dtype=np.float32) * U
    v_star = np.zeros((Ny, Nx), dtype=np.float32)
    if cfg["solver"].get("use_initial_velocity", False):
        noise = 0.5 * (np.random.rand(Ny, Nx) - 0.5)
        u_star += noise * mask
        v_star += noise * mask
    u_star[mask == 0] = 0.0
    v_star[mask == 0] = 0.0

    dudx = (np.roll(u_star, -1, 1) - np.roll(u_star, 1, 1)) / (2 * dx)
    dvdy = (np.roll(v_star, -1, 0) - np.roll(v_star, 1, 0)) / (2 * dy)
    rhs = rho / dt * (dudx + dvdy)
    rhs[mask == 0] = 0.0

    pressure = spsolve(L, rhs.ravel()).reshape(Ny, Nx).astype(np.float32)
    pressure[mask == 0] = 0.0

    np.savez_compressed(
        os.path.join(out_dir, f"sample_{seed}.npz"),
        rhs=rhs.astype(np.float32),
        mask=mask.astype(np.float32),
        pressure=pressure,
        u_star=u_star,
        v_star=v_star,
    )
    return seed  # for progress bar

# ── main (launch parallel workers) ───────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=os.cpu_count(),
                        help="Number of CPU processes to spawn")
    args = parser.parse_args()

    cfg = load_config()
    X, Y, dx, dy, *_ = create_grid(cfg)

    out_dir = cfg["output"]["out_dir"]
    os.makedirs(out_dir, exist_ok=True)

    n_samples = cfg["output"]["n_samples"]
    seed0 = cfg["output"].get("seed", 0)
    seeds = [seed0 + i for i in range(n_samples)]

    partial_fn = partial(generate_one,
                         cfg=cfg, X=X, Y=Y, dx=dx, dy=dy, out_dir=out_dir)

    process_map(partial_fn, seeds, max_workers=args.workers,
                desc="Generating")

if __name__ == "__main__":
    import os
    main()
