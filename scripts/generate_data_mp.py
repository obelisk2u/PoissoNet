import os, shutil, yaml, argparse
import numpy as np
from tqdm.contrib.concurrent import process_map
from scipy.sparse.linalg import spsolve
from utils.laplacian import build_laplacian
from skimage.draw import polygon2mask, disk
from skimage.morphology import binary_dilation, disk as morph_disk
from scipy.ndimage import gaussian_filter
import multiprocessing as mp
from functools import partial


# ─────────────────── utility helpers ────────────────────
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


# ───────────────── obstacle generator ────────────────────
def generate_fun_mask(X, Y, mode="mixed", max_obstacles=3, seed=None):
    if seed is not None:
        np.random.seed(seed)

    Ny, Nx = X.shape

    def create_mask():
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
            poly = polygon2mask((Ny, Nx), np.stack([yp * Ny, xp * Nx], -1))
            mask[poly] = 0.0

        def add_blob():
            noise = np.random.rand(Ny, Nx)
            blurred = gaussian_filter(noise, sigma=6)
            blob_mask = (blurred > 0.9)  # boolean mask
            blob_mask = binary_dilation(blob_mask, morph_disk(2))
            mask[blob_mask] = 0.0

        for _ in range(np.random.randint(1, max_obstacles + 1)):
            t = np.random.choice(["circle", "polygon", "blob"]) if mode == "mixed" else mode
            {"circle": add_circle, "polygon": add_polygon, "blob": add_blob}[t]()

        return mask

    for _ in range(20):
        m = create_mask()
        fluid = m.mean()
        if 0.3 <= fluid <= 0.99:
            return m
    return np.ones_like(X, dtype=np.float32)  # fallback


# ─────────────── per-sample generation worker ─────────────
def generate_one(seed, cfg, X, Y, dx, dy, out_dir):
    np.random.seed(seed)

    # (Re)build small Laplacian locally – cheap vs Poisson solve
    L = build_laplacian(cfg["domain"]["Nx"], cfg["domain"]["Ny"], dx, dy)

    Ny, Nx = cfg["domain"]["Ny"], cfg["domain"]["Nx"]
    dt, rho, U = cfg["solver"]["dt"], cfg["physics"]["density"], cfg["boundary"]["inflow_velocity"]

    mask = generate_fun_mask(X, Y, mode="mixed", seed=seed)

    u = np.ones((Ny, Nx), dtype=np.float32) * U
    v = np.zeros((Ny, Nx), dtype=np.float32)
    if cfg["solver"].get("use_initial_velocity", False):
        noise = 0.5 * (np.random.rand(Ny, Nx) - 0.5)
        u += noise * mask
        v += noise * mask

    u[mask == 0] = 0
    v[mask == 0] = 0

    dudx = (np.roll(u, -1, 1) - np.roll(u, 1, 1)) / (2 * dx)
    dvdy = (np.roll(v, -1, 0) - np.roll(v, 1, 0)) / (2 * dy)
    rhs = rho / dt * (dudx + dvdy)
    rhs[mask == 0] = 0

    p = spsolve(L, rhs.ravel())
    p = p.reshape(Ny, Nx)
    p[mask == 0] = 0

    np.savez_compressed(
        os.path.join(out_dir, f"sample_{seed}.npz"),
        rhs=rhs.astype(np.float32),
        mask=mask.astype(np.float32),
        pressure=p.astype(np.float32),
    )
    return seed  # for tqdm counting


# ───────────────────────── main ──────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--workers",
        type=int,
        default=os.cpu_count() or mp.cpu_count(),  # auto-detect
        help="Number of CPU processes (default: all cores)"
    )
    args = parser.parse_args()

    cfg = load_config()
    X, Y, dx, dy, *_ = create_grid(cfg)
    out_dir = cfg["output"]["out_dir"]
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    n_samples = cfg["output"]["n_samples"]
    seed0 = cfg["output"].get("seed", 0)
    seeds = [seed0 + i for i in range(n_samples)]

    # process_map shows a neat progress bar across processes
    partial_fn = partial(
            generate_one,
            cfg=cfg,
            X=X,
            Y=Y,
            dx=dx,
            dy=dy,
            out_dir=out_dir,
        )

    process_map(
        partial_fn,           # ← only seed is left as a free arg
        seeds,                # iterable over seeds
        max_workers=args.workers,
        desc="Generating"
    )


if __name__ == "__main__":
    main()
