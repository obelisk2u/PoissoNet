import os, shutil
import numpy as np
import yaml
from tqdm import trange
from scipy.sparse.linalg import spsolve
from utils.laplacian import build_laplacian
from skimage.draw import polygon2mask, disk
from skimage.morphology import binary_dilation, disk as morph_disk
from scipy.ndimage import gaussian_filter

def load_config(path="configs/sim_config.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)

def create_grid(config):
    Nx = config["domain"]["Nx"]
    Ny = config["domain"]["Ny"]
    Lx = config["domain"]["Lx"]
    Ly = config["domain"]["Ly"]
    dx = Lx / (Nx - 1)
    dy = Ly / (Ny - 1)

    x = np.linspace(0, Lx, Nx)
    y = np.linspace(0, Ly, Ny)
    X, Y = np.meshgrid(x, y)

    return X, Y, dx, dy, Nx, Ny



def generate_fun_mask(X, Y, mode="mixed", max_obstacles=3, seed=None):
    if seed is not None:
        np.random.seed(seed)

    Ny, Nx = X.shape

    def create_mask():
        mask = np.ones((Ny, Nx), dtype=np.float32)

        def add_random_circle():
            r = np.random.uniform(0.05, 0.15)
            cx = np.random.uniform(r, 1.0 - r)
            cy = np.random.uniform(r, 1.0 - r)
            rr, cc = disk((cy * Ny, cx * Nx), r * Nx)
            mask[rr.clip(0, Ny - 1), cc.clip(0, Nx - 1)] = 0.0

        def add_random_polygon():
            num_vertices = np.random.randint(3, 8)
            angles = np.sort(np.random.rand(num_vertices) * 2 * np.pi)
            radii = np.random.uniform(0.05, 0.15, size=num_vertices)
            cx = np.random.uniform(0.2, 0.8)
            cy = np.random.uniform(0.2, 0.8)
            x_pts = cx + radii * np.cos(angles)
            y_pts = cy + radii * np.sin(angles)
            coords = np.stack([y_pts * Ny, x_pts * Nx], axis=-1)
            poly_mask = polygon2mask((Ny, Nx), coords)
            mask[poly_mask] = 0.0

        def add_blob():
            noise = np.random.rand(Ny, Nx)
            blurred = gaussian_filter(noise, sigma=6)
            blob_mask = (blurred > 0.9)  # boolean mask
            blob_mask = binary_dilation(blob_mask, morph_disk(2))
            mask[blob_mask] = 0.0

        for _ in range(np.random.randint(1, max_obstacles + 1)):
            shape_type = np.random.choice(["circle", "polygon", "blob"]) if mode == "mixed" else mode
            if shape_type == "circle":
                add_random_circle()
            elif shape_type == "polygon":
                add_random_polygon()
            elif shape_type == "blob":
                add_blob()

        return mask

    # Keep generating until at least 30% of the domain is fluid
    for _ in range(20):  # max retries
        mask = create_mask()
        fluid_fraction = mask.mean()
        if 0.3 <= fluid_fraction <= 0.99:
            return mask

    # fallback: fully open domain
    return np.ones((Ny, Nx), dtype=np.float32)





def generate_sample(seed, config, X, Y, dx, dy, L, output_dir):
    np.random.seed(seed)

    Nx = config["domain"]["Nx"]
    Ny = config["domain"]["Ny"]
    dt = config["solver"]["dt"]
    rho = config["physics"]["density"]
    U = config["boundary"]["inflow_velocity"]

    mask = generate_fun_mask(X, Y, mode="mixed")

    u = np.ones((Ny, Nx)) * U
    v = np.zeros((Ny, Nx))
    if config["solver"].get("use_initial_velocity", False):
        u += 0.5 * (np.random.rand(Ny, Nx) - 0.5)
        v += 0.5 * (np.random.rand(Ny, Nx) - 0.5)

    u[mask == 0] = 0.0
    v[mask == 0] = 0.0

    dudx = (np.roll(u, -1, axis=1) - np.roll(u, 1, axis=1)) / (2 * dx)
    dvdy = (np.roll(v, -1, axis=0) - np.roll(v, 1, axis=0)) / (2 * dy)
    rhs = rho / dt * (dudx + dvdy)
    rhs[mask == 0] = 0.0

    rhs_flat = rhs.ravel()
    rhs_flat[mask.ravel() == 0] = 0.0
    pressure_flat = spsolve(L, rhs_flat)
    pressure = pressure_flat.reshape(Ny, Nx)
    pressure[mask == 0] = 0.0

    out_path = os.path.join(output_dir, f"sample_{seed}.npz")
    np.savez_compressed(out_path,
                        rhs=rhs.astype(np.float32),
                        mask=mask.astype(np.float32),
                        pressure=pressure.astype(np.float32))

def main():
    config = load_config()
    X, Y, dx, dy, Nx, Ny = create_grid(config)
    output_dir = config["output"]["out_dir"]
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    L = build_laplacian(Nx, Ny, dx, dy)

    n_samples = config["output"]["n_samples"]
    seed_offset = config["output"].get("seed", 0)

    for seed in trange(n_samples, desc="Generating data"):
        generate_sample(seed + seed_offset, config, X, Y, dx, dy, L, output_dir)

if __name__ == "__main__":
    main()
