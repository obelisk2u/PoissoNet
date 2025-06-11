import os
import numpy as np
import yaml
from tqdm import trange
from scipy.sparse.linalg import spsolve

from utils.laplacian import build_laplacian

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

def generate_sample(seed, config, X, Y, dx, dy, L, output_dir):
    np.random.seed(seed)

    Nx = config["domain"]["Nx"]
    Ny = config["domain"]["Ny"]
    dt = config["solver"]["dt"]
    rho = config["physics"]["density"]
    U = config["boundary"]["inflow_velocity"]

    # Cylinder position
    r = config["cylinder"]["radius"]
    cx = np.random.uniform(*config["cylinder"]["cx_range"])
    cy = np.random.uniform(*config["cylinder"]["cy_range"])
    mask = ((X - cx)**2 + (Y - cy)**2 >= r**2).astype(np.float32)

    # Initial velocity
    u = np.ones((Ny, Nx)) * U
    v = np.zeros((Ny, Nx))
    if config["solver"].get("use_initial_velocity", False):
        u += 0.5 * (np.random.rand(Ny, Nx) - 0.5)
        v += 0.5 * (np.random.rand(Ny, Nx) - 0.5)

    u[mask == 0] = 0.0
    v[mask == 0] = 0.0

    # Compute RHS (divergence of intermediate velocity)
    dudx = (np.roll(u, -1, axis=1) - np.roll(u, 1, axis=1)) / (2 * dx)
    dvdy = (np.roll(v, -1, axis=0) - np.roll(v, 1, axis=0)) / (2 * dy)
    rhs = rho / dt * (dudx + dvdy)
    rhs[mask == 0] = 0.0

    # Solve Poisson equation
    rhs_flat = rhs.ravel()
    rhs_flat[mask.ravel() == 0] = 0.0
    pressure_flat = spsolve(L, rhs_flat)
    pressure = pressure_flat.reshape(Ny, Nx)
    pressure[mask == 0] = 0.0

    # Save sample
    out_path = os.path.join(output_dir, f"sample_{seed}.npz")
    np.savez_compressed(out_path,
                        rhs=rhs.astype(np.float32),
                        mask=mask.astype(np.float32),
                        pressure=pressure.astype(np.float32))
    print(f"Saved: {out_path}")

def main():
    config = load_config()
    X, Y, dx, dy, Nx, Ny = create_grid(config)
    output_dir = config["output"]["out_dir"]
    os.makedirs(output_dir, exist_ok=True)

    # Precompute Laplacian
    L = build_laplacian(Nx, Ny, dx, dy)

    n_samples = config["output"]["n_samples"]
    seed_offset = config["output"].get("seed", 0)

    for seed in trange(n_samples, desc="Generating data"):
        generate_sample(seed + seed_offset, config, X, Y, dx, dy, L, output_dir)

if __name__ == "__main__":
    main()
