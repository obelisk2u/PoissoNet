import numpy as np
import matplotlib.pyplot as plt
import glob
import random
import os

def load_sample(path):
    data = np.load(path)
    return data["rhs"], data["mask"], data["pressure"]

def main():
    sample_paths = glob.glob("data/raw/*.npz")
    if not sample_paths:
        print("No .npz files found in data/raw/")
        return

    selected = random.sample(sample_paths, min(4, len(sample_paths)))

    fig, axes = plt.subplots(len(selected), 3, figsize=(12, 3 * len(selected)))
    if len(selected) == 1:
        axes = np.expand_dims(axes, axis=0)  # keep 2D

    for i, path in enumerate(selected):
        rhs, mask, pressure = load_sample(path)

        axes[i, 0].imshow(rhs, cmap="RdBu", origin="lower")
        axes[i, 0].set_title(f"RHS\n{os.path.basename(path)}")
        axes[i, 1].imshow(mask, cmap="gray", origin="lower")
        axes[i, 1].set_title("Mask")
        im = axes[i, 2].imshow(pressure, cmap="viridis", origin="lower")
        axes[i, 2].set_title("Pressure")
        fig.colorbar(im, ax=axes[i, 2], shrink=0.8)

        for ax in axes[i]:
            ax.axis("off")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
