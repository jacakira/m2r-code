from programming.generate_datasets import make_point_clouds
import numpy as np
import matplotlib.pyplot as plt

from ripser import ripser
from persim import plot_diagrams, bottleneck, wasserstein


# Core functions for application
def plot_3d_point_clouds(point_clouds, titles):
    """Plot multiple 3D point clouds side-by-side with titles."""
    n = len(point_clouds)
    fig = plt.figure(figsize=(18, 5))

    for i in range(n):
        ax = fig.add_subplot(1, n, i + 1, projection='3d')
        ax.scatter(
            point_clouds[i][:, 0],
            point_clouds[i][:, 1],
            point_clouds[i][:, 2],
            s=5
        )
        ax.set_title(f"{titles[i]} Point Cloud")
        ax.set_axis_off()
    plt.tight_layout()
    plt.show()


def compute_persistence_ripser(point_clouds, maxdim=2):
    """Compute Persistence Diagrams using ripser"""
    diagrams = []
    for pc in point_clouds:
        result = ripser(pc, maxdim=maxdim)
        diagrams.append(result['dgms'])
    return diagrams


def plot_persistence_diagrams_persim(diagrams, titles):
    """Plot Persistentence Diagrams using persim"""
    n = len(diagrams)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4))
    if n == 1:
        axes = [axes]
    for i, (dgm, title) in enumerate(zip(diagrams, titles)):
        plot_diagrams(dgm, ax=axes[i], title=f"Persistence Diagram - {title}", size=40)
    plt.tight_layout()
    plt.show()


def plot_barcodes_manual(diagram, title="Barcode"):
    fig, ax = plt.subplots(figsize=(6, 4))
    for i, (birth, death) in enumerate(diagram):
        if death == np.inf:
            death = max(diagram[diagram != np.inf].flatten()) + 1  # cap infinite bars
        ax.plot([birth, death], [i, i], color='b', lw=4)
    ax.set_xlabel("Scale")
    ax.set_ylabel("Interval Index")
    ax.set_title(title)
    plt.show()


def add_noise(pc, noise_level=0.3):
    """Add Gaussian noise to a point cloud."""
    return pc + noise_level * np.random.randn(*pc.shape)


def analyze_stability_with_noise(base_pc, noise_levels=[0.0, 0.1, 0.2, 0.3, 0.4],
                                 homology_dim=1, n_trials=5, plot_example_diagrams=True):
    """Analyze stability of persistence diagrams with averaging over multiple noise trials (no error bars)."""

    clean_diag = ripser(base_pc, maxdim=homology_dim)['dgms'][homology_dim]

    mean_bottleneck_dists = []
    mean_wasserstein_dists = []
    mean_max_pers = []

    example_diagrams = []

    for nl in noise_levels:
        bn_list = []
        ws_list = []
        max_pers = []

        for _ in range(n_trials):
            pc_noisy = add_noise(base_pc, noise_level=nl)
            diag = ripser(pc_noisy, maxdim=homology_dim)['dgms'][homology_dim]

            bn_list.append(bottleneck(clean_diag, diag))
            ws_list.append(wasserstein(clean_diag, diag))

            finite = diag[np.isfinite(diag[:, 1])]
            if len(finite) > 0:
                persistence = finite[:, 1] - finite[:, 0]
                max_pers.append(np.max(persistence))
            else:
                max_pers.append(0)

        mean_bottleneck_dists.append(np.mean(bn_list))
        mean_wasserstein_dists.append(np.mean(ws_list))
        mean_max_pers.append(np.mean(max_pers))

        if plot_example_diagrams:
            example_diagrams.append(diag)

    # Plot example diagrams
    if plot_example_diagrams:
        fig, axs = plt.subplots(1, len(example_diagrams), figsize=(5 * len(example_diagrams), 4))
        for i, (dgm, nl) in enumerate(zip(example_diagrams, noise_levels)):
            plot_diagrams([dgm], labels=[f"H{homology_dim}"], ax=axs[i], title=f"Noise = {nl}", size=40)
        plt.tight_layout()
        plt.show()

    # Plot Bottleneck and Wasserstein distances
    plt.figure(figsize=(6, 4))
    plt.plot(noise_levels, mean_bottleneck_dists, marker='o', label='Bottleneck')
    plt.plot(noise_levels, mean_wasserstein_dists, marker='s', label='Wasserstein')
    plt.title(f"Stability Theorem (H{homology_dim}) - Averaged over {n_trials} trials")
    plt.xlabel("Noise Level")
    plt.ylabel("Distance")
    plt.legend()
    plt.grid(True)
    plt.show()

    # Plot Max Persistence
    plt.figure(figsize=(6, 4))
    plt.plot(noise_levels, mean_max_pers, marker='^', color='green', label='Max Persistence')
    plt.xlabel("Noise Level")
    plt.ylabel("Max Persistence Length")
    plt.title(f"Max Persistence of Dominant H{homology_dim} Feature")
    plt.grid(True)
    plt.legend()
    plt.show()


# 1.
# Generate Point clouds
point_clouds, labels = make_point_clouds(n_samples_per_shape=1, n_points=20, noise=0.1)
titles = ["Circle", "Sphere", "Torus"]

# Plot point clouds
plot_3d_point_clouds(point_clouds, titles)

# 2.
# Compute and plot persistence diagrams
diagrams = compute_persistence_ripser(point_clouds, maxdim=2)
plot_persistence_diagrams_persim(diagrams, titles)

# Plot barcodes for all shapes
for i, title in enumerate(titles):
    if "Sphere" in title:
        hom_dim = 2
    else:
        hom_dim = 1

    print(f"Barcodes for {title} (H{hom_dim}): illustrating Structure Theorem")
    plot_barcodes_manual(diagrams[i][hom_dim], title=f"{title} H{hom_dim} Barcode")

# Generate clean and noisy point clouds
point_clouds_clean, label1 = make_point_clouds(n_samples_per_shape=1, n_points=20, noise=0.0)
point_clouds_noisy, label2 = make_point_clouds(n_samples_per_shape=1, n_points=20, noise=0.4)

# 3.
# Verify Stability theorem for torii
torus_pc = point_clouds_clean[2]
analyze_stability_with_noise(torus_pc, noise_levels=[0.0, 0.02, 0.04, 0.06, 0.08], homology_dim=2)
