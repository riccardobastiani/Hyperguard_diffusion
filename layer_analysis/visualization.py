from pathlib import Path
from typing import Dict, Sequence

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


def save_layer_score_plot(layer_scores: Dict[int, float], output_path: Path) -> None:
    """Save a layer index versus silhouette score curve."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    layers = sorted(layer_scores)
    scores = [layer_scores[layer] for layer in layers]

    plt.figure(figsize=(10, 5))
    plt.plot(layers, scores, marker="o", linewidth=2)
    plt.xlabel("Transformer layer")
    plt.ylabel("Silhouette score")
    plt.title("Layer-wise safe/unsafe separability")
    plt.grid(True, alpha=0.3)
    plt.xticks(layers)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def save_layer_projection(
    features: np.ndarray,
    labels: Sequence[int],
    output_path: Path,
    method: str = "pca",
    seed: int = 42,
    title: str | None = None,
) -> None:
    """Reduce layer features to 2D and save a safe/unsafe scatter plot."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    labels = np.asarray(labels)
    method = method.lower()

    if method == "pca":
        reducer = PCA(n_components=2, random_state=seed)
        reduced = reducer.fit_transform(features)
        plot_title = title or "Layer PCA projection"
        x_label = "PC1"
        y_label = "PC2"
    elif method == "tsne":
        perplexity = min(30, max(1, (features.shape[0] - 1) // 3))
        perplexity = min(perplexity, features.shape[0] - 1)
        reducer = TSNE(n_components=2, init="pca", learning_rate="auto", perplexity=perplexity, random_state=seed)
        reduced = reducer.fit_transform(features)
        plot_title = title or "Layer t-SNE projection"
        x_label = "t-SNE 1"
        y_label = "t-SNE 2"
    else:
        raise ValueError("method must be 'pca' or 'tsne'.")

    plt.figure(figsize=(7, 6))
    safe = labels == 0
    unsafe = labels == 1
    plt.scatter(reduced[safe, 0], reduced[safe, 1], c="#2563eb", label="safe", alpha=0.8, edgecolors="none")
    plt.scatter(reduced[unsafe, 0], reduced[unsafe, 1], c="#dc2626", label="unsafe", alpha=0.8, edgecolors="none")
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(plot_title)
    plt.legend()
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def save_best_layer_projection(
    features: np.ndarray,
    labels: Sequence[int],
    output_path: Path,
    method: str = "pca",
    seed: int = 42,
) -> None:
    """Reduce best-layer features to 2D and save a safe/unsafe scatter plot."""
    save_layer_projection(
        features=features,
        labels=labels,
        output_path=output_path,
        method=method,
        seed=seed,
        title=f"Best layer {method.upper()} projection",
    )
