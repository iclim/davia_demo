from davia import Davia
import matplotlib.pyplot as plt
import numpy as np
from pydantic import BaseModel
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans

app = Davia()


class GeneratedData(BaseModel):
    data: list[tuple[float, float]]
    true_centers: list[tuple[float, float]]
    true_labels: list[float]


class KMeansModel(BaseModel):
    k: int
    kmeans_centers: list[tuple[float, float]]
    kmeans_labels: list[float]
    inertia: float


@app.task
def generate_data(n_clusters: int, sample_size: int, noise: float) -> GeneratedData:
    """
    Function to generate synthetic data in clusters.
    Parameters
    ----------
    n_clusters: int - number of data cluster to generate
    sample_size: int - number of total data points
    noise: float - standard deviation of noise

    Returns
    -------
    GeneratedData - synthetic data to apply k-means clustering
    """
    true_centers = np.random.uniform(-10, 10, size=(n_clusters, 2))

    data, true_labels = make_blobs(
        n_samples=sample_size,
        centers=true_centers,
        n_features=2,
        cluster_std=noise,
    )

    generated_clusters = GeneratedData(
        data=data.tolist(),
        true_labels=true_labels.tolist(),
        true_centers=true_centers.tolist(),
    )

    return generated_clusters


@app.task
def run_kmeans(generated_clusters: GeneratedData, k: int, max_iter: int) -> KMeansModel:
    """
    Function to apply k-means clustering on the result of generate_data.
    Parameters
    ----------
    generated_clusters: GeneratedData - synthetic data to apply k-means clustering
    k: int - number of clusters to find
    max_iter: int - maximum number of iterations

    Returns
    -------
    KMeansModel - k-means model results
    """
    data = np.array(generated_clusters.data)

    kmeans = KMeans(n_clusters=k, max_iter=max_iter)
    kmeans_labels = kmeans.fit_predict(data)
    kmeans_centers = kmeans.cluster_centers_
    inertia = kmeans.inertia_

    kmeans_model = KMeansModel(
        k=k,
        kmeans_centers=kmeans_centers.tolist(),
        kmeans_labels=kmeans_labels.tolist(),
        inertia=inertia,
    )

    return kmeans_model


@app.task
def plot_results(generated_clusters: GeneratedData, kmeans_results: KMeansModel) -> None:
    """
    Function to visualize the true clusters and the results of the k-means clustering.
    Parameters
    ----------
    generated_clusters: GeneratedData - synthetic data to apply k-means clustering
    kmeans_results: KMeansModel - k-means model results

    Returns
    -------
    None
    """
    data = np.array(generated_clusters.data)
    true_centers = np.array(generated_clusters.true_centers)
    kmeans_centers = np.array(kmeans_results.kmeans_centers)
    true_labels = np.array(generated_clusters.true_labels)
    kmeans_labels = np.array(kmeans_results.kmeans_labels)

    fig, axes = plt.subplots(1, 2, figsize=(18, 6))

    n_clusters = max(len(true_centers), kmeans_results.k)
    colors = plt.cm.Set3(np.linspace(0, 1, n_clusters))

    label_mapping = {}
    for kmeans_label in np.unique(kmeans_labels):
        mask = kmeans_labels == kmeans_label
        true_labels_for_this_cluster = true_labels[mask]
        most_common_true_label = np.bincount(true_labels_for_this_cluster.astype(int)).argmax()
        label_mapping[kmeans_label] = most_common_true_label

    mapped_kmeans_labels = np.array([label_mapping[label] for label in kmeans_labels])

    ax1 = axes[0]
    ax1.scatter(data[:, 0], data[:, 1], c=[colors[int(label)] for label in true_labels],
                           alpha=0.7, s=50)
    ax1.scatter(true_centers[:, 0], true_centers[:, 1],
                c='red', marker='x', s=200, linewidths=3, label='True Centers')
    ax1.set_title('True Clusters', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Feature 1')
    ax1.set_ylabel('Feature 2')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2 = axes[1]
    ax2.scatter(data[:, 0], data[:, 1], c=[colors[int(label)] for label in mapped_kmeans_labels],
                           alpha=0.7, s=50)
    ax2.scatter(true_centers[:, 0], true_centers[:, 1],
                c='red', marker='x', s=200, linewidths=3, label='True Centers')
    ax2.scatter(kmeans_centers[:, 0], kmeans_centers[:, 1],
                c='blue', marker='o', s=200, linewidths=2,
                edgecolors='black', label='K-means Centers')

    for i, true_center in enumerate(true_centers):
        if i < len(kmeans_centers):
            distances = np.linalg.norm(kmeans_centers - true_center, axis=1)
            nearest_kmeans_idx = np.argmin(distances)
            ax2.plot([true_center[0], kmeans_centers[nearest_kmeans_idx][0]],
                     [true_center[1], kmeans_centers[nearest_kmeans_idx][1]],
                     'k--', alpha=0.5, linewidth=1)

    ax2.set_title('Comparison: True vs K-means Centers', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Feature 1')
    ax2.set_ylabel('Feature 2')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig.suptitle(f'K-means Clustering Analysis\nInertia: {kmeans_results.inertia:.2f}',
                 fontsize=16, fontweight='bold')

    plt.tight_layout()

    return fig


if __name__ == "__main__":
    app.run()

    # generated_data = generate_data(n_clusters=3, sample_size=100, noise=1.5)
    # kmeans_model = run_kmeans(generated_data, k=3, max_iter=100)
    # plot_results(generated_data, kmeans_model)