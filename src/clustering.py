"""Clustering algorithms and medoid extraction for semantic issue grouping."""

from typing import Dict, List, Optional, Tuple
import numpy as np
from sklearn.cluster import AgglomerativeClustering, HDBSCAN
from sklearn.metrics import silhouette_score

from src.config import NOISE_CLUSTER_LABEL


class SemanticClusterer:
    """
    Groups dense review vectors into semantically coherent issue clusters.
    Uses HDBSCAN by default to automatically detect cluster count and isolate noise,
    with an Agglomerative fallback for fine-grained distance control.
    """

    def __init__(
        self,
        method: str = "hdbscan",
        min_cluster_size: int = 3,
        distance_threshold: float = 0.40,
    ):
        self.method = method.lower()
        self.min_cluster_size = max(2, min_cluster_size)
        self.distance_threshold = distance_threshold

    def fit_predict(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Clusters embeddings and returns an array of integer cluster IDs.
        Noise / unclustered points receive label NOISE_CLUSTER_LABEL (-1).
        """
        n_samples = len(embeddings)
        if n_samples == 0:
            return np.array([], dtype=int)
        if n_samples < self.min_cluster_size:
            # Too few points to form clusters; label all as 0 (single group)
            return np.zeros(n_samples, dtype=int)

        if self.method == "hdbscan":
            try:
                clusterer = HDBSCAN(
                    min_cluster_size=self.min_cluster_size,
                    min_samples=max(1, self.min_cluster_size - 1),
                    metric="euclidean",
                    cluster_selection_method="eom",
                )
                labels = clusterer.fit_predict(embeddings)
                # If HDBSCAN found only noise (-1) for all samples, fall back to Agglomerative
                unique_labels = set(labels)
                if unique_labels == {NOISE_CLUSTER_LABEL} or len(unique_labels) <= 1:
                    return self._fit_agglomerative(embeddings)
                return labels
            except Exception:
                return self._fit_agglomerative(embeddings)
        else:
            return self._fit_agglomerative(embeddings)

    def _fit_agglomerative(self, embeddings: np.ndarray) -> np.ndarray:
        """Fallback hierarchical clustering with distance threshold."""
        clusterer = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=self.distance_threshold,
            metric="cosine",
            linkage="average",
        )
        return clusterer.fit_predict(embeddings)

    @staticmethod
    def find_cluster_medoids(
        embeddings: np.ndarray, cluster_labels: np.ndarray
    ) -> Dict[int, int]:
        """
        Identifies the central review index (the medoid) for each cluster.
        The medoid is the review whose embedding has the highest cosine similarity
        to all other members of the cluster.

        Returns:
            Dictionary mapping cluster_id -> row_index_of_medoid
        """
        medoids: Dict[int, int] = {}
        unique_clusters = set(cluster_labels)

        for cluster_id in unique_clusters:
            member_indices = np.where(cluster_labels == cluster_id)[0]
            if len(member_indices) == 1:
                medoids[cluster_id] = int(member_indices[0])
                continue

            member_vecs = embeddings[member_indices]
            # Center of the cluster
            centroid = np.mean(member_vecs, axis=0, keepdims=True)
            centroid_norm = centroid / (np.linalg.norm(centroid) + 1e-12)

            # Cosine similarity to centroid
            similarities = np.dot(member_vecs, centroid_norm.T).flatten()
            best_local_idx = int(np.argmax(similarities))
            medoids[cluster_id] = int(member_indices[best_local_idx])

        return medoids

    @staticmethod
    def evaluate_clustering(
        embeddings: np.ndarray, cluster_labels: np.ndarray
    ) -> Dict[str, float]:
        """
        Computes clustering quality metrics (excluding noise points).
        """
        total_samples = len(cluster_labels)
        non_noise_mask = cluster_labels != NOISE_CLUSTER_LABEL
        non_noise_count = int(np.sum(non_noise_mask))
        unique_non_noise_clusters = len(set(cluster_labels[non_noise_mask]))

        metrics = {
            "total_samples": float(total_samples),
            "clustered_samples": float(non_noise_count),
            "noise_samples": float(total_samples - non_noise_count),
            "cluster_count": float(unique_non_noise_clusters),
            "silhouette_score": 0.0,
        }

        # Silhouette score requires at least 2 clusters and more samples than clusters
        if unique_non_noise_clusters >= 2 and non_noise_count > unique_non_noise_clusters:
            try:
                score = silhouette_score(
                    embeddings[non_noise_mask],
                    cluster_labels[non_noise_mask],
                    metric="cosine",
                )
                metrics["silhouette_score"] = round(float(score), 3)
            except Exception:
                metrics["silhouette_score"] = 0.0

        return metrics
