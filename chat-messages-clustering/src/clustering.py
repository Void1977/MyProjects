"""Понижение размерности (UMAP) и кластеризация (HDBSCAN)."""

from __future__ import annotations

import hdbscan
import numpy as np
import pandas as pd
import umap
from sklearn.metrics.pairwise import cosine_similarity

NOISE = -1  # метка HDBSCAN для сообщений, не попавших ни в один кластер


def reduce_dimensions(embeddings: np.ndarray, n_components: int, n_neighbors: int = 15) -> np.ndarray:
    """Понижает размерность UMAP с косинусной метрикой.

    min_dist=0 позволяет похожим сообщениям собираться в плотные группы, что упрощает кластеризацию.
    """
    reducer = umap.UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=0.0,
        metric="cosine",
        random_state=42,
    )
    return reducer.fit_transform(embeddings)


def cluster_points(
    points: np.ndarray,
    min_cluster_size: int = 10,
    min_samples: int = 3,
    epsilon: float = 0.12,
) -> tuple[np.ndarray, np.ndarray]:
    """Кластеризует точки HDBSCAN.

    Возвращает метки кластеров (NOISE — шум) и уверенность принадлежности точки к своему кластеру.
    epsilon (cluster_selection_epsilon) объединяет кластеры, которые ближе этого расстояния,
    чтобы одна тема не дробилась на много мелких кластеров.
    """
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        cluster_selection_epsilon=epsilon,
        cluster_selection_method="eom",
        metric="euclidean",
        core_dist_n_jobs=-1,
    )
    labels = clusterer.fit_predict(points)
    return labels, clusterer.probabilities_


def describe_clustering(labels: np.ndarray) -> dict:
    """Считает число кластеров, долю шума и размеры кластеров."""
    labels = np.asarray(labels)
    sizes = pd.Series(labels[labels != NOISE]).value_counts()
    return {
        "n_clusters": len(sizes),
        "noise_share": (labels == NOISE).mean(),
        "min_size": sizes.min(),
        "median_size": sizes.median(),
        "max_size": sizes.max(),
    }


def grid_search(embeddings: np.ndarray, configs: list[dict], n_components: int = 3) -> pd.DataFrame:
    """Перебирает параметры UMAP + HDBSCAN. Одна строка результата — один вариант параметров.

    Каждый config — словарь с ключами n_neighbors, min_cluster_size, min_samples, epsilon.
    """
    rows = []
    for config in configs:
        points = reduce_dimensions(embeddings, n_components, n_neighbors=config["n_neighbors"])
        labels, _ = cluster_points(
            points,
            min_cluster_size=config["min_cluster_size"],
            min_samples=config["min_samples"],
            epsilon=config["epsilon"],
        )
        rows.append({**config, **describe_clustering(labels)})
    return pd.DataFrame(rows)


def top_examples(texts, embeddings: np.ndarray, labels: np.ndarray, n: int = 10) -> pd.DataFrame:
    """Для каждого кластера возвращает n сообщений, ближайших к его центроиду.

    Кластеры идут по убыванию размера. По этим сообщениям удобно понять, о чём кластер.
    """
    texts = np.asarray(texts)
    labels = np.asarray(labels)
    cluster_ids = pd.Series(labels[labels != NOISE]).value_counts().index

    rows = []
    for cluster_id in cluster_ids:
        in_cluster = labels == cluster_id
        centroid = embeddings[in_cluster].mean(axis=0, keepdims=True)
        similarity = cosine_similarity(centroid, embeddings[in_cluster])[0]
        for rank, idx in enumerate(similarity.argsort()[::-1][:n], start=1):
            rows.append({
                "cluster": cluster_id,
                "cluster_size": in_cluster.sum(),
                "rank": rank,
                "text": texts[in_cluster][idx],
            })
    return pd.DataFrame(rows)
