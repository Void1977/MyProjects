"""Интерактивная 3D-визуализация кластеров."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from .clustering import NOISE

NOISE_LABEL = "шум"


def plot_clusters_3d(points: np.ndarray, labels: np.ndarray, hover_texts, title: str | None = None) -> go.Figure:
    """Рисует сообщения в 3D: цвет — кластер, при наведении — текст. Шум рисуется бледно-серым."""
    labels = np.asarray(labels)
    plot_data = pd.DataFrame({
        "x": points[:, 0],
        "y": points[:, 1],
        "z": points[:, 2],
        "cluster": [NOISE_LABEL if label == NOISE else f"кластер {label}" for label in labels],
        "text": list(hover_texts),
    })
    # Шум — в конце легенды, после всех кластеров
    plot_data = plot_data.sort_values("cluster", key=lambda s: s == NOISE_LABEL, kind="stable")

    fig = px.scatter_3d(
        plot_data,
        x="x", y="y", z="z",
        color="cluster",
        hover_name="text",
        opacity=0.6,
        title=title,
        width=1400, height=900,
    )
    fig.update_traces(marker_size=2)
    fig.update_traces(selector={"name": NOISE_LABEL}, marker={"size": 1.5, "color": "lightgrey", "opacity": 0.15})
    fig.update_layout(
        scene={"xaxis_title": "UMAP 1", "yaxis_title": "UMAP 2", "zaxis_title": "UMAP 3"},
        legend={"font_size": 7, "itemsizing": "constant"},
    )
    return fig
