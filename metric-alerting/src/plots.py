"""Графики по результатам ежедневного запуска: выбросы и алерты плавного сдвига."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

DIRECTIONS = {"up": ("вверх", "green"), "down": ("вниз", "red")}


def plot_metrics(panels: dict, draw, title: str, n_cols: int = 2):
    """Рисует сетку графиков, по одному на метрику.

    panels — {метрика: данные}; draw(ax, data, metric) рисует один график.
    """
    n_rows = int(np.ceil(len(panels) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 3.8 * n_rows), squeeze=False)
    axes = axes.ravel()

    for ax, (metric, data) in zip(axes, panels.items()):
        draw(ax, data, metric)
        ax.grid(alpha=0.3)
        ax.tick_params(axis="x", rotation=45, labelsize=8)
        ax.legend(fontsize=7, loc="upper left")
    for ax in axes[len(panels):]:
        ax.set_visible(False)

    fig.suptitle(title, fontsize=15, fontweight="bold")
    fig.tight_layout()
    return fig


def draw_spikes(ax, history, metric: str):
    """Факт, ожидаемый уровень, коридор нормы и выбросы — по результату backtest.

    Уровень и коридор на каждый день — такие, какими их видел детектор в этот день.
    """
    dates = history["date"]
    ax.plot(dates, history["value"], color="lightcoral", lw=0.9, alpha=0.55, label="исходный")
    ax.plot(dates, history["cleaned"], color="seagreen", lw=1.4, label="очищенный")
    ax.plot(dates, history["expected"], color="steelblue", lw=1.5, label="LOWESS")
    ax.fill_between(dates, history["lower"], history["upper"], color="steelblue", alpha=0.1, label="коридор нормы")

    for direction, (name, color) in DIRECTIONS.items():
        spikes = history[history["spike"] == direction]
        if len(spikes):
            ax.scatter(spikes["date"], spikes["value"], color=color, s=24, zorder=5,
                       label=f"выброс {name} ({len(spikes)})")

    n_spikes = (history["spike"] != "normal").sum()
    ax.set_title(f"{metric}: выбросов {n_spikes}", fontsize=11, fontweight="bold")


def draw_trend_alerts(ax, history, metric: str):
    """Факт, сигналы плавного сдвига и подтверждённые алерты — по результату backtest."""
    ax.plot(history["date"], history["value"], color="lightcoral", lw=1.0, alpha=0.55, label="факт")

    for direction, (name, color) in DIRECTIONS.items():
        signals = history[history["trend"] == direction]
        alerts = signals[signals["trend_alert"]]
        if len(signals):
            ax.scatter(signals["date"], signals["expected"], color=color, s=18, alpha=0.2, zorder=5,
                       label=f"сигнал {name} ({len(signals)})")
        if len(alerts):
            ax.scatter(alerts["date"], alerts["expected"], facecolors="none", edgecolors=color, s=100, lw=2.4,
                       zorder=7, label=f"алерт {name} ({len(alerts)})")

    ax.set_title(metric, fontsize=11, fontweight="bold")
