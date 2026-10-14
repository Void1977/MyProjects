"""Сводка по метрикам на последний день и текст сообщения для рабочего чата."""

from __future__ import annotations

import numpy as np
import pandas as pd

NORMAL = "В норме"
SPIKE_NAMES = {"up": "Выброс вверх", "down": "Выброс вниз"}
TREND_NAMES = {"up": "Плавный рост", "down": "Плавное снижение"}


def build_status_table(history: pd.DataFrame) -> pd.DataFrame:
    """Статус каждой метрики на последний день: выброс, подтверждённый сдвиг уровня или норма.

    history — результаты backtest по всем рядам с колонками channel и metric.
    Колонки результата:
    - status_code — alert / norm;
    - status_nm — расшифровка статуса;
    - fact — фактическое значение;
    - delta — отклонение факта от ожидаемого уровня;
    - delta_dyn — относительное изменение ожидаемого уровня за trend_lookback_days дней.
    """
    latest = history.sort_values("date").groupby(["channel", "metric"]).tail(1)

    status = pd.DataFrame({
        "channel": latest["channel"],
        "metric": latest["metric"],
        "date": latest["date"],
        "status_nm": [
            _status_name(spike, trend, confirmed)
            for spike, trend, confirmed in zip(latest["spike"], latest["trend"], latest["trend_confirmed"])
        ],
        "fact": latest["value"],
        "delta": latest["value"] - latest["expected"],
        "delta_dyn": latest["trend_change"],
    })
    status.insert(3, "status_code", np.where(status["status_nm"] == NORMAL, "norm", "alert"))
    return status.sort_values(["status_code", "channel", "metric"]).reset_index(drop=True)


def format_message(status: pd.DataFrame, lookback_days: int = 7) -> str:
    """Текст сводки для рабочего чата: сначала отклонения, потом метрики в норме."""
    alerts = status[status["status_code"] == "alert"]
    normal = status[status["status_code"] == "norm"]

    lines = [
        f"Сводка по метрикам за {status['date'].max():%Y-%m-%d}",
        f"Всего: {len(status)} | ⚠️ отклонения: {len(alerts)} | ✅ норма: {len(normal)}",
    ]
    for title, block in [("⚠️ ОТКЛОНЕНИЯ", alerts), ("✅ В НОРМЕ", normal)]:
        if block.empty:
            continue
        lines += ["", title]
        for channel, rows in block.groupby("channel", sort=False):
            lines.append(f"── {channel} ──")
            lines += [_format_row(row) for row in rows.itertuples()]

    lines += ["", f"(динамика — изменение сглаженного уровня за {lookback_days} дн.)"]
    return "\n".join(lines)


def _status_name(spike: str, trend: str, trend_confirmed: bool) -> str:
    parts = []
    if spike in SPIKE_NAMES:
        parts.append(SPIKE_NAMES[spike])
    if trend_confirmed and trend in TREND_NAMES:
        parts.append(TREND_NAMES[trend])
    return " + ".join(parts) or NORMAL


def _format_row(row) -> str:
    parts = [row.status_nm, f"факт {_number(row.fact)}", f"к ожиданию {_number(row.delta, signed=True)}"]
    if pd.notna(row.delta_dyn):
        parts.append(f"динамика {row.delta_dyn:+.1%}")
    return f"• {row.metric}: " + " | ".join(parts)


def _number(x: float, signed: bool = False) -> str:
    """Счётчики — целым числом с пробелами между разрядами, доли — с четырьмя знаками."""
    sign = "+" if signed else ""
    if abs(x) >= 1:
        return f"{x:{sign},.0f}".replace(",", " ")
    return f"{x:{sign}.4f}"
