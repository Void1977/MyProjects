"""Подготовка входных данных: длинная таблица date / metric / value / channel."""

from __future__ import annotations

import warnings

import pandas as pd

REQUIRED_COLUMNS = {"date", "metric", "value"}
KEY = ["channel", "date", "metric"]


def prepare_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Проверяет колонки, приводит типы и убирает дубли.

    Если колонки channel нет, все метрики попадают в один канал "all".
    При дублях по (channel, date, metric) остаётся последняя строка.
    """
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Во входной таблице нет колонок: {sorted(missing)}")

    data = df.copy()
    data["date"] = pd.to_datetime(data["date"]).dt.normalize()
    data["value"] = pd.to_numeric(data["value"], errors="coerce")
    if "channel" not in data.columns:
        data["channel"] = "all"
    data["channel"] = data["channel"].fillna("all").astype(str)
    data = data.dropna(subset=["date", "metric", "value"])

    n_dupes = data.duplicated(KEY).sum()
    if n_dupes:
        warnings.warn(f"Найдено {n_dupes} дублей по (channel, date, metric), оставлены последние")

    return (
        data.sort_values(["channel", "metric", "date"])
        .drop_duplicates(KEY, keep="last")
        .reset_index(drop=True)
    )


def iter_series(data: pd.DataFrame, freq: str = "D"):
    """Перебирает ряды метрик: (channel, metric, series).

    Ряд строится на равномерной сетке дат с шагом freq, пропущенные даты — NaN.
    """
    for (channel, metric), group in data.groupby(["channel", "metric"]):
        series = group.set_index("date")["value"].sort_index().asfreq(freq)
        yield channel, metric, series
