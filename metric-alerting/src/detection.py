"""Детектор выбросов и плавных сдвигов во временном ряду метрики (LOWESS + MAD)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from tsmoothie.smoother import LowessSmoother

MIN_POINTS = 8  # на более коротких рядах LOWESS не строим
CLIP_LIMITS = {"count": (0, None), "rate": (0, 1)}


@dataclass
class Params:
    """Параметры детектора. Значения по умолчанию — рекомендуемые (подробнее — в README)."""

    smooth_fraction: float = 0.17      # доля ряда в локальном окне LOWESS
    iterations: int = 6                # итерации LOWESS внутри tsmoothie
    passes: int = 3                    # сколько раз чистим выбросы перед финальным сглаживанием
    clean_sigma: float = 2.5           # ширина коридора при очистке
    n_sigma: float = 2.0               # порог выброса по MAD-score
    trend_lookback_days: int = 7       # с каким днём сравниваем текущий сглаженный уровень
    trend_pct_threshold: float = 0.05  # минимальное относительное изменение уровня
    confirm_days: int = 3              # сколько дней подряд должен держаться сигнал сдвига


def detect_spikes(series: pd.Series, params: Params | None = None, kind: str | None = None) -> pd.DataFrame:
    """Ищет точечные выбросы в ряду.

    1. Несколько раз сглаживаем ряд LOWESS и убираем точки вне коридора ±clean_sigma,
       чтобы выбросы не искажали ожидаемый уровень.
    2. LOWESS по очищенному ряду — ожидаемое значение метрики.
    3. Отклонение факта от ожидания делим на робастную оценку разброса (MAD-σ).
       Если |score| > n_sigma — выброс.

    kind: "count" (значения ≥ 0) или "rate" (значения в [0, 1]) — чтобы коридор нормы не выходил за эти пределы.

    Возвращает таблицу с индексом series: value, cleaned (выбросы заменены ожидаемым значением),
    expected, lower / upper (коридор нормы), score, spike (up / down / normal).
    """
    params = params or Params()
    values = fill_gaps(series.to_numpy(dtype=float))

    work = values.copy()
    for _ in range(params.passes):
        _, lower, upper = _lowess(work, params)
        outside = (work < lower) | (work > upper)
        if not outside.any():
            break
        work = fill_gaps(np.where(outside, np.nan, work))

    expected, _, _ = _lowess(work, params)
    residual = values - expected
    sigma = _mad_sigma(residual)

    if sigma < 1e-9 * (np.max(np.abs(values)) or 1.0):
        score = np.full(len(values), np.nan)  # ряд почти постоянный — выбросов не ищем
    else:
        score = residual / sigma
    is_spike = np.abs(np.nan_to_num(score)) > params.n_sigma

    result = pd.DataFrame({
        "value": values,
        "cleaned": np.where(is_spike, expected, values),
        "expected": expected,
        "lower": expected - params.n_sigma * sigma,
        "upper": expected + params.n_sigma * sigma,
        "score": score,
        "spike": np.where(is_spike, np.where(residual > 0, "up", "down"), "normal"),
    }, index=series.index)

    low, high = CLIP_LIMITS.get(kind, (None, None))
    bounded = ["cleaned", "lower", "upper"]
    result[bounded] = result[bounded].clip(low, high)
    return result


def backtest(
    series: pd.Series,
    params: Params | None = None,
    kind: str | None = None,
    start: int | None = None,
) -> pd.DataFrame:
    """Имитирует ежедневный запуск детектора на истории.

    Для каждого дня детектор видит только данные до этого дня включительно и оценивает последнюю точку:
    - expected, lower / upper, cleaned — ожидаемый уровень, коридор нормы и очищенное значение,
      какими их видел детектор в этот день;
    - spike — выброс в этот день (up / down / normal);
    - trend — сигнал плавного сдвига: сглаженный уровень изменился за trend_lookback_days дней
      больше чем на trend_pct_threshold (up / down / normal);
    - trend_streak — сколько дней подряд держится один и тот же сигнал;
    - trend_confirmed — сигнал держится не меньше confirm_days дней;
    - trend_alert — первый день, когда сигнал подтвердился: в этот день уходит алерт.

    start — номер первой точки, для которой запускать детектор. Для ежедневного отчёта
    достаточно последних confirm_days точек: start=len(series) - params.confirm_days.
    """
    params = params or Params()
    first = max(start or 0, MIN_POINTS + params.trend_lookback_days)

    rows = []
    for end in range(first, len(series)):
        run = detect_spikes(series.iloc[:end + 1], params, kind)
        today = run.iloc[-1]
        change = _pct_change(today["expected"], run["expected"].iloc[-1 - params.trend_lookback_days])
        rows.append({
            "date": run.index[-1],
            "value": today["value"],
            "cleaned": today["cleaned"],
            "expected": today["expected"],
            "lower": today["lower"],
            "upper": today["upper"],
            "spike": today["spike"],
            "trend_change": change,
            "trend": _trend_signal(change, params.trend_pct_threshold),
        })

    if not rows:
        return pd.DataFrame()
    return _add_trend_streaks(pd.DataFrame(rows), params.confirm_days)


def fill_gaps(values) -> np.ndarray:
    """Заполняет пропуски линейной интерполяцией, по краям — ближайшим значением."""
    series = pd.Series(values, dtype="float")
    if series.isna().all():
        return np.zeros(len(series))
    return series.interpolate(limit_direction="both").bfill().ffill().to_numpy()


def _lowess(values: np.ndarray, params: Params):
    """LOWESS-сглаживание и коридор ±clean_sigma вокруг него."""
    if len(values) < MIN_POINTS:
        return values.copy(), np.full(len(values), -np.inf), np.full(len(values), np.inf)

    smoother = LowessSmoother(smooth_fraction=params.smooth_fraction, iterations=params.iterations)
    smoother.smooth(values.reshape(1, -1))
    lower, upper = smoother.get_intervals("sigma_interval", n_sigma=params.clean_sigma)
    return smoother.smooth_data[0], lower[0], upper[0]


def _mad_sigma(residuals: np.ndarray) -> float:
    """Робастная оценка разброса: 1.4826 · MAD.

    Для нормального распределения совпадает со стандартным отклонением, но в отличие от него
    почти не реагирует на единичные выбросы.
    """
    median = np.nanmedian(residuals)
    sigma = 1.4826 * np.nanmedian(np.abs(residuals - median))
    if not np.isfinite(sigma) or sigma <= 1e-12:
        sigma = np.nanstd(residuals, ddof=1)
    if not np.isfinite(sigma) or sigma <= 1e-12:
        sigma = 1e-12
    return sigma


def _pct_change(now: float, past: float) -> float:
    """Относительное изменение; NaN, если прошлое значение около нуля."""
    if not (np.isfinite(now) and np.isfinite(past)) or abs(past) <= 1e-12:
        return np.nan
    return (now - past) / abs(past)


def _trend_signal(change: float, threshold: float) -> str:
    if change >= threshold:
        return "up"
    if change <= -threshold:
        return "down"
    return "normal"


def _add_trend_streaks(history: pd.DataFrame, confirm_days: int) -> pd.DataFrame:
    """Считает, сколько дней подряд держится один и тот же сигнал тренда, и отмечает подтверждённые сдвиги."""
    trend = history["trend"]
    run_id = (trend != trend.shift()).cumsum()
    streak = trend.groupby(run_id).cumcount() + 1

    history["trend_streak"] = streak.where(trend != "normal", 0)
    history["trend_confirmed"] = history["trend_streak"] >= confirm_days
    history["trend_alert"] = history["trend_streak"] == confirm_days
    return history
