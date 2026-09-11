"""
forecast_metrics.py
===================
Shared evaluation library for the small-data healthcare forecasting benchmark.

This is deliberately ONE module imported by every downstream script (EDA,
AutoML track, Snowflake ML FORECAST track, comparison report, learning loop).
If each track computed its own metrics, the head-to-head comparison would be
meaningless - a "benchmark" where the two sides use subtly different sMAPE
conventions or a different MASE denominator proves nothing.

Contents
--------
1. Rolling-origin (expanding-window) backtest split generation
2. Point-accuracy metrics        - MASE, RMSSE, sMAPE, MAE, RMSE, MAPE
3. Probabilistic metrics         - weighted quantile loss, CRPS, PICP, MIS, Winkler
4. Bias metrics                  - ME, MPE
5. Robustness breakdowns         - per-horizon, per-series, per-regime
6. Split-conformal prediction intervals with per-horizon calibration
7. Naive baseline forecasters

Why MASE is the primary metric here
-----------------------------------
The six departments differ in scale by roughly 8x (ONCOLOGY ~18/day vs
EMERGENCY ~148/day). Any absolute metric (MAE, RMSE) averaged across the panel
would be dominated almost entirely by EMERGENCY and would tell us nothing about
whether the model handles the small series. MASE is scale-free because it
divides by the in-sample seasonal-naive error of that same series.

MAPE is reported but must NOT be used for ranking: ELECTIVE_SURGERY and
ONCOLOGY contain true zeros (weekends/holidays), where MAPE is undefined and
its usual fixes silently bias the result.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# Seasonal period for daily hospital data. Weekly is the dominant cycle
# (see the weekday/weekend ratios in step 1), so m=7 is the right MASE
# denominator - not m=1, which would flatter every model that learns
# day-of-week and make the scores incomparable to the literature.
SEASONAL_PERIOD = 7

DEFAULT_QUANTILES = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)


# ==========================================================================
# 1. Rolling-origin backtest splits
# ==========================================================================
@dataclass
class BacktestWindow:
    """One forecast origin: fit on <= train_end, predict the following horizon."""
    window_id: int
    train_end: pd.Timestamp     # inclusive last date of training data
    test_start: pd.Timestamp
    test_end: pd.Timestamp

    def __repr__(self) -> str:
        return (f"W{self.window_id}(train<={self.train_end.date()}, "
                f"test {self.test_start.date()}..{self.test_end.date()})")


def make_backtest_windows(
    dates: pd.Series | pd.DatetimeIndex,
    horizon: int = 28,
    n_windows: int = 6,
    step: int | None = None,
) -> list[BacktestWindow]:
    """Expanding-window (rolling-origin) splits, oldest origin first.

    A SINGLE train/test split on ~1,100 observations gives one realisation of a
    noisy quantity: swap which 28 days you held out and the ranking of two close
    models can flip. With n_windows origins each score is an average over
    n_windows independent forecast tasks, which is what makes small-data model
    selection trustworthy.

    Windows are non-overlapping in their test sets by default (step=horizon),
    so no observation is scored twice.
    """
    d = pd.DatetimeIndex(pd.Series(pd.to_datetime(dates)).sort_values().unique())
    step = horizon if step is None else step

    need = horizon + n_windows * step
    if len(d) < need:
        raise ValueError(
            f"need >= {need} timestamps for {n_windows} windows at horizon "
            f"{horizon} (step {step}); got {len(d)}"
        )

    windows: list[BacktestWindow] = []
    # Build from the most recent origin backwards, then reverse, so the final
    # window always ends exactly on the last observed date.
    for k in range(n_windows):
        test_end_i = len(d) - 1 - k * step
        test_start_i = test_end_i - horizon + 1
        train_end_i = test_start_i - 1
        if train_end_i < 0:
            raise ValueError("not enough history for the requested windows")
        windows.append(BacktestWindow(
            window_id=n_windows - k,
            train_end=d[train_end_i],
            test_start=d[test_start_i],
            test_end=d[test_end_i],
        ))
    windows.reverse()
    for i, w in enumerate(windows, start=1):
        w.window_id = i
    return windows


# ==========================================================================
# 2. Point-accuracy metrics
# ==========================================================================
def _seasonal_naive_mae(y_insample: np.ndarray, m: int = SEASONAL_PERIOD) -> float:
    """In-sample seasonal-naive MAE - the MASE denominator."""
    y = np.asarray(y_insample, dtype=float)
    if len(y) <= m:
        return float("nan")
    return float(np.mean(np.abs(y[m:] - y[:-m])))


def _seasonal_naive_mse(y_insample: np.ndarray, m: int = SEASONAL_PERIOD) -> float:
    """In-sample seasonal-naive MSE - the RMSSE denominator."""
    y = np.asarray(y_insample, dtype=float)
    if len(y) <= m:
        return float("nan")
    return float(np.mean((y[m:] - y[:-m]) ** 2))


def mae(y: np.ndarray, f: np.ndarray) -> float:
    return float(np.mean(np.abs(np.asarray(y, float) - np.asarray(f, float))))


def rmse(y: np.ndarray, f: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(y, float) - np.asarray(f, float)) ** 2)))


def mape(y: np.ndarray, f: np.ndarray) -> float:
    """Mean absolute percentage error, zeros EXCLUDED (not imputed).

    Reported for familiarity only. Excluding zeros makes MAPE non-comparable
    across series with different zero rates, which is precisely why it must not
    be used to rank models on this panel.
    """
    y = np.asarray(y, float)
    f = np.asarray(f, float)
    m = y != 0
    if not m.any():
        return float("nan")
    return float(np.mean(np.abs((y[m] - f[m]) / y[m])) * 100.0)


def smape(y: np.ndarray, f: np.ndarray) -> float:
    """Symmetric MAPE in [0, 200]. Defined at zero when y and f are not both 0."""
    y = np.asarray(y, float)
    f = np.asarray(f, float)
    denom = (np.abs(y) + np.abs(f)) / 2.0
    m = denom != 0
    if not m.any():
        return 0.0
    return float(np.mean(np.abs(y[m] - f[m]) / denom[m]) * 100.0)


def mase(y: np.ndarray, f: np.ndarray, y_insample: np.ndarray,
         m: int = SEASONAL_PERIOD) -> float:
    """Mean absolute scaled error. <1 beats in-sample seasonal naive."""
    d = _seasonal_naive_mae(y_insample, m)
    if not np.isfinite(d) or d == 0:
        return float("nan")
    return mae(y, f) / d


def rmsse(y: np.ndarray, f: np.ndarray, y_insample: np.ndarray,
          m: int = SEASONAL_PERIOD) -> float:
    """Root mean squared scaled error (the M5 competition metric)."""
    d = _seasonal_naive_mse(y_insample, m)
    if not np.isfinite(d) or d == 0:
        return float("nan")
    return float(np.sqrt(np.mean((np.asarray(y, float) - np.asarray(f, float)) ** 2) / d))


# ==========================================================================
# 3. Bias
# ==========================================================================
def me(y: np.ndarray, f: np.ndarray) -> float:
    """Mean error. Sign matters: <0 = over-forecasting, >0 = under-forecasting.

    A model can have excellent MAE and still be systematically biased, which for
    capacity planning is a different and often worse failure than random error.
    """
    return float(np.mean(np.asarray(y, float) - np.asarray(f, float)))


def mpe(y: np.ndarray, f: np.ndarray) -> float:
    """Mean percentage error (signed), zeros excluded."""
    y = np.asarray(y, float)
    f = np.asarray(f, float)
    m = y != 0
    if not m.any():
        return float("nan")
    return float(np.mean((y[m] - f[m]) / y[m]) * 100.0)


# ==========================================================================
# 4. Probabilistic metrics
# ==========================================================================
def pinball_loss(y: np.ndarray, q_pred: np.ndarray, q: float) -> float:
    """Quantile (pinball) loss at level q."""
    y = np.asarray(y, float)
    p = np.asarray(q_pred, float)
    d = y - p
    return float(np.mean(np.maximum(q * d, (q - 1.0) * d)))


def weighted_quantile_loss(
    y: np.ndarray, quantile_preds: dict[float, np.ndarray],
) -> float:
    """Mean weighted quantile loss (AutoGluon's WQL), scale-free.

    wQL_q = 2 * sum(pinball_q) / sum(|y|), averaged over quantiles.
    Normalising by sum(|y|) makes this comparable across departments of very
    different volume, the same reason MASE is preferred over MAE here.
    """
    y = np.asarray(y, float)
    denom = np.sum(np.abs(y))
    if denom == 0:
        return float("nan")
    vals = [
        2.0 * np.sum(np.maximum(q * (y - np.asarray(p, float)),
                                (q - 1.0) * (y - np.asarray(p, float)))) / denom
        for q, p in sorted(quantile_preds.items())
    ]
    return float(np.mean(vals))


def crps_from_quantiles(y: np.ndarray, quantile_preds: dict[float, np.ndarray]) -> float:
    """CRPS approximated by averaging pinball loss over a quantile grid.

    Exact as the grid becomes dense; with 9 deciles it is a close approximation
    and is the standard practical estimator.
    """
    if not quantile_preds:
        return float("nan")
    return float(np.mean([
        2.0 * pinball_loss(y, p, q) for q, p in sorted(quantile_preds.items())
    ]))


def picp(y: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> float:
    """Prediction Interval Coverage Probability - fraction of actuals inside.

    Compare against the NOMINAL level: an 80% interval covering 55% is
    dangerously overconfident, and covering 99% is uselessly wide. Coverage is
    the metric most often skipped and most often wrong on small data, because
    asymptotic intervals assume more data than we have.
    """
    y = np.asarray(y, float)
    return float(np.mean((y >= np.asarray(lower, float)) &
                         (y <= np.asarray(upper, float))))


def interval_score(y: np.ndarray, lower: np.ndarray, upper: np.ndarray,
                   alpha: float) -> float:
    """Winkler / mean interval score for a central (1-alpha) interval.

    Penalises width AND miss distance, so it cannot be gamed by making the
    interval enormous (unlike coverage alone).
    """
    y = np.asarray(y, float)
    lo = np.asarray(lower, float)
    hi = np.asarray(upper, float)
    width = hi - lo
    below = (2.0 / alpha) * np.maximum(lo - y, 0.0)
    above = (2.0 / alpha) * np.maximum(y - hi, 0.0)
    return float(np.mean(width + below + above))


# ==========================================================================
# 5. Full metric bundle
# ==========================================================================
def evaluate_point(
    y: np.ndarray, f: np.ndarray, y_insample: np.ndarray,
    m: int = SEASONAL_PERIOD,
) -> dict[str, float]:
    """All point-accuracy and bias metrics for one series/window."""
    return {
        "MASE": mase(y, f, y_insample, m),
        "RMSSE": rmsse(y, f, y_insample, m),
        "sMAPE": smape(y, f),
        "MAE": mae(y, f),
        "RMSE": rmse(y, f),
        "MAPE": mape(y, f),
        "ME": me(y, f),
        "MPE": mpe(y, f),
    }


def evaluate_probabilistic(
    y: np.ndarray,
    quantile_preds: dict[float, np.ndarray] | None = None,
    intervals: dict[float, tuple[np.ndarray, np.ndarray]] | None = None,
) -> dict[str, float]:
    """Probabilistic metrics.

    `intervals` maps nominal coverage (e.g. 0.80) -> (lower, upper).
    """
    out: dict[str, float] = {}
    if quantile_preds:
        out["WQL"] = weighted_quantile_loss(y, quantile_preds)
        out["CRPS"] = crps_from_quantiles(y, quantile_preds)
    if intervals:
        for lvl, (lo, hi) in sorted(intervals.items()):
            tag = int(round(lvl * 100))
            out[f"PICP{tag}"] = picp(y, lo, hi)
            out[f"MIS{tag}"] = interval_score(y, lo, hi, alpha=1.0 - lvl)
            out[f"WIDTH{tag}"] = float(np.mean(np.asarray(hi, float) -
                                               np.asarray(lo, float)))
    return out


def per_horizon_errors(
    y: np.ndarray, f: np.ndarray, horizons: np.ndarray,
) -> pd.DataFrame:
    """MAE / signed error by forecast step.

    Error should grow with h. A FLAT curve usually means the model is ignoring
    recent history and just predicting a seasonal climatology - which can look
    acceptable on average while being useless for next-day staffing.
    """
    df = pd.DataFrame({"h": np.asarray(horizons, int),
                       "y": np.asarray(y, float),
                       "f": np.asarray(f, float)})
    df["abs_err"] = (df["y"] - df["f"]).abs()
    df["err"] = df["y"] - df["f"]
    return (df.groupby("h")
              .agg(MAE=("abs_err", "mean"), ME=("err", "mean"), N=("y", "size"))
              .reset_index())


# ==========================================================================
# 6. Split-conformal prediction intervals
# ==========================================================================
@dataclass
class ConformalCalibrator:
    """Split-conformal intervals with PER-HORIZON calibration.

    Motivation: model-native intervals rest on distributional assumptions
    (Gaussian errors, correct likelihood, enough data to estimate variance).
    On ~1,100 observations per series with overdispersed counts and a regime
    shift, those assumptions are shaky and coverage drifts away from nominal.

    Split conformal instead takes the empirical quantile of absolute residuals
    on a held-out calibration set. It is distribution-free and gives finite-
    sample coverage under exchangeability - no asymptotics required.

    Calibrating SEPARATELY PER HORIZON matters: day-28 residuals are far larger
    than day-1 residuals, so one pooled quantile would produce intervals that
    are too wide early and too narrow late.

    Caveat, stated plainly: time-series residuals are not strictly exchangeable,
    so the guarantee is approximate here. That is exactly why we still measure
    PICP empirically rather than assuming the guarantee holds.
    """
    quantile_by_h: dict[int, float] = field(default_factory=dict)
    pooled_quantile: float = float("nan")
    level: float = 0.80

    def fit(self, residuals: np.ndarray, horizons: np.ndarray,
            level: float = 0.80) -> "ConformalCalibrator":
        r = np.abs(np.asarray(residuals, float))
        h = np.asarray(horizons, int)
        self.level = level
        # Finite-sample corrected rank: ceil((n+1)*level)/n
        def _q(a: np.ndarray) -> float:
            n = len(a)
            if n == 0:
                return float("nan")
            k = min(int(np.ceil((n + 1) * level)), n)
            return float(np.sort(a)[k - 1])

        self.pooled_quantile = _q(r)
        self.quantile_by_h = {}
        for hh in np.unique(h):
            sel = r[h == hh]
            # Fall back to the pooled quantile when a horizon has too few
            # calibration points to estimate its own quantile reliably.
            self.quantile_by_h[int(hh)] = _q(sel) if len(sel) >= 10 else self.pooled_quantile
        return self

    def intervals(self, point: np.ndarray, horizons: np.ndarray,
                  non_negative: bool = True) -> tuple[np.ndarray, np.ndarray]:
        p = np.asarray(point, float)
        h = np.asarray(horizons, int)
        q = np.array([self.quantile_by_h.get(int(x), self.pooled_quantile) for x in h])
        lo, hi = p - q, p + q
        if non_negative:
            lo = np.maximum(lo, 0.0)   # admissions cannot be negative
        return lo, hi


# ==========================================================================
# 7. Naive baselines
# ==========================================================================
def baseline_forecasts(
    y_train: np.ndarray, horizon: int, m: int = SEASONAL_PERIOD,
) -> dict[str, np.ndarray]:
    """Trivial baselines every real model must beat to justify its complexity.

    seasonal_naive (repeat the last observed week) is the one that matters: on
    daily hospital data with a strong weekly cycle it is a genuinely strong
    competitor, and a model that cannot beat it has learned nothing useful.
    """
    y = np.asarray(y_train, float)
    out: dict[str, np.ndarray] = {}

    out["naive"] = np.repeat(y[-1], horizon)
    out["mean"] = np.repeat(np.mean(y), horizon)

    if len(y) >= m:
        last = y[-m:]
        out["seasonal_naive"] = np.array([last[i % m] for i in range(horizon)])
    else:
        out["seasonal_naive"] = out["naive"]

    # Drift: extrapolate the average per-step change over the whole history.
    if len(y) >= 2:
        slope = (y[-1] - y[0]) / (len(y) - 1)
        out["drift"] = y[-1] + slope * np.arange(1, horizon + 1)
    else:
        out["drift"] = out["naive"]

    # Seasonal naive on a 4-week average - lower variance than a single week,
    # a slightly stronger baseline on noisy count data.
    if len(y) >= 4 * m:
        blk = y[-4 * m:].reshape(4, m).mean(axis=0)
        out["seasonal_mean_4w"] = np.array([blk[i % m] for i in range(horizon)])

    return {k: np.maximum(v, 0.0) for k, v in out.items()}
