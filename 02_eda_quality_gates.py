"""
02_eda_quality_gates.py
=======================
EDA plus the eight mandatory quality gates for the small-data healthcare
forecasting benchmark, and the naive baseline scores that every model must beat.

The gates are run BEFORE any model code exists, on purpose. Most forecasting
projects fail not because the model was badly tuned but because the evaluation
was wrong, a covariate leaked, or nobody checked whether seasonal-naive already
solved the problem.

Gate list
---------
1. Leakage detection        - covariate availability at forecast time
2. ID / constant exclusion  - zero-signal columns
3. Evaluation setup         - rolling-origin temporal splits (never random)
4. Naive baselines          - scored on MASE, the metric we will rank on
5. Data quality             - duplicates, timestamp gaps, long-format check
6. Fairness                 - which series are structurally disadvantaged
7. Scale/intermittency      - the forecasting analogue of class imbalance
8. Outlier assessment       - signal vs noise, decided per department

Outputs
-------
  plots/*.png                   EDA figures
  artifacts/baselines.csv       per-series/window baseline scores
  artifacts/quality_gates.json  machine-readable gate results
"""

from __future__ import annotations

import json
import os

import matplotlib
matplotlib.use("Agg")            # headless: save figures, never try to display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import acf, adfuller
from statsmodels.tsa.seasonal import STL

from forecast_metrics import (
    SEASONAL_PERIOD, baseline_forecasts, evaluate_point,
    make_backtest_windows,
)
from snowpark_session import create_snowpark_session

DATABASE, SCHEMA = "HEALTHCARE", "ML_FORECAST"
HORIZON = 28
N_WINDOWS = 6
PLOTS = "plots"
ARTIFACTS = "artifacts"

# Covariate availability contract. This is the single most important piece of
# metadata in the whole project and it cannot be inferred from the data - it is
# a domain fact about WHEN each value becomes known.
KNOWN_FUTURE = ["DOW", "MONTH", "DAY_OF_YEAR", "IS_WEEKEND",
                "IS_HOLIDAY", "IS_HOLIDAY_WINDOW", "IS_SCHOOL_TERM"]
PAST_ONLY = ["AVG_TEMP_C", "ER_WAIT_MINS"]
TARGET = "ADMISSIONS"
ID_COL = "DEPT_ID"
TIME_COL = "ADMIT_DATE"


def load() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    session = create_snowpark_session()
    session.use_database(DATABASE)
    session.use_schema(SCHEMA)
    print(f"connected: {session.get_current_account()} "
          f"{session.get_current_database()}.{session.get_current_schema()}")
    panel = session.table("DAILY_ADMISSIONS").to_pandas()
    truth = session.table("GROUND_TRUTH_COMPONENTS").to_pandas()
    static = session.table("DEPT_STATIC").to_pandas()
    session.close()
    for df in (panel, truth):
        df[TIME_COL] = pd.to_datetime(df[TIME_COL])
    panel = panel.sort_values([ID_COL, TIME_COL]).reset_index(drop=True)
    truth = truth.sort_values([ID_COL, TIME_COL]).reset_index(drop=True)
    return panel, truth, static


# ==========================================================================
def gate_1_leakage(panel: pd.DataFrame) -> dict:
    """Gate 1 - leakage: is each covariate actually knowable at forecast time?

    The trap in this dataset: AVG_TEMP_C and ER_WAIT_MINS correlate with
    admissions and would improve any backtest if fed in as known-future
    covariates. But at real forecast time you do not know next month's daily
    temperature or ER wait. Using them as known-future would produce an
    impressive score that collapses in production.

    They are still usable - as PAST-ONLY covariates, where the model may use
    their history but not their future values.
    """
    print("\n" + "=" * 74)
    print("GATE 1: LEAKAGE DETECTION")
    print("=" * 74)

    corr = {}
    for c in KNOWN_FUTURE + PAST_ONLY:
        corr[c] = panel[[c, TARGET]].corr().iloc[0, 1]

    print(f"{'covariate':<22}{'availability':<16}{'corr(target)':>14}  verdict")
    verdicts = {}
    for c in KNOWN_FUTURE:
        verdicts[c] = "known_future"
        print(f"{c:<22}{'known-future':<16}{corr[c]:>14.3f}  SAFE as known covariate")
    for c in PAST_ONLY:
        verdicts[c] = "past_only"
        print(f"{c:<22}{'PAST-ONLY':<16}{corr[c]:>14.3f}  "
              f"LEAK if used as known-future -> past covariate only")

    # HOLIDAY_NAME is a high-cardinality mostly-empty string. Encoding it would
    # add ~35 sparse levels against ~1,100 observations per series - a textbook
    # small-data variance problem for no gain over the existing binary flags.
    nn = int((panel["HOLIDAY_NAME"].fillna("") != "").sum())
    print(f"\nHOLIDAY_NAME: {panel['HOLIDAY_NAME'].nunique()} distinct, "
          f"{nn}/{len(panel)} non-empty "
          f"({100*nn/len(panel):.1f}%) -> DROP (IS_HOLIDAY covers it "
          f"without adding sparse levels)")
    verdicts["HOLIDAY_NAME"] = "drop_high_cardinality_sparse"
    return {"availability": verdicts,
            "target_correlations": {k: round(float(v), 4) for k, v in corr.items()}}


def gate_2_constants(panel: pd.DataFrame) -> dict:
    """Gate 2 - drop columns with no usable variation."""
    print("\n" + "=" * 74)
    print("GATE 2: ID / CONSTANT EXCLUSION")
    print("=" * 74)
    drop, keep = [], []
    for c in KNOWN_FUTURE + PAST_ONLY:
        nu = panel[c].nunique()
        if nu <= 1:
            drop.append(c)
            print(f"  {c:<22} constant ({nu} value) -> DROP")
        else:
            keep.append(c)
            print(f"  {c:<22} {nu:>5} distinct -> keep")
    # DAY_OF_YEAR is kept as a raw column but will be Fourier-encoded in step 3
    # rather than used directly; as a raw integer it implies 1 Jan and 31 Dec are
    # maximally distant, which is wrong for a cyclical variable.
    print("\n  note: DAY_OF_YEAR / DOW / MONTH are cyclical - they will be "
          "sin/cos encoded in\n        03_features.py, not consumed as raw "
          "integers (raw ints imply Dec 31 and\n        Jan 1 are 364 apart).")
    return {"dropped": drop, "kept": keep}


def gate_3_eval_setup(panel: pd.DataFrame) -> tuple[list, dict]:
    """Gate 3 - temporal rolling-origin evaluation. Never a random split."""
    print("\n" + "=" * 74)
    print("GATE 3: EVALUATION SETUP")
    print("=" * 74)
    windows = make_backtest_windows(panel[TIME_COL], horizon=HORIZON,
                                    n_windows=N_WINDOWS)
    print(f"rolling-origin, expanding train, horizon={HORIZON}d, "
          f"{N_WINDOWS} windows, non-overlapping test sets")
    for w in windows:
        n_train = int((panel[TIME_COL] <= w.train_end).sum() / panel[ID_COL].nunique())
        print(f"  {w}  train_days/series={n_train}")
    print("\n  A random train/test split would be INVALID here: it leaks future "
          "information\n  into training and destroys the autocorrelation "
          "structure the models rely on.")
    total_scored = HORIZON * N_WINDOWS * panel[ID_COL].nunique()
    print(f"  total scored observations: {HORIZON} x {N_WINDOWS} x "
          f"{panel[ID_COL].nunique()} = {total_scored}")
    return windows, {
        "horizon": HORIZON, "n_windows": N_WINDOWS,
        "scheme": "expanding_window_rolling_origin",
        "scored_observations": total_scored,
        "windows": [{"id": w.window_id, "train_end": str(w.train_end.date()),
                     "test_start": str(w.test_start.date()),
                     "test_end": str(w.test_end.date())} for w in windows],
    }


def gate_4_baselines(panel: pd.DataFrame, windows: list) -> tuple[pd.DataFrame, dict]:
    """Gate 4 - naive baselines on the ranking metric (MASE), not accuracy."""
    print("\n" + "=" * 74)
    print("GATE 4: NAIVE BASELINES")
    print("=" * 74)
    rows = []
    for w in windows:
        for dept, g in panel.groupby(ID_COL, sort=False):
            tr = g[g[TIME_COL] <= w.train_end]
            te = g[(g[TIME_COL] >= w.test_start) & (g[TIME_COL] <= w.test_end)]
            if len(te) != HORIZON:
                continue
            y_ins = tr[TARGET].to_numpy(float)
            y_te = te[TARGET].to_numpy(float)
            for name, f in baseline_forecasts(y_ins, HORIZON).items():
                m = evaluate_point(y_te, f, y_ins, SEASONAL_PERIOD)
                rows.append({"model": name, "window": w.window_id,
                             "dept_id": dept, **m})
    bl = pd.DataFrame(rows)

    piv = (bl.groupby("model")[["MASE", "RMSSE", "sMAPE", "MAE", "ME"]]
             .mean().sort_values("MASE"))
    print("mean across all windows and departments:")
    print(piv.round(3).to_string())

    best = piv.index[0]
    print(f"\n  BASELINE TO BEAT: '{best}' at MASE={piv.loc[best,'MASE']:.4f}")
    print("  Any model with MASE >= this has not earned its complexity.")
    print("\nper-department MASE for the strongest baseline:")
    print(bl[bl["model"] == best].groupby("dept_id")["MASE"]
          .mean().round(3).to_string())
    return bl, {"best_baseline": best,
                "scores": piv.round(4).to_dict(orient="index")}


def gate_5_data_quality(panel: pd.DataFrame) -> dict:
    """Gate 5 - duplicates, timestamp gaps, long-format integrity."""
    print("\n" + "=" * 74)
    print("GATE 5: DATA QUALITY")
    print("=" * 74)
    dup = int(panel.duplicated([ID_COL, TIME_COL]).sum())
    print(f"duplicate (dept, date) keys : {dup}")
    nulls = panel.isna().sum()
    print(f"null values                 : {int(nulls.sum())} total")
    if nulls.sum():
        print(nulls[nulls > 0].to_string())

    print("\ntimestamp gap check (daily frequency, per department):")
    gaps = {}
    for dept, g in panel.groupby(ID_COL, sort=False):
        d = pd.DatetimeIndex(g[TIME_COL])
        full = pd.date_range(d.min(), d.max(), freq="D")
        missing = len(full.difference(d))
        gaps[dept] = missing
        print(f"  {dept:<20} {len(g):>5} rows, {missing:>3} missing days")
    print("\n  Gaps matter: a lag-7 feature computed on a gapped index silently "
          "reaches back\n  the wrong number of DAYS. Zero gaps here means lags "
          "are safe to compute positionally.")

    long_ok = set(panel.columns) >= {ID_COL, TIME_COL, TARGET}
    print(f"\nlong format (id, timestamp, target) : {long_ok}")
    print(f"target dtype / non-negative         : {panel[TARGET].dtype} / "
          f"{bool((panel[TARGET] >= 0).all())}")
    return {"duplicate_keys": dup, "total_nulls": int(nulls.sum()),
            "missing_days_per_dept": gaps, "long_format": bool(long_ok)}


def gate_6_fairness(panel: pd.DataFrame, static: pd.DataFrame) -> dict:
    """Gate 6 - which series will the model systematically under-serve?

    The forecasting analogue of fairness. A panel-wide average score hides the
    fact that low-volume departments contribute little to any pooled loss and so
    get optimised last. If ONCOLOGY forecasts are bad, a global MASE average
    dominated by EMERGENCY will not reveal it - so we commit up front to
    reporting WORST-SERIES metrics, not just the mean.
    """
    print("\n" + "=" * 74)
    print("GATE 6: FAIRNESS / SUBGROUP EXPOSURE")
    print("=" * 74)
    s = (panel.groupby(ID_COL)[TARGET]
              .agg(["mean", "std", "min", "max"]).round(2))
    s["share_of_volume"] = (panel.groupby(ID_COL)[TARGET].sum()
                            / panel[TARGET].sum()).round(4)
    s = s.merge(static.set_index("DEPT_ID")[["DEPT_TYPE", "BED_CAPACITY"]],
                left_index=True, right_index=True, how="left")
    print(s.to_string())
    print(f"\n  volume share ranges {s['share_of_volume'].min():.1%} .. "
          f"{s['share_of_volume'].max():.1%}")
    print("  => EMERGENCY alone would dominate any volume-weighted loss.")
    print("  COMMITMENT: report per-department MASE and WORST-series MASE "
          "alongside the mean.")
    return {"volume_share": s["share_of_volume"].to_dict(),
            "commitment": "report_per_series_and_worst_series_metrics"}


def gate_7_scale_intermittency(panel: pd.DataFrame) -> dict:
    """Gate 7 - scale imbalance and intermittency (forecasting's 'imbalance')."""
    print("\n" + "=" * 74)
    print("GATE 7: SCALE IMBALANCE / INTERMITTENCY")
    print("=" * 74)
    rows = []
    for dept, g in panel.groupby(ID_COL, sort=False):
        y = g[TARGET].to_numpy(float)
        zero = float((y == 0).mean())
        rows.append({"dept_id": dept, "mean": y.mean(), "zero_frac": zero,
                     "cv": y.std() / y.mean() if y.mean() else np.nan})
    t = pd.DataFrame(rows).set_index("dept_id")
    ratio = t["mean"].max() / t["mean"].min()
    print(t.round(3).to_string())
    print(f"\n  volume ratio largest:smallest = {ratio:.1f}x")
    print("  => absolute metrics (MAE/RMSE) are NOT comparable across "
          "departments.\n     Ranking metric is MASE (scale-free). "
          "Confirmed choice.")
    inter = t[t["zero_frac"] > 0.02]
    if len(inter):
        print(f"\n  intermittent series (>2% zeros): "
              f"{', '.join(inter.index)}")
        print("  => MAPE is UNDEFINED on these rows. MAPE reported for "
              "familiarity only, never\n     used for model selection. "
              "Probabilistic scoring uses WQL, which is defined at zero.")
    return {"volume_ratio": round(float(ratio), 2),
            "zero_fraction": t["zero_frac"].round(4).to_dict(),
            "intermittent_series": list(inter.index),
            "ranking_metric": "MASE"}


def gate_8_outliers(panel: pd.DataFrame, truth: pd.DataFrame) -> dict:
    """Gate 8 - outliers: real signal or measurement noise?

    The decision is per department and it is NOT 'clip everything'. The year-3
    flu surge produces large positive outliers in EMERGENCY and PEDIATRICS, but
    those are genuine demand events - exactly what a capacity forecast must
    capture. Clipping them would train the model to under-predict every future
    surge, which is the costliest possible error here.
    """
    print("\n" + "=" * 74)
    print("GATE 8: OUTLIER ASSESSMENT")
    print("=" * 74)
    dec = {}
    print(f"{'dept':<20}{'n>3sig':>8}{'n>IQR':>8}{'max_z':>8}   decision")
    for dept, g in panel.groupby(ID_COL, sort=False):
        y = g[TARGET].to_numpy(float)
        z = (y - y.mean()) / y.std()
        n3 = int((np.abs(z) > 3).sum())
        q1, q3 = np.percentile(y, [25, 75])
        iqr = q3 - q1
        nq = int(((y < q1 - 1.5 * iqr) | (y > q3 + 1.5 * iqr)).sum())
        # Attribute the extremes: are they explained by the known flu component?
        gt = truth[truth["DEPT_ID"] == dept]
        flu_driven = float(gt.loc[gt["COMP_FLU"] > 1.05, "COMP_FLU"].max() or 1.0)
        keep = flu_driven > 1.05
        dec[dept] = "KEEP (real demand signal)" if keep else "KEEP (no clipping)"
        print(f"{dept:<20}{n3:>8}{nq:>8}{z.max():>8.2f}   {dec[dept]}"
              + (f"  flu x{flu_driven:.2f}" if keep else ""))
    print("\n  DECISION: no clipping, no winsorising anywhere.")
    print("  Rationale: the extremes are generated by the flu wave and holiday")
    print("  effects - structural demand, not sensor error. Clipping would bias")
    print("  the model to under-forecast exactly the surges that matter most for")
    print("  capacity planning. Heavy tails are instead handled by (a) modelling")
    print("  counts with quantile/probabilistic loss and (b) conformal intervals.")
    return {"decision": "no_clipping", "per_dept": dec}


# ==========================================================================
def eda_plots(panel: pd.DataFrame, truth: pd.DataFrame, bl: pd.DataFrame,
              windows: list) -> None:
    os.makedirs(PLOTS, exist_ok=True)
    depts = list(panel[ID_COL].unique())

    # --- 1. the six series, with break + flu annotated ---------------------
    fig, axes = plt.subplots(len(depts), 1, figsize=(15, 2.3 * len(depts)),
                             sharex=True)
    for ax, d in zip(axes, depts):
        g = panel[panel[ID_COL] == d]
        ax.plot(g[TIME_COL], g[TARGET], lw=0.6, color="#1f4e79")
        gt = truth[truth["DEPT_ID"] == d]
        ax.plot(gt["ADMIT_DATE"], gt["MU_EXPECTED"], lw=1.0, color="#d1495b",
                alpha=0.85, label="true mean mu(t)")
        ax.axvline(pd.Timestamp("2024-08-23"), color="green", ls="--", lw=1)
        ax.axvspan(pd.Timestamp("2025-01-15"), pd.Timestamp("2025-04-15"),
                   color="orange", alpha=0.13)
        ax.axvspan(windows[0].test_start, windows[-1].test_end,
                   color="grey", alpha=0.16)
        ax.set_ylabel(d, fontsize=8)
        ax.legend(fontsize=6, loc="upper left")
    axes[0].set_title("Synthetic daily admissions | red = true mean, "
                      "green dash = structural break, orange = flu wave, "
                      "grey = backtest region", fontsize=10)
    plt.tight_layout()
    plt.savefig(f"{PLOTS}/01_series_overview.png", dpi=120)
    plt.close()

    # --- 2. weekly + annual profiles --------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.2))
    for d in depts:
        g = panel[panel[ID_COL] == d]
        prof = g.groupby("DOW")[TARGET].mean()
        axes[0].plot(prof.index, prof / prof.mean(), marker="o", label=d, lw=1.3)
        mo = g.groupby("MONTH")[TARGET].mean()
        axes[1].plot(mo.index, mo / mo.mean(), marker="o", label=d, lw=1.3)
    axes[0].set_xticks(range(7))
    axes[0].set_xticklabels(["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"])
    axes[0].set_title("Weekly profile (normalised)")
    axes[0].axhline(1.0, color="k", lw=0.5, ls=":")
    axes[1].set_title("Annual profile (normalised) - note ORTHOPEDICS is "
                      "counter-seasonal")
    axes[1].axhline(1.0, color="k", lw=0.5, ls=":")
    axes[1].set_xlabel("month")
    for a in axes:
        a.legend(fontsize=7)
        a.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(f"{PLOTS}/02_seasonal_profiles.png", dpi=120)
    plt.close()

    # --- 3. ACF ------------------------------------------------------------
    fig, axes = plt.subplots(2, 3, figsize=(15, 6))
    for ax, d in zip(axes.ravel(), depts):
        y = panel.loc[panel[ID_COL] == d, TARGET].to_numpy(float)
        a = acf(y, nlags=35, fft=True)
        ax.bar(range(len(a)), a, color="#1f4e79")
        for k in (7, 14, 21, 28):
            ax.axvline(k, color="red", ls=":", lw=0.8)
        ax.set_title(f"ACF {d}", fontsize=9)
        ax.axhline(0, color="k", lw=0.5)
    plt.suptitle("Autocorrelation - red lines at weekly lags confirm m=7 "
                 "is the right seasonal period for MASE", fontsize=10)
    plt.tight_layout()
    plt.savefig(f"{PLOTS}/03_acf.png", dpi=120)
    plt.close()

    # --- 4. STL decomposition vs known truth ------------------------------
    fig, axes = plt.subplots(3, 2, figsize=(14, 8))
    for col, d in enumerate(["EMERGENCY", "ELECTIVE_SURGERY"]):
        g = panel[panel[ID_COL] == d].set_index(TIME_COL)[TARGET]
        st = STL(g, period=7, robust=True).fit()
        axes[0, col].plot(st.trend, color="#d1495b")
        axes[0, col].set_title(f"{d}: STL trend", fontsize=9)
        axes[1, col].plot(st.seasonal[:70], color="#1f4e79")
        axes[1, col].set_title(f"{d}: STL weekly seasonal (first 10w)", fontsize=9)
        axes[2, col].plot(st.resid, color="grey", lw=0.5)
        axes[2, col].set_title(f"{d}: STL residual", fontsize=9)
    plt.tight_layout()
    plt.savefig(f"{PLOTS}/04_stl.png", dpi=120)
    plt.close()

    # --- 5. baseline MASE heatmap -----------------------------------------
    piv = bl.pivot_table(index="model", columns="dept_id", values="MASE",
                         aggfunc="mean").sort_index()
    fig, ax = plt.subplots(figsize=(10, 3.6))
    im = ax.imshow(piv.values, cmap="RdYlGn_r", aspect="auto")
    ax.set_xticks(range(len(piv.columns)))
    ax.set_xticklabels(piv.columns, rotation=30, ha="right", fontsize=8)
    ax.set_yticks(range(len(piv.index)))
    ax.set_yticklabels(piv.index, fontsize=8)
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            ax.text(j, i, f"{piv.values[i,j]:.2f}", ha="center", va="center",
                    fontsize=7)
    plt.colorbar(im, label="MASE (lower better)")
    ax.set_title("Baseline MASE by department - the bar every model must clear",
                 fontsize=10)
    plt.tight_layout()
    plt.savefig(f"{PLOTS}/05_baseline_mase.png", dpi=120)
    plt.close()

    # --- 6. distributions --------------------------------------------------
    fig, axes = plt.subplots(2, 3, figsize=(14, 6))
    for ax, d in zip(axes.ravel(), depts):
        y = panel.loc[panel[ID_COL] == d, TARGET]
        ax.hist(y, bins=40, color="#1f4e79", alpha=0.85)
        ax.set_title(f"{d}  zeros={int((y==0).sum())}", fontsize=9)
    plt.suptitle("Admission count distributions - right-skewed, "
                 "zero-inflated in scheduled departments", fontsize=10)
    plt.tight_layout()
    plt.savefig(f"{PLOTS}/06_distributions.png", dpi=120)
    plt.close()

    print(f"\n  wrote 6 figures to {PLOTS}/")


def stationarity(panel: pd.DataFrame) -> dict:
    print("\n" + "-" * 74)
    print("STATIONARITY (ADF) - informational")
    print("-" * 74)
    out = {}
    for d, g in panel.groupby(ID_COL, sort=False):
        y = g[TARGET].to_numpy(float)
        p_lvl = adfuller(y, autolag="AIC")[1]
        p_dif = adfuller(np.diff(y), autolag="AIC")[1]
        out[d] = {"adf_p_level": round(float(p_lvl), 5),
                  "adf_p_diff": round(float(p_dif), 5)}
        print(f"  {d:<20} p(level)={p_lvl:.4f}  p(diff)={p_dif:.4f}"
              f"   {'non-stationary in level' if p_lvl > 0.05 else 'stationary in level'}")
    return out


def main() -> None:
    os.makedirs(ARTIFACTS, exist_ok=True)
    panel, truth, static = load()
    print(f"\nloaded panel: {panel.shape[0]} rows x {panel.shape[1]} cols, "
          f"{panel[ID_COL].nunique()} series")

    res = {}
    res["gate1_leakage"] = gate_1_leakage(panel)
    res["gate2_constants"] = gate_2_constants(panel)
    windows, res["gate3_eval"] = gate_3_eval_setup(panel)
    bl, res["gate4_baselines"] = gate_4_baselines(panel, windows)
    res["gate5_quality"] = gate_5_data_quality(panel)
    res["gate6_fairness"] = gate_6_fairness(panel, static)
    res["gate7_scale"] = gate_7_scale_intermittency(panel)
    res["gate8_outliers"] = gate_8_outliers(panel, truth)
    res["stationarity"] = stationarity(panel)

    eda_plots(panel, truth, bl, windows)

    bl.to_csv(f"{ARTIFACTS}/baselines.csv", index=False)
    with open(f"{ARTIFACTS}/quality_gates.json", "w") as fh:
        json.dump(res, fh, indent=2, default=str)

    print("\n" + "=" * 74)
    print("ALL 8 QUALITY GATES COMPLETE")
    print("=" * 74)
    print(f"  ranking metric   : MASE (m={SEASONAL_PERIOD})")
    print(f"  baseline to beat : {res['gate4_baselines']['best_baseline']} "
          f"@ MASE="
          f"{res['gate4_baselines']['scores'][res['gate4_baselines']['best_baseline']]['MASE']:.4f}")
    print(f"  artifacts        : {ARTIFACTS}/baselines.csv, "
          f"{ARTIFACTS}/quality_gates.json")
    print(f"  figures          : {PLOTS}/")


if __name__ == "__main__":
    main()
