"""
06_snowflake_forecast.py
========================
Track B of the benchmark: SNOWFLAKE.ML.FORECAST, evaluated on the IDENTICAL
rolling-origin windows, horizon, and metrics as Track A.

Fairness of the comparison
--------------------------
Everything that could bias the result is held constant:
  * same 6 expanding windows and 28-day horizon
  * same MASE denominator (in-sample seasonal-naive, m=7) from the same history
  * same metric implementations, imported from forecast_metrics.py
  * same known-future covariate set (the leakage contract from Gate 1 applies
    equally - past-only covariates are withheld from both tracks)

One genuine asymmetry, stated rather than hidden: Track A refits nothing across
windows (fit once, re-condition), whereas SNOWFLAKE.ML.FORECAST has no
"predict from new history without retraining" API - a model is bound to the
data it was created from. So Track B necessarily CREATES A FRESH MODEL PER
WINDOW, which means it sees more recent training data at each origin than the
fixed Track A model does.

That advantages Track B, and we report it that way. Step 8's learning loop
quantifies exactly how much refitting is worth, which is what makes this
asymmetry interpretable instead of a confound.

Variants
--------
  B1  multi-series, no exogenous     - FORECAST's own seasonality detection
  B2  multi-series with exogenous    - our 17 known-future engineered features

Metric coverage caveat
----------------------
FORECAST returns a point forecast plus ONE prediction interval, not a quantile
grid. So WQL and CRPS - which need multiple quantiles - are not computable for
Track B and are reported as NaN rather than silently approximated. Point
metrics, bias, and 80% interval coverage are fully comparable.

Outputs
-------
  artifacts/snowflake_forecast_scores.csv
  artifacts/snowflake_forecast_predictions.csv
"""

from __future__ import annotations

import json
import os
import time

import numpy as np
import pandas as pd
import holidays as _holidays
import importlib.util as _ilu

_spec = _ilu.spec_from_file_location(
    "features03", os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "03_features.py"))
feats = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(feats)

from forecast_metrics import (
    SEASONAL_PERIOD, evaluate_point, interval_score, make_backtest_windows,
    picp,
)
from snowpark_session import create_snowpark_session

DATABASE, SCHEMA = "HEALTHCARE", "ML_FORECAST"
HORIZON, N_WINDOWS = 28, 6
TARGET, ID_COL, TIME_COL = "ADMISSIONS", "DEPT_ID", "ADMIT_DATE"
ARTIFACTS = "artifacts"

# Feature table staged once in Snowflake so FORECAST reads the SAME engineered
# known-future covariates Track A used.
FEATURE_TABLE = "FORECAST_FEATURES"


def stage_features(session):
    """Materialise the engineered feature frame in Snowflake for FORECAST."""
    panel = session.table("DAILY_ADMISSIONS").to_pandas()
    static = session.table("DEPT_STATIC").to_pandas()
    panel[TIME_COL] = pd.to_datetime(panel[TIME_COL])
    yrs = sorted(panel[TIME_COL].dt.year.unique())
    hol = set(_holidays.US(years=list(yrs) + [max(yrs) + 1]).keys())

    frame, kf_cols, past_cols = feats.build_features(panel, hol)
    keep = [ID_COL, TIME_COL, TARGET] + kf_cols
    w = frame[keep].copy()
    w[TIME_COL] = pd.to_datetime(w[TIME_COL]).dt.strftime("%Y-%m-%d")

    session.write_pandas(w, FEATURE_TABLE, auto_create_table=True,
                         overwrite=True, quote_identifiers=False)
    cols = ", ".join(f"TO_DATE({c}) AS {c}" if c == TIME_COL else c
                     for c in w.columns)
    session.sql(f"CREATE OR REPLACE TABLE {FEATURE_TABLE} AS "
                f"SELECT {cols} FROM {FEATURE_TABLE}").collect()
    n = session.table(FEATURE_TABLE).count()
    print(f"  staged {FEATURE_TABLE}: {n} rows, {len(kf_cols)} known-future cols")
    return panel, frame, kf_cols, hol


def run_variant(session, window, kf_cols, use_exog: bool, tag: str):
    """Create a FORECAST model on history <= train_end, forecast the horizon."""
    model = f"FC_{tag}_W{window.window_id}"
    tr_end = window.train_end.strftime("%Y-%m-%d")
    ts_start = window.test_start.strftime("%Y-%m-%d")
    ts_end = window.test_end.strftime("%Y-%m-%d")

    exog = ", ".join(kf_cols) if use_exog else ""
    sel_exog = (", " + exog) if use_exog else ""

    session.sql(f"DROP SNOWFLAKE.ML.FORECAST IF EXISTS {model}").collect()

    create = f"""
    CREATE SNOWFLAKE.ML.FORECAST {model}(
      INPUT_DATA => TABLE(
        SELECT {ID_COL}, {TIME_COL}, {TARGET}{sel_exog}
        FROM {FEATURE_TABLE}
        WHERE {TIME_COL} <= '{tr_end}'
      ),
      SERIES_COLNAME => '{ID_COL}',
      TIMESTAMP_COLNAME => '{TIME_COL}',
      TARGET_COLNAME => '{TARGET}',
      CONFIG_OBJECT => {{'ON_ERROR': 'SKIP'}}
    )
    """
    t0 = time.time()
    session.sql(create).collect()
    fit_s = time.time() - t0

    if use_exog:
        call = f"""
        CALL {model}!FORECAST(
          INPUT_DATA => TABLE(
            SELECT {ID_COL}, {TIME_COL}{sel_exog}
            FROM {FEATURE_TABLE}
            WHERE {TIME_COL} BETWEEN '{ts_start}' AND '{ts_end}'
          ),
          SERIES_COLNAME => '{ID_COL}',
          TIMESTAMP_COLNAME => '{TIME_COL}',
          CONFIG_OBJECT => {{'prediction_interval': 0.8}}
        )
        """
    else:
        call = f"""
        CALL {model}!FORECAST(
          FORECASTING_PERIODS => {HORIZON},
          CONFIG_OBJECT => {{'prediction_interval': 0.8}}
        )
        """
    res = session.sql(call).to_pandas()
    session.sql(f"DROP SNOWFLAKE.ML.FORECAST IF EXISTS {model}").collect()
    return res, fit_s


def normalise(res: pd.DataFrame) -> pd.DataFrame:
    """FORECAST output -> (DEPT_ID, ADMIT_DATE, mean, lo, hi).

    Snowpark returns the CALL result with DOUBLE QUOTES EMBEDDED IN BOTH the
    column names and the string values - literally '"SERIES"' as a column and
    '"EMERGENCY"' as a value, not SERIES / EMERGENCY. Stripping both is
    required; matching on the raw names raises KeyError.
    """
    c = {x.strip('"').upper(): x for x in res.columns}
    missing = {"SERIES", "TS", "FORECAST", "LOWER_BOUND", "UPPER_BOUND"} - set(c)
    if missing:
        raise KeyError(f"FORECAST output missing {missing}; got {list(res.columns)}")

    out = pd.DataFrame({
        ID_COL: res[c["SERIES"]].astype(str).str.strip('"'),
        TIME_COL: pd.to_datetime(res[c["TS"]]).dt.tz_localize(None).dt.normalize(),
        "mean": res[c["FORECAST"]].astype(float),
        "lo": res[c["LOWER_BOUND"]].astype(float),
        "hi": res[c["UPPER_BOUND"]].astype(float),
    })
    # Admissions are non-negative counts; FORECAST can emit negative bounds
    # (visible in the probe output above), so clip rather than pass them through.
    out["mean"] = out["mean"].clip(lower=0)
    out["lo"] = out["lo"].clip(lower=0)
    return out


def score(pred, frame, window, tag):
    rows = []
    actual = frame[(frame[TIME_COL] >= window.test_start) &
                   (frame[TIME_COL] <= window.test_end)][[ID_COL, TIME_COL, TARGET]]
    hist = frame[frame[TIME_COL] <= window.train_end]
    m = actual.merge(pred, on=[ID_COL, TIME_COL], how="inner")
    if m.empty:
        return rows
    for dept, g in m.groupby(ID_COL, sort=False):
        g = g.sort_values(TIME_COL)
        y = g[TARGET].to_numpy(float)
        f = g["mean"].to_numpy(float)
        y_ins = hist.loc[hist[ID_COL] == dept, TARGET].to_numpy(float)
        pt = evaluate_point(y, f, y_ins, SEASONAL_PERIOD)
        rows.append({
            "trial": tag, "model": "SNOWFLAKE.ML.FORECAST",
            "window": window.window_id, "dept_id": dept, **pt,
            # FORECAST returns one interval, not a quantile grid -> WQL/CRPS
            # are genuinely not computable. Report NaN, do not fake them.
            "WQL": np.nan, "CRPS": np.nan,
            "PICP80": picp(y, g["lo"].to_numpy(float), g["hi"].to_numpy(float)),
            "MIS80": interval_score(y, g["lo"].to_numpy(float),
                                    g["hi"].to_numpy(float), alpha=0.20),
            "WIDTH80": float(np.mean(g["hi"] - g["lo"])),
        })
    return rows


def main():
    os.makedirs(ARTIFACTS, exist_ok=True)
    print("=" * 78)
    print("TRACK B - SNOWFLAKE.ML.FORECAST")
    print("=" * 78)

    session = create_snowpark_session()
    session.use_database(DATABASE)
    session.use_schema(SCHEMA)
    print(f"connected: {session.get_current_account()} "
          f"{session.get_current_database()}.{session.get_current_schema()}")

    panel, frame, kf_cols, hol = stage_features(session)
    windows = make_backtest_windows(panel[TIME_COL], HORIZON, N_WINDOWS)
    print(f"  windows: {N_WINDOWS} x {HORIZON}d "
          f"({windows[0].test_start.date()} .. {windows[-1].test_end.date()})")
    print("\n  NOTE: FORECAST has no re-condition-without-refit API, so a fresh")
    print("  model is created per window. It therefore sees MORE RECENT training")
    print("  data than the fixed Track A model. This favours Track B and is")
    print("  reported as such.")

    variants = [("sf_forecast_base", False,
                 "multi-series, FORECAST's own seasonality detection"),
                ("sf_forecast_exog", True,
                 f"multi-series + {len(kf_cols)} engineered known-future covariates")]

    all_rows, all_preds, timings = [], [], []
    for tag, use_exog, desc in variants:
        print("\n" + "-" * 78)
        print(f"VARIANT {tag}: {desc}")
        print("-" * 78)
        for w in windows:
            try:
                res, fit_s = run_variant(session, w, kf_cols, use_exog, tag)
                pred = normalise(res)
                pred["trial"] = tag
                pred["window"] = w.window_id
                all_preds.append(pred)
                rows = score(pred, frame, w, tag)
                all_rows += rows
                timings.append({"trial": tag, "window": w.window_id,
                                "fit_seconds": round(fit_s, 2)})
                mm = np.mean([r["MASE"] for r in rows]) if rows else float("nan")
                print(f"  W{w.window_id} train<={w.train_end.date()}  "
                      f"fit={fit_s:5.1f}s  rows={len(pred):3d}  "
                      f"mean MASE={mm:.4f}", flush=True)
                # Save incrementally. FORECAST model creation with exogenous
                # features is slow, and writing only at the end means an
                # interruption throws away every completed window.
                pd.DataFrame(all_rows).to_csv(
                    f"{ARTIFACTS}/snowflake_forecast_scores.csv", index=False)
                pd.concat(all_preds, ignore_index=True).to_csv(
                    f"{ARTIFACTS}/snowflake_forecast_predictions.csv", index=False)
            except Exception as e:
                print(f"  W{w.window_id} FAILED: {type(e).__name__}: "
                      f"{str(e)[:220]}", flush=True)

    if not all_rows:
        print("\nTRACK B PRODUCED NO RESULTS")
        session.close()
        return

    sc = pd.DataFrame(all_rows)
    pr = pd.concat(all_preds, ignore_index=True)
    sc.to_csv(f"{ARTIFACTS}/snowflake_forecast_scores.csv", index=False)
    pr.to_csv(f"{ARTIFACTS}/snowflake_forecast_predictions.csv", index=False)

    print("\n" + "=" * 78)
    print("TRACK B RESULTS")
    print("=" * 78)
    agg = (sc.groupby("trial")[["MASE", "RMSSE", "sMAPE", "MAE", "RMSE",
                                "ME", "PICP80", "MIS80", "WIDTH80"]]
             .mean().sort_values("MASE"))
    print(agg.round(4).to_string())

    ws = sc.groupby(["trial", "dept_id"])["MASE"].mean().groupby("trial").max()
    print("\nworst-series MASE:")
    print(ws.round(4).to_string())

    print("\nper-department MASE:")
    print(sc.pivot_table(index="trial", columns="dept_id", values="MASE",
                         aggfunc="mean").round(3).to_string())

    try:
        bl = pd.read_csv(f"{ARTIFACTS}/baselines.csv")
        bm = bl.groupby("model")["MASE"].mean().min()
        best = agg.index[0]
        print(f"\nbest FORECAST variant: {best} @ MASE={agg.iloc[0]['MASE']:.4f}")
        print(f"baseline to beat     : {bm:.4f} -> "
              f"{'BEATS' if agg.iloc[0]['MASE'] < bm else 'LOSES TO'} baseline")
    except FileNotFoundError:
        pass

    with open(f"{ARTIFACTS}/snowflake_forecast_summary.json", "w") as fh:
        json.dump({"scores": agg.round(4).to_dict(orient="index"),
                   "worst_series_MASE": ws.round(4).to_dict(),
                   "timings": timings,
                   "known_future_covariates": kf_cols,
                   "note": "fresh model per window; WQL/CRPS not computable "
                           "(single interval, no quantile grid)"},
                  fh, indent=2)
    print(f"\nartifacts written to {ARTIFACTS}/")
    session.close()


if __name__ == "__main__":
    main()
