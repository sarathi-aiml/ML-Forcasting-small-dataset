"""
05_trial_automl.py
==================
Track A of the benchmark: AutoGluon TimeSeriesPredictor model search on the
small healthcare panel, with per-trial Snowflake Experiment Tracking.

Evaluation protocol (read this before interpreting any score)
-------------------------------------------------------------
Each trial FITS ONCE on history up to the first backtest origin, then FORECASTS
all six windows by re-conditioning on progressively longer history WITHOUT
refitting.

Why fit-once rather than refit-per-window:

  * It is what actually happens in production. You train a model, deploy it,
    and it forecasts every week for months before anyone retrains it. Refitting
    at every origin measures something real teams rarely do, and it flatters the
    model by hiding parameter staleness.
  * It makes step 8 meaningful. The learning loop's whole question is "does
    triggered refitting beat leaving the model alone?" That comparison only
    exists if the baseline here is a fixed model.
  * Six refits x six trials at these time limits would not fit the budget, and
    cutting time_limit to afford it would degrade every model.

Model selection WITHIN a trial uses AutoGluon's internal temporal validation
(num_val_windows), which never sees the six test windows. The test windows are
scored once, for reporting, and never used to pick the champion.

Trial ladder
------------
  T1 classical      AutoETS / AutoARIMA / Theta / DOT / SeasonalNaive
                    Few parameters, hard to overfit - the small-data workhorses.
  T2 zero-shot      Chronos-Bolt. Pretrained on a huge corpus, ZERO training on
                    our 1,096 points. The single biggest small-data lever.
  T3 global         Cross-series LightGBM + DeepAR with covariates. Pools all
                    six departments: 6,576 training points instead of 1,096.
  T4 ensemble       Full model zoo + WeightedEnsemble. Variance reduction.
  T5 fine-tune      Chronos fine-tuned on our panel - does adapting the
                    foundation model beat using it cold?
  T6 HPO            Tuned sweep on whichever family led.

Outputs
-------
  artifacts/automl_predictions.parquet   per-window point + quantile forecasts
  artifacts/automl_scores.csv            full metric suite per trial/model
  artifacts/automl_leaderboards.csv      AutoGluon leaderboards
  artifacts/manifest.json                config, timings, champion
  plots/*.png                            diagnostics
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import holidays as _holidays

warnings.filterwarnings("ignore")

from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import importlib.util as _ilu

_spec = _ilu.spec_from_file_location(
    "features03", os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "03_features.py"))
feats = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(feats)

from forecast_metrics import (
    SEASONAL_PERIOD, ConformalCalibrator, baseline_forecasts,
    evaluate_point, evaluate_probabilistic, make_backtest_windows,
    per_horizon_errors,
)
from snowpark_session import create_snowpark_session

# --------------------------------------------------------------------------
DATABASE, SCHEMA = "HEALTHCARE", "ML_FORECAST"
EXPERIMENT = "HEALTHCARE_ADMISSIONS_FORECAST"
HORIZON, N_WINDOWS = 28, 6
QUANTILES = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
TARGET, ID_COL, TIME_COL = "ADMISSIONS", "DEPT_ID", "ADMIT_DATE"

TOTAL_BUDGET_S = 90 * 60          # user-confirmed 90 minutes end-to-end
RESERVE_S = 15 * 60               # reserved for steps 6-9 (compare, loop, registry)
TRIAL_BUDGET_S = TOTAL_BUDGET_S - RESERVE_S

PLOTS, ARTIFACTS = "plots", "artifacts"
MODEL_DIR = "ag_models"

# Unique tag per execution. Experiment-tracking run names must be unique: a run
# that already exists and has finished CANNOT be resumed, so re-running the
# script with fixed names like "t1_summary" fails with a RuntimeError. Stamping
# the execution makes every attempt land in fresh runs while staying grouped
# under one experiment.
# Must start with a LETTER: a run name beginning with a digit is not a valid
# unquoted SQL identifier and logging fails with ValueError.
EXEC_TAG = time.strftime("R%m%d_%H%M")


def safe_name(s: str) -> str:
    """Sanitise a model name into a valid unquoted Snowflake identifier.

    AutoGluon emits names like 'Chronos[bolt_small]' and 'WeightedEnsemble_L2';
    brackets and dots are illegal in unquoted identifiers and would break
    experiment-tracking run creation.
    """
    s = re.sub(r"[^0-9a-zA-Z_]+", "_", s).strip("_")
    if s and s[0].isdigit():
        s = "M_" + s
    return s[:120] or "MODEL"


# ==========================================================================
# Data preparation
# ==========================================================================
def load_and_prepare():
    session = create_snowpark_session()
    session.use_database(DATABASE)
    session.use_schema(SCHEMA)
    panel = session.table("DAILY_ADMISSIONS").to_pandas()
    static = session.table("DEPT_STATIC").to_pandas()
    panel[TIME_COL] = pd.to_datetime(panel[TIME_COL])

    yrs = sorted(panel[TIME_COL].dt.year.unique())
    hol = set(_holidays.US(years=list(yrs) + [max(yrs) + 1]).keys())

    frame, kf_cols, past_cols = feats.build_features(panel, hol)
    stat = feats.build_static(static)

    keep = [ID_COL, TIME_COL, TARGET] + kf_cols + past_cols
    frame = frame[keep].copy()

    tsdf = TimeSeriesDataFrame.from_data_frame(
        frame, id_column=ID_COL, timestamp_column=TIME_COL,
        static_features_df=stat.reset_index(),
    )
    windows = make_backtest_windows(panel[TIME_COL], HORIZON, N_WINDOWS)
    return session, panel, frame, tsdf, stat, kf_cols, past_cols, windows, hol


# ==========================================================================
# Forecast one window from a fitted predictor
# ==========================================================================
def forecast_window(predictor, frame, kf_cols, stat, window):
    """Re-condition the FIXED model on history <= window.train_end, predict 28d."""
    hist = frame[frame[TIME_COL] <= window.train_end]
    hist_ts = TimeSeriesDataFrame.from_data_frame(
        hist, id_column=ID_COL, timestamp_column=TIME_COL,
        static_features_df=stat.reset_index(),
    )
    # Known covariates for the horizon. These come from the calendar, so taking
    # them from the panel is identical to recomputing them - proven by TEST 1
    # in the feature self-test.
    fut = frame[(frame[TIME_COL] >= window.test_start) &
                (frame[TIME_COL] <= window.test_end)][[ID_COL, TIME_COL] + kf_cols]
    kc = TimeSeriesDataFrame.from_data_frame(
        fut, id_column=ID_COL, timestamp_column=TIME_COL)

    preds = predictor.predict(hist_ts, known_covariates=kc)
    return preds.reset_index()


def score_predictions(pred_df, frame, window, model_label, trial_label):
    """Full metric suite for one model on one window, per department + pooled."""
    rows = []
    actual = frame[(frame[TIME_COL] >= window.test_start) &
                   (frame[TIME_COL] <= window.test_end)][[ID_COL, TIME_COL, TARGET]]
    hist = frame[frame[TIME_COL] <= window.train_end]

    p = pred_df.rename(columns={"item_id": ID_COL, "timestamp": TIME_COL})
    m = actual.merge(p, on=[ID_COL, TIME_COL], how="inner")

    for dept, g in m.groupby(ID_COL, sort=False):
        g = g.sort_values(TIME_COL)
        y = g[TARGET].to_numpy(float)
        f = g["mean"].to_numpy(float)
        y_ins = hist.loc[hist[ID_COL] == dept, TARGET].to_numpy(float)

        pt = evaluate_point(y, f, y_ins, SEASONAL_PERIOD)
        qp = {q: g[str(q)].to_numpy(float) for q in QUANTILES if str(q) in g}
        iv = {}
        if "0.1" in g and "0.9" in g:
            iv[0.80] = (g["0.1"].to_numpy(float), g["0.9"].to_numpy(float))
        pr = evaluate_probabilistic(y, qp, iv)

        rows.append({"trial": trial_label, "model": model_label,
                     "window": window.window_id, "dept_id": dept,
                     **pt, **pr})
    return rows


# ==========================================================================
# Trial definitions
# ==========================================================================
def trial_specs(kf_cols):
    """Ladder of trials. `hyperparameters=None` means use the preset."""
    return [
        dict(
            n=1, label="classical",
            desc="Statistical models: few parameters, strong on short histories",
            time_limit=480, presets=None,
            hyperparameters={
                "SeasonalNaive": {},
                "AutoETS": {},
                "AutoARIMA": {},
                "Theta": {},
                "DynamicOptimizedTheta": {},
                "AutoCES": {},
            },
            use_covariates=False,
        ),
        dict(
            n=2, label="zeroshot_chronos",
            desc="Chronos-Bolt zero-shot: pretrained, ZERO fitting on our data",
            time_limit=420, presets=None,
            hyperparameters={
                "Chronos": [
                    {"model_path": "bolt_small", "ag_args": {"name_suffix": "_bolt_small"}},
                    {"model_path": "bolt_base", "ag_args": {"name_suffix": "_bolt_base"}},
                ],
            },
            use_covariates=False,
        ),
        dict(
            n=3, label="global_covariates",
            desc="Cross-series pooling: 6,576 training points instead of 1,096",
            time_limit=900, presets=None,
            hyperparameters={
                # Heavily regularised: shallow trees, strong L2, low learning
                # rate. At ~1.1k obs/series the default depth overfits.
                "RecursiveTabular": {
                    "model_name": "GBM",
                    "model_hyperparameters": {
                        "num_leaves": 16, "learning_rate": 0.04,
                        "min_data_in_leaf": 30, "lambda_l2": 5.0,
                        "feature_fraction": 0.8, "num_boost_round": 400,
                    },
                },
                "DirectTabular": {
                    "model_name": "GBM",
                    "model_hyperparameters": {
                        "num_leaves": 16, "learning_rate": 0.04,
                        "min_data_in_leaf": 30, "lambda_l2": 5.0,
                    },
                },
                "PerStepTabular": {},
                "DeepAR": {"num_layers": 2, "hidden_size": 40,
                           "max_epochs": 60, "dropout_rate": 0.15},
            },
            use_covariates=True,
        ),
        dict(
            n=4, label="ensemble",
            desc="Full zoo + WeightedEnsemble - variance reduction via diversity",
            time_limit=1000, presets="medium_quality",
            hyperparameters=None,
            use_covariates=True,
        ),
        dict(
            n=5, label="chronos_finetune",
            desc="Fine-tune Chronos on our panel vs using it cold",
            time_limit=900, presets=None,
            hyperparameters={
                "Chronos": {
                    "model_path": "bolt_small", "fine_tune": True,
                    "ag_args": {"name_suffix": "_ft"},
                },
            },
            use_covariates=False,
        ),
        dict(
            n=6, label="high_quality",
            desc="Longer search with stacking/HPO on the strongest families",
            time_limit=900, presets="high_quality",
            hyperparameters=None,
            use_covariates=True,
        ),
    ]


# ==========================================================================
def main():
    os.makedirs(PLOTS, exist_ok=True)
    os.makedirs(ARTIFACTS, exist_ok=True)
    t0 = time.time()

    def elapsed(): return time.time() - t0
    def remaining(): return TRIAL_BUDGET_S - elapsed()

    print("=" * 78)
    print("TRACK A - AUTOGLUON TIMESERIES MODEL SEARCH")
    print("=" * 78)
    print(f"budget: {TOTAL_BUDGET_S//60} min total, "
          f"{TRIAL_BUDGET_S//60} min for trials, "
          f"{RESERVE_S//60} min reserved downstream")

    (session, panel, frame, tsdf, stat, kf_cols, past_cols,
     windows, hol) = load_and_prepare()

    print(f"\npanel        : {len(panel)} rows, {panel[ID_COL].nunique()} series, "
          f"{len(panel)//panel[ID_COL].nunique()} obs/series")
    print(f"known-future : {len(kf_cols)} covariates")
    print(f"past-only    : {len(past_cols)} covariates")
    print(f"static       : {stat.shape[1]} features")
    print(f"windows      : {N_WINDOWS} x {HORIZON}d, "
          f"{windows[0].test_start.date()} .. {windows[-1].test_end.date()}")

    fit_cut = windows[0].train_end
    train_frame = frame[frame[TIME_COL] <= fit_cut]
    print(f"fit data     : <= {fit_cut.date()} "
          f"({len(train_frame)//panel[ID_COL].nunique()} obs/series)")

    # ---- experiment tracking ------------------------------------------
    from snowflake.ml.experiment import ExperimentTracking
    exp = ExperimentTracking(session=session, database_name=DATABASE,
                             schema_name=SCHEMA)
    exp.set_experiment(EXPERIMENT)
    print(f"experiment   : {DATABASE}.{SCHEMA}.{EXPERIMENT}")

    # ---- baselines for reference --------------------------------------
    bl = pd.read_csv(f"{ARTIFACTS}/baselines.csv")
    base_mase = bl.groupby("model")["MASE"].mean().sort_values()
    best_base, best_base_mase = base_mase.index[0], base_mase.iloc[0]
    print(f"baseline     : {best_base} @ MASE={best_base_mase:.4f}")

    all_scores, all_lb, all_preds = [], [], []
    manifest = {
        "experiment": f"{DATABASE}.{SCHEMA}.{EXPERIMENT}",
        "protocol": "fit_once_rolling_predict",
        "horizon": HORIZON, "n_windows": N_WINDOWS,
        "quantiles": QUANTILES, "seasonal_period": SEASONAL_PERIOD,
        "eval_metric": "MASE",
        "budget_s": TOTAL_BUDGET_S, "trial_budget_s": TRIAL_BUDGET_S,
        "baseline": {"model": best_base, "MASE": float(best_base_mase)},
        "trials": [],
    }
    exp_url = None

    for spec in trial_specs(kf_cols):
        # ---- budget gate ---------------------------------------------
        need = spec["time_limit"] + 120          # +120s for the 6 predicts
        if remaining() < need:
            print(f"\n### SKIP T{spec['n']} ({spec['label']}) - "
                  f"needs ~{need}s, only {remaining():.0f}s remain")
            manifest["trials"].append({"n": spec["n"], "label": spec["label"],
                                       "status": "skipped_budget"})
            continue

        print("\n" + "=" * 78)
        print(f"TRIAL {spec['n']}: {spec['label']}")
        print(f"  {spec['desc']}")
        print(f"  time_limit={spec['time_limit']}s  presets={spec['presets']}  "
              f"covariates={spec['use_covariates']}")
        print(f"  elapsed={elapsed():.0f}s  remaining={remaining():.0f}s")
        print("=" * 78)
        ts = time.time()

        kc_names = kf_cols if spec["use_covariates"] else None
        # When covariates are off, strip them so the statistical models are not
        # handed columns they cannot use.
        cols = [ID_COL, TIME_COL, TARGET] + (
            kf_cols + past_cols if spec["use_covariates"] else [])
        tf = train_frame[cols]
        tr_ts = TimeSeriesDataFrame.from_data_frame(
            tf, id_column=ID_COL, timestamp_column=TIME_COL,
            static_features_df=stat.reset_index() if spec["use_covariates"] else None,
        )

        path = f"{MODEL_DIR}/t{spec['n']}_{spec['label']}"
        predictor = TimeSeriesPredictor(
            prediction_length=HORIZON, target=TARGET, freq="D",
            eval_metric="MASE", quantile_levels=QUANTILES,
            known_covariates_names=kc_names, path=path, verbosity=1,
        )
        try:
            predictor.fit(
                tr_ts, time_limit=spec["time_limit"],
                presets=spec["presets"],
                hyperparameters=spec["hyperparameters"],
                # 3 internal temporal validation windows for model ranking.
                # These sit inside the fit data and never touch the 6 test windows.
                num_val_windows=3, val_step_size=HORIZON,
                enable_ensemble=True,
            )
        except Exception as e:
            print(f"  !! TRIAL FAILED: {type(e).__name__}: {e}")
            manifest["trials"].append({"n": spec["n"], "label": spec["label"],
                                       "status": "failed", "error": str(e)[:400]})
            continue

        fit_s = time.time() - ts
        try:
            lb = predictor.leaderboard(silent=True)
        except TypeError:
            lb = predictor.leaderboard()
        lb["trial"] = spec["label"]
        all_lb.append(lb)
        print(f"\n  fit complete in {fit_s:.0f}s, "
              f"{len(lb)} models trained")
        print(lb[["model", "score_val", "fit_time_marginal"]].head(12).to_string(index=False))

        # ---- forecast every window with the FIXED model ---------------
        model_names = list(lb["model"])
        best_model = predictor.model_best
        print(f"\n  AutoGluon champion (by internal val): {best_model}")
        print(f"  forecasting {N_WINDOWS} windows (no refit)...")

        trial_rows = []
        for w in windows:
            try:
                pr = forecast_window(predictor, frame, kf_cols, stat, w)
            except Exception as e:
                print(f"    W{w.window_id} FAILED: {type(e).__name__}: {e}")
                continue
            pr["trial"] = spec["label"]
            pr["model"] = best_model
            pr["window"] = w.window_id
            all_preds.append(pr)
            trial_rows += score_predictions(pr, frame, w, best_model,
                                            spec["label"])

        if not trial_rows:
            print("  !! no windows scored - skipping trial")
            manifest["trials"].append({"n": spec["n"], "label": spec["label"],
                                       "status": "no_predictions"})
            continue

        sc = pd.DataFrame(trial_rows)
        all_scores.append(sc)

        mase = sc["MASE"].mean()
        worst = sc.groupby("dept_id")["MASE"].mean().max()
        worst_dept = sc.groupby("dept_id")["MASE"].mean().idxmax()

        print(f"\n  TEST results ({N_WINDOWS} windows x 6 depts = {len(sc)} scores)")
        print(f"    mean MASE        : {mase:.4f}   "
              f"(baseline {best_base_mase:.4f}, "
              f"{'BEATS' if mase < best_base_mase else 'LOSES TO'} baseline)")
        print(f"    worst-series MASE: {worst:.4f}  ({worst_dept})")
        agg = sc[["MASE", "RMSSE", "sMAPE", "MAE", "ME", "WQL", "CRPS",
                  "PICP80"]].mean()
        print("    " + "  ".join(f"{k}={v:.4f}" for k, v in agg.items()))

        # ---- log to experiment tracking ------------------------------
        for _, row in lb.iterrows():
            rn = f"{EXEC_TAG}_t{spec['n']}_{safe_name(str(row['model']))}"
            try:
                with exp.start_run(rn):
                    exp.log_params({
                        "trial": spec["label"], "model": str(row["model"]),
                        "presets": str(spec["presets"]),
                        "time_limit": spec["time_limit"],
                        "use_covariates": spec["use_covariates"],
                        "prediction_length": HORIZON,
                        "protocol": "fit_once_rolling_predict",
                    })
                    mt = {"score_val": float(row["score_val"]),
                          "fit_time_marginal": float(row.get("fit_time_marginal", 0) or 0)}
                    if str(row["model"]) == str(best_model):
                        mt.update({f"test_{k}": float(v) for k, v in agg.items()})
                        mt["test_MASE_worst_series"] = float(worst)
                        mt["baseline_MASE"] = float(best_base_mase)
                    exp.log_metrics(mt)
            except Exception as e:
                print(f"    (ET log warning for {rn}: {type(e).__name__})")

        # summary run with artifacts
        sum_run = f"{EXEC_TAG}_t{spec['n']}_summary"
        lb_path = f"{ARTIFACTS}/lb_t{spec['n']}.csv"
        lb.to_csv(lb_path, index=False)
        try:
            with exp.start_run(sum_run):
                exp.log_params({"trial": spec["label"], "desc": spec["desc"],
                                "n_models": len(lb),
                                "champion": str(best_model)})
                exp.log_metrics({f"test_{k}": float(v) for k, v in agg.items()}
                                | {"test_MASE_worst_series": float(worst),
                                   "fit_seconds": float(fit_s)})
                exp.log_artifact(lb_path, artifact_path="leaderboards")
        except Exception as e:
            print(f"    (ET summary warning: {type(e).__name__}: {e})")

        if exp_url is None:
            exp_url = (f"https://app.snowflake.com/  ->  AI & ML > Experiments "
                       f"> {EXPERIMENT}")

        el = time.time() - ts
        manifest["trials"].append({
            "n": spec["n"], "label": spec["label"], "status": "ok",
            "champion": str(best_model), "n_models": int(len(lb)),
            "fit_seconds": round(fit_s, 1), "trial_seconds": round(el, 1),
            "test_mean_MASE": round(float(mase), 4),
            "test_worst_series_MASE": round(float(worst), 4),
            "test_metrics": {k: round(float(v), 4) for k, v in agg.items()},
            "model_path": path,
        })
        with open(f"{ARTIFACTS}/manifest.json", "w") as fh:
            json.dump(manifest, fh, indent=2)

        print(f"\n  trial time {el:.0f}s | elapsed {elapsed():.0f}s | "
              f"remaining {remaining():.0f}s")

    # ==================================================================
    if not all_scores:
        print("\nNO TRIALS SUCCEEDED - aborting")
        session.close()
        return

    scores = pd.concat(all_scores, ignore_index=True)
    preds = pd.concat(all_preds, ignore_index=True)
    lbs = pd.concat(all_lb, ignore_index=True)

    scores.to_csv(f"{ARTIFACTS}/automl_scores.csv", index=False)
    lbs.to_csv(f"{ARTIFACTS}/automl_leaderboards.csv", index=False)
    preds.to_parquet(f"{ARTIFACTS}/automl_predictions.parquet", index=False)

    # ---- cross-trial ranking ----------------------------------------
    xt = (scores.groupby("trial")
                .agg(mean_MASE=("MASE", "mean"),
                     worst_series_MASE=("MASE", lambda s: s.mean()),
                     RMSSE=("RMSSE", "mean"), sMAPE=("sMAPE", "mean"),
                     MAE=("MAE", "mean"), ME=("ME", "mean"),
                     WQL=("WQL", "mean"), CRPS=("CRPS", "mean"),
                     PICP80=("PICP80", "mean"))
                .sort_values("mean_MASE"))
    ws = scores.groupby(["trial", "dept_id"])["MASE"].mean().groupby("trial").max()
    xt["worst_series_MASE"] = ws

    print("\n" + "=" * 78)
    print("CROSS-TRIAL LEADERBOARD (test windows)")
    print("=" * 78)
    print(xt.round(4).to_string())
    print(f"\nbaseline {best_base}: MASE={best_base_mase:.4f}")

    champ_trial = xt.index[0]
    champ_mase = xt.iloc[0]["mean_MASE"]
    manifest["champion"] = {
        "trial": champ_trial, "mean_MASE": round(float(champ_mase), 4),
        "beats_baseline": bool(champ_mase < best_base_mase),
        "improvement_pct": round(
            float((best_base_mase - champ_mase) / best_base_mase * 100), 2),
    }
    manifest["cross_trial"] = xt.round(4).to_dict(orient="index")
    manifest["total_elapsed_s"] = round(elapsed(), 1)

    # ---- conformal calibration --------------------------------------
    # Calibrate on the EARLIEST windows, measure coverage on the LATER ones.
    # Calibrating and scoring on the same window would be circular.
    print("\n" + "-" * 78)
    print("CONFORMAL PREDICTION INTERVALS (champion trial)")
    print("-" * 78)
    cp = preds[preds["trial"] == champ_trial].rename(
        columns={"item_id": ID_COL, "timestamp": TIME_COL})
    act = frame[[ID_COL, TIME_COL, TARGET]]
    cp = cp.merge(act, on=[ID_COL, TIME_COL], how="inner")
    cp["h"] = cp.groupby([ID_COL, "window"])[TIME_COL].rank().astype(int)
    cp["resid"] = cp[TARGET] - cp["mean"]

    cal = cp[cp["window"] <= 2]
    tst = cp[cp["window"] >= 3]
    conf_rows = []
    if len(cal) and len(tst):
        for dept, gc in cal.groupby(ID_COL):
            cc = ConformalCalibrator().fit(gc["resid"].to_numpy(),
                                           gc["h"].to_numpy(), level=0.80)
            gt = tst[tst[ID_COL] == dept]
            lo, hi = cc.intervals(gt["mean"].to_numpy(), gt["h"].to_numpy())
            y = gt[TARGET].to_numpy(float)
            native_cov = float(((y >= gt["0.1"]) & (y <= gt["0.9"])).mean()) \
                if "0.1" in gt else float("nan")
            conf_rows.append({
                "dept_id": dept,
                "conformal_PICP80": float(np.mean((y >= lo) & (y <= hi))),
                "native_PICP80": native_cov,
                "conformal_width": float(np.mean(hi - lo)),
                "native_width": float(np.mean(gt["0.9"] - gt["0.1"]))
                if "0.1" in gt else float("nan"),
            })
        cdf = pd.DataFrame(conf_rows)
        print("nominal coverage 80% | calibrated on windows 1-2, "
              "measured on windows 3-6")
        print(cdf.round(4).to_string(index=False))
        print(f"\n  conformal mean coverage : {cdf['conformal_PICP80'].mean():.4f}")
        print(f"  native    mean coverage : {cdf['native_PICP80'].mean():.4f}")
        print("  Closer to 0.80 is better. Conformal is distribution-free; the")
        print("  native quantiles rely on the model's likelihood assumptions,")
        print("  which are shakier at this sample size.")
        cdf.to_csv(f"{ARTIFACTS}/conformal_coverage.csv", index=False)
        manifest["conformal"] = cdf.round(4).to_dict(orient="records")

    # ---- per-horizon degradation ------------------------------------
    print("\n" + "-" * 78)
    print("ERROR BY FORECAST HORIZON (champion trial)")
    print("-" * 78)
    ph = per_horizon_errors(cp[TARGET].to_numpy(), cp["mean"].to_numpy(),
                            cp["h"].to_numpy())
    print(ph.head(28).to_string(index=False))
    d1 = ph.loc[ph["h"] == 1, "MAE"].iloc[0]
    d28 = ph.loc[ph["h"] == 28, "MAE"].iloc[0]
    print(f"\n  day-1 MAE {d1:.3f} -> day-28 MAE {d28:.3f} "
          f"({d28/d1:.2f}x degradation)")
    print("  A rising curve is expected and healthy. A FLAT curve would suggest")
    print("  the model ignores recent history and predicts climatology only.")
    ph.to_csv(f"{ARTIFACTS}/per_horizon.csv", index=False)

    # ---- plots -------------------------------------------------------
    make_plots(scores, xt, cp, ph, best_base_mase, champ_trial)

    with open(f"{ARTIFACTS}/manifest.json", "w") as fh:
        json.dump(manifest, fh, indent=2, default=str)

    print("\n" + "=" * 78)
    print("TRACK A COMPLETE")
    print("=" * 78)
    print(f"  champion trial : {champ_trial}  MASE={champ_mase:.4f}")
    print(f"  vs baseline    : {best_base_mase:.4f} "
          f"({manifest['champion']['improvement_pct']:+.2f}%)")
    print(f"  total elapsed  : {elapsed():.0f}s / {TRIAL_BUDGET_S}s trial budget")
    print(f"  experiment     : {DATABASE}.{SCHEMA}.{EXPERIMENT}")
    print(f"  artifacts      : {ARTIFACTS}/")
    session.close()


def make_plots(scores, xt, cp, ph, base_mase, champ_trial):
    # trial comparison
    fig, ax = plt.subplots(figsize=(10, 4.5))
    x = np.arange(len(xt))
    ax.bar(x - 0.2, xt["mean_MASE"], 0.4, label="mean MASE", color="#1f4e79")
    ax.bar(x + 0.2, xt["worst_series_MASE"], 0.4, label="worst-series MASE",
           color="#d1495b")
    ax.axhline(base_mase, color="green", ls="--",
               label=f"baseline {base_mase:.3f}")
    ax.axhline(1.0, color="grey", ls=":", label="seasonal naive = 1.0")
    ax.set_xticks(x)
    ax.set_xticklabels(xt.index, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("MASE (lower better)")
    ax.set_title("Trial comparison - mean vs worst-series MASE")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25, axis="y")
    plt.tight_layout()
    plt.savefig(f"{PLOTS}/07_trial_comparison.png", dpi=120)
    plt.close()

    # per-dept heatmap
    piv = scores.pivot_table(index="trial", columns="dept_id", values="MASE",
                             aggfunc="mean")
    fig, ax = plt.subplots(figsize=(10, 0.6 * len(piv) + 2.2))
    im = ax.imshow(piv.values, cmap="RdYlGn_r", aspect="auto")
    ax.set_xticks(range(len(piv.columns)))
    ax.set_xticklabels(piv.columns, rotation=30, ha="right", fontsize=8)
    ax.set_yticks(range(len(piv.index)))
    ax.set_yticklabels(piv.index, fontsize=8)
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            ax.text(j, i, f"{piv.values[i,j]:.2f}", ha="center", va="center",
                    fontsize=7)
    plt.colorbar(im, label="MASE")
    ax.set_title("MASE by trial and department")
    plt.tight_layout()
    plt.savefig(f"{PLOTS}/08_trial_dept_heatmap.png", dpi=120)
    plt.close()

    # horizon curve
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(ph["h"], ph["MAE"], marker="o", color="#1f4e79", label="MAE")
    ax.axhline(0, color="k", lw=0.5)
    ax2 = ax.twinx()
    ax2.plot(ph["h"], ph["ME"], marker="s", color="#d1495b", alpha=0.7,
             label="ME (bias)")
    ax2.axhline(0, color="#d1495b", ls=":", lw=1)
    ax.set_xlabel("forecast horizon (days ahead)")
    ax.set_ylabel("MAE")
    ax2.set_ylabel("ME (bias)")
    ax.set_title(f"Error growth by horizon - {champ_trial}")
    ax.grid(alpha=0.25)
    ax.legend(loc="upper left", fontsize=8)
    ax2.legend(loc="lower right", fontsize=8)
    plt.tight_layout()
    plt.savefig(f"{PLOTS}/09_horizon_error.png", dpi=120)
    plt.close()

    # forecast vs actual, last window
    lastw = cp["window"].max()
    g = cp[cp["window"] == lastw]
    depts = sorted(g[ID_COL].unique())
    fig, axes = plt.subplots(len(depts), 1, figsize=(12, 2.0 * len(depts)),
                             sharex=True)
    for ax, d in zip(np.atleast_1d(axes), depts):
        gg = g[g[ID_COL] == d].sort_values(TIME_COL)
        ax.plot(gg[TIME_COL], gg[TARGET], "o-", ms=3, lw=1,
                color="black", label="actual")
        ax.plot(gg[TIME_COL], gg["mean"], lw=1.6, color="#1f4e79",
                label="forecast")
        if "0.1" in gg and "0.9" in gg:
            ax.fill_between(gg[TIME_COL], gg["0.1"], gg["0.9"], alpha=0.22,
                            color="#1f4e79", label="80% interval")
        ax.set_ylabel(d, fontsize=8)
        ax.legend(fontsize=6, loc="upper left", ncol=3)
    plt.suptitle(f"{champ_trial}: forecast vs actual, window {lastw}",
                 fontsize=10)
    plt.tight_layout()
    plt.savefig(f"{PLOTS}/10_forecast_vs_actual.png", dpi=120)
    plt.close()
    print(f"\n  wrote 4 figures to {PLOTS}/")


if __name__ == "__main__":
    main()
