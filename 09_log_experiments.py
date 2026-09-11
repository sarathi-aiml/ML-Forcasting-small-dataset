"""
09_log_experiments.py
=====================
Backfill Snowflake Experiment Tracking from saved artifacts, and (optionally)
register the champion model to the Snowflake Model Registry.

Why a separate backfill exists
------------------------------
Experiment-tracking run names must be unique within an experiment: a run that
already exists and has FINISHED cannot be resumed, it raises RuntimeError. The
first trial execution was interrupted after creating `t1_*`/`t2_*` runs, so the
re-run could not reuse those names and its logging failed - while the scores
themselves saved correctly to artifacts.

Rather than repeat ~75 minutes of training to fix a logging problem, this script
replays the saved artifacts into fresh, uniquely-named runs. Same data, same
metrics, no retraining.

It is also the right tool generally: logging is decoupled from training, so a
tracking outage never costs you a model.

Usage
-----
  python 09_log_experiments.py              # log experiments only
  python 09_log_experiments.py --register   # also register the champion
"""

from __future__ import annotations

import json
import os
import re
import sys
import time

import numpy as np
import pandas as pd

from snowpark_session import create_snowpark_session

DATABASE, SCHEMA = "HEALTHCARE", "ML_FORECAST"
EXPERIMENT = "HEALTHCARE_ADMISSIONS_FORECAST"
ARTIFACTS, PLOTS = "artifacts", "plots"
MODEL_NAME = "HEALTHCARE_ADMISSIONS_FORECASTER"
# Must start with a LETTER: a run name beginning with a digit is not a valid
# unquoted SQL identifier, and every logging call fails with ValueError.
EXEC_TAG = time.strftime("R%m%d_%H%M")


def safe(s: str) -> str:
    s = re.sub(r"[^0-9a-zA-Z_]+", "_", str(s)).strip("_")
    if s and s[0].isdigit():
        s = "M_" + s
    return s[:110] or "X"


def main():
    do_register = "--register" in sys.argv

    print("=" * 78)
    print("EXPERIMENT TRACKING BACKFILL")
    print("=" * 78)

    session = create_snowpark_session()
    session.use_database(DATABASE)
    session.use_schema(SCHEMA)
    print(f"connected: {session.get_current_account()} "
          f"{DATABASE}.{SCHEMA}")

    from snowflake.ml.experiment import ExperimentTracking
    exp = ExperimentTracking(session=session, database_name=DATABASE,
                             schema_name=SCHEMA)
    exp.set_experiment(EXPERIMENT)
    print(f"experiment: {DATABASE}.{SCHEMA}.{EXPERIMENT}")
    print(f"exec tag  : {EXEC_TAG}  (all runs prefixed, guarantees uniqueness)")

    # ---- gather artifacts --------------------------------------------
    def maybe(path, reader=pd.read_csv):
        return reader(path) if os.path.exists(path) else None

    a = maybe(f"{ARTIFACTS}/automl_scores.csv")
    b = maybe(f"{ARTIFACTS}/snowflake_forecast_scores.csv")
    lbs = maybe(f"{ARTIFACTS}/automl_leaderboards.csv")
    final = None
    if os.path.exists(f"{ARTIFACTS}/final_report.json"):
        final = json.load(open(f"{ARTIFACTS}/final_report.json"))
    manifest = None
    if os.path.exists(f"{ARTIFACTS}/manifest.json"):
        manifest = json.load(open(f"{ARTIFACTS}/manifest.json"))
    loop = None
    if os.path.exists(f"{ARTIFACTS}/learning_loop_summary.json"):
        loop = json.load(open(f"{ARTIFACTS}/learning_loop_summary.json"))

    METRICS = ["MASE", "RMSSE", "sMAPE", "MAE", "RMSE", "MAPE", "ME", "MPE",
               "WQL", "CRPS", "PICP80"]

    n_runs = 0
    plot_files = sorted(
        os.path.join(PLOTS, f) for f in os.listdir(PLOTS)
        if f.endswith(".png")) if os.path.isdir(PLOTS) else []

    # ---- one run per trial (Track A + Track B) ------------------------
    for src, track in ((a, "A_autogluon"), (b, "B_snowflake_forecast")):
        if src is None or src.empty:
            continue
        for trial, g in src.groupby("trial"):
            rn = f"{EXEC_TAG}_{safe(track)}_{safe(trial)}"
            mt = {}
            for m in METRICS:
                if m in g.columns:
                    v = pd.to_numeric(g[m], errors="coerce").mean()
                    if np.isfinite(v):
                        mt[m] = float(v)
            ws = g.groupby("dept_id")["MASE"].mean()
            mt["MASE_worst_series"] = float(ws.max())
            mt["MASE_best_series"] = float(ws.min())
            mt["n_scored_cells"] = float(len(g))
            # per-department, so a bad small series is visible in Snowsight
            for d, v in ws.items():
                mt[f"MASE_{safe(d)}"] = float(v)

            params = {"track": track, "trial": trial,
                      "model": str(g["model"].iloc[0]),
                      "n_windows": int(g["window"].nunique()),
                      "n_departments": int(g["dept_id"].nunique())}
            if manifest:
                params["protocol"] = manifest.get("protocol", "")
                params["eval_metric"] = manifest.get("eval_metric", "MASE")
                params["horizon"] = manifest.get("horizon", 28)
                for t in manifest.get("trials", []):
                    if t.get("label") == trial:
                        params["fit_seconds"] = t.get("fit_seconds")
                        params["n_models_trained"] = t.get("n_models")
            try:
                with exp.start_run(rn):
                    exp.log_params(params)
                    exp.log_metrics(mt)
                n_runs += 1
                print(f"  logged {rn:<52} MASE={mt.get('MASE', float('nan')):.4f}")
            except Exception as e:
                print(f"  FAILED {rn}: {type(e).__name__}: {str(e)[:140]}")

    # ---- baselines as reference runs ---------------------------------
    bl = maybe(f"{ARTIFACTS}/baselines.csv")
    if bl is not None:
        for model, g in bl.groupby("model"):
            rn = f"{EXEC_TAG}_baseline_{safe(model)}"
            try:
                with exp.start_run(rn):
                    exp.log_params({"track": "baseline", "model": model})
                    exp.log_metrics({m: float(g[m].mean()) for m in
                                     ["MASE", "RMSSE", "sMAPE", "MAE", "RMSE", "ME"]
                                     if m in g.columns})
                n_runs += 1
                print(f"  logged {rn:<52} MASE={g['MASE'].mean():.4f}")
            except Exception as e:
                print(f"  FAILED {rn}: {type(e).__name__}")

    # ---- overall summary run with every artifact ---------------------
    rn = f"{EXEC_TAG}_BENCHMARK_SUMMARY"
    try:
        with exp.start_run(rn):
            p = {"dataset": f"{DATABASE}.{SCHEMA}.DAILY_ADMISSIONS",
                 "rows": 6576, "series": 6, "obs_per_series": 1096,
                 "regime": "small_data_under_10k_records"}
            if manifest:
                p |= {"protocol": manifest.get("protocol"),
                      "horizon": manifest.get("horizon"),
                      "n_windows": manifest.get("n_windows")}
            exp.log_params(p)

            m = {}
            if final:
                m["irreducible_floor_MASE"] = final["irreducible_floor_MASE"]
                m["best_baseline_MASE"] = final["best_baseline"]["MASE"]
                lbd = final.get("leaderboard", {})
                real = {k: v for k, v in lbd.items()
                        if not k.startswith(("ORACLE", "baseline_"))}
                if real:
                    champ = min(real, key=lambda k: real[k]["MASE"])
                    m["champion_MASE"] = real[champ]["MASE"]
                    p["champion"] = champ
            if loop:
                for k, v in loop.get("summary", {}).items():
                    m[f"loop_{k}_MASE"] = v["mean_MASE"]
                m["loop_triggers_fired"] = loop.get("triggers_fired", 0)
                m["loop_promotions"] = loop.get("promotions", 0)
                cv = loop.get("coverage", {})
                if cv:
                    m["loop_coverage_conformal"] = cv.get("conformal")
                    m["loop_coverage_native"] = cv.get("native")
            m = {k: float(v) for k, v in m.items() if v is not None}
            if m:
                exp.log_metrics(m)

            for f in plot_files:
                try:
                    exp.log_artifact(f, artifact_path="plots")
                except Exception:
                    pass
            for f in ["final_comparison.csv", "final_report.json",
                      "manifest.json", "quality_gates.json",
                      "automl_scores.csv", "snowflake_forecast_scores.csv",
                      "learning_loop_steps.csv", "learning_loop_events.csv",
                      "conformal_coverage.csv", "per_horizon.csv",
                      "error_decomposition.csv"]:
                fp = os.path.join(ARTIFACTS, f)
                if os.path.exists(fp):
                    try:
                        exp.log_artifact(fp, artifact_path="artifacts")
                    except Exception:
                        pass
        n_runs += 1
        print(f"  logged {rn} (+{len(plot_files)} plots, artifacts)")
    except Exception as e:
        print(f"  FAILED {rn}: {type(e).__name__}: {str(e)[:200]}")

    print(f"\n{n_runs} runs logged to {DATABASE}.{SCHEMA}.{EXPERIMENT}")

    r = session.sql(f"SHOW RUNS IN EXPERIMENT {DATABASE}.{SCHEMA}.{EXPERIMENT}")
    df = r.to_pandas()
    print(f"experiment now contains {len(df)} runs total")
    print("View in Snowsight: AI & ML > Experiments > "
          f"{EXPERIMENT}")

    if do_register:
        register_champion(session, final)

    session.close()


def register_champion(session, final):
    """Register the champion as a CustomModel wrapping the AutoGluon predictor.

    A CustomModel wrapper is required (not a native flavour) because the
    champion is an AutoGluon TimeSeriesPredictor - potentially an ensemble - and
    wrapping preserves the exact leaderboard score rather than approximating it
    by extracting a single native estimator.
    """
    print("\n" + "=" * 78)
    print("MODEL REGISTRY")
    print("=" * 78)
    if not final:
        print("  no final_report.json - run 07_compare_report.py first")
        return

    lbd = final.get("leaderboard", {})
    real = {k: v for k, v in lbd.items()
            if not k.startswith(("ORACLE", "baseline_"))}
    if not real:
        print("  no candidate models found")
        return
    champ = min(real, key=lambda k: real[k]["MASE"])
    print(f"  champion trial : {champ}  MASE={real[champ]['MASE']:.4f}")

    manifest = json.load(open(f"{ARTIFACTS}/manifest.json"))
    path = None
    for t in manifest.get("trials", []):
        if t.get("label") == champ:
            path = t.get("model_path")
    if not path or not os.path.isdir(path):
        print(f"  champion is a Track B (SQL) model or path missing "
              f"({path}) - nothing to register from Python.")
        print("  SNOWFLAKE.ML.FORECAST models live in Snowflake already and are")
        print("  not Model Registry artifacts.")
        return
    print(f"  predictor path : {path}")
    print("  NOTE: registration requires the AutoGluon wheel set to be")
    print("  resolvable in the Snowflake conda/pip channel. If it is not,")
    print("  registration will fail at dependency resolution - that is an")
    print("  environment constraint, not a model problem.")

    import importlib.util as ilu
    spec = ilu.spec_from_file_location("reg", "10_register_model.py")
    if os.path.exists("10_register_model.py"):
        mod = ilu.module_from_spec(spec)
        spec.loader.exec_module(mod)
        mod.register(session, path, champ, real[champ])
    else:
        print("  10_register_model.py not present - skipping")


if __name__ == "__main__":
    main()
