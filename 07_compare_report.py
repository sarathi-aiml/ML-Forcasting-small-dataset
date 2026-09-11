"""
07_compare_report.py
====================
Head-to-head comparison of Track A (AutoGluon AutoML) and Track B
(SNOWFLAKE.ML.FORECAST) against the naive baselines, plus two analyses that are
only possible because the data is synthetic.

1. The irreducible-error floor
------------------------------
Because we generated the data we know the true conditional mean mu(t). Every
forecast error decomposes exactly:

    y - f  =  (y - mu)  +  (mu - f)
              \______/     \______/
             irreducible    model
             NegBinomial    error
             noise

A perfect model predicts mu and STILL scores MASE > 0, because the counts are
random around mu. So "MASE 0.75" is meaningless on its own - the real question
is how close it is to the floor. This computes the floor and reports the
remaining headroom, which is the only honest way to say whether a model is
nearly optimal or still leaving signal on the table.

2. Paired significance testing
------------------------------
Two models scoring 0.780 and 0.785 across 36 (window x department) cells are
almost certainly indistinguishable. Declaring a champion on a 0.005 gap is
noise-chasing. We run a Wilcoxon signed-rank test on the PAIRED per-cell MASE
differences - paired because both models forecast the identical cells, and
signed-rank rather than a t-test because MASE differences are skewed and the
sample is small.

Outputs
-------
  artifacts/final_comparison.csv
  artifacts/final_report.json
  plots/13_final_comparison.png
  plots/14_error_decomposition.png
"""

from __future__ import annotations

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from forecast_metrics import SEASONAL_PERIOD, evaluate_point, make_backtest_windows
from snowpark_session import create_snowpark_session

ARTIFACTS, PLOTS = "artifacts", "plots"
DATABASE, SCHEMA = "HEALTHCARE", "ML_FORECAST"
TARGET, ID_COL, TIME_COL = "ADMISSIONS", "DEPT_ID", "ADMIT_DATE"
HORIZON, N_WINDOWS = 28, 6

METRICS = ["MASE", "RMSSE", "sMAPE", "MAE", "RMSE", "MAPE", "ME", "MPE",
           "WQL", "CRPS", "PICP80"]


def load_all():
    a = pd.read_csv(f"{ARTIFACTS}/automl_scores.csv") \
        if os.path.exists(f"{ARTIFACTS}/automl_scores.csv") else pd.DataFrame()
    b = pd.read_csv(f"{ARTIFACTS}/snowflake_forecast_scores.csv") \
        if os.path.exists(f"{ARTIFACTS}/snowflake_forecast_scores.csv") \
        else pd.DataFrame()
    bl = pd.read_csv(f"{ARTIFACTS}/baselines.csv")
    bl["trial"] = "baseline_" + bl["model"]
    return a, b, bl


def noise_floor():
    """MASE achievable by a model that predicts the TRUE mean mu exactly."""
    session = create_snowpark_session()
    session.use_database(DATABASE)
    session.use_schema(SCHEMA)
    panel = session.table("DAILY_ADMISSIONS").to_pandas()
    truth = session.table("GROUND_TRUTH_COMPONENTS").to_pandas()
    session.close()
    panel[TIME_COL] = pd.to_datetime(panel[TIME_COL])
    truth["ADMIT_DATE"] = pd.to_datetime(truth["ADMIT_DATE"])

    m = panel[[ID_COL, TIME_COL, TARGET]].merge(
        truth[["DEPT_ID", "ADMIT_DATE", "MU_EXPECTED"]],
        left_on=[ID_COL, TIME_COL], right_on=["DEPT_ID", "ADMIT_DATE"],
        how="inner")
    windows = make_backtest_windows(panel[TIME_COL], HORIZON, N_WINDOWS)

    rows = []
    for w in windows:
        te = m[(m[TIME_COL] >= w.test_start) & (m[TIME_COL] <= w.test_end)]
        hist = m[m[TIME_COL] <= w.train_end]
        for dept, g in te.groupby(ID_COL, sort=False):
            g = g.sort_values(TIME_COL)
            y = g[TARGET].to_numpy(float)
            mu = g["MU_EXPECTED"].to_numpy(float)
            y_ins = hist.loc[hist[ID_COL] == dept, TARGET].to_numpy(float)
            pt = evaluate_point(y, mu, y_ins, SEASONAL_PERIOD)
            rows.append({"trial": "ORACLE_true_mean", "window": w.window_id,
                         "dept_id": dept, **pt})
    return pd.DataFrame(rows), m, windows


def main():
    os.makedirs(PLOTS, exist_ok=True)
    print("=" * 78)
    print("FINAL HEAD-TO-HEAD COMPARISON")
    print("=" * 78)

    a, b, bl = load_all()
    if a.empty and b.empty:
        print("no track results found - run 05 and 06 first")
        return

    oracle, merged, windows = noise_floor()

    frames = [x for x in (a, b, bl, oracle) if not x.empty]
    allsc = pd.concat(frames, ignore_index=True)
    for c in METRICS:
        if c not in allsc.columns:
            allsc[c] = np.nan

    # ---------- unified leaderboard -----------------------------------
    lb = (allsc.groupby("trial")[METRICS].mean()
               .sort_values("MASE"))
    ws = allsc.groupby(["trial", "dept_id"])["MASE"].mean().groupby("trial").max()
    lb.insert(1, "worst_dept_MASE", ws)
    lb.insert(0, "track", [
        "ORACLE" if t.startswith("ORACLE") else
        "baseline" if t.startswith("baseline_") else
        "B: SNOWFLAKE.ML.FORECAST" if t.startswith("sf_forecast") else
        "A: AutoGluon" for t in lb.index])

    print("\nUNIFIED LEADERBOARD - all tracks, 6 windows x 6 departments")
    print("(ORACLE = predicts the TRUE mean; its MASE is the irreducible floor)")
    print("-" * 78)
    show = ["track", "MASE", "worst_dept_MASE", "RMSSE", "sMAPE", "MAE",
            "ME", "WQL", "PICP80"]
    print(lb[show].round(4).to_string())

    floor = float(lb.loc["ORACLE_true_mean", "MASE"])
    print(f"\nIRREDUCIBLE FLOOR (MASE of a perfect model) : {floor:.4f}")

    # best real model per track
    real = lb[~lb.index.str.startswith(("ORACLE", "baseline_"))]
    best_a = real[real["track"] == "A: AutoGluon"]
    best_b = real[real["track"] == "B: SNOWFLAKE.ML.FORECAST"]
    base_best = lb[lb.index.str.startswith("baseline_")]["MASE"].min()
    base_name = lb[lb.index.str.startswith("baseline_")]["MASE"].idxmin()

    print(f"BEST BASELINE  {base_name:<32} MASE={base_best:.4f}")
    if len(best_a):
        na, va = best_a.index[0], best_a.iloc[0]["MASE"]
        print(f"BEST TRACK A   {na:<32} MASE={va:.4f}")
    if len(best_b):
        nb, vb = best_b.index[0], best_b.iloc[0]["MASE"]
        print(f"BEST TRACK B   {nb:<32} MASE={vb:.4f}")

    # ---------- headroom analysis -------------------------------------
    print("\n" + "-" * 78)
    print("HEADROOM vs THE IRREDUCIBLE FLOOR")
    print("-" * 78)
    print(f"{'model':<34}{'MASE':>8}{'excess':>9}{'captured':>10}")
    for t, r in real.sort_values("MASE").iterrows():
        excess = r["MASE"] - floor
        # Fraction of the gap between the best baseline and the floor that the
        # model closed. 100% = as good as knowing the true mean.
        captured = (base_best - r["MASE"]) / (base_best - floor) * 100 \
            if base_best > floor else np.nan
        print(f"{t:<34}{r['MASE']:>8.4f}{excess:>9.4f}{captured:>9.1f}%")
    print("\n  'captured' = share of the achievable improvement (baseline -> floor)")
    print("  that the model actually realised. This is the number that matters:")
    print("  it is scale-free AND accounts for the noise you can never remove.")

    # ---------- paired significance ----------------------------------
    print("\n" + "-" * 78)
    print("PAIRED SIGNIFICANCE (Wilcoxon signed-rank on per-cell MASE)")
    print("-" * 78)
    key = ["window", "dept_id"]
    sig = []
    cands = list(real.sort_values("MASE").index[:6])
    if len(cands) >= 2:
        top = cands[0]
        t1 = allsc[allsc["trial"] == top].set_index(key)["MASE"]
        for other in cands[1:]:
            t2 = allsc[allsc["trial"] == other].set_index(key)["MASE"]
            j = t1.to_frame("a").join(t2.to_frame("b"), how="inner").dropna()
            if len(j) < 8:
                continue
            d = j["a"] - j["b"]
            try:
                st, p = stats.wilcoxon(j["a"], j["b"])
            except ValueError:
                st, p = np.nan, np.nan
            sig.append({"champion": top, "vs": other, "n_cells": len(j),
                        "mean_diff": float(d.mean()),
                        "champion_wins": int((d < 0).sum()),
                        "p_value": float(p) if p == p else None,
                        "significant_at_0.05": bool(p < 0.05) if p == p else None})
        sg = pd.DataFrame(sig)
        print(f"champion: {top}")
        print(sg.round(5).to_string(index=False))
        ns = sg[sg["significant_at_0.05"] == False]
        if len(ns):
            print(f"\n  NOT statistically distinguishable from the champion: "
                  f"{', '.join(ns['vs'])}")
            print("  Treat these as tied. Prefer whichever is simpler to operate")
            print("  and retrain - a 0.00x MASE gap is not a reason to choose a")
            print("  harder-to-serve model.")
        else:
            print("\n  Champion significantly beats every compared alternative.")

    # ---------- per-department ----------------------------------------
    print("\n" + "-" * 78)
    print("PER-DEPARTMENT MASE")
    print("-" * 78)
    piv = allsc.pivot_table(index="trial", columns="dept_id", values="MASE",
                            aggfunc="mean")
    piv = piv.loc[lb.index]
    print(piv.round(3).to_string())

    # ---------- error decomposition ----------------------------------
    print("\n" + "-" * 78)
    print("ERROR DECOMPOSITION (champion vs the noise floor)")
    print("-" * 78)
    dec = None
    if len(best_a) or len(best_b):
        champ = real.sort_values("MASE").index[0]
        pth = f"{ARTIFACTS}/automl_predictions.parquet"
        if champ.startswith("sf_forecast"):
            pf = pd.read_csv(f"{ARTIFACTS}/snowflake_forecast_predictions.csv")
            pf[TIME_COL] = pd.to_datetime(pf[TIME_COL])
            cp = pf[pf["trial"] == champ]
        elif os.path.exists(pth):
            pf = pd.read_parquet(pth)
            pf = pf.rename(columns={"item_id": ID_COL, "timestamp": TIME_COL})
            pf[TIME_COL] = pd.to_datetime(pf[TIME_COL])
            cp = pf[pf["trial"] == champ]
        else:
            cp = pd.DataFrame()

        if not cp.empty:
            j = cp.merge(merged[[ID_COL, TIME_COL, TARGET, "MU_EXPECTED"]],
                         on=[ID_COL, TIME_COL], how="inner")
            j["total_err"] = (j[TARGET] - j["mean"]).abs()
            j["noise_err"] = (j[TARGET] - j["MU_EXPECTED"]).abs()
            j["model_err"] = (j["MU_EXPECTED"] - j["mean"]).abs()
            dec = j.groupby(ID_COL)[["total_err", "noise_err", "model_err"]].mean()
            dec["model_share_pct"] = (dec["model_err"] /
                                      (dec["model_err"] + dec["noise_err"]) * 100)
            print(f"champion: {champ}")
            print(dec.round(3).to_string())
            print(f"\n  pooled  noise MAE={j['noise_err'].mean():.3f}   "
                  f"model MAE={j['model_err'].mean():.3f}")
            share = j["model_err"].mean() / (j["model_err"].mean() +
                                             j["noise_err"].mean()) * 100
            print(f"  model error is {share:.1f}% of total avoidable+unavoidable "
                  f"error")
            if share < 40:
                print("  => Most remaining error is IRREDUCIBLE count noise.")
                print("     Further model tuning has limited upside; effort is")
                print("     better spent on interval calibration and monitoring.")
            else:
                print("  => A substantial share is MODEL error, so there is real")
                print("     headroom left for better features or model families.")
            dec.to_csv(f"{ARTIFACTS}/error_decomposition.csv")

    # ---------- save + plots ------------------------------------------
    lb.to_csv(f"{ARTIFACTS}/final_comparison.csv")
    report = {
        "irreducible_floor_MASE": round(floor, 4),
        "best_baseline": {"model": base_name, "MASE": round(float(base_best), 4)},
        "leaderboard": lb[show].round(4).to_dict(orient="index"),
        "per_department_MASE": piv.round(4).to_dict(orient="index"),
        "significance": sig,
    }
    if dec is not None:
        report["error_decomposition"] = dec.round(4).to_dict(orient="index")
    with open(f"{ARTIFACTS}/final_report.json", "w") as fh:
        json.dump(report, fh, indent=2, default=str)

    make_plots(lb, piv, floor, base_best, dec)
    print(f"\nartifacts: {ARTIFACTS}/final_comparison.csv, final_report.json")


def make_plots(lb, piv, floor, base_best, dec):
    real = lb[~lb.index.str.startswith("ORACLE")]
    colors = {"A: AutoGluon": "#1f4e79", "B: SNOWFLAKE.ML.FORECAST": "#e07a5f",
              "baseline": "#adb5bd", "ORACLE": "#2a9d8f"}

    fig, ax = plt.subplots(figsize=(11, 0.42 * len(real) + 2.5))
    y = np.arange(len(real))
    ax.barh(y, real["MASE"], color=[colors[t] for t in real["track"]])
    ax.axvline(floor, color="#2a9d8f", ls="--", lw=2,
               label=f"irreducible floor {floor:.3f}")
    ax.axvline(base_best, color="black", ls=":", lw=1.5,
               label=f"best baseline {base_best:.3f}")
    ax.axvline(1.0, color="grey", ls="-.", lw=1, label="seasonal naive = 1.0")
    ax.set_yticks(y)
    ax.set_yticklabels(real.index, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("MASE (lower better)")
    ax.set_title("Track A vs Track B vs baselines, against the irreducible floor")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25, axis="x")
    from matplotlib.patches import Patch
    h, _ = ax.get_legend_handles_labels()
    ax.legend(handles=h + [Patch(color=v, label=k) for k, v in colors.items()
                           if k in set(real["track"])], fontsize=7)
    plt.tight_layout()
    plt.savefig(f"{PLOTS}/13_final_comparison.png", dpi=120)
    plt.close()

    if dec is not None:
        fig, ax = plt.subplots(figsize=(10, 4.5))
        x = np.arange(len(dec))
        ax.bar(x, dec["noise_err"], 0.55, label="irreducible noise",
               color="#adb5bd")
        ax.bar(x, dec["model_err"], 0.55, bottom=dec["noise_err"],
               label="model error", color="#d1495b")
        ax.set_xticks(x)
        ax.set_xticklabels(dec.index, rotation=25, ha="right", fontsize=8)
        ax.set_ylabel("mean absolute error (admissions)")
        ax.set_title("Error decomposition - grey is the floor no model can beat")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.25, axis="y")
        for i, v in enumerate(dec["model_share_pct"]):
            ax.text(i, dec["noise_err"].iloc[i] + dec["model_err"].iloc[i],
                    f"{v:.0f}%", ha="center", va="bottom", fontsize=7)
        plt.tight_layout()
        plt.savefig(f"{PLOTS}/14_error_decomposition.png", dpi=120)
        plt.close()
    print(f"wrote comparison plots to {PLOTS}/")


if __name__ == "__main__":
    main()
