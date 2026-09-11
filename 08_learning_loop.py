"""
08_learning_loop.py
===================
Walk-forward learning loop with drift detection and champion/challenger
promotion, run across 2025 so it traverses the year-3 flu surge.

Why the loop starts in January, not July
----------------------------------------
The six benchmark windows in step 2 all landed in H2 2025, which put the flu
surge (Jan-Apr 2025) inside TRAINING for every window. That makes the benchmark
a test of "don't over-extrapolate a past shock" - useful, but it never tests
adaptation to a shock in flight.

Running the loop from 2025-01-01 fixes that: it walks through surge onset, peak,
decay, and the return to normal, which is the only way the drift detector has
something real to catch.

The question the loop answers
-----------------------------
Retraining is usually assumed to be good. It is not free, and on small data it
can be actively harmful: refitting on a transient shock bakes the shock into the
model just as it fades. So we run three strategies over the SAME timeline and
the SAME data:

  NEVER      fit once at the start, never refit          (staleness cost)
  ALWAYS     refit at every step                         (churn cost)
  TRIGGERED  refit only when drift or accuracy breach    (the proposal)

If TRIGGERED does not beat NEVER, the honest conclusion is that the loop is not
worth its complexity - and this script will say so.

Drift detection (two independent signals, either can fire)
----------------------------------------------------------
  Page-Hinkley  - sequential change-point test on the running mean of
                  residuals. Catches gradual BIAS drift, which is what a rising
                  epidemic looks like: the model is consistently too low.
  KS two-sample - compares the recent residual DISTRIBUTION to the reference
                  distribution. Catches variance/shape change even when the
                  mean is stable.

Both operate on residuals, not on the raw series, because a seasonal peak is
not drift - the model is supposed to predict that. Only unexplained error is.

Conformal intervals are recalibrated every step on a trailing residual window,
so coverage tracks the current regime rather than a stale one.

Outputs
-------
  artifacts/learning_loop_steps.csv     per-step scores for all 3 strategies
  artifacts/learning_loop_events.csv    drift alarms and promotion decisions
  artifacts/learning_loop_summary.json
  plots/11_learning_loop.png
  plots/12_drift_detection.png
"""

from __future__ import annotations

import json
import os
import time
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import holidays as _holidays
from scipy import stats

warnings.filterwarnings("ignore")

from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor

import importlib.util as _ilu
_spec = _ilu.spec_from_file_location(
    "features03", os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "03_features.py"))
feats = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(feats)

from forecast_metrics import (
    SEASONAL_PERIOD, ConformalCalibrator, evaluate_point, picp,
)
from snowpark_session import create_snowpark_session

DATABASE, SCHEMA = "HEALTHCARE", "ML_FORECAST"
TARGET, ID_COL, TIME_COL = "ADMISSIONS", "DEPT_ID", "ADMIT_DATE"
ARTIFACTS, PLOTS = "artifacts", "plots"

STEP_DAYS = 28                       # forecast cadence
LOOP_START = pd.Timestamp("2025-01-01")
LOOP_END = pd.Timestamp("2025-12-31")
QUANTILES = [0.1, 0.5, 0.9]

# Fast, trainable model set. Chronos is deliberately EXCLUDED here: it is
# zero-shot, so "retraining" it is close to a no-op and the never/always/
# triggered comparison would be vacuous. The loop needs models that actually
# learn from data for the question to mean anything.
LOOP_HP = {
    "SeasonalNaive": {},
    "AutoETS": {},
    "RecursiveTabular": {
        "model_name": "GBM",
        "model_hyperparameters": {"num_leaves": 16, "learning_rate": 0.05,
                                  "min_data_in_leaf": 30, "lambda_l2": 5.0,
                                  "num_boost_round": 250},
    },
}
FIT_TIME_LIMIT = 45

# Trigger thresholds
# Page-Hinkley thresholds.
#
# CALIBRATION NOTE: lambda=3.0 (a common textbook default) is badly wrong at
# this update rate and saturated the detector - it fired on 12 of 13 steps in
# the first run, making the trigger useless. The reason is scale: PH accumulates
# one update PER OBSERVATION, and each step contributes 168 observations
# (6 departments x 28 days). Under no drift the cumulative sum random-walks with
# standard deviation ~sqrt(n), i.e. ~13 within a single step and larger across
# steps. Any threshold below that fires on noise alone.
#
# lambda=35 sits comfortably above the sqrt(n) noise band while still catching a
# sustained bias shift.
PH_DELTA = 0.01       # Page-Hinkley slack: ignore drift smaller than this
PH_LAMBDA = 35.0      # alarm threshold, calibrated to 168 updates/step
KS_ALPHA = 0.01       # KS p-value below this => distribution shifted
MASE_BREACH = 1.15    # or: accuracy degraded past 1.15x the reference MASE


# ==========================================================================
class PageHinkley:
    """Sequential change-point detector on the running mean.

    Tracks cumulative deviation of standardised residuals from their reference
    mean, minus a slack term. When the excess over the running minimum passes
    lambda, a persistent shift has occurred rather than a one-off outlier.
    """

    def __init__(self, delta=PH_DELTA, lam=PH_LAMBDA):
        self.delta, self.lam = delta, lam
        self.reset()

    def reset(self):
        self.cum = 0.0
        self.min_cum = 0.0
        self.n = 0

    def update(self, standardised_resid: float) -> tuple[bool, float]:
        self.n += 1
        self.cum += standardised_resid - self.delta
        self.min_cum = min(self.min_cum, self.cum)
        stat = self.cum - self.min_cum
        return stat > self.lam, stat


def fit_predictor(frame, stat_df, kf_cols, past_cols, cutoff, path):
    cols = [ID_COL, TIME_COL, TARGET] + kf_cols + past_cols
    tf = frame[frame[TIME_COL] <= cutoff][cols]
    ts = TimeSeriesDataFrame.from_data_frame(
        tf, id_column=ID_COL, timestamp_column=TIME_COL,
        static_features_df=stat_df.reset_index())
    p = TimeSeriesPredictor(
        prediction_length=STEP_DAYS, target=TARGET, freq="D",
        eval_metric="MASE", quantile_levels=QUANTILES,
        known_covariates_names=kf_cols, path=path, verbosity=0)
    p.fit(ts, hyperparameters=LOOP_HP, time_limit=FIT_TIME_LIMIT,
          num_val_windows=2, val_step_size=STEP_DAYS, enable_ensemble=True)
    return p


def predict_range(predictor, frame, stat_df, kf_cols, past_cols,
                  hist_end, t_start, t_end):
    cols = [ID_COL, TIME_COL, TARGET] + kf_cols + past_cols
    hist = frame[frame[TIME_COL] <= hist_end][cols]
    hts = TimeSeriesDataFrame.from_data_frame(
        hist, id_column=ID_COL, timestamp_column=TIME_COL,
        static_features_df=stat_df.reset_index())
    fut = frame[(frame[TIME_COL] >= t_start) & (frame[TIME_COL] <= t_end)][
        [ID_COL, TIME_COL] + kf_cols]
    kc = TimeSeriesDataFrame.from_data_frame(
        fut, id_column=ID_COL, timestamp_column=TIME_COL)
    return predictor.predict(hts, known_covariates=kc).reset_index()


def score_step(pred, frame, hist_end, t_start, t_end):
    """Pooled + per-dept metrics for one step."""
    act = frame[(frame[TIME_COL] >= t_start) & (frame[TIME_COL] <= t_end)][
        [ID_COL, TIME_COL, TARGET]]
    hist = frame[frame[TIME_COL] <= hist_end]
    p = pred.rename(columns={"item_id": ID_COL, "timestamp": TIME_COL})
    m = act.merge(p, on=[ID_COL, TIME_COL], how="inner")
    rows = []
    for dept, g in m.groupby(ID_COL, sort=False):
        g = g.sort_values(TIME_COL)
        y = g[TARGET].to_numpy(float)
        f = g["mean"].to_numpy(float)
        y_ins = hist.loc[hist[ID_COL] == dept, TARGET].to_numpy(float)
        pt = evaluate_point(y, f, y_ins, SEASONAL_PERIOD)
        cov = picp(y, g["0.1"].to_numpy(float), g["0.9"].to_numpy(float)) \
            if "0.1" in g else np.nan
        rows.append({"dept_id": dept, **pt, "PICP80_native": cov})
    return pd.DataFrame(rows), m


def main():
    os.makedirs(ARTIFACTS, exist_ok=True)
    os.makedirs(PLOTS, exist_ok=True)
    t0 = time.time()

    print("=" * 78)
    print("WALK-FORWARD LEARNING LOOP")
    print("=" * 78)

    session = create_snowpark_session()
    session.use_database(DATABASE)
    session.use_schema(SCHEMA)
    panel = session.table("DAILY_ADMISSIONS").to_pandas()
    static = session.table("DEPT_STATIC").to_pandas()
    truth = session.table("GROUND_TRUTH_COMPONENTS").to_pandas()
    session.close()
    panel[TIME_COL] = pd.to_datetime(panel[TIME_COL])
    truth["ADMIT_DATE"] = pd.to_datetime(truth["ADMIT_DATE"])

    yrs = sorted(panel[TIME_COL].dt.year.unique())
    hol = set(_holidays.US(years=list(yrs) + [max(yrs) + 1]).keys())
    frame, kf_cols, past_cols = feats.build_features(panel, hol)
    stat_df = feats.build_static(static)

    # Step schedule
    steps = []
    cur = LOOP_START
    while cur + pd.Timedelta(days=STEP_DAYS - 1) <= LOOP_END:
        steps.append((cur, cur + pd.Timedelta(days=STEP_DAYS - 1)))
        cur = cur + pd.Timedelta(days=STEP_DAYS)
    print(f"steps       : {len(steps)} x {STEP_DAYS}d, "
          f"{steps[0][0].date()} .. {steps[-1][1].date()}")
    print(f"strategies  : NEVER / ALWAYS / TRIGGERED")
    print(f"model set   : {list(LOOP_HP)} (Chronos excluded - zero-shot, "
          f"refit is a no-op)")
    print(f"triggers    : Page-Hinkley(delta={PH_DELTA}, lambda={PH_LAMBDA}), "
          f"KS(p<{KS_ALPHA}), MASE>{MASE_BREACH}x ref")

    # flu surge reference for annotation
    flu = truth.groupby("ADMIT_DATE")["COMP_FLU"].mean()

    # ---- initial fit (shared starting point for all strategies) ----------
    init_cut = LOOP_START - pd.Timedelta(days=1)
    print(f"\ninitial fit on data <= {init_cut.date()} "
          f"({len(frame[frame[TIME_COL] <= init_cut])//6} obs/series)")
    t = time.time()
    p_never = fit_predictor(frame, stat_df, kf_cols, past_cols, init_cut,
                            "ag_models/loop_never")
    print(f"  fitted in {time.time()-t:.0f}s, best={p_never.model_best}")

    # ALWAYS and TRIGGERED start from the same weights
    p_always = p_never
    p_trig = p_never
    trig_cut = init_cut          # data the triggered champion was fitted on

    ph = PageHinkley()
    ref_resid = None
    ref_mase = None
    rows, events = [], []
    conf = ConformalCalibrator()
    resid_hist = []              # trailing (resid, horizon) for conformal

    for i, (s, e) in enumerate(steps, start=1):
        hist_end = s - pd.Timedelta(days=1)
        flu_lvl = float(flu.loc[s:e].mean()) if len(flu.loc[s:e]) else 1.0
        print("\n" + "-" * 78)
        print(f"STEP {i}/{len(steps)}  forecast {s.date()} .. {e.date()}  "
              f"(flu factor {flu_lvl:.3f})")
        print("-" * 78)

        # ---------- NEVER ------------------------------------------------
        pr_n = predict_range(p_never, frame, stat_df, kf_cols, past_cols,
                             hist_end, s, e)
        sc_n, _ = score_step(pr_n, frame, hist_end, s, e)

        # ---------- ALWAYS: refit on everything up to now -----------------
        p_always = fit_predictor(frame, stat_df, kf_cols, past_cols, hist_end,
                                 f"ag_models/loop_always_{i}")
        pr_a = predict_range(p_always, frame, stat_df, kf_cols, past_cols,
                             hist_end, s, e)
        sc_a, _ = score_step(pr_a, frame, hist_end, s, e)

        # ---------- TRIGGERED --------------------------------------------
        pr_t = predict_range(p_trig, frame, stat_df, kf_cols, past_cols,
                             hist_end, s, e)
        sc_t, m_t = score_step(pr_t, frame, hist_end, s, e)

        mase_n, mase_a, mase_t = (sc_n["MASE"].mean(), sc_a["MASE"].mean(),
                                  sc_t["MASE"].mean())
        print(f"  MASE   NEVER={mase_n:.4f}   ALWAYS={mase_a:.4f}   "
              f"TRIGGERED={mase_t:.4f}")

        # ---------- drift detection on TRIGGERED residuals ----------------
        m_t = m_t.sort_values([ID_COL, TIME_COL])
        resid = (m_t[TARGET] - m_t["mean"]).to_numpy(float)

        if ref_resid is None:
            ref_resid = resid.copy()
            ref_mase = mase_t
            ph.reset()
            ks_p, ph_stat, ph_alarm = np.nan, 0.0, False
        else:
            # Standardise against the REFERENCE distribution, not the current
            # one. Dividing by the current std (the first implementation)
            # normalises away the very scale change the detector exists to
            # find, and subtracting the current mean would hide bias drift.
            ref_mu = float(np.mean(ref_resid))
            ref_sd = float(np.std(ref_resid)) or 1.0
            z = (resid - ref_mu) / ref_sd
            ks_p = float(stats.ks_2samp(ref_resid, resid).pvalue)
            ph_alarm = False
            ph_stat = 0.0
            for v in z:
                a, ph_stat = ph.update(v)
                ph_alarm = ph_alarm or a

        mase_breach = (ref_mase is not None) and (mase_t > MASE_BREACH * ref_mase)
        ks_alarm = (not np.isnan(ks_p)) and (ks_p < KS_ALPHA)
        fire = bool(ph_alarm or ks_alarm or mase_breach)

        print(f"  drift  PH_stat={ph_stat:.3f} (alarm={ph_alarm})  "
              f"KS_p={ks_p:.4g} (alarm={ks_alarm})  "
              f"MASE_breach={mase_breach}  -> {'RETRAIN' if fire else 'hold'}")

        promoted = False
        if fire:
            # Champion/challenger with an HONEST comparison window.
            #
            # The challenger is fitted on data up to (hist_end - STEP), so the
            # most recent observed window is genuinely out-of-sample for it.
            # Both models then forecast that same window and are scored on it.
            # Fitting the challenger on data that INCLUDES the comparison window
            # would guarantee it wins and make the gate meaningless.
            val_start = hist_end - pd.Timedelta(days=STEP_DAYS - 1)
            val_hist_end = val_start - pd.Timedelta(days=1)
            try:
                chal = fit_predictor(frame, stat_df, kf_cols, past_cols,
                                     val_hist_end, f"ag_models/loop_chal_{i}")
                pv_c = predict_range(chal, frame, stat_df, kf_cols, past_cols,
                                     val_hist_end, val_start, hist_end)
                sv_c, _ = score_step(pv_c, frame, val_hist_end, val_start, hist_end)
                pv_ch = predict_range(p_trig, frame, stat_df, kf_cols, past_cols,
                                      val_hist_end, val_start, hist_end)
                sv_ch, _ = score_step(pv_ch, frame, val_hist_end, val_start,
                                      hist_end)
                c_m, ch_m = sv_c["MASE"].mean(), sv_ch["MASE"].mean()
                promoted = bool(c_m < ch_m)
                print(f"  challenger gate: challenger MASE={c_m:.4f} vs "
                      f"champion MASE={ch_m:.4f} -> "
                      f"{'PROMOTE' if promoted else 'KEEP CHAMPION'}")
                if promoted:
                    # Refit the promoted configuration on ALL data up to now.
                    p_trig = fit_predictor(frame, stat_df, kf_cols, past_cols,
                                           hist_end, f"ag_models/loop_trig_{i}")
                    trig_cut = hist_end
                    ref_resid = resid.copy()
                    ref_mase = mase_t
                    ph.reset()
            except Exception as ex:
                print(f"  challenger failed: {type(ex).__name__}: {ex}")

        events.append({"step": i, "start": s.date(), "end": e.date(),
                       "flu_factor": round(flu_lvl, 4),
                       "PH_stat": round(float(ph_stat), 4),
                       "PH_alarm": bool(ph_alarm),
                       "KS_p": round(float(ks_p), 6) if not np.isnan(ks_p) else None,
                       "KS_alarm": bool(ks_alarm),
                       "MASE_breach": bool(mase_breach),
                       "trigger_fired": fire, "promoted": promoted,
                       "champion_data_cutoff": str(trig_cut.date())})

        # ---------- conformal recalibration ------------------------------
        h = m_t.groupby(ID_COL)[TIME_COL].rank().astype(int).to_numpy()
        resid_hist.append(pd.DataFrame({"resid": resid, "h": h}))
        trail = pd.concat(resid_hist[-3:], ignore_index=True)   # trailing 3 steps
        conf.fit(trail["resid"].to_numpy(), trail["h"].to_numpy(), level=0.80)
        lo, hi = conf.intervals(m_t["mean"].to_numpy(), h)
        conf_cov = picp(m_t[TARGET].to_numpy(float), lo, hi)
        nat_cov = sc_t["PICP80_native"].mean()
        print(f"  coverage@80  conformal={conf_cov:.3f}  native={nat_cov:.3f}")

        for strat, sc, mm in (("NEVER", sc_n, mase_n), ("ALWAYS", sc_a, mase_a),
                              ("TRIGGERED", sc_t, mase_t)):
            rows.append({
                "step": i, "start": s.date(), "end": e.date(),
                "strategy": strat, "flu_factor": round(flu_lvl, 4),
                "MASE": float(mm),
                "worst_dept_MASE": float(sc["MASE"].max()),
                "sMAPE": float(sc["sMAPE"].mean()),
                "MAE": float(sc["MAE"].mean()),
                "ME": float(sc["ME"].mean()),
                "PICP80_native": float(sc["PICP80_native"].mean()),
                "PICP80_conformal": float(conf_cov) if strat == "TRIGGERED" else np.nan,
            })

    df = pd.DataFrame(rows)
    ev = pd.DataFrame(events)
    df.to_csv(f"{ARTIFACTS}/learning_loop_steps.csv", index=False)
    ev.to_csv(f"{ARTIFACTS}/learning_loop_events.csv", index=False)

    # ================= results =======================================
    print("\n" + "=" * 78)
    print("LEARNING LOOP RESULTS")
    print("=" * 78)
    summ = df.groupby("strategy").agg(
        mean_MASE=("MASE", "mean"), worst_step_MASE=("MASE", "max"),
        mean_worst_dept=("worst_dept_MASE", "mean"),
        sMAPE=("sMAPE", "mean"), ME=("ME", "mean"),
        PICP80=("PICP80_native", "mean")).sort_values("mean_MASE")
    print(summ.round(4).to_string())

    n_ref = int(ev["trigger_fired"].sum())
    n_prom = int(ev["promoted"].sum())
    print(f"\nrefit events   : ALWAYS={len(steps)}  "
          f"TRIGGERED fired={n_ref}, promoted={n_prom}")
    print(f"compute saved  : TRIGGERED did {n_prom} refits vs "
          f"{len(steps)} for ALWAYS "
          f"({100*(1-n_prom/max(len(steps),1)):.0f}% fewer)")

    mn = summ.loc["NEVER", "mean_MASE"]
    ma = summ.loc["ALWAYS", "mean_MASE"]
    mt = summ.loc["TRIGGERED", "mean_MASE"]
    print(f"\nNEVER    {mn:.4f}")
    print(f"ALWAYS   {ma:.4f}  ({(mn-ma)/mn*100:+.2f}% vs NEVER)")
    print(f"TRIGGERED{mt:.4f}  ({(mn-mt)/mn*100:+.2f}% vs NEVER)")

    print("\nVERDICT:")
    if mt < mn and mt <= ma * 1.02:
        print("  TRIGGERED retraining is justified - it beats NEVER and matches")
        print(f"  ALWAYS within 2% while doing {n_prom} refits instead of "
              f"{len(steps)}.")
    elif mt < mn:
        print("  TRIGGERED beats NEVER but ALWAYS is clearly better. If compute")
        print("  is cheap, refit every cycle; the trigger is costing accuracy.")
    else:
        print("  TRIGGERED does NOT beat leaving the model alone. On this data")
        print("  the loop is not worth its complexity - the honest recommendation")
        print("  is to keep the fixed model and monitor, not retrain.")

    # surge vs normal
    surge = df[df["flu_factor"] > 1.05]
    normal = df[df["flu_factor"] <= 1.05]
    if len(surge) and len(normal):
        print("\nMASE during flu surge vs normal periods:")
        cmp = pd.DataFrame({
            "surge": surge.groupby("strategy")["MASE"].mean(),
            "normal": normal.groupby("strategy")["MASE"].mean()})
        cmp["degradation"] = (cmp["surge"] / cmp["normal"]).round(3)
        print(cmp.round(4).to_string())
        print("  Lower degradation = more robust to the regime shift.")

    cc = df["PICP80_conformal"].dropna().mean()
    nc = df.loc[df["strategy"] == "TRIGGERED", "PICP80_native"].mean()
    print(f"\ncoverage@80 (nominal 0.80): conformal={cc:.4f}  native={nc:.4f}")
    print(f"  |error| conformal={abs(cc-0.8):.4f}  native={abs(nc-0.8):.4f} -> "
          f"{'conformal' if abs(cc-0.8)<abs(nc-0.8) else 'native'} closer")

    print("\ndrift events:")
    print(ev[["step", "start", "flu_factor", "PH_alarm", "KS_p", "KS_alarm",
              "MASE_breach", "trigger_fired", "promoted"]].to_string(index=False))

    with open(f"{ARTIFACTS}/learning_loop_summary.json", "w") as fh:
        json.dump({"summary": summ.round(4).to_dict(orient="index"),
                   "n_steps": len(steps), "triggers_fired": n_ref,
                   "promotions": n_prom,
                   "coverage": {"conformal": round(float(cc), 4),
                                "native": round(float(nc), 4)},
                   "elapsed_s": round(time.time() - t0, 1)}, fh, indent=2)

    make_plots(df, ev)
    print(f"\ntotal elapsed {time.time()-t0:.0f}s")


def make_plots(df, ev):
    fig, axes = plt.subplots(3, 1, figsize=(13, 9), sharex=True)

    ax = axes[0]
    for strat, c in (("NEVER", "#d1495b"), ("ALWAYS", "#1f4e79"),
                     ("TRIGGERED", "#2a9d8f")):
        g = df[df["strategy"] == strat]
        ax.plot(g["step"], g["MASE"], marker="o", label=strat, color=c, lw=1.6)
    ax.axhline(1.0, color="grey", ls=":", label="seasonal naive")
    for _, r in ev[ev["promoted"]].iterrows():
        ax.axvline(r["step"], color="green", ls="--", alpha=0.5)
    ax.set_ylabel("MASE")
    ax.set_title("Walk-forward accuracy by retraining strategy "
                 "(green dashed = model promoted)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)

    ax = axes[1]
    ax.fill_between(ev["step"], ev["flu_factor"], 1.0, color="orange",
                    alpha=0.35, label="flu factor")
    ax.set_ylabel("flu factor")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)
    ax.set_title("Flu-surge intensity over the loop timeline")

    ax = axes[2]
    ax.plot(ev["step"], ev["PH_stat"], marker="o", color="#1f4e79",
            label="Page-Hinkley stat")
    ax.axhline(PH_LAMBDA, color="red", ls="--", label=f"lambda={PH_LAMBDA}")
    ax2 = ax.twinx()
    ax2.plot(ev["step"], ev["KS_p"].astype(float), marker="s", color="#d1495b",
             alpha=0.75, label="KS p-value")
    ax2.axhline(KS_ALPHA, color="#d1495b", ls=":", label=f"alpha={KS_ALPHA}")
    ax2.set_yscale("log")
    ax.set_xlabel("step")
    ax.set_ylabel("PH statistic")
    ax2.set_ylabel("KS p-value (log)")
    ax.set_title("Drift detector signals")
    ax.legend(loc="upper left", fontsize=8)
    ax2.legend(loc="upper right", fontsize=8)
    ax.grid(alpha=0.25)

    plt.tight_layout()
    plt.savefig(f"{PLOTS}/11_learning_loop.png", dpi=120)
    plt.close()

    # coverage tracking
    fig, ax = plt.subplots(figsize=(11, 4))
    g = df[df["strategy"] == "TRIGGERED"]
    ax.plot(g["step"], g["PICP80_conformal"], marker="o", color="#2a9d8f",
            label="conformal")
    ax.plot(g["step"], g["PICP80_native"], marker="s", color="#d1495b",
            label="model-native")
    ax.axhline(0.80, color="k", ls="--", label="nominal 0.80")
    ax.set_xlabel("step")
    ax.set_ylabel("80% interval coverage")
    ax.set_title("Interval coverage over time - conformal recalibrates each step")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(f"{PLOTS}/12_drift_detection.png", dpi=120)
    plt.close()
    print(f"wrote plots to {PLOTS}/")


if __name__ == "__main__":
    main()
