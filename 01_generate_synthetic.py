"""
01_generate_synthetic.py
========================
Generate a deliberately SMALL synthetic healthcare-volume panel for time-series
forecasting research, and land it in HEALTHCARE.ML_FORECAST.

Design intent
-------------
6 hospital departments x 1,096 daily observations (2023-01-01 .. 2025-12-31)
= 6,576 rows total, which sits under the 10,000-record budget and reproduces
the target small-data regime: ~1,100 observations per series is
enough for two full seasonal cycles but NOT enough to train a deep sequence
model from scratch.

Every series is built from an explicit, KNOWN generative decomposition:

    mu(t) = base
            * trend(t)          # slow growth + one structural break
            * weekly(t)         # day-of-week profile, department-specific
            * yearly(t)         # Fourier annual seasonality, dept-specific phase
            * holiday(t)        # ED spikes, elective surgery collapses
            * flu_surge(t)      # a bad flu wave in year 3 -> regime shift
            * temp_effect(t)    # lagged cold-weather respiratory driver

    y(t) ~ NegativeBinomial(mean=mu(t), r=dispersion)

Two deliberate modelling choices:

1. MULTIPLICATIVE composition, not additive. Admission volumes scale
   proportionally - a holiday removes ~70% of elective surgery regardless of
   the underlying level. Additive seasonality would be wrong here.

2. NEGATIVE BINOMIAL noise, not Gaussian. Admissions are non-negative integer
   counts and they are overdispersed relative to Poisson (variance > mean).
   Gaussian noise would produce negative admissions and understate tail risk.

Because the decomposition is known, GROUND_TRUTH_COMPONENTS lets us later ask a
question that is impossible on real data: did the model actually recover the
trend and seasonality, or did it merely score well?

Outputs (HEALTHCARE.ML_FORECAST)
--------------------------------
  DAILY_ADMISSIONS        long-format panel + covariates  (6,576 rows)
  GROUND_TRUTH_COMPONENTS per-row component decomposition  (6,576 rows)
  DEPT_STATIC             static per-department features   (6 rows)

Covariates are split by AVAILABILITY AT FORECAST TIME, which matters because
feeding a past-only covariate to the model as if it were known-future is a
classic leakage bug:

  known-future : calendar features - known for any future date
  past-only    : temperature, ER wait - only observed historically
  static       : department metadata - constant per series
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import holidays

from snowpark_session import create_snowpark_session

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------
SEED = 20260910
DATABASE = "HEALTHCARE"
SCHEMA = "ML_FORECAST"

START_DATE = "2023-01-01"
END_DATE = "2025-12-31"          # 1,096 days inclusive

# Structural break: a capacity expansion partway through the panel. Real
# hospital series are full of these and they are what break naive extrapolation.
BREAK_DAY = 600

# A severe flu season in year 3, centred ~Feb 2025. Deliberately placed late in
# the panel so it lands inside the backtest windows and the learning loop has a
# genuine regime shift to detect rather than a synthetic toy.
FLU_CENTER_DAY = 790
FLU_WIDTH_DAYS = 45

# --------------------------------------------------------------------------
# Department specifications
#
# amp_yearly  : annual seasonality amplitude (0 = flat)
# phase_yearly: day-of-year of the seasonal PEAK. Winter-peaking respiratory
#               departments sit near day 15; ORTHOPEDICS is deliberately
#               counter-seasonal (summer sports injuries) so the models cannot
#               assume one shared seasonal shape across the panel.
# holiday_mult: <1 means the holiday suppresses volume (elective procedures get
#               cancelled), >1 means it amplifies it (ED absorbs the overflow).
# flu_sens    : multiplier sensitivity to the year-3 flu wave.
# temp_sens   : sensitivity to LAGGED cold weather (respiratory admissions
#               follow a cold snap by roughly a week, not the same day).
# disp_r      : NegBinomial dispersion. Lower r = heavier overdispersion.
#               Low-volume series are noisier in relative terms.
# --------------------------------------------------------------------------
DEPARTMENTS = [
    dict(
        dept_id="EMERGENCY", dept_type="UNSCHEDULED", bed_capacity=64,
        base=120.0, slope=0.030, break_step=1.12, break_slope=0.055,
        # ED runs 7 days; Monday is the well-known post-weekend peak.
        dow=[1.14, 1.02, 0.98, 0.97, 1.03, 0.94, 0.92],
        amp_yearly=0.20, phase_yearly=15,
        holiday_mult=1.28, holiday_window_mult=1.10,
        flu_sens=0.55, temp_sens=-0.011, school_sens=0.00, disp_r=45.0,
    ),
    dict(
        dept_id="ELECTIVE_SURGERY", dept_type="SCHEDULED", bed_capacity=28,
        base=28.0, slope=0.022, break_step=1.15, break_slope=0.040,
        # Scheduled theatre lists: essentially weekday-only. The near-zero
        # weekend creates a hard, non-smooth weekly pattern that punishes
        # models which assume smooth seasonality.
        dow=[1.18, 1.22, 1.20, 1.15, 0.95, 0.06, 0.04],
        amp_yearly=0.10, phase_yearly=250,
        holiday_mult=0.22, holiday_window_mult=0.70,
        flu_sens=-0.18, temp_sens=0.000, school_sens=0.00, disp_r=30.0,
    ),
    dict(
        dept_id="CARDIOLOGY", dept_type="MIXED", bed_capacity=40,
        base=45.0, slope=0.018, break_step=1.05, break_slope=0.022,
        dow=[1.12, 1.10, 1.08, 1.06, 1.00, 0.38, 0.30],
        amp_yearly=0.14, phase_yearly=25,
        holiday_mult=0.72, holiday_window_mult=0.90,
        flu_sens=0.18, temp_sens=-0.006, school_sens=0.00, disp_r=38.0,
    ),
    dict(
        dept_id="PEDIATRICS", dept_type="MIXED", bed_capacity=34,
        base=38.0, slope=0.012, break_step=1.02, break_slope=0.014,
        dow=[1.15, 1.08, 1.05, 1.03, 1.02, 0.85, 0.78],
        # Strongest annual swing in the panel: RSV/flu season plus a deep
        # summer trough when school is out.
        amp_yearly=0.34, phase_yearly=10,
        holiday_mult=0.85, holiday_window_mult=0.95,
        flu_sens=0.70, temp_sens=-0.014, school_sens=0.12, disp_r=28.0,
    ),
    dict(
        dept_id="ONCOLOGY", dept_type="SCHEDULED", bed_capacity=22,
        base=22.0, slope=0.035, break_step=1.03, break_slope=0.038,
        dow=[1.20, 1.18, 1.16, 1.12, 0.98, 0.14, 0.10],
        # Treatment cycles are booked regardless of season: near-flat annual
        # profile, but the strongest secular growth trend in the panel.
        amp_yearly=0.05, phase_yearly=180,
        holiday_mult=0.35, holiday_window_mult=0.80,
        flu_sens=-0.05, temp_sens=0.000, school_sens=0.00, disp_r=26.0,
    ),
    dict(
        dept_id="ORTHOPEDICS", dept_type="MIXED", bed_capacity=30,
        base=32.0, slope=0.015, break_step=1.06, break_slope=0.020,
        dow=[1.16, 1.12, 1.10, 1.08, 1.02, 0.42, 0.34],
        # COUNTER-SEASONAL on purpose: peaks in summer (day 200), opposite to
        # the respiratory departments.
        amp_yearly=0.22, phase_yearly=200,
        holiday_mult=0.80, holiday_window_mult=0.92,
        flu_sens=-0.08, temp_sens=+0.005, school_sens=-0.06, disp_r=32.0,
    ),
]


# --------------------------------------------------------------------------
# Calendar / exogenous drivers (shared across all departments)
# --------------------------------------------------------------------------
def build_calendar() -> pd.DataFrame:
    """Daily calendar with holiday windows, school terms, and weather.

    Weather is generated ONCE for the whole hospital (all departments share the
    same city), which is what makes it a genuine cross-series driver.
    """
    rng = np.random.default_rng(SEED)
    dates = pd.date_range(START_DATE, END_DATE, freq="D")
    n = len(dates)
    cal = pd.DataFrame({"admit_date": dates})
    cal["t"] = np.arange(n)
    cal["dow"] = cal["admit_date"].dt.dayofweek          # 0=Mon .. 6=Sun
    cal["month"] = cal["admit_date"].dt.month
    cal["day_of_year"] = cal["admit_date"].dt.dayofyear
    cal["is_weekend"] = (cal["dow"] >= 5).astype(int)

    # --- US federal holidays, plus a +/-3 day shoulder window ---------------
    yrs = sorted(cal["admit_date"].dt.year.unique())
    us_hol = holidays.US(years=yrs)
    cal["is_holiday"] = cal["admit_date"].dt.date.map(
        lambda d: 1 if d in us_hol else 0
    ).astype(int)
    cal["holiday_name"] = cal["admit_date"].dt.date.map(
        lambda d: us_hol.get(d) or ""
    )
    # Behaviour changes in the days AROUND a holiday, not just on the day
    # itself - elective lists thin out before Christmas, not only on the 25th.
    hol_idx = np.flatnonzero(cal["is_holiday"].to_numpy() == 1)
    window = np.zeros(n, dtype=int)
    for i in hol_idx:
        window[max(0, i - 3): min(n, i + 4)] = 1
    cal["is_holiday_window"] = window

    # --- School term (drives paediatric respiratory transmission) ----------
    # Out of term: summer (Jul-Aug) and the winter break.
    doy = cal["day_of_year"].to_numpy()
    month = cal["month"].to_numpy()
    out_of_term = (
        np.isin(month, [7, 8])
        | (doy >= 356)
        | (doy <= 6)
    )
    cal["is_school_term"] = (~out_of_term).astype(int)

    # --- Temperature: annual sinusoid + AR(1) weather persistence ----------
    # AR(1) matters. Independent daily noise would give the model an easy,
    # unrealistically clean signal; real weather arrives in multi-day spells.
    seasonal_temp = 14.0 - 12.0 * np.cos(2 * np.pi * (doy - 15) / 365.25)
    innov = rng.normal(0, 2.6, n)
    ar = np.zeros(n)
    for i in range(1, n):
        ar[i] = 0.72 * ar[i - 1] + innov[i]
    cal["avg_temp_c"] = np.round(seasonal_temp + ar, 2)

    # --- ER wait time: congestion proxy, partly weather / weekday driven ---
    cal["er_wait_mins"] = np.round(
        38.0
        + 9.0 * (cal["dow"] == 0)                     # Monday crush
        - 0.45 * (cal["avg_temp_c"] - 14.0)           # cold -> busier -> longer
        + rng.normal(0, 5.0, n),
        1,
    ).clip(8, None)

    # --- Year-3 flu wave: Gaussian bump, shared shock across the panel -----
    cal["flu_wave"] = np.exp(
        -0.5 * ((cal["t"] - FLU_CENTER_DAY) / FLU_WIDTH_DAYS) ** 2
    )
    cal.loc[cal["t"] < 731, "flu_wave"] = 0.0   # confine it to year 3

    return cal


# --------------------------------------------------------------------------
# Per-department series generation
# --------------------------------------------------------------------------
def generate_department(spec: dict, cal: pd.DataFrame, rng) -> pd.DataFrame:
    """Build one department's series from the known multiplicative components."""
    t = cal["t"].to_numpy()
    n = len(t)

    # --- Trend: linear growth with a structural BREAK (level step + new slope)
    trend = 1.0 + spec["slope"] * (t / 365.25)
    post = t >= BREAK_DAY
    trend[post] = (
        spec["break_step"]
        * (1.0 + spec["slope"] * (BREAK_DAY / 365.25))
        + spec["break_slope"] * ((t[post] - BREAK_DAY) / 365.25)
    )

    # --- Weekly profile (department-specific day-of-week multipliers) -------
    weekly = np.array(spec["dow"])[cal["dow"].to_numpy()]

    # --- Annual seasonality (cosine peaking at phase_yearly) ---------------
    yearly = 1.0 + spec["amp_yearly"] * np.cos(
        2 * np.pi * (cal["day_of_year"].to_numpy() - spec["phase_yearly"]) / 365.25
    )

    # --- Holidays: day-of and shoulder-window effects -----------------------
    hol = np.ones(n)
    win = cal["is_holiday_window"].to_numpy() == 1
    day = cal["is_holiday"].to_numpy() == 1
    hol[win] = spec["holiday_window_mult"]
    hol[day] = spec["holiday_mult"]      # exact day overrides the window

    # --- Flu wave -----------------------------------------------------------
    flu = 1.0 + spec["flu_sens"] * cal["flu_wave"].to_numpy()

    # --- Lagged temperature effect -----------------------------------------
    # 5-day lag: a cold snap shows up in respiratory admissions about a week
    # later. Using same-day temperature would be physiologically wrong and
    # would also make the covariate artificially easy to exploit.
    temp_lag = pd.Series(cal["avg_temp_c"]).shift(5).bfill().to_numpy()
    temp_eff = 1.0 + spec["temp_sens"] * (temp_lag - 14.0)

    # --- School term --------------------------------------------------------
    school = 1.0 + spec["school_sens"] * cal["is_school_term"].to_numpy()

    mu = (
        spec["base"] * trend * weekly * yearly
        * hol * flu * temp_eff * school
    ).clip(0.35, None)   # keep the NegBinomial mean strictly positive

    # --- Negative Binomial counts ------------------------------------------
    # numpy parameterises NB by (n_success, p). For a target mean mu and
    # dispersion r:  p = r / (r + mu)  =>  E[y] = r(1-p)/p = mu
    # and Var[y] = mu + mu^2/r, i.e. overdispersed relative to Poisson.
    r = spec["disp_r"]
    p = r / (r + mu)
    y = rng.negative_binomial(r, p)

    out = pd.DataFrame({
        "dept_id": spec["dept_id"],
        "admit_date": cal["admit_date"],
        "admissions": y.astype(int),
        # known-future covariates
        "dow": cal["dow"], "month": cal["month"],
        "day_of_year": cal["day_of_year"], "is_weekend": cal["is_weekend"],
        "is_holiday": cal["is_holiday"],
        "is_holiday_window": cal["is_holiday_window"],
        "is_school_term": cal["is_school_term"],
        "holiday_name": cal["holiday_name"],
        # past-only covariates
        "avg_temp_c": cal["avg_temp_c"], "er_wait_mins": cal["er_wait_mins"],
    })

    truth = pd.DataFrame({
        "dept_id": spec["dept_id"],
        "admit_date": cal["admit_date"],
        "mu_expected": np.round(mu, 4),
        "comp_base": spec["base"],
        "comp_trend": np.round(trend, 5),
        "comp_weekly": np.round(weekly, 5),
        "comp_yearly": np.round(yearly, 5),
        "comp_holiday": np.round(hol, 5),
        "comp_flu": np.round(flu, 5),
        "comp_temp": np.round(temp_eff, 5),
        "comp_school": np.round(school, 5),
        "nb_dispersion_r": r,
    })
    return out, truth


def main() -> None:
    rng = np.random.default_rng(SEED)
    cal = build_calendar()

    print("=" * 74)
    print("SYNTHETIC HEALTHCARE ADMISSIONS PANEL")
    print("=" * 74)
    print(f"date range      : {cal['admit_date'].min().date()} .. "
          f"{cal['admit_date'].max().date()}")
    print(f"days per series : {len(cal)}")
    print(f"departments     : {len(DEPARTMENTS)}")
    print(f"TOTAL ROWS      : {len(cal) * len(DEPARTMENTS)}  "
          f"(budget 10,000)")
    print(f"holidays        : {int(cal['is_holiday'].sum())} days, "
          f"{int(cal['is_holiday_window'].sum())} in +/-3d window")
    print(f"structural break: day {BREAK_DAY} "
          f"({cal.loc[cal['t'] == BREAK_DAY, 'admit_date'].iloc[0].date()})")
    print(f"flu wave        : centred day {FLU_CENTER_DAY} "
          f"({cal.loc[cal['t'] == FLU_CENTER_DAY, 'admit_date'].iloc[0].date()})"
          f", sigma={FLU_WIDTH_DAYS}d")
    print(f"temp range      : {cal['avg_temp_c'].min():.1f} .. "
          f"{cal['avg_temp_c'].max():.1f} C")

    panels, truths = [], []
    for spec in DEPARTMENTS:
        p, tr = generate_department(spec, cal, rng)
        panels.append(p)
        truths.append(tr)

    panel = pd.concat(panels, ignore_index=True)
    truth = pd.concat(truths, ignore_index=True)

    static = pd.DataFrame([
        {"dept_id": s["dept_id"], "dept_type": s["dept_type"],
         "bed_capacity": s["bed_capacity"],
         "seasonal_peak_doy": s["phase_yearly"],
         "weekend_operating": int(min(s["dow"][5], s["dow"][6]) > 0.25)}
        for s in DEPARTMENTS
    ])

    # ---- per-series summary, including the overdispersion check ------------
    #
    # NOTE on measuring overdispersion correctly: the MARGINAL var/mean of the
    # raw series is inflated by trend + seasonality and would overstate the
    # noise badly. The meaningful quantity is the CONDITIONAL dispersion given
    # the known mean mu, i.e. Var[y | mu] / mu, which for our NegBinomial
    # should be approximately 1 + mu/r  (and exactly 1.0 under Poisson).
    # Because we generated the data we can compute this exactly, which is a
    # check that is simply unavailable on real data.
    truth_mu = truth.set_index(["dept_id", "admit_date"])["mu_expected"]
    panel_mu = panel.set_index(["dept_id", "admit_date"]).index.map(truth_mu)
    panel = panel.assign(_mu=panel_mu.to_numpy())

    print("\n" + "-" * 74)
    print("PER-DEPARTMENT SUMMARY")
    print("-" * 74)
    print(f"{'dept':<18}{'n':>6}{'mean':>9}{'std':>8}{'min':>6}{'max':>7}"
          f"{'zeros':>7}{'cond.disp':>11}{'expected':>10}")
    for d, g in panel.groupby("dept_id", sort=False):
        a = g["admissions"]
        mu = g["_mu"]
        # Var[y|mu]/mu estimated as mean((y-mu)^2)/mean(mu)
        cond_disp = ((a - mu) ** 2).mean() / mu.mean()
        r = next(s["disp_r"] for s in DEPARTMENTS if s["dept_id"] == d)
        expected = 1.0 + mu.mean() / r
        print(f"{d:<18}{len(g):>6}{a.mean():>9.2f}{a.std():>8.2f}"
              f"{a.min():>6}{a.max():>7}{int((a == 0).sum()):>7}"
              f"{cond_disp:>11.2f}{expected:>10.2f}")
    print("  cond.disp = Var[y|mu]/mu  (1.0 would mean Poisson; >1 = "
          "overdispersed, as intended)")
    panel = panel.drop(columns=["_mu"])

    print("\nweekday vs weekend mean (exposes the hard weekly pattern):")
    wk = panel.groupby(["dept_id", "is_weekend"])["admissions"].mean().unstack()
    wk.columns = ["weekday", "weekend"]
    wk["weekend_ratio"] = (wk["weekend"] / wk["weekday"]).round(3)
    print(wk.round(2).to_string())

    print("\nflu-wave impact (year-3 peak +/-30d vs same calendar window y1/y2):")
    peak = cal.loc[(cal["t"] >= FLU_CENTER_DAY - 30) &
                   (cal["t"] <= FLU_CENTER_DAY + 30), "admit_date"]
    pk = panel[panel["admit_date"].isin(peak)]
    base_doy = set(peak.dt.dayofyear)
    bl = panel[(panel["admit_date"].dt.dayofyear.isin(base_doy)) &
               (panel["admit_date"].dt.year < 2025)]
    cmp = pd.DataFrame({
        "y1y2_mean": bl.groupby("dept_id")["admissions"].mean(),
        "y3_surge_mean": pk.groupby("dept_id")["admissions"].mean(),
    })
    cmp["lift"] = (cmp["y3_surge_mean"] / cmp["y1y2_mean"]).round(3)
    print(cmp.round(2).to_string())

    # ---- write to Snowflake ----------------------------------------------
    print("\n" + "-" * 74)
    print(f"WRITING TO {DATABASE}.{SCHEMA}")
    print("-" * 74)
    session = create_snowpark_session()
    session.use_database(DATABASE)
    session.use_schema(SCHEMA)
    print(f"connected: {session.get_current_account()} "
          f"{session.get_current_database()}.{session.get_current_schema()} "
          f"wh={session.get_current_warehouse()}")

    # Uppercase column names so the Snowflake identifiers are unquoted and
    # therefore queryable without escaping downstream.
    #
    # DATE HANDLING: write_pandas maps pandas datetime64 to a NUMBER column
    # holding nanosecond epochs, not a DATE. That silently breaks
    # SNOWFLAKE.ML.FORECAST, which requires a real DATE/TIMESTAMP for
    # TIMESTAMP_COLNAME. So we write the date as an ISO string and then
    # rebuild the table with an explicit TO_DATE cast.
    date_cols = {"DAILY_ADMISSIONS": "ADMIT_DATE",
                 "GROUND_TRUTH_COMPONENTS": "ADMIT_DATE",
                 "DEPT_STATIC": None}

    for df, name in [(panel, "DAILY_ADMISSIONS"),
                     (truth, "GROUND_TRUTH_COMPONENTS"),
                     (static, "DEPT_STATIC")]:
        w = df.copy()
        w.columns = [c.upper() for c in w.columns]
        dcol = date_cols[name]
        if dcol:
            w[dcol] = pd.to_datetime(w[dcol]).dt.strftime("%Y-%m-%d")

        session.write_pandas(
            w, name, auto_create_table=True, overwrite=True,
            quote_identifiers=False,
        )

        if dcol:
            # Rebuild with the date column properly typed, preserving column
            # order so downstream SELECT * stays stable.
            cols = ", ".join(
                f"TO_DATE({c}) AS {c}" if c == dcol else c for c in w.columns
            )
            session.sql(
                f"CREATE OR REPLACE TABLE {name} AS "
                f"SELECT {cols} FROM {name}"
            ).collect()

        cnt = session.table(name).count()
        print(f"  {name:<26} {cnt:>6} rows"
              + (f"  ({dcol} cast to DATE)" if dcol else ""))

    print("\nsanity check from Snowflake:")
    session.sql(
        "SELECT DEPT_ID, COUNT(*) AS N, MIN(ADMIT_DATE) AS FROM_DT, "
        "MAX(ADMIT_DATE) AS TO_DT, ROUND(AVG(ADMISSIONS),2) AS AVG_ADM "
        "FROM DAILY_ADMISSIONS GROUP BY DEPT_ID ORDER BY DEPT_ID"
    ).show(10)

    session.close()
    print("=" * 74)
    print("STEP 1 COMPLETE")
    print("=" * 74)


if __name__ == "__main__":
    main()
