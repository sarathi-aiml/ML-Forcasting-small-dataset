"""
03_features.py
==============
Leak-safe feature engineering for the small-data healthcare forecasting panel.

Importable library (used by trials 05/06/08) plus a `--selftest` mode that
proves the leakage and dimensionality properties rather than asserting them.

THE governing constraint: dimensionality
----------------------------------------
Each series has ~1,100 observations. Standard "throw features at it" practice
is actively harmful at this size. One-hot encoding the calendar the obvious way
costs:

    day-of-week dummies      7
    month dummies           12
    day-of-year dummies    365
    ------------------------------
    total                  384 features against ~1,100 rows

That is a variance catastrophe, and it also encodes a falsehood: as raw
integers or independent dummies, 31 Dec and 1 Jan are maximally distant when
they are in fact adjacent.

Cyclical + Fourier encoding buys the same information for ~14 features:

    dow sin/cos              2
    month sin/cos            2
    annual Fourier (K=3)     6
    holiday proximity        4
    ------------------------------
    total                   14

That is a ~27x reduction with no loss of expressiveness for smooth seasonality,
which is the single highest-leverage feature decision in this whole project.

The covariate availability contract
-----------------------------------
Features are returned in three STRICTLY separated groups, because mixing them
is the leakage bug that quietly inflates backtest scores:

  known-future : computable for any future date from the calendar alone.
                 Safe to pass to AutoGluon as `known_covariates_names` and to
                 SNOWFLAKE.ML.FORECAST as exogenous features.
  past-only    : temperature, ER wait, and transforms of them. Observed history
                 only - you do NOT know next month's daily temperature.
                 AutoGluon may use their history; they must NEVER be declared
                 known-future.
  static       : constant per department.

Target lags are deliberately NOT built here. AutoGluon's tabular forecasters
generate target lags internally with correct horizon alignment; hand-rolling
them alongside would be redundant and is an easy way to introduce a lag shorter
than the forecast horizon, which leaks.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Reference temperature for degree-day construction (deg C).
BASE_TEMP_C = 18.0

# Number of annual Fourier harmonics. K=3 captures the winter peak, the summer
# trough, and one asymmetry term. Raising K adds 2 features each and starts
# fitting noise at this sample size.
ANNUAL_HARMONICS = 3

# Respiratory admissions follow a cold snap with a lag of roughly a week.
TEMP_LAGS = (3, 5, 7)
TEMP_ROLL_WINDOWS = (7, 14)

TIME_COL = "ADMIT_DATE"
ID_COL = "DEPT_ID"
TARGET = "ADMISSIONS"


# ==========================================================================
# Known-future calendar features
# ==========================================================================
def _fourier_terms(day_of_year: np.ndarray, k: int, period: float = 365.25
                   ) -> dict[str, np.ndarray]:
    """Fourier basis for smooth annual seasonality."""
    out = {}
    for h in range(1, k + 1):
        ang = 2.0 * np.pi * h * day_of_year / period
        out[f"ANN_SIN{h}"] = np.sin(ang)
        out[f"ANN_COS{h}"] = np.cos(ang)
    return out


def build_known_future(dates: pd.Series, holiday_dates: set) -> pd.DataFrame:
    """Calendar features computable for ANY date, past or future.

    Takes a bare date series so it can be called for future horizon dates that
    do not exist in the training panel - which is exactly the property that
    makes a covariate legitimately "known future".
    """
    d = pd.to_datetime(pd.Series(dates)).reset_index(drop=True)
    doy = d.dt.dayofyear.to_numpy(float)
    dow = d.dt.dayofweek.to_numpy(float)
    month = d.dt.month.to_numpy(float)

    f = pd.DataFrame({TIME_COL: d})

    # --- cyclical encodings (2 features each instead of 7 / 12 dummies) ----
    f["DOW_SIN"] = np.sin(2 * np.pi * dow / 7.0)
    f["DOW_COS"] = np.cos(2 * np.pi * dow / 7.0)
    f["MONTH_SIN"] = np.sin(2 * np.pi * month / 12.0)
    f["MONTH_COS"] = np.cos(2 * np.pi * month / 12.0)

    # --- annual Fourier basis (6 features instead of 365 dummies) ---------
    for k, v in _fourier_terms(doy, ANNUAL_HARMONICS).items():
        f[k] = v

    # --- weekend / weekday-position -----------------------------------------
    # IS_WEEKEND is kept as an explicit binary despite the sin/cos pair because
    # the weekend drop in scheduled departments is a STEP, not a smooth cycle -
    # ELECTIVE_SURGERY falls to ~4% of its weekday level. Fourier terms alone
    # cannot represent a discontinuity that sharp without many harmonics.
    f["IS_WEEKEND"] = (dow >= 5).astype(int)
    f["IS_MONDAY"] = (dow == 0).astype(int)

    # --- holiday flags and PROXIMITY ---------------------------------------
    hol = np.array([1 if x.date() in holiday_dates else 0 for x in d])
    f["IS_HOLIDAY"] = hol
    f["IS_HOLIDAY_WINDOW"] = _window_flag(hol, 3)
    # Signed distance to the nearest holiday. A plain flag cannot express the
    # run-up/run-down around Christmas; distance can, in 2 features.
    nxt, prv = _holiday_distances(d, holiday_dates)
    f["DAYS_TO_HOLIDAY"] = np.clip(nxt, 0, 14)
    f["DAYS_SINCE_HOLIDAY"] = np.clip(prv, 0, 14)

    # --- school term (paediatric transmission driver) ----------------------
    f["IS_SCHOOL_TERM"] = (~(
        np.isin(month, [7, 8]) | (doy >= 356) | (doy <= 6)
    )).astype(int)

    return f


def _window_flag(flag: np.ndarray, half_width: int) -> np.ndarray:
    n = len(flag)
    out = np.zeros(n, dtype=int)
    for i in np.flatnonzero(flag == 1):
        out[max(0, i - half_width): min(n, i + half_width + 1)] = 1
    return out


def _holiday_distances(d: pd.Series, holiday_dates: set
                       ) -> tuple[np.ndarray, np.ndarray]:
    """Days to next / since previous holiday, computed on the calendar only."""
    if not holiday_dates:
        n = len(d)
        return np.full(n, 14), np.full(n, 14)
    hd = np.array(sorted(pd.Timestamp(x).value for x in holiday_dates))
    vals = d.astype("int64").to_numpy()
    day_ns = 86_400_000_000_000
    nxt, prv = [], []
    for v in vals:
        after = hd[hd >= v]
        before = hd[hd <= v]
        nxt.append((after[0] - v) // day_ns if len(after) else 14)
        prv.append((v - before[-1]) // day_ns if len(before) else 14)
    return np.array(nxt), np.array(prv)


# ==========================================================================
# Past-only covariate features
# ==========================================================================
def build_past_only(panel: pd.DataFrame) -> pd.DataFrame:
    """Transforms of observed-history-only covariates (temperature, ER wait).

    These are legitimate PAST covariates. They must never be declared
    known-future: at a real forecast origin you do not have next month's daily
    temperature. Passing them as known-future would inflate the backtest and
    then fail in production.
    """
    out = []
    for dept, g in panel.groupby(ID_COL, sort=False):
        g = g.sort_values(TIME_COL).copy()

        # Degree days: the temperature response is U-shaped (cold drives
        # respiratory admissions, heat drives heat-related ones), which a single
        # linear temperature term cannot represent. Two hinge features can.
        g["HDD"] = np.maximum(BASE_TEMP_C - g["AVG_TEMP_C"], 0.0)
        g["CDD"] = np.maximum(g["AVG_TEMP_C"] - BASE_TEMP_C, 0.0)

        # Lagged temperature - the physiological delay between a cold snap and
        # the resulting admissions.
        for L in TEMP_LAGS:
            g[f"TEMP_LAG{L}"] = g["AVG_TEMP_C"].shift(L)
            g[f"HDD_LAG{L}"] = g["HDD"].shift(L)

        # Rolling means capture a sustained cold SPELL rather than one cold day.
        # shift(1) first so the window never includes the current observation.
        for W in TEMP_ROLL_WINDOWS:
            g[f"TEMP_ROLL{W}"] = g["AVG_TEMP_C"].shift(1).rolling(W).mean()
            g[f"HDD_ROLL{W}"] = g["HDD"].shift(1).rolling(W).mean()

        g["ER_WAIT_ROLL7"] = g["ER_WAIT_MINS"].shift(1).rolling(7).mean()

        out.append(g)

    res = pd.concat(out, ignore_index=True)
    # Warm-up NaNs from lags/rolls: back-fill WITHIN each department only, so no
    # value ever crosses a department boundary.
    cols = [c for c in res.columns if c.startswith(
        ("TEMP_", "HDD", "CDD", "ER_WAIT_ROLL"))]
    res[cols] = res.groupby(ID_COL)[cols].bfill()
    return res


# ==========================================================================
# Static features
# ==========================================================================
def build_static(static: pd.DataFrame) -> pd.DataFrame:
    """Per-department constants, indexed by DEPT_ID.

    The index name stays DEPT_ID (not item_id) because
    TimeSeriesDataFrame.from_data_frame uses the SAME `id_column` name for both
    the long frame and static_features_df, and renames to item_id itself.
    """
    s = static.copy()
    s["LOG_CAPACITY"] = np.log1p(s["BED_CAPACITY"])
    keep = [ID_COL, "DEPT_TYPE", "BED_CAPACITY", "LOG_CAPACITY",
            "SEASONAL_PEAK_DOY", "WEEKEND_OPERATING"]
    return s[[c for c in keep if c in s.columns]].set_index(ID_COL)


# ==========================================================================
# Assembly
# ==========================================================================
def known_future_names() -> list[str]:
    """Exact known-future feature names, in a stable order."""
    names = ["DOW_SIN", "DOW_COS", "MONTH_SIN", "MONTH_COS"]
    for h in range(1, ANNUAL_HARMONICS + 1):
        names += [f"ANN_SIN{h}", f"ANN_COS{h}"]
    names += ["IS_WEEKEND", "IS_MONDAY", "IS_HOLIDAY", "IS_HOLIDAY_WINDOW",
              "DAYS_TO_HOLIDAY", "DAYS_SINCE_HOLIDAY", "IS_SCHOOL_TERM"]
    return names


def past_only_names() -> list[str]:
    names = ["AVG_TEMP_C", "ER_WAIT_MINS", "HDD", "CDD", "ER_WAIT_ROLL7"]
    for L in TEMP_LAGS:
        names += [f"TEMP_LAG{L}", f"HDD_LAG{L}"]
    for W in TEMP_ROLL_WINDOWS:
        names += [f"TEMP_ROLL{W}", f"HDD_ROLL{W}"]
    return names


def build_features(
    panel: pd.DataFrame,
    holiday_dates: set,
    include_past_only: bool = True,
) -> tuple[pd.DataFrame, list[str], list[str]]:
    """Assemble the modelling frame.

    Returns (frame, known_future_cols, past_only_cols).
    """
    base = panel.copy()
    base[TIME_COL] = pd.to_datetime(base[TIME_COL])

    # Drop the raw calendar columns that the encoded versions replace, plus the
    # sparse high-cardinality holiday name flagged in Gate 1.
    drop = ["DOW", "MONTH", "DAY_OF_YEAR", "HOLIDAY_NAME",
            "IS_WEEKEND", "IS_HOLIDAY", "IS_HOLIDAY_WINDOW", "IS_SCHOOL_TERM"]
    base = base.drop(columns=[c for c in drop if c in base.columns])

    kf = build_known_future(
        pd.Series(sorted(base[TIME_COL].unique())), holiday_dates
    )
    frame = base.merge(kf, on=TIME_COL, how="left")

    past_cols: list[str] = []
    if include_past_only:
        frame = build_past_only(frame)
        past_cols = [c for c in past_only_names() if c in frame.columns]

    kf_cols = [c for c in known_future_names() if c in frame.columns]
    return frame.sort_values([ID_COL, TIME_COL]).reset_index(drop=True), \
        kf_cols, past_cols


def future_known_frame(
    last_date: pd.Timestamp, horizon: int, dept_ids: list[str],
    holiday_dates: set,
) -> pd.DataFrame:
    """Known-future covariates for the horizon AFTER `last_date`.

    Existence of this function IS the leakage test: if a feature cannot be
    produced here from the calendar alone, it is not known-future and does not
    belong in that group.
    """
    dates = pd.date_range(last_date + pd.Timedelta(days=1),
                          periods=horizon, freq="D")
    kf = build_known_future(pd.Series(dates), holiday_dates)
    rows = []
    for d in dept_ids:
        t = kf.copy()
        t[ID_COL] = d
        rows.append(t)
    return pd.concat(rows, ignore_index=True)


# ==========================================================================
# Self-test
# ==========================================================================
def _selftest() -> None:
    import holidays as _h
    from snowpark_session import create_snowpark_session

    print("=" * 74)
    print("FEATURE LIBRARY SELF-TEST")
    print("=" * 74)

    session = create_snowpark_session()
    session.use_database("HEALTHCARE")
    session.use_schema("ML_FORECAST")
    panel = session.table("DAILY_ADMISSIONS").to_pandas()
    static = session.table("DEPT_STATIC").to_pandas()
    session.close()
    panel[TIME_COL] = pd.to_datetime(panel[TIME_COL])

    yrs = sorted(panel[TIME_COL].dt.year.unique())
    hol = set(_h.US(years=list(yrs) + [max(yrs) + 1]).keys())

    frame, kf, past = build_features(panel, hol)
    stat = build_static(static)

    print(f"\ninput panel        : {panel.shape[0]} rows x {panel.shape[1]} cols")
    print(f"feature frame      : {frame.shape[0]} rows x {frame.shape[1]} cols")
    print(f"known-future feats : {len(kf)}")
    print(f"past-only feats    : {len(past)}")
    print(f"static feats       : {stat.shape[1]}")
    print(f"TOTAL model feats  : {len(kf) + len(past) + stat.shape[1]}")

    obs_per_series = len(panel) // panel[ID_COL].nunique()
    tot = len(kf) + len(past) + stat.shape[1]
    print(f"\nobs per series     : {obs_per_series}")
    print(f"obs : feature ratio: {obs_per_series / tot:.1f} : 1")
    print("  (>10:1 is a reasonable floor for small-data tabular forecasting)")

    naive_dims = 7 + 12 + 365
    print(f"\ndimensionality win : {len(kf)} encoded vs {naive_dims} "
          f"one-hot equivalents = {naive_dims/len(kf):.1f}x reduction")

    # --- TEST 1: known-future computable beyond the panel -------------------
    print("\n[TEST 1] known-future features computable for UNSEEN future dates")
    last = frame[TIME_COL].max()
    fut = future_known_frame(last, 28, sorted(frame[ID_COL].unique()), hol)
    missing = [c for c in kf if c not in fut.columns]
    nan_ct = int(fut[kf].isna().sum().sum())
    print(f"  horizon frame     : {fut.shape[0]} rows "
          f"({fut[ID_COL].nunique()} depts x 28 days)")
    print(f"  dates             : {fut[TIME_COL].min().date()} .. "
          f"{fut[TIME_COL].max().date()}  (all > {last.date()})")
    print(f"  missing kf cols   : {missing or 'none'}")
    print(f"  NaNs in kf block  : {nan_ct}")
    assert not missing and nan_ct == 0, "known-future block is not future-computable"
    print("  PASS - every known-future feature is derivable from the calendar alone")

    # --- TEST 2: past-only features are NOT future-computable --------------
    print("\n[TEST 2] past-only features correctly ABSENT from the future frame")
    leaked = [c for c in past if c in fut.columns]
    print(f"  past-only cols present in future frame: {leaked or 'none'}")
    assert not leaked, f"LEAK: past-only cols exposed as known-future: {leaked}"
    print("  PASS - temperature / ER wait cannot be fabricated for the future")

    # --- TEST 3: no NaNs left in the modelling frame -----------------------
    print("\n[TEST 3] no residual NaNs in engineered columns")
    n = frame[kf + past].isna().sum()
    bad = n[n > 0]
    print(f"  columns with NaNs : {dict(bad) if len(bad) else 'none'}")
    assert len(bad) == 0, "NaNs remain after within-department bfill"
    print("  PASS")

    # --- TEST 4: rolling features never peek at the current row ------------
    print("\n[TEST 4] rolling features exclude the current observation")
    g = frame[frame[ID_COL] == "EMERGENCY"].sort_values(TIME_COL).reset_index(drop=True)
    # TEMP_ROLL7 at row i must equal mean(AVG_TEMP_C[i-7 : i]) - not including i.
    i = 400
    expect = g["AVG_TEMP_C"].iloc[i - 7:i].mean()
    got = g["TEMP_ROLL7"].iloc[i]
    print(f"  row {i}: expected {expect:.4f}, got {got:.4f}, "
          f"diff {abs(expect-got):.2e}")
    assert abs(expect - got) < 1e-9, "rolling window includes current row (leak)"
    # And it must NOT match the window that includes row i.
    incl = g["AVG_TEMP_C"].iloc[i - 6:i + 1].mean()
    print(f"  window including current row would be {incl:.4f} "
          f"-> correctly NOT used")
    print("  PASS - shift(1) before rolling is effective")

    # --- TEST 5: cyclical encoding preserves adjacency ---------------------
    print("\n[TEST 5] cyclical encoding makes Dec 31 and Jan 1 adjacent")
    a = build_known_future(pd.Series([pd.Timestamp("2025-12-31")]), hol)
    b = build_known_future(pd.Series([pd.Timestamp("2026-01-01")]), hol)
    ann = [f"ANN_SIN1", "ANN_COS1"]
    dist = float(np.linalg.norm(a[ann].to_numpy() - b[ann].to_numpy()))
    mid = build_known_future(pd.Series([pd.Timestamp("2025-07-01")]), hol)
    dist_far = float(np.linalg.norm(a[ann].to_numpy() - mid[ann].to_numpy()))
    print(f"  |Dec31 - Jan1| in annual Fourier space : {dist:.5f}")
    print(f"  |Dec31 - Jul01| in annual Fourier space : {dist_far:.5f}")
    assert dist < dist_far, "cyclical encoding failed to wrap the year"
    print(f"  PASS - adjacent dates are {dist_far/max(dist,1e-9):.0f}x closer "
          f"than opposite dates")

    print("\n" + "=" * 74)
    print("ALL FEATURE SELF-TESTS PASSED")
    print("=" * 74)
    print("known-future:", ", ".join(kf))
    print("past-only   :", ", ".join(past))
    print("static      :", ", ".join(stat.columns))


if __name__ == "__main__":
    import sys
    if "--selftest" in sys.argv:
        _selftest()
    else:
        print(__doc__)
        print("run with --selftest to validate leakage and dimensionality "
              "properties")
