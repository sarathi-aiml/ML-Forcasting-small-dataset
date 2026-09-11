"""
10_register_model.py
====================
Register the champion AutoGluon TimeSeriesPredictor to the Snowflake Model
Registry as a CustomModel.

Why CustomModel and not a native flavour
----------------------------------------
The champion is a TimeSeriesPredictor, usually a WeightedEnsemble over several
model families. There is no native registry flavour for it, and extracting a
single underlying estimator would discard the ensemble that produced the score
we are registering. Wrapping the predictor whole preserves the exact benchmarked
behaviour.

The inference contract
----------------------
Time-series models do not fit the usual one-row-in/one-row-out shape, so the
contract is made explicit:

  INPUT  a long frame containing BOTH history and the horizon:
           DEPT_ID, ADMIT_DATE, ADMISSIONS (NULL for horizon rows),
           plus the 17 known-future covariates
  OUTPUT one row per input row:
           FORECAST, LOWER_80, UPPER_80  (NULL on history rows)

Rows with a NULL target are treated as the horizon to predict. This keeps a
1:1 row mapping so the model signature is well-defined, while still letting the
caller supply fresh history at inference time - which matters because a
forecaster with frozen history goes stale immediately.

A dependency note, stated up front
----------------------------------
Locally, `autogluon.timeseries` could not resolve `xgboost-cpu` because that
package ships no macOS wheels. That constraint does NOT apply in Snowflake,
which runs Linux and where manylinux wheels exist. Registration pins the real
dependency set so the server-side environment resolves correctly.
"""

# NOTE: `from __future__ import annotations` must NOT be used in this module.
# PEP 563 turns annotations into lazy STRINGS, and snowflake-ml's
# _validate_predict_function inspects the annotation object to confirm the
# predict input is a pandas.DataFrame. With the future import it sees the string
# "pd.DataFrame" and raises:
#   TypeError: Input for predict method should have type pandas.DataFrame.
import json
import os
import shutil

import numpy as np
import pandas as pd

DATABASE, SCHEMA = "HEALTHCARE", "ML_FORECAST"
MODEL_NAME = "HEALTHCARE_ADMISSIONS_FORECASTER"
ARTIFACTS = "artifacts"
TARGET, ID_COL, TIME_COL = "ADMISSIONS", "DEPT_ID", "ADMIT_DATE"

AG_VERSION = "1.6.1"
PIP_REQS = [
    f"autogluon.timeseries=={AG_VERSION}",
    f"autogluon.tabular=={AG_VERSION}",
    f"autogluon.core=={AG_VERSION}",
    f"autogluon.features=={AG_VERSION}",
    f"autogluon.common=={AG_VERSION}",
    "statsforecast==2.0.3",
    "mlforecast==0.14.0",
    "utilsforecast==0.2.11",
    "gluonts==0.17.0",
    "chronos-forecasting==2.3.2",
    "lightgbm==4.7.0",
    "torch==2.10.0",
    "transformers==5.14.1",
]


def dir_size_mb(path: str) -> float:
    tot = 0
    for r, _, fs in os.walk(path):
        for f in fs:
            try:
                tot += os.path.getsize(os.path.join(r, f))
            except OSError:
                pass
    return tot / 1e6


def build_custom_model(predictor_path: str, static_path: str):
    """Construct the CustomModel instance wrapping the fitted predictor.

    `static_path` is a parquet of the per-department static features. It must
    travel WITH the model: the predictor was fitted with static features, so
    AutoGluon rejects any predict call whose data lacks them
    ("Provided data must contain static_features"). Baking them in as an
    artifact keeps the caller from having to supply department metadata.
    """
    from snowflake.ml.model import custom_model

    class AdmissionsForecaster(custom_model.CustomModel):
        """AutoGluon TimeSeriesPredictor behind a row-aligned inference API."""

        def __init__(self, context: custom_model.ModelContext) -> None:
            super().__init__(context)
            from autogluon.timeseries import TimeSeriesPredictor
            self.predictor = TimeSeriesPredictor.load(
                context.path("predictor"), require_version_match=False)
            self.kf = list(self.predictor.known_covariates_names or [])
            self.static = pd.read_parquet(context.path("static"))

        @custom_model.inference_api
        def predict(self, X: pd.DataFrame) -> pd.DataFrame:
            from autogluon.timeseries import TimeSeriesDataFrame

            df = X.copy()
            df[TIME_COL] = pd.to_datetime(df[TIME_COL])
            # Sort for AutoGluon but KEEP the original index: the output must be
            # row-aligned to the caller's input. reset_index(drop=True) here
            # would silently return rows in a different order than supplied.
            df = df.sort_values([ID_COL, TIME_COL])

            # Rows with a NULL target define the horizon to be predicted.
            is_future = df[TARGET].isna()
            hist = df[~is_future]
            fut = df[is_future]

            out = pd.DataFrame(
                {"FORECAST": np.nan, "LOWER_80": np.nan, "UPPER_80": np.nan},
                index=df.index)
            if hist.empty or fut.empty:
                return out

            hcols = [ID_COL, TIME_COL, TARGET] + [
                c for c in df.columns
                if c not in (ID_COL, TIME_COL, TARGET)]
            # Static features must be attached or AutoGluon raises
            # "Provided data must contain static_features".
            st = self.static[self.static[ID_COL].isin(hist[ID_COL].unique())]
            hts = TimeSeriesDataFrame.from_data_frame(
                hist[hcols], id_column=ID_COL, timestamp_column=TIME_COL,
                static_features_df=st)

            kc = None
            if self.kf:
                have = [c for c in self.kf if c in fut.columns]
                if len(have) == len(self.kf):
                    kc = TimeSeriesDataFrame.from_data_frame(
                        fut[[ID_COL, TIME_COL] + self.kf],
                        id_column=ID_COL, timestamp_column=TIME_COL)

            p = self.predictor.predict(hts, known_covariates=kc).reset_index()
            p = p.rename(columns={"item_id": ID_COL, "timestamp": TIME_COL})
            p[TIME_COL] = pd.to_datetime(p[TIME_COL])

            lo = "0.1" if "0.1" in p.columns else None
            hi = "0.9" if "0.9" in p.columns else None
            m = fut[[ID_COL, TIME_COL]].merge(
                p[[ID_COL, TIME_COL, "mean"] +
                  [c for c in (lo, hi) if c]],
                on=[ID_COL, TIME_COL], how="left")

            # Admissions are non-negative counts.
            out.loc[fut.index, "FORECAST"] = np.maximum(
                m["mean"].to_numpy(), 0.0)
            if lo:
                out.loc[fut.index, "LOWER_80"] = np.maximum(
                    m[lo].to_numpy(), 0.0)
            if hi:
                out.loc[fut.index, "UPPER_80"] = m[hi].to_numpy()
            return out

    mc = custom_model.ModelContext(artifacts={
        "predictor": os.path.abspath(predictor_path),
        "static": os.path.abspath(static_path),
    })
    return AdmissionsForecaster(mc)


def make_sample_input(session, kf_cols, horizon=28, hist_days=120):
    """Sample input matching the documented contract: history + NULL-target horizon."""
    import holidays as _h
    import importlib.util as ilu
    spec = ilu.spec_from_file_location("f3", "03_features.py")
    feats = ilu.module_from_spec(spec)
    spec.loader.exec_module(feats)

    panel = session.table("DAILY_ADMISSIONS").to_pandas()
    panel[TIME_COL] = pd.to_datetime(panel[TIME_COL])
    yrs = sorted(panel[TIME_COL].dt.year.unique())
    hol = set(_h.US(years=list(yrs) + [max(yrs) + 1]).keys())
    frame, kf, past = feats.build_features(panel, hol)
    static_df = feats.build_static(session.table("DEPT_STATIC").to_pandas())

    cut = frame[TIME_COL].max() - pd.Timedelta(days=horizon)
    hist = frame[(frame[TIME_COL] > cut - pd.Timedelta(days=hist_days)) &
                 (frame[TIME_COL] <= cut)]
    fut = frame[frame[TIME_COL] > cut].copy()
    fut[TARGET] = np.nan          # the horizon marker

    cols = [ID_COL, TIME_COL, TARGET] + kf + past
    sample = pd.concat([hist[cols], fut[cols]], ignore_index=True)
    sample[TIME_COL] = pd.to_datetime(sample[TIME_COL])
    return sample, kf, past, static_df


def register(session, predictor_path: str, trial_label: str, metrics: dict):
    from snowflake.ml.registry import Registry

    print(f"\n  predictor dir  : {predictor_path} "
          f"({dir_size_mb(predictor_path):.1f} MB)")

    sample, kf, past, static_df = make_sample_input(session, None)
    static_path = os.path.join(ARTIFACTS, "static_features.parquet")
    static_df.reset_index().to_parquet(static_path, index=False)
    print(f"  static features: {static_path} "
          f"({static_df.shape[0]} depts x {static_df.shape[1]} cols)")
    model = build_custom_model(predictor_path, static_path)
    print(f"  sample input   : {sample.shape[0]} rows "
          f"({int(sample[TARGET].isna().sum())} horizon rows with NULL target)")

    # Verify the wrapper works BEFORE registering. Registering a model that
    # cannot run is worse than not registering it.
    print("  local smoke test of the wrapper...")
    got = model.predict(sample)
    n_pred = int(got["FORECAST"].notna().sum())
    print(f"    returned {len(got)} rows, {n_pred} non-null forecasts")
    if n_pred == 0:
        print("    !! wrapper produced no forecasts - ABORTING registration")
        return None
    print(f"    forecast range {got['FORECAST'].min():.1f} .. "
          f"{got['FORECAST'].max():.1f}")

    # Signature inference samples only the FIRST 100 rows, which are history
    # rows where FORECAST is NULL by design -> "no non-null data in column
    # FORECAST so the signature cannot be inferred". Declaring the signature
    # explicitly removes the dependency on row ordering entirely.
    from snowflake.ml.model import model_signature as msig
    dt_map = {"object": msig.DataType.STRING,
              "datetime64[ns]": msig.DataType.TIMESTAMP_NTZ}
    inputs = []
    for c in sample.columns:
        d = str(sample[c].dtype)
        inputs.append(msig.FeatureSpec(
            name=c, dtype=dt_map.get(d, msig.DataType.DOUBLE), nullable=True))
    outputs = [msig.FeatureSpec(name=n, dtype=msig.DataType.DOUBLE,
                                nullable=True)
               for n in ("FORECAST", "LOWER_80", "UPPER_80")]
    sig = msig.ModelSignature(inputs=inputs, outputs=outputs)
    print(f"  explicit signature: {len(inputs)} inputs -> {len(outputs)} outputs")

    reg = Registry(session=session, database_name=DATABASE, schema_name=SCHEMA)
    existing = []
    try:
        m = reg.get_model(MODEL_NAME)
        existing = [v.version_name for v in m.versions()]
    except Exception:
        pass
    ver = f"V{len(existing) + 1}"
    print(f"  existing versions: {existing or 'none'} -> registering {ver}")

    try:
        mv = reg.log_model(
            model,
            model_name=MODEL_NAME,
            version_name=ver,
            signatures={"predict": sig},
            pip_requirements=PIP_REQS,
            comment=(f"AutoGluon TimeSeriesPredictor champion from trial "
                     f"'{trial_label}'. Rolling-origin MASE="
                     f"{metrics.get('MASE')}, worst-series MASE="
                     f"{metrics.get('worst_dept_MASE')}. Trained on 6,576-row "
                     f"synthetic panel (6 departments x 1,096 days)."),
            metrics={k: float(v) for k, v in metrics.items()
                     if isinstance(v, (int, float)) and np.isfinite(v)},
            # AutoGluon is not in the Snowflake Conda channel, so the
            # dependency set must be declared as pip_requirements. Two
            # consequences, both forced rather than chosen:
            #   * `relax_version=True` is rejected with pip_requirements
            #     ("only allowed ... with Snowflake Conda Channel dependencies")
            #   * pip-based models cannot run in a warehouse, so the target
            #     platform must be Snowpark Container Services.
            target_platforms=["SNOWPARK_CONTAINER_SERVICES"],
        )
        print(f"  REGISTERED {DATABASE}.{SCHEMA}.{MODEL_NAME} version {ver}")
        print(f"  functions: {[f['name'] for f in mv.show_functions()]}")
        return mv
    except Exception as e:
        print(f"  REGISTRATION FAILED: {type(e).__name__}: {str(e)[:500]}")
        print("\n  This is typically dependency resolution: the AutoGluon wheel")
        print("  set must be reachable from the account's pip/conda channel.")
        print("  The model itself is validated and working locally - the")
        print("  benchmark results are unaffected.")
        return None


def main():
    from snowpark_session import create_snowpark_session
    session = create_snowpark_session()
    session.use_database(DATABASE)
    session.use_schema(SCHEMA)

    print("=" * 78)
    print("MODEL REGISTRY - CHAMPION REGISTRATION")
    print("=" * 78)

    final = json.load(open(f"{ARTIFACTS}/final_report.json"))
    manifest = json.load(open(f"{ARTIFACTS}/manifest.json"))
    lbd = final["leaderboard"]
    real = {k: v for k, v in lbd.items()
            if not k.startswith(("ORACLE", "baseline_"))}
    champ = min(real, key=lambda k: real[k]["MASE"])
    print(f"  champion: {champ}  MASE={real[champ]['MASE']:.4f}")

    path = None
    for t in manifest.get("trials", []):
        if t.get("label") == champ:
            path = t.get("model_path")

    if not path or not os.path.isdir(path):
        # If the SQL track won, prefer the best Python-side model so we still
        # register something deployable rather than silently doing nothing.
        py = {k: v for k, v in real.items() if not k.startswith("sf_forecast")}
        if not py:
            print("  no registrable Python model available")
            session.close()
            return
        champ = min(py, key=lambda k: py[k]["MASE"])
        for t in manifest.get("trials", []):
            if t.get("label") == champ:
                path = t.get("model_path")
        print(f"  champion is a SQL-side FORECAST model (already lives in "
              f"Snowflake); registering best Python model instead: {champ}")

    register(session, path, champ, real[champ])
    session.close()


if __name__ == "__main__":
    main()
