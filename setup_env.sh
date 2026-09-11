#!/bin/bash
# Environment setup for small-data healthcare time-series forecasting.
# AutoGluon's timeseries extra pulls torch + Chronos deps, so this is a multi-GB install.
set -o pipefail
cd "$(dirname "$0")"   # run from the repo root, wherever it is cloned

AG_PIN="1.5.1b20260415"

echo "=== [1/4] creating venv (python 3.10) ==="
uv venv .venv --python 3.10 || exit 1

echo "=== [2/4] core data + snowflake packages ==="
uv pip install --python .venv/bin/python \
  "snowflake-ml-python" "numba" "tomli" "snowflake-snowpark-python" \
  "pandas" "numpy" "scikit-learn" "matplotlib" "seaborn" "plotly" \
  "statsmodels" "holidays" "scipy" "psutil" "pyarrow" || exit 1

echo "=== [3/4] autogluon (pinned $AG_PIN, components only - NOT the umbrella pkg) ==="
# The skill pins a beta build. If that build is not on PyPI, fall back to latest stable
# so the workflow is not blocked; the exact version gets recorded in the manifest either way.
if uv pip install --python .venv/bin/python \
     "autogluon.tabular[all]==$AG_PIN" \
     "autogluon.core==$AG_PIN" \
     "autogluon.features==$AG_PIN" \
     "autogluon.timeseries==$AG_PIN"; then
  echo "AUTOGLUON_MODE=pinned"
else
  echo "!!! pinned $AG_PIN unavailable - falling back to latest stable autogluon components"
  uv pip install --python .venv/bin/python \
    "autogluon.tabular[all]" "autogluon.core" "autogluon.features" "autogluon.timeseries" || exit 1
  echo "AUTOGLUON_MODE=latest_stable"
fi

echo "=== [4/4] verifying imports ==="
.venv/bin/python - <<'PY'
import importlib.metadata as md
pkgs = ["snowflake-ml-python","snowflake-snowpark-python","autogluon.core",
        "autogluon.timeseries","autogluon.tabular","pandas","numpy",
        "scikit-learn","statsmodels","holidays"]
for p in pkgs:
    try:
        print(f"  {p:32s} {md.version(p)}")
    except Exception as e:
        print(f"  {p:32s} MISSING ({type(e).__name__})")

# Confirm the timeseries predictor and the zero-shot Chronos model actually import,
# since Chronos is the main small-data lever in this plan.
from autogluon.timeseries import TimeSeriesPredictor, TimeSeriesDataFrame  # noqa
print("  TimeSeriesPredictor import OK")
from autogluon.timeseries.models import ChronosModel  # noqa
print("  ChronosModel import OK")
PY
echo "=== SETUP COMPLETE ==="
