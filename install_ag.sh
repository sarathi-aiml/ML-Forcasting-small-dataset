#!/bin/bash
cd "$(dirname "$0")"   # run from the repo root, wherever it is cloned
PY=.venv/bin/python

# autogluon.timeseries declares autogluon.tabular[catboost,lightgbm,xgboost],
# and the xgboost extra resolves to xgboost-cpu, which publishes NO macOS
# wheels (linux/windows only) and therefore needs a cmake source build.
#
# Workaround: install every real dependency explicitly (all have arm64 wheels),
# substituting the regular `xgboost` package - which provides the identical
# `xgboost` import module - for xgboost-cpu. Then install autogluon.timeseries
# with --no-deps so the unsatisfiable xgboost-cpu edge is never resolved.
# Functionally equivalent; avoids a 10+ minute native build.

echo "### [1/3] timeseries runtime deps (torch is the big one)"
uv pip install --python $PY \
  "joblib<1.7,>=1.2" "scipy<1.19,>=1.5.4" \
  "torch>=2.10,<2.11" "lightning<2.7,>=2.5.1" \
  "transformers[sentencepiece]<5.15,>=5.3" "accelerate<2.0,>=1.1.0" \
  "huggingface_hub[torch]<2.0,>=1.3" "safetensors<1,>=0.4" \
  "gluonts<0.18.0,>=0.17.0" "networkx<4,>=3.0" \
  "statsforecast<2.1.2,>=1.7.0" "mlforecast<0.15.0,>=0.14.0" \
  "utilsforecast<0.2.12,>=0.2.3" "coreforecast<0.0.17,>=0.0.12" \
  "fugue>=0.9.0" "tqdm<5,>=4.38" "orjson~=3.9" "einops<1,>=0.7" \
  "chronos-forecasting<2.4,>=2.3.1" "peft<0.20,>=0.18.1" \
  "tensorboard<3,>=2.9" "lightgbm" || { echo "DEPS_FAILED"; exit 1; }

echo "### [2/3] catboost (optional - not used by TimeSeriesPredictor)"
uv pip install --python $PY "catboost" || echo "catboost unavailable - continuing (not needed for forecasting)"

echo "### [3/3] autogluon.timeseries --no-deps"
uv pip install --python $PY --no-deps "autogluon.timeseries==1.6.1" \
  || { echo "TS_FAILED"; exit 1; }

echo "### verify"
$PY - <<'PYV'
import importlib.metadata as md
for p in ["autogluon.timeseries","autogluon.tabular","autogluon.core",
          "torch","lightgbm","xgboost","statsforecast","mlforecast",
          "gluonts","chronos-forecasting","transformers"]:
    try: print(f"  {p:24s} {md.version(p)}")
    except Exception: print(f"  {p:24s} MISSING")
from autogluon.timeseries import TimeSeriesPredictor, TimeSeriesDataFrame
print("  TimeSeriesPredictor OK")
from autogluon.timeseries.models import ChronosModel
print("  ChronosModel OK")
PYV
echo "=== AG_INSTALL_DONE ==="
