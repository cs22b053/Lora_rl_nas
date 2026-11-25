#!/bin/bash
# Run on the login node once. Downloads packages; does not submit jobs or train.
# Isolate this run from changes to the cluster's shared pytorch_gpu1 packages.
set -euo pipefail
cd "${AUTODEEPLAB_DIR:-/scratch/ajin/Autodeeplab/modified}"
retrain_env="${RETRAIN_VENV:-$PWD/.venvs/retrain_cu118}"
retrain_base_python="${RETRAIN_BASE_PYTHON:-/apps/anaconda3/envs/pytorch_gpu1/bin/python}"
unset LD_LIBRARY_PATH PYTHONPATH
export PYTHONNOUSERSITE=1
# The cluster/user pip configuration may inject pypi.ngc.nvidia.com even when
# --index-url is supplied. Isolate only this script's pip settings; do not edit
# the user's/global configuration. Use public binary wheels, never source builds.
unset PIP_INDEX_URL PIP_EXTRA_INDEX_URL PIP_NO_INDEX PIP_FIND_LINKS
unset PIP_NO_BINARY PIP_ONLY_BINARY PIP_TARGET PIP_PREFIX PIP_USER
unset PIP_REQUIREMENT PIP_CONSTRAINT
export PIP_CONFIG_FILE=/dev/null
export PIP_DISABLE_PIP_VERSION_CHECK=1
if [[ -e "$retrain_env" && ! -f "$retrain_env/pyvenv.cfg" ]]; then
  echo "Refusing to install into a non-venv directory: $retrain_env" >&2
  exit 1
fi
if [[ ! -f "$retrain_env/pyvenv.cfg" ]]; then
  "$retrain_base_python" -m venv "$retrain_env"
fi
echo 'Install on an internet-connected login node; the GPU job performs no downloads.'
"$retrain_env/bin/python" -m pip install --only-binary=:all: \
  --index-url https://pypi.org/simple \
  --extra-index-url https://download.pytorch.org/whl/cu118 \
  'torch==2.3.1+cu118' 'typing_extensions==4.12.2' \
  'numpy==1.26.4' 'Pillow==10.4.0'
"$retrain_env/bin/python" -m pip check
"$retrain_env/bin/python" - <<'PY'
import sys
import torch
import numpy
import PIL
from reference_training import read_checkpoint
from retrain_data import CityscapesFine
assert sys.prefix != sys.base_prefix, 'Expected an isolated virtual environment'
assert torch.__version__ == '2.3.1+cu118', torch.__version__
assert torch.version.cuda == '11.8', torch.version.cuda
print('Environment ready:', sys.executable)
print('PyTorch:', torch.__version__, 'CUDA:', torch.version.cuda)
print('NumPy:', numpy.__version__, 'Pillow:', PIL.__version__)
print('No CUDA initialization is expected on the login node.')
PY
"$retrain_env/bin/python" -m pip freeze > "$retrain_env/requirements-installed.txt"
touch "$retrain_env/.autodeeplab-ready"
echo 'Setup complete. Submit: sbatch run_retrain_selected_4000.sh'
