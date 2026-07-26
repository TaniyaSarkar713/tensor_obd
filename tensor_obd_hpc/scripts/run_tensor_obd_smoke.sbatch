#!/bin/bash
#SBATCH --job-name=tensor_obd_sweep
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=backfill2
#SBATCH -A backfill2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00

set -euo pipefail

module load webproxy

SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
if [[ -n "${PROJECT_DIR:-}" ]]; then
  PROJECT_DIR="${PROJECT_DIR}"
elif [[ -d "${SUBMIT_DIR}/tensor_obd_experiments" ]]; then
  PROJECT_DIR="${SUBMIT_DIR}"
elif [[ -d "${SUBMIT_DIR}/tensor_obd_hpc/tensor_obd_experiments" ]]; then
  PROJECT_DIR="${SUBMIT_DIR}/tensor_obd_hpc"
else
  echo "Could not find tensor_obd_experiments from submit dir: ${SUBMIT_DIR}" >&2
  echo "Submit from tensor_obd_hpc, or set PROJECT_DIR=/path/to/tensor_obd_hpc." >&2
  exit 1
fi
cd "${PROJECT_DIR}"
export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}"
mkdir -p logs

export HF_HOME="${HF_HOME:-/gpfs/home/ts22u/.cache/huggingface}"
LOCAL_FILES_ONLY_ARGS=()
if [[ "${HF_OFFLINE:-0}" == "1" ]]; then
  export HF_HUB_OFFLINE=1
  export TRANSFORMERS_OFFLINE=1
  export HF_DATASETS_OFFLINE=1
  LOCAL_FILES_ONLY_ARGS=(--local_files_only)
else
  unset HF_HUB_OFFLINE
  unset TRANSFORMERS_OFFLINE
  unset HF_DATASETS_OFFLINE
fi

PYTHON_BIN="${PYTHON_BIN:-/gpfs/home/ts22u/.conda/envs/tensor_obd/bin/python}"
if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Python executable not found or not executable: ${PYTHON_BIN}" >&2
  echo "Set PYTHON_BIN=/path/to/python when submitting, or edit this script." >&2
  exit 1
fi

echo "Host: $(hostname)"
echo "Date: $(date)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "Project dir: ${PROJECT_DIR}"
echo "Python: ${PYTHON_BIN}"
echo "HF_HOME: ${HF_HOME}"
echo "HF_OFFLINE: ${HF_OFFLINE:-0}"

"${PYTHON_BIN}" - <<'PY'
import torch
print('torch:', torch.__version__)
print('cuda available:', torch.cuda.is_available())
if torch.cuda.is_available(): print('gpu:', torch.cuda.get_device_name(0))
PY

if [[ "${RUN_TINY_SMOKE:-0}" == "1" ]]; then
  # Optional smoke test. Enable with RUN_TINY_SMOKE=1.
  "${PYTHON_BIN}" -m tensor_obd_experiments.evaluate \
    --model sshleifer/tiny-gpt2 \
    --group attn.c_proj \
    --rank 8 \
    --seq_length 64 \
    --calib_batches 2 \
    --eval_batches 2 \
    --batch_size 1 \
    "${LOCAL_FILES_ONLY_ARGS[@]}" \
    --methods matrix_svd obd_llm tensor_llm tensor_obd
fi

RESULTS_CSV="logs/opt125m_fc2_rank_sweep_${SLURM_JOB_ID:-manual}.csv"

# Budget-matched OPT-125M fc2 rank sweep.
# Matrix ranks 64/128/192 and tensor ranks 240/480/720 have similar target-group ratios.
"${PYTHON_BIN}" -m tensor_obd_experiments.evaluate \
  --model facebook/opt-125m \
  --group fc2 \
  --seq_length 256 \
  --calib_batches 64 \
  --eval_batches 128 \
  --batch_size 1 \
  --matrix_ranks 64 128 192 \
  --tensor_ranks 240 480 720 \
  --results_csv "${RESULTS_CSV}" \
  "${LOCAL_FILES_ONLY_ARGS[@]}" \
  --methods matrix_svd obd_llm tensor_llm tensor_obd
