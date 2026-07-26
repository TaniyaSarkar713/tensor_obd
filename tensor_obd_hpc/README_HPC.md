# Tensor-OBD HPC smoke test

This package runs a small comparison of matrix SVD, OBD-LLM, loss-oblivious tensor decomposition, and Tensor-OBD.

## Files

- `tensor_obd_experiments/decomposition.py`: algorithms.
- `tensor_obd_experiments/evaluate.py`: experiment runner.
- `scripts/run_tensor_obd_smoke.sbatch`: Slurm job script.
- `requirements.txt`: Python packages.

## Quick run

```bash
unzip tensor_obd_hpc.zip
cd tensor_obd_hpc
pip install -r requirements.txt
python -m tensor_obd_experiments.evaluate --model sshleifer/tiny-gpt2 --group attn.c_proj --rank 8 --seq_length 64 --calib_batches 2 --eval_batches 2 --batch_size 1
```

## Budget-matched ranks

Tensor methods and per-layer matrix methods have different parameter-count formulas, so using the same `--rank` is not a fair comparison. Use `--budget_reference_method` to treat `--rank` as the reference method's rank and automatically convert the other methods to similar parameter budgets.

For example, this uses tensor rank 64 as the budget and lowers the matrix-method ranks accordingly:

```bash
python -m tensor_obd_experiments.evaluate \
  --model facebook/opt-125m \
  --group self_attn.out_proj \
  --rank 64 \
  --budget_reference_method tensor_obd \
  --seq_length 128 \
  --calib_batches 8 \
  --eval_batches 8 \
  --batch_size 1 \
  --methods matrix_svd obd_llm tensor_llm tensor_obd
```

For OPT-125M `self_attn.out_proj`, tensor rank 64 corresponds to matrix rank about 35. If you instead want matrix rank 64 as the budget, use `--budget_reference_method matrix_svd`; the tensor ranks will be about 118.

You can also set ranks manually:

```bash
python -m tensor_obd_experiments.evaluate \
  --model facebook/opt-125m \
  --group self_attn.out_proj \
  --method_ranks matrix_svd=35 obd_llm=35 tensor_llm=64 tensor_obd=64 \
  --seq_length 128 \
  --calib_batches 8 \
  --eval_batches 8 \
  --batch_size 1 \
  --methods matrix_svd obd_llm tensor_llm tensor_obd
```

## Rank sweep table

For the OPT-125M `fc2` target, matrix ranks 64/128/192 and tensor ranks 240/480/720 are budget-matched. Run:

```bash
python -m tensor_obd_experiments.evaluate \
  --model facebook/opt-125m \
  --group fc2 \
  --seq_length 256 \
  --calib_batches 64 \
  --eval_batches 128 \
  --batch_size 1 \
  --matrix_ranks 64 128 192 \
  --tensor_ranks 240 480 720 \
  --results_csv logs/opt125m_fc2_rank_sweep.csv \
  --methods matrix_svd obd_llm tensor_llm tensor_obd
```

The output prints one wide table with perplexity at each matched compression ratio and writes a long-form CSV for plotting.

## Slurm

```bash
mkdir -p logs
sbatch scripts/run_tensor_obd_smoke.sbatch
squeue -u $USER
```

Check results:

```bash
ls logs
cat logs/tensor_obd_sweep_<JOBID>.out
cat logs/tensor_obd_sweep_<JOBID>.err
```

## Notes

For OPT models, `self_attn.out_proj` is a simple first target, while `fc2` is a better target for showing tensor-sharing benefits.
For GPT-2 style tiny model, use `attn.c_proj`.
The script prints available linear module names before compression.
