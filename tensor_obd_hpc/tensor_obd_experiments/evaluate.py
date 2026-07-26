from __future__ import annotations

import argparse
import csv
import math
import os
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, default_data_collator

from .decomposition import compute_covariances, apply_decomposition_to_group


MATRIX_METHODS = {"matrix_svd", "obd_llm"}
TENSOR_METHODS = {"tensor_llm", "tensor_obd"}
ALL_METHODS = MATRIX_METHODS | TENSOR_METHODS


@dataclass
class ExperimentConfig:
    model_name: str = "sshleifer/tiny-gpt2"
    dataset_name: str = "wikitext"
    dataset_config: str = "wikitext-2-raw-v1"
    split: str = "test"
    group_name: str = "attn.c_proj"
    rank: int = 16
    method_ranks: Dict[str, int] = field(default_factory=dict)
    budget_reference_method: str | None = None
    matrix_ranks: Tuple[int, ...] = ()
    tensor_ranks: Tuple[int, ...] = ()
    results_csv: str | None = None
    methods: Tuple[str, ...] = ("matrix_svd", "obd_llm", "tensor_llm", "tensor_obd")
    seq_length: int = 128
    calib_batches: int = 4
    eval_batches: int = 4
    batch_size: int = 1
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    local_files_only: bool = False


def parse_method_ranks(items: List[str]) -> Dict[str, int]:
    ranks: Dict[str, int] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"Invalid method rank {item!r}; expected METHOD=RANK")
        method, rank_text = item.split("=", 1)
        method = method.strip().lower()
        if method not in ALL_METHODS:
            raise ValueError(f"Unknown method {method!r} in --method_ranks")
        try:
            rank = int(rank_text)
        except ValueError as exc:
            raise ValueError(f"Invalid rank {rank_text!r} for method {method!r}") from exc
        if rank <= 0:
            raise ValueError(f"Rank for method {method!r} must be positive")
        ranks[method] = rank
    return ranks


def validate_methods(methods: Tuple[str, ...]) -> None:
    for method in methods:
        if method.lower() not in ALL_METHODS:
            raise ValueError(f"Unknown method {method!r}")


def validate_rank_list(name: str, ranks: Tuple[int, ...]) -> None:
    for rank in ranks:
        if rank <= 0:
            raise ValueError(f"{name} ranks must be positive; got {rank}")


def compressed_param_count(method: str, rank: int, n: int, out_features: int, in_features: int) -> int:
    method = method.lower()
    if method in MATRIX_METHODS:
        return n * rank * (out_features + in_features)
    if method in TENSOR_METHODS:
        return rank * (n * out_features + in_features)
    raise ValueError(f"Unknown method {method!r}")


def rank_for_budget(method: str, target_params: int, n: int, out_features: int, in_features: int) -> int:
    method = method.lower()
    if method in MATRIX_METHODS:
        denom = n * (out_features + in_features)
    elif method in TENSOR_METHODS:
        denom = n * out_features + in_features
    else:
        raise ValueError(f"Unknown method {method!r}")
    return max(1, int(round(target_params / denom)))


def resolve_method_ranks(cfg: ExperimentConfig, n: int, out_features: int, in_features: int) -> Dict[str, int]:
    ranks: Dict[str, int] = {}
    if cfg.budget_reference_method is not None:
        ref_method = cfg.budget_reference_method.lower()
        if ref_method not in ALL_METHODS:
            raise ValueError(f"Unknown --budget_reference_method {cfg.budget_reference_method!r}")
        target_params = compressed_param_count(ref_method, cfg.rank, n, out_features, in_features)
        for method in cfg.methods:
            ranks[method.lower()] = rank_for_budget(method, target_params, n, out_features, in_features)
    ranks.update(cfg.method_ranks)
    return ranks


def print_rank_budget_notes(cfg: ExperimentConfig, ranks: Dict[str, int], n: int, out_features: int, in_features: int) -> None:
    orig_params = n * out_features * in_features
    tensor_rank_for_matrix_rank = cfg.rank * n * (out_features + in_features) / (n * out_features + in_features)
    matrix_rank_for_tensor_rank = cfg.rank * (n * out_features + in_features) / (n * (out_features + in_features))
    matrix_ratio = compressed_param_count("matrix_svd", cfg.rank, n, out_features, in_features) / orig_params
    tensor_ratio = compressed_param_count("tensor_obd", cfg.rank, n, out_features, in_features) / orig_params

    print("\nBudget note for shared --rank:")
    print(f"  matrix methods at rank {cfg.rank}: ratio {matrix_ratio:.4f}")
    print(f"  tensor methods at rank {cfg.rank}: ratio {tensor_ratio:.4f}")
    print(f"  tensor rank matching matrix rank {cfg.rank}: ~{tensor_rank_for_matrix_rank:.1f}")
    print(f"  matrix rank matching tensor rank {cfg.rank}: ~{matrix_rank_for_tensor_rank:.1f}")
    if ranks:
        print("Resolved per-method ranks:")
        for method in cfg.methods:
            method_key = method.lower()
            rank = ranks.get(method_key, cfg.rank)
            ratio = compressed_param_count(method_key, rank, n, out_features, in_features) / orig_params
            print(f"  {method:13s}: rank {rank:<4d} ratio {ratio:.4f}")


def print_sweep_budget_notes(cfg: ExperimentConfig, n: int, out_features: int, in_features: int) -> None:
    orig_params = n * out_features * in_features
    print("\nSweep budget notes:")
    print("  budget | matrix_rank | matrix_ratio | tensor_rank | tensor_ratio")
    for budget, (matrix_rank, tensor_rank) in enumerate(zip(cfg.matrix_ranks, cfg.tensor_ranks), start=1):
        matrix_ratio = compressed_param_count("matrix_svd", matrix_rank, n, out_features, in_features) / orig_params
        tensor_ratio = compressed_param_count("tensor_obd", tensor_rank, n, out_features, in_features) / orig_params
        print(f"  {budget:6d} | {matrix_rank:11d} | {matrix_ratio:12.4f} | {tensor_rank:11d} | {tensor_ratio:12.4f}")


def is_rank_sweep(cfg: ExperimentConfig) -> bool:
    return bool(cfg.matrix_ranks or cfg.tensor_ranks)


def get_rank_for_method(method: str, matrix_rank: int, tensor_rank: int) -> int:
    method = method.lower()
    if method in MATRIX_METHODS:
        return matrix_rank
    if method in TENSOR_METHODS:
        return tensor_rank
    raise ValueError(f"Unknown method {method!r}")


def prepare_dataloader(tokenizer, dataset, seq_length: int, batch_size: int) -> DataLoader:
    def tokenize_function(examples):
        return tokenizer(examples["text"], return_attention_mask=False)

    tokenized = dataset.map(tokenize_function, batched=True, remove_columns=["text"])

    def group_texts(examples):
        ids = []
        for row in examples["input_ids"]:
            ids.extend(row)
        total_length = (len(ids) // seq_length) * seq_length
        ids = ids[:total_length]
        return {"input_ids": [ids[i:i + seq_length] for i in range(0, total_length, seq_length)]}

    lm_dataset = tokenized.map(group_texts, batched=True)
    lm_dataset = lm_dataset.map(lambda batch: {"labels": batch["input_ids"]})
    return DataLoader(lm_dataset, batch_size=batch_size, shuffle=False, collate_fn=default_data_collator)


def get_group_modules(model: nn.Module, group_name: str) -> List[nn.Linear]:
    modules: List[nn.Linear] = []
    for name, module in model.named_modules():
        if name.endswith(group_name) and isinstance(module, nn.Linear):
            modules.append(module)
    return modules


def print_linear_module_names(model: nn.Module) -> None:
    print("\nAvailable nn.Linear module suffixes/names:")
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            print("  ", name, tuple(module.weight.shape))


def evaluate_perplexity(model: nn.Module, dataloader: DataLoader, device: torch.device, max_batches: int) -> float:
    model.eval()
    losses: List[float] = []
    for i, batch in enumerate(dataloader):
        if i >= max_batches:
            break
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        with torch.no_grad():
            outputs = model(input_ids=input_ids, labels=labels)
        losses.append(float(outputs.loss.detach().cpu()))
    if not losses:
        raise RuntimeError("No evaluation batches were available. Increase dataset subset or reduce seq_length.")
    return math.exp(sum(losses) / len(losses))


def load_small_dataset(cfg: ExperimentConfig, tokenizer):
    raw = load_dataset(cfg.dataset_name, cfg.dataset_config, split=cfg.split)
    # keep more rows than batches because many Wikitext rows are short/empty
    n_rows = min(len(raw), max(200, 40 * (cfg.calib_batches + cfg.eval_batches) * cfg.batch_size))
    raw = raw.shuffle(seed=cfg.seed).select(range(n_rows))
    return raw


def maybe_compute_curvature(
    cfg: ExperimentConfig,
    calib_data,
    device: torch.device,
) -> Tuple[torch.Tensor | None, torch.Tensor | None]:
    if not any(method.lower() in {"obd_llm", "tensor_obd"} for method in cfg.methods):
        return None, None
    print("\nEstimating curvature once for OBD methods.")
    model = AutoModelForCausalLM.from_pretrained(cfg.model_name, local_files_only=cfg.local_files_only).to(device)
    modules = get_group_modules(model, cfg.group_name)
    calib_loader = DataLoader(calib_data, batch_size=1, shuffle=False)
    C_out, C_in = compute_covariances(model, calib_loader, modules, device=device, num_batches=cfg.calib_batches)
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("Curvature estimated.")
    return C_out, C_in


def run_method(
    cfg: ExperimentConfig,
    method: str,
    rank: int,
    eval_data,
    device: torch.device,
    C_out: torch.Tensor | None,
    C_in: torch.Tensor | None,
) -> Dict[str, float]:
    method = method.lower()
    print(f"\n*** Method: {method} ***")
    print(f"rank: {rank}")
    model = AutoModelForCausalLM.from_pretrained(cfg.model_name, local_files_only=cfg.local_files_only).to(device)
    modules = get_group_modules(model, cfg.group_name)
    if method in {"obd_llm", "tensor_obd"} and (C_out is None or C_in is None):
        raise RuntimeError(f"Curvature was not available for {method}")
    orig_params, comp_params = apply_decomposition_to_group(modules, method, rank, C_out=C_out, C_in=C_in)
    ratio = comp_params / orig_params
    ppl = evaluate_perplexity(model, DataLoader(eval_data, batch_size=1, shuffle=False), device, cfg.eval_batches)
    print(f"compressed/original parameter ratio for target group: {ratio:.4f}")
    print(f"perplexity: {ppl:.4f}")
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {"rank": rank, "ratio": ratio, "ppl": ppl}


def print_sweep_table(rows: List[Dict[str, float]], base_ppl: float) -> None:
    print("\n=== RANK SWEEP TABLE ===")
    print(f"baseline | ratio = 1.0000 | ppl = {base_ppl:.4f}")
    print("budget | ratio  | matrix_rank | tensor_rank | matrix_svd | obd_llm | tensor_llm | tensor_obd")
    print("-" * 94)
    budget_ids = sorted({int(row["budget"]) for row in rows})
    methods = ["matrix_svd", "obd_llm", "tensor_llm", "tensor_obd"]
    for budget in budget_ids:
        budget_rows = [row for row in rows if int(row["budget"]) == budget]
        by_method = {str(row["method"]): row for row in budget_rows}
        matrix_rank = int(budget_rows[0]["matrix_rank"])
        tensor_rank = int(budget_rows[0]["tensor_rank"])
        ratio = sum(float(row["ratio"]) for row in budget_rows) / len(budget_rows)
        ppl_cells = []
        for method in methods:
            row = by_method.get(method)
            ppl_cells.append("-" if row is None else f"{float(row['ppl']):.4f}")
        print(
            f"{budget:6d} | {ratio:.4f} | {matrix_rank:11d} | {tensor_rank:11d} | "
            f"{ppl_cells[0]:>10s} | {ppl_cells[1]:>7s} | {ppl_cells[2]:>10s} | {ppl_cells[3]:>10s}"
        )


def write_sweep_csv(path: str, rows: List[Dict[str, float]], base_ppl: float) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    fieldnames = ["budget", "matrix_rank", "tensor_rank", "method", "rank", "ratio", "ppl", "baseline_ppl"]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            out = {name: row.get(name, "") for name in fieldnames}
            out["baseline_ppl"] = base_ppl
            writer.writerow(out)
    print(f"\nWrote sweep CSV: {path}")


def run_rank_sweep(
    cfg: ExperimentConfig,
    eval_data,
    device: torch.device,
    C_out: torch.Tensor | None,
    C_in: torch.Tensor | None,
    base_ppl: float,
) -> None:
    if not cfg.matrix_ranks or not cfg.tensor_ranks:
        raise ValueError("Rank sweep requires both --matrix_ranks and --tensor_ranks")
    if len(cfg.matrix_ranks) != len(cfg.tensor_ranks):
        raise ValueError("--matrix_ranks and --tensor_ranks must have the same length")
    if cfg.method_ranks or cfg.budget_reference_method is not None:
        raise ValueError("Do not combine rank sweep with --method_ranks or --budget_reference_method")

    rows: List[Dict[str, float]] = []
    for budget, (matrix_rank, tensor_rank) in enumerate(zip(cfg.matrix_ranks, cfg.tensor_ranks), start=1):
        print(f"\n=== Sweep budget {budget}: matrix rank {matrix_rank}, tensor rank {tensor_rank} ===")
        for method in cfg.methods:
            method_key = method.lower()
            rank = get_rank_for_method(method_key, matrix_rank, tensor_rank)
            result = run_method(cfg, method_key, rank, eval_data, device, C_out, C_in)
            result.update(
                {
                    "budget": budget,
                    "matrix_rank": matrix_rank,
                    "tensor_rank": tensor_rank,
                    "method": method_key,
                }
            )
            rows.append(result)
    print_sweep_table(rows, base_ppl)
    if cfg.results_csv is not None:
        write_sweep_csv(cfg.results_csv, rows, base_ppl)


def run_experiments(cfg: ExperimentConfig) -> None:
    validate_methods(cfg.methods)
    validate_rank_list("matrix", cfg.matrix_ranks)
    validate_rank_list("tensor", cfg.tensor_ranks)
    if is_rank_sweep(cfg):
        if not cfg.matrix_ranks or not cfg.tensor_ranks:
            raise ValueError("Rank sweep requires both --matrix_ranks and --tensor_ranks")
        if len(cfg.matrix_ranks) != len(cfg.tensor_ranks):
            raise ValueError("--matrix_ranks and --tensor_ranks must have the same length")
        if cfg.method_ranks or cfg.budget_reference_method is not None:
            raise ValueError("Do not combine rank sweep with --method_ranks or --budget_reference_method")
    torch.manual_seed(cfg.seed)
    device = torch.device(cfg.device)
    print("Device:", device)
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, local_files_only=cfg.local_files_only)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    raw_dataset = load_small_dataset(cfg, tokenizer)
    dataloader = prepare_dataloader(tokenizer, raw_dataset, cfg.seq_length, cfg.batch_size)
    batches = list(dataloader)
    need_batches = cfg.calib_batches + cfg.eval_batches
    if len(batches) < need_batches:
        raise RuntimeError(f"Only {len(batches)} tokenized batches found, but need {need_batches}. Reduce --seq_length or increase dataset rows.")
    calib_data = batches[:cfg.calib_batches]
    eval_data = batches[cfg.calib_batches:cfg.calib_batches + cfg.eval_batches]
    print(f"Prepared {len(calib_data)} calibration and {len(eval_data)} evaluation batches.")

    results: Dict[str, Dict[str, float]] = {}

    # Uncompressed baseline
    base_model = AutoModelForCausalLM.from_pretrained(cfg.model_name, local_files_only=cfg.local_files_only).to(device)
    print_linear_module_names(base_model)
    base_modules = get_group_modules(base_model, cfg.group_name)
    if not base_modules:
        raise RuntimeError(f"No nn.Linear modules found with suffix {cfg.group_name!r}. Choose one from the printed list.")
    print(f"Compressing group suffix {cfg.group_name!r}: {len(base_modules)} modules, shape {tuple(base_modules[0].weight.shape)}")
    n_modules = len(base_modules)
    out_features, in_features = base_modules[0].weight.shape
    method_ranks: Dict[str, int] = {}
    if is_rank_sweep(cfg):
        print_sweep_budget_notes(cfg, n_modules, out_features, in_features)
    else:
        method_ranks = resolve_method_ranks(cfg, n_modules, out_features, in_features)
        print_rank_budget_notes(cfg, method_ranks, n_modules, out_features, in_features)
    base_ppl = evaluate_perplexity(base_model, DataLoader(eval_data, batch_size=1, shuffle=False), device, cfg.eval_batches)
    print(f"\nUncompressed baseline perplexity: {base_ppl:.4f}")
    del base_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    C_out, C_in = maybe_compute_curvature(cfg, calib_data, device)

    if is_rank_sweep(cfg):
        run_rank_sweep(cfg, eval_data, device, C_out, C_in, base_ppl)
        return

    for method in cfg.methods:
        method_key = method.lower()
        method_rank = method_ranks.get(method.lower(), cfg.rank)
        results[method_key] = run_method(cfg, method_key, method_rank, eval_data, device, C_out, C_in)

    print("\n=== SUMMARY ===")
    print(f"baseline      | ratio = 1.0000 | ppl = {base_ppl:.4f}")
    for method, r in results.items():
        print(f"{method:13s} | rank = {r['rank']:4.0f} | ratio = {r['ratio']:.4f} | ppl = {r['ppl']:.4f}")
    best = min(results.items(), key=lambda kv: kv[1]["ppl"])
    print(f"\nBest compressed method by perplexity: {best[0]} (ppl={best[1]['ppl']:.4f})")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="sshleifer/tiny-gpt2")
    p.add_argument("--dataset", default="wikitext")
    p.add_argument("--dataset_config", default="wikitext-2-raw-v1")
    p.add_argument("--split", default="test")
    p.add_argument("--group", default="attn.c_proj")
    p.add_argument("--rank", type=int, default=16)
    p.add_argument("--method_ranks", nargs="*", default=[], metavar="METHOD=RANK")
    p.add_argument("--budget_reference_method", choices=sorted(ALL_METHODS), default=None)
    p.add_argument("--matrix_ranks", nargs="*", type=int, default=[])
    p.add_argument("--tensor_ranks", nargs="*", type=int, default=[])
    p.add_argument("--results_csv", default=None)
    p.add_argument("--methods", nargs="*", default=["matrix_svd", "obd_llm", "tensor_llm", "tensor_obd"])
    p.add_argument("--seq_length", type=int, default=128)
    p.add_argument("--calib_batches", type=int, default=4)
    p.add_argument("--eval_batches", type=int, default=4)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--local_files_only", action="store_true")
    args = p.parse_args()
    try:
        method_ranks = parse_method_ranks(args.method_ranks)
    except ValueError as exc:
        p.error(str(exc))
    cfg = ExperimentConfig(
        model_name=args.model,
        dataset_name=args.dataset,
        dataset_config=args.dataset_config,
        split=args.split,
        group_name=args.group,
        rank=args.rank,
        method_ranks=method_ranks,
        budget_reference_method=args.budget_reference_method,
        matrix_ranks=tuple(args.matrix_ranks),
        tensor_ranks=tuple(args.tensor_ranks),
        results_csv=args.results_csv,
        methods=tuple(args.methods),
        seq_length=args.seq_length,
        calib_batches=args.calib_batches,
        eval_batches=args.eval_batches,
        batch_size=args.batch_size,
        seed=args.seed,
        device=args.device,
        local_files_only=args.local_files_only,
    )
    run_experiments(cfg)


if __name__ == "__main__":
    main()
