from __future__ import annotations
from typing import Iterable, List, Tuple
import torch
from torch import nn


def _sym_eigh_sqrt(A: torch.Tensor, eps: float = 1e-5, inverse: bool = False) -> torch.Tensor:
    A = (A + A.T) / 2
    vals, vecs = torch.linalg.eigh(A)
    vals = torch.clamp(vals, min=0) + eps
    vals = vals.rsqrt() if inverse else vals.sqrt()
    return (vecs * vals.unsqueeze(0)) @ vecs.T


def apply_matrix_svd(W: torch.Tensor, rank: int) -> torch.Tensor:
    U, S, Vh = torch.linalg.svd(W.float(), full_matrices=False)
    r = min(rank, S.numel())
    out = (U[:, :r] * S[:r].unsqueeze(0)) @ Vh[:r]
    return out.to(dtype=W.dtype, device=W.device)


def obd_llm_decompose(W: torch.Tensor, C_out: torch.Tensor, C_in: torch.Tensor, rank: int, eps: float = 1e-5) -> torch.Tensor:
    device = W.device
    dtype = W.dtype
    Wf = W.float()
    C_out = C_out.to(device=device, dtype=torch.float32)
    C_in = C_in.to(device=device, dtype=torch.float32)
    Co_inv = _sym_eigh_sqrt(C_out, eps=eps, inverse=True)
    Ci_inv = _sym_eigh_sqrt(C_in, eps=eps, inverse=True)
    Co = _sym_eigh_sqrt(C_out, eps=eps, inverse=False)
    Ci = _sym_eigh_sqrt(C_in, eps=eps, inverse=False)
    Y = Co @ Wf @ Ci
    Y_hat = apply_matrix_svd(Y, rank).float()
    return (Co_inv @ Y_hat @ Ci_inv).to(device=device, dtype=dtype)


def tensor_llm_decompose(Ws: torch.Tensor, rank: int) -> torch.Tensor:
    # Ws: (n_layers, out, in). Flatten layer/out and use shared right basis.
    n, out, inn = Ws.shape
    flat = Ws.float().reshape(n * out, inn)
    flat_hat = apply_matrix_svd(flat, rank).float()
    return flat_hat.reshape(n, out, inn).to(device=Ws.device, dtype=Ws.dtype)


def tensor_obd_decompose(Ws: torch.Tensor, C_out: torch.Tensor, C_in: torch.Tensor, rank: int, eps: float = 1e-5) -> torch.Tensor:
    # Loss-aware multiway approximation. Shared C_out/C_in, shared right basis across layers.
    n, out, inn = Ws.shape
    device, dtype = Ws.device, Ws.dtype
    C_out = C_out.to(device=device, dtype=torch.float32)
    C_in = C_in.to(device=device, dtype=torch.float32)
    Co_inv = _sym_eigh_sqrt(C_out, eps=eps, inverse=True)
    Ci_inv = _sym_eigh_sqrt(C_in, eps=eps, inverse=True)
    Co = _sym_eigh_sqrt(C_out, eps=eps, inverse=False)
    Ci = _sym_eigh_sqrt(C_in, eps=eps, inverse=False)
    weighted = torch.stack([Co @ Ws[i].float() @ Ci for i in range(n)], dim=0)
    flat_hat = apply_matrix_svd(weighted.reshape(n * out, inn), rank).float().reshape(n, out, inn)
    out_ws = torch.stack([Co_inv @ flat_hat[i] @ Ci_inv for i in range(n)], dim=0)
    return out_ws.to(device=device, dtype=dtype)


def _flatten_2d(x: torch.Tensor) -> torch.Tensor:
    # Linear modules in LMs often receive (batch, seq, hidden); flatten all but last.
    return x.detach().reshape(-1, x.shape[-1]).float().cpu()


def compute_covariances(model: nn.Module, data_loader: Iterable, modules: Iterable[nn.Linear], device: torch.device, num_batches: int = 8, eps: float = 1e-5) -> Tuple[torch.Tensor, torch.Tensor]:
    modules = list(modules)
    if not modules:
        raise ValueError('No modules supplied')
    in_features = modules[0].in_features
    out_features = modules[0].out_features
    for m in modules:
        if m.in_features != in_features or m.out_features != out_features:
            raise ValueError('All modules in a tensor group must have identical shapes')

    acts: List[torch.Tensor] = []
    grads: List[torch.Tensor] = []

    def fwd_hook(mod, inp, out):
        acts.append(_flatten_2d(inp[0]))

    def bwd_hook(mod, grad_input, grad_output):
        grads.append(_flatten_2d(grad_output[0]))

    handles = []
    for m in modules:
        handles.append(m.register_forward_hook(fwd_hook))
        handles.append(m.register_full_backward_hook(bwd_hook))

    model.train()
    model.zero_grad(set_to_none=True)
    for i, batch in enumerate(data_loader):
        if i >= num_batches:
            break
        input_ids = batch['input_ids'].to(device)
        labels = batch['labels'].to(device)
        loss = model(input_ids=input_ids, labels=labels).loss
        loss.backward()
        model.zero_grad(set_to_none=True)

    for h in handles:
        h.remove()
    if not acts or not grads:
        raise RuntimeError('No activations/gradients captured. Check module group name.')

    A = torch.cat(acts, dim=0).to(device)
    G = torch.cat(grads, dim=0).to(device)
    # Limit rows for speed/memory if hooks capture many token positions.
    max_rows = 8192
    if A.shape[0] > max_rows:
        idx = torch.linspace(0, A.shape[0]-1, steps=max_rows).long().to(device)
        A = A[idx]
    if G.shape[0] > max_rows:
        idx = torch.linspace(0, G.shape[0]-1, steps=max_rows).long().to(device)
        G = G[idx]
    C_in = (A.T @ A) / max(1, A.shape[0])
    C_out = (G.T @ G) / max(1, G.shape[0])
    C_in = C_in + eps * torch.eye(C_in.shape[0], device=device)
    C_out = C_out + eps * torch.eye(C_out.shape[0], device=device)
    return C_out.detach(), C_in.detach()


def apply_decomposition_to_group(modules: Iterable[nn.Linear], method: str, rank: int, C_out: torch.Tensor | None = None, C_in: torch.Tensor | None = None, eps: float = 1e-5) -> Tuple[int, int]:
    modules = list(modules)
    if not modules:
        raise ValueError('No modules supplied')
    out_features, in_features = modules[0].weight.shape
    for m in modules:
        if tuple(m.weight.shape) != (out_features, in_features):
            raise ValueError('All modules must have identical weight shapes')
    n = len(modules)
    orig_params = n * out_features * in_features
    method = method.lower()
    with torch.no_grad():
        if method == 'matrix_svd':
            for m in modules:
                m.weight.copy_(apply_matrix_svd(m.weight.data, rank))
            comp_params = n * rank * (out_features + in_features)
        elif method == 'obd_llm':
            if C_out is None or C_in is None:
                raise ValueError('C_out/C_in required for obd_llm')
            for m in modules:
                m.weight.copy_(obd_llm_decompose(m.weight.data, C_out, C_in, rank, eps))
            comp_params = n * rank * (out_features + in_features)
        elif method == 'tensor_llm':
            Ws = torch.stack([m.weight.data for m in modules], dim=0)
            Wh = tensor_llm_decompose(Ws, rank)
            for i, m in enumerate(modules):
                m.weight.copy_(Wh[i])
            comp_params = rank * (n * out_features + in_features)
        elif method == 'tensor_obd':
            if C_out is None or C_in is None:
                raise ValueError('C_out/C_in required for tensor_obd')
            Ws = torch.stack([m.weight.data for m in modules], dim=0)
            Wh = tensor_obd_decompose(Ws, C_out, C_in, rank, eps)
            for i, m in enumerate(modules):
                m.weight.copy_(Wh[i])
            comp_params = rank * (n * out_features + in_features)
        else:
            raise ValueError(f'Unknown method {method}')
    return orig_params, comp_params
