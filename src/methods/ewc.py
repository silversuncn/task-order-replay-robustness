"""Elastic Weight Consolidation helpers."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn


def config(ewc_lambda: float = 100.0) -> dict[str, Any]:
    return {
        "method": "ewc",
        "memory_budget": 0,
        "method_cell": "ewc",
        "ewc_lambda": float(ewc_lambda),
        "use_replay": False,
        "use_ewc": True,
        "matched": False,
    }


def estimate_fisher(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    indices: torch.Tensor,
    device: torch.device,
    batch_size: int,
) -> dict[str, torch.Tensor]:
    model.eval()
    fisher = {name: torch.zeros_like(param, device="cpu") for name, param in model.named_parameters() if param.requires_grad}
    if indices.numel() == 0:
        return fisher

    try:
        from torch.func import functional_call, grad, vmap

        params = {name: param.detach() for name, param in model.named_parameters() if param.requires_grad}
        buffers = {name: buffer.detach() for name, buffer in model.named_buffers()}

        def loss_for_one(params_arg, buffers_arg, sample, target):  # type: ignore[no-untyped-def]
            logits = functional_call(model, (params_arg, buffers_arg), (sample.unsqueeze(0),))
            return nn.functional.cross_entropy(logits, target.unsqueeze(0), reduction="sum")

        grad_fn = grad(loss_for_one)
        count = 0
        ordered = indices.detach().cpu()
        for start in range(0, ordered.numel(), batch_size):
            batch_idx = ordered[start : start + batch_size]
            batch_x = x[batch_idx].to(device)
            batch_y = y[batch_idx].to(device)
            grads = vmap(grad_fn, in_dims=(None, None, 0, 0))(params, buffers, batch_x, batch_y)
            for name, grad_values in grads.items():
                fisher[name] += grad_values.detach().pow(2).sum(dim=0).cpu()
            count += int(batch_y.numel())
        return {name: value / max(1, count) for name, value in fisher.items()}
    except Exception:
        criterion = nn.CrossEntropyLoss()
        count = 0
        for idx in indices.detach().cpu():
            model.zero_grad(set_to_none=True)
            sample_x = x[idx].view(1, -1).to(device)
            sample_y = y[idx].view(1).to(device)
            loss = criterion(model(sample_x), sample_y)
            loss.backward()
            for name, param in model.named_parameters():
                if param.grad is not None:
                    fisher[name] += param.grad.detach().cpu().pow(2)
            count += 1
        return {name: value / max(1, count) for name, value in fisher.items()}


def penalty(model: nn.Module, anchors: list[dict[str, dict[str, torch.Tensor]]], device: torch.device) -> torch.Tensor:
    value = torch.tensor(0.0, device=device)
    for anchor in anchors:
        for name, param in model.named_parameters():
            if name in anchor["params"]:
                old_param = anchor["params"][name].to(device)
                fisher = anchor["fisher"][name].to(device)
                value = value + (fisher * (param - old_param).pow(2)).sum()
    return value
