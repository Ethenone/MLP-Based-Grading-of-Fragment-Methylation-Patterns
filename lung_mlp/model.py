"""Exact two-hidden-layer MLP used for fragment evidence scoring.

Copied from workflow/models.py in the research implementation.
"""

from __future__ import annotations
import numpy as np
from .patterns import PatternSpace


def _require_torch():
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as functional
    except ImportError as exc:
        raise RuntimeError("PyTorch is required for MLP/Transformer models") from exc
    return torch, nn, functional


def _one_hot_inputs(pattern_ids: np.ndarray, space: PatternSpace) -> np.ndarray:
    states = space.digits[np.asarray(pattern_ids, dtype=np.int64)]
    return np.eye(3, dtype=np.float32)[states].reshape(len(states), -1)


def fit_small_mlp(
    train_ids: np.ndarray,
    targets: np.ndarray,
    weights: np.ndarray,
    space: PatternSpace,
    epochs: int = 30,
    learning_rate: float = 1e-3,
    seed: int = 42,
) -> tuple[object, list[dict[str, float]]]:
    torch, nn, functional = _require_torch()
    np.random.seed(seed)
    torch.manual_seed(seed)

    class SmallPatternMLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.network = nn.Sequential(
                nn.Linear(space.seq_len * 3, 64),
                nn.ReLU(),
                nn.Dropout(0.05),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, 1),
            )

        def forward(self, values):
            return self.network(values).squeeze(-1)

    model = SmallPatternMLP()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    x = torch.from_numpy(_one_hot_inputs(train_ids, space))
    y = torch.from_numpy(np.asarray(targets, dtype=np.float32))
    w = torch.from_numpy(np.clip(np.asarray(weights, dtype=np.float32), 1e-4, None))
    generator = torch.Generator().manual_seed(seed)
    history: list[dict[str, float]] = []

    for epoch in range(1, epochs + 1):
        permutation = torch.randperm(len(x), generator=generator)
        model.train()
        weighted_loss_sum = 0.0
        weight_sum = 0.0
        for start in range(0, len(x), 256):
            indices = permutation[start : start + 256]
            prediction = model(x[indices])
            loss_each = functional.huber_loss(prediction, y[indices], reduction="none", delta=1.0)
            loss = (loss_each * w[indices]).sum() / w[indices].sum().clamp(min=1e-8)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            weighted_loss_sum += float((loss_each.detach() * w[indices]).sum())
            weight_sum += float(w[indices].sum())
        history.append({"epoch": epoch, "train_weighted_huber": weighted_loss_sum / weight_sum})
    return model, history


def predict_small_mlp(model: object, pattern_ids: np.ndarray, space: PatternSpace) -> np.ndarray:
    torch, _, _ = _require_torch()
    model.eval()
    outputs: list[np.ndarray] = []
    x = _one_hot_inputs(pattern_ids, space)
    with torch.no_grad():
        for start in range(0, len(x), 1024):
            outputs.append(model(torch.from_numpy(x[start : start + 1024])).cpu().numpy())
    return np.concatenate(outputs).astype(np.float64)
