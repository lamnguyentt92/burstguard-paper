"""Local training loop of one site (used by benign and simulated malicious clients and by baselines)."""
from __future__ import annotations

import time
from typing import Any

import numpy as np
import torch

from burstguard import model as M


def local_train(model, X: np.ndarray, Y: np.ndarray, scale: float | np.ndarray, cfg: dict[str, Any],
                seed: int = 0, cache: "M.HeadCache | None" = None, epochs: int | None = None,
                grad_mask: list[torch.Tensor] | None = None, prox_anchor: list[torch.Tensor] | None = None,
                prox_weight: float = 0.0, extra_loss=None, epoch_hook=None) -> dict[str, float]:
    """Train the requires_grad parameters in place.

    Loss = MSE on values normalised by the site's train std (scale may be per-window for pooled baselines).
    grad_mask: per-parameter 0/1 masks applied to gradients each step (Neurotoxin-style projection).
    prox_anchor/prox_weight: adds prox_weight * ||theta - anchor||^2 (distance-aware malicious training).
    extra_loss(model) -> tensor: added to every step's loss; epoch_hook(model, epoch): called before each epoch
    (v0.3.12 score-aware attacker: refresh the worst bursts, then penalise the model's differential response).
    """
    tcfg = cfg["train"]
    H = Y.shape[1]
    params = [p for _, p in M.trainable_named(model)]
    if not params:
        raise RuntimeError("no trainable parameters - check train.peft")
    opt = torch.optim.AdamW(params, lr=float(tcfg["lr"]), weight_decay=float(tcfg["weight_decay"]))
    g = torch.Generator().manual_seed(int(seed))
    Yt = torch.from_numpy(np.ascontiguousarray(Y))
    Xt = torch.from_numpy(np.ascontiguousarray(X))
    inv = torch.as_tensor(1.0 / np.asarray(scale, dtype=np.float32)).reshape(-1, 1, 1) \
        if np.ndim(scale) else torch.tensor(1.0 / float(scale))
    bs = int(tcfg["batch_size"])
    use_cache = cache is not None
    if use_cache:
        M.ttm_core(model).eval()
        M.ttm_core(model).head.train()
    else:
        model.train()
    t0 = time.process_time()
    tot, nb = 0.0, 0
    for ep in range(int(epochs if epochs is not None else tcfg["local_epochs"])):
        if epoch_hook is not None:
            epoch_hook(model, ep)
            if use_cache:     # the hook may switch modules to eval mode
                M.ttm_core(model).eval()
                M.ttm_core(model).head.train()
            else:
                model.train()
        perm = torch.randperm(len(X), generator=g)
        for i in range(0, len(X), bs):
            idx = perm[i:i + bs]
            pred = M.head_forward(model, cache, idx, H) if use_cache else M.forward(model, Xt[idx], cfg["model"]["freq"])
            w = inv[idx] if inv.ndim else inv
            loss = torch.mean(((pred - Yt[idx]) * w) ** 2)
            if prox_weight and prox_anchor is not None:
                loss = loss + prox_weight * sum(((p - a) ** 2).sum() for p, a in zip(params, prox_anchor))
            if extra_loss is not None:
                loss = loss + extra_loss(model)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if grad_mask is not None:
                for p, m in zip(params, grad_mask):
                    if p.grad is not None:
                        p.grad.mul_(m)
            opt.step()
            tot += float(loss.detach()); nb += 1
    model.eval()
    return {"train_loss": tot / max(nb, 1), "train_batches": nb, "train_cpu_s": time.process_time() - t0}
