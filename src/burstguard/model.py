"""TinyTimeMixer (TTM) loading, fine-tuning modes, flat parameter vectors and the cached head-only path."""
from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

FREQ_ALIASES = {"10min": "10min", "10t": "10min", "10m": "10min"}


@dataclass
class ModelInfo:
    source: str
    revision: str
    context_length: int
    prediction_length: int
    horizon: int
    freq_prefix: bool
    n_params: int


def _silence_loaders() -> None:
    from burstguard.logutil import silence_tsfm
    silence_tsfm()


def _random_ttm(L: int, H: int, size: str = "small"):
    from tsfm_public.models.tinytimemixer import TinyTimeMixerConfig, TinyTimeMixerForPrediction
    patch = 64 if L % 64 == 0 else next(p for p in (32, 16, 8, 4, 2) if L % p == 0)
    pl = 48 if H <= 48 else 96
    dims = {"small": (256, 2, 128, 2), "tiny": (64, 1, 32, 1)}.get(size, (256, 2, 128, 2))
    cfg = TinyTimeMixerConfig(
        context_length=L, prediction_length=pl, prediction_filter_length=H,
        patch_length=patch, patch_stride=patch, d_model=dims[0], num_layers=dims[1],
        adaptive_patching_levels=3 if dims[0] >= 128 else 0,
        decoder_d_model=dims[2], decoder_num_layers=dims[3], num_input_channels=1,
        resolution_prefix_tuning=True, scaling="std")
    return TinyTimeMixerForPrediction(cfg)


def load_model(cfg: dict[str, Any], seed: int = 0):
    """Return (model, ModelInfo). Model in eval mode with all parameters frozen."""
    mcfg, dcfg = cfg["model"], cfg["data"]
    L, H = int(dcfg["context"]), int(dcfg["horizon"])
    src = mcfg["source"]
    torch.manual_seed(seed)
    _silence_loaders()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if src == "random":
            model, rev = _random_ttm(L, H, mcfg.get("random_size", "small")), "random-init"
        else:
            from tsfm_public.toolkit.get_model import get_model
            path, rev = mcfg["path"], (mcfg.get("revision", "auto") or "auto")
            if rev == "auto":
                key = get_model(path, context_length=L, prediction_length=H, return_model_key=True) \
                    if src == "hf" else "local"
                model = get_model(path, context_length=L, prediction_length=H)
                rev = str(key)
            else:
                from tsfm_public.models.tinytimemixer import TinyTimeMixerForPrediction
                model = TinyTimeMixerForPrediction.from_pretrained(path, revision=rev, prediction_filter_length=H)
    if int(model.config.context_length) != L:
        raise ValueError(f"model context_length={model.config.context_length} differs from data.context={L}")
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    info = ModelInfo(source=src, revision=rev, context_length=L,
                     prediction_length=int(model.config.prediction_length), horizon=H,
                     freq_prefix=bool(getattr(model.config, "resolution_prefix_tuning", False)),
                     n_params=sum(p.numel() for p in model.parameters()))
    return model, info


# ---------------------------------------------------------------- forward / predict

def _freq_token(model, n: int, freq: str = "10min") -> torch.Tensor | None:
    conf = getattr(model, "config", None)
    if conf is None or not getattr(conf, "resolution_prefix_tuning", False):
        return None
    from tsfm_public.toolkit.time_series_preprocessor import DEFAULT_FREQUENCY_MAPPING
    code = DEFAULT_FREQUENCY_MAPPING.get(FREQ_ALIASES.get(freq.lower(), freq), 0)
    return torch.full((n,), int(code), dtype=torch.long)


def forward(model, X: torch.Tensor, freq: str = "10min") -> torch.Tensor:
    """X [B, L, 1] -> forecast [B, H, 1] (original units)."""
    ft = _freq_token(model, X.shape[0], freq)
    out = model(past_values=X, freq_token=ft) if ft is not None else model(past_values=X)
    return out.prediction_outputs


@torch.no_grad()
def predict(model, X: np.ndarray, batch: int = 256, freq: str = "10min") -> np.ndarray:
    model.eval()
    outs = [forward(model, torch.from_numpy(np.ascontiguousarray(X[i:i + batch])), freq).numpy()
            for i in range(0, len(X), batch)]
    return np.concatenate(outs) if outs else np.zeros((0, 0, 1), np.float32)


# ---------------------------------------------------------------- fine-tuning modes

def ttm_core(model):
    return model.get_base_model() if hasattr(model, "get_base_model") else model


def configure_peft(model, cfg: dict[str, Any]):
    """Set requires_grad according to train.peft; returns the (possibly LoRA-wrapped) model."""
    mode = cfg["train"]["peft"]
    for p in model.parameters():
        p.requires_grad_(False)
    if mode == "head":
        for p in model.head.parameters():
            p.requires_grad_(True)
    elif mode == "decoder":
        for mod in (model.decoder, model.head):
            for p in mod.parameters():
                p.requires_grad_(True)
    elif mode == "full":
        for p in model.parameters():
            p.requires_grad_(True)
    elif mode == "lora":
        from peft import LoraConfig, get_peft_model
        lcfg = LoraConfig(r=int(cfg["train"]["lora_r"]), lora_alpha=int(cfg["train"]["lora_alpha"]),
                          target_modules=["fc1", "fc2"], lora_dropout=0.0, bias="none", modules_to_save=["head"])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = get_peft_model(model, lcfg)
    else:
        raise ValueError(mode)
    return model


def trainable_named(model) -> list[tuple[str, torch.nn.Parameter]]:
    return sorted(((n, p) for n, p in model.named_parameters() if p.requires_grad), key=lambda t: t[0])


def layout(model) -> list[tuple[str, tuple[int, ...], int]]:
    return [(n, tuple(p.shape), p.numel()) for n, p in trainable_named(model)]


def get_vec(model) -> np.ndarray:
    return np.concatenate([p.detach().cpu().numpy().ravel() for _, p in trainable_named(model)]).astype(np.float32)


def set_vec(model, vec: np.ndarray) -> None:
    named = trainable_named(model)
    total = sum(p.numel() for _, p in named)
    vec = np.asarray(vec, dtype=np.float32)
    if vec.size != total:
        raise ValueError(f"parameter vector size {vec.size} != model trainable size {total}")
    off = 0
    with torch.no_grad():
        for _, p in named:
            k = p.numel()
            p.copy_(torch.from_numpy(vec[off:off + k].reshape(p.shape)))
            off += k


def capture_params(model) -> list[torch.nn.Parameter]:
    """Trainable parameters in vector order (stays valid after requires_grad is switched off)."""
    return [p for _, p in trainable_named(model)]


def set_params_vec(params: list[torch.nn.Parameter], vec: np.ndarray) -> None:
    vec = np.asarray(vec, dtype=np.float32)
    off = 0
    with torch.no_grad():
        for p in params:
            k = p.numel()
            p.copy_(torch.from_numpy(vec[off:off + k].reshape(p.shape)))
            off += k
    if off != vec.size:
        raise ValueError(f"parameter vector size {vec.size} != captured size {off}")


def split_vec(vec: np.ndarray, lay) -> list[np.ndarray]:
    out, off = [], 0
    for _, shape, k in lay:
        out.append(vec[off:off + k].reshape(shape)); off += k
    return out


def join_arrays(arrays) -> np.ndarray:
    return np.concatenate([np.asarray(a, dtype=np.float32).ravel() for a in arrays])


def n_trainable(model) -> int:
    return sum(p.numel() for _, p in trainable_named(model))


def lora_segments(model) -> list[tuple[str, int, int, tuple[int, ...]]]:
    """(name, offset, size, shape) of LoRA A/B matrices inside the flat trainable vector."""
    segs, off = [], 0
    for n, p in trainable_named(model):
        if ".lora_A." in n or ".lora_B." in n:
            segs.append((n, off, p.numel(), tuple(p.shape)))
        off += p.numel()
    return segs


# ---------------------------------------------------------------- cached head-only path

@dataclass
class HeadCache:
    dec: torch.Tensor
    loc: torch.Tensor
    scale: torch.Tensor
    X: torch.Tensor


@torch.no_grad()
def build_head_cache(model, X: np.ndarray, batch: int = 256, freq: str = "10min") -> HeadCache:
    core = ttm_core(model)
    core.eval()
    decs, locs, scales = [], [], []
    for i in range(0, len(X), batch):
        xb = torch.from_numpy(np.ascontiguousarray(X[i:i + batch]))
        ft = _freq_token(core, xb.shape[0], freq)
        out = core(past_values=xb, freq_token=ft) if ft is not None else core(past_values=xb)
        if out.decoder_hidden_state is None:
            raise RuntimeError("model returned no decoder_hidden_state - set train.head_cache=false")
        decs.append(out.decoder_hidden_state); locs.append(out.loc); scales.append(out.scale)
    return HeadCache(torch.cat(decs), torch.cat(locs), torch.cat(scales), torch.from_numpy(np.ascontiguousarray(X)))


def head_forward(model, cache: HeadCache, idx: torch.Tensor, H: int) -> torch.Tensor:
    core = ttm_core(model)
    y = core.head(cache.dec[idx], past_values=cache.X[idx])
    if isinstance(y, tuple):
        y = y[0]
    return core.backbone.scaler.inverse(data=y[:, :H, :], loc=cache.loc[idx], scale=cache.scale[idx])


@torch.no_grad()
def check_head_cache(model, X: np.ndarray, H: int, freq: str = "10min") -> float:
    """Max relative deviation between the cached path and the full forward pass on a few windows."""
    Xs = X[: min(8, len(X))]
    cache = build_head_cache(model, Xs, freq=freq)
    core = ttm_core(model)
    core.eval()
    a = head_forward(model, cache, torch.arange(len(Xs)), H)
    b = forward(core, torch.from_numpy(np.ascontiguousarray(Xs)), freq)
    return float((a - b).abs().max() / (b.abs().max() + 1e-8))
