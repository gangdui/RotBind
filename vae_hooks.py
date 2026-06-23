"""VAE encode hook for eval_freq_anchor.py.

Example:

FREQ_ANCHOR_DEVICE=cuda \
FREQ_ANCHOR_DTYPE=fp16 \
FREQ_ANCHOR_VAE_MODEL=stabilityai/stable-diffusion-2-1-base \
FREQ_ANCHOR_VAE_SUBFOLDER=vae \
python eval_freq_anchor.py \
  --image-dir path/to/watermarked_images \
  --outdir freq_anchor_eval_real_vae \
  --size 512 \
  --alphas 0.001,0.002,0.003,0.005 \
  --angles 5,10,15,30,45,60,75,90,120,150,180 \
  --enable-vae-metrics \
  --vae-encoder vae_hooks:encode_sd_vae
"""

from __future__ import annotations

import os
from typing import Dict, Tuple

import numpy as np
import torch

AutoencoderKL = None


_VAE_CACHE: Dict[Tuple[str, str, str, torch.dtype], torch.nn.Module] = {}


def _resolve_device() -> torch.device:
    device_name = os.environ.get("FREQ_ANCHOR_DEVICE")
    if not device_name:
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(device_name)


def _resolve_dtype(device: torch.device) -> torch.dtype:
    dtype_name = os.environ.get("FREQ_ANCHOR_DTYPE")
    if not dtype_name:
        dtype_name = "fp16" if device.type == "cuda" else "fp32"

    normalized = dtype_name.lower()
    if normalized in {"fp16", "float16", "half"}:
        return torch.float16
    if normalized in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if normalized in {"fp32", "float32", "full"}:
        return torch.float32
    raise ValueError(f"Unsupported FREQ_ANCHOR_DTYPE: {dtype_name}")


def _load_vae():
    global AutoencoderKL
    if AutoencoderKL is None:
        try:
            from diffusers import AutoencoderKL as DiffusersAutoencoderKL
        except Exception as exc:  # pragma: no cover - depends on local env.
            raise ImportError("diffusers.AutoencoderKL is required for encode_sd_vae") from exc
        AutoencoderKL = DiffusersAutoencoderKL

    device = _resolve_device()
    dtype = _resolve_dtype(device)
    model_name = os.environ.get("FREQ_ANCHOR_VAE_MODEL", "stabilityai/sd-vae-ft-mse")
    subfolder = os.environ.get("FREQ_ANCHOR_VAE_SUBFOLDER", "")
    cache_key = (model_name, subfolder, str(device), dtype)

    if cache_key not in _VAE_CACHE:
        kwargs = {"torch_dtype": dtype}
        if subfolder:
            kwargs["subfolder"] = subfolder
        vae = AutoencoderKL.from_pretrained(model_name, **kwargs)
        vae = vae.to(device).eval()
        for param in vae.parameters():
            param.requires_grad_(False)
        _VAE_CACHE[cache_key] = vae
    return _VAE_CACHE[cache_key], device, dtype


def encode_sd_vae(img_rgb: np.ndarray):
    """Encode a float32 RGB image in [0, 1] with a Stable Diffusion VAE."""

    img = np.asarray(img_rgb, dtype=np.float32)
    if img.ndim != 3 or img.shape[-1] != 3:
        raise ValueError(f"Expected image shape [H, W, 3], got {img.shape}")

    vae, device, dtype = _load_vae()
    x = torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1).unsqueeze(0)
    x = x.to(device=device, dtype=dtype)
    x = x * 2.0 - 1.0

    with torch.inference_mode():
        latents = vae.encode(x).latent_dist.mode()
        scaling_factor = getattr(vae.config, "scaling_factor", 1.0)
        latents = latents * scaling_factor
    return latents
