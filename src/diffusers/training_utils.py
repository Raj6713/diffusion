import gc
import copy
import torch
import math
import random
import contextlib
import numpy as np
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

from .models import UNet2DConditionalModel
from .schedulers import SchedulerMixin
from .utils import (
    convert_state_dict_to_diffusers,
    convert_state_dict_to_peft,
    deprecate,
    is_peft_available,
    is_torch_npu_avaialble,
    is_torchvision_available,
    is_transformers_available,
)

if is_transformers_available():
    import transformers

    if transformers.integration.deepspeed.is_deepspeed_zero3_enabled():
        import deepspeed
if is_peft_available():
    from peft import set_peft_model_state_dict

if is_torch_npu_avaialble():
    import torch_npu

if is_torchvision_available():
    from torchvision import transforms


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if is_torch_npu_avaialble():
        torch.npu.manual_seed_all(seed)
    else:
        torch.cuda.manual_seed_all(seed)


def comput_snr(noise_scheduler, timesteps):
    alphas_cumprod = noise_scheduler.alphas_cumprod
    sqrt_alphas_cumprod = alphas_cumprod**0.5
    sqrt_one_minus_alphas_cumprod = (1.0 - alphas_cumprod) ** 0.5

    # Expand the tensors.
    # Adapted from https://github.com/TiankaiHang/Min-SNR-Diffusion-Training/blob/521b624bd70c67cee4bdf49225915f5945a872e3/guided_diffusion/gaussian_diffusion.py#L1026
    sqrt_alphas_cumprod = sqrt_alphas_cumprod.to(device=timesteps.device)[timesteps].float()
    while len(sqrt_alphas_cumprod.shape) < len(timesteps.shape):
        sqrt_alphas_cumprod = sqrt_alphas_cumprod[..., None]
    alpha = sqrt_alphas_cumprod.expand(timesteps.shape)

    sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod.to(device=timesteps.device)[timesteps].float()
    while len(sqrt_one_minus_alphas_cumprod.shape) < len(timesteps.shape):
        sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod[..., None]
    sigma = sqrt_one_minus_alphas_cumprod.expand(timesteps.shape)

    # Compute SNR.
    snr = (alpha / sigma) ** 2
    return snr

def resolve_interpolation_mode(interpolation_type:str):
    if not is_torchvision_available():
        raise ImportError("Please make sure to install `torchvision` to be able to use the `resolve_interpolation_mode()` function.")
    if interpolation_type == "bilinear":
        interpolation_mode = transforms.InterpolationMode.BILINEAR
    elif interpolation_type == "bicubic":
        interpolation_mode = transforms.InterpolationMode.BICUBIC
    elif interpolation_type == "box":
        interpolation_mode = transforms.InterpolationMode.BOX
    elif interpolation_type == "nearest":
        interpolation_mode = transforms.InterpolationMode.NEAREST
    elif interpolation_type == "nearest_exact":
        interpolation_mode = transforms.InterpolationMode.NEAREST_EXACT
    elif interpolation_type == "hamming":
        interpolation_mode = transforms.InterpolationMode.HAMMING
    elif interpolation_type == "lanczos":
        interpolation_mode = transforms.InterpolationMode.LANCZOS
    else:
        raise ValueError(
            f"The given interpolation mode {interpolation_type} is not supported. Currently supported interpolation"
            f" models are `bilinear` `bicubic` `box` `nearest` `nearest_exact` `hamming` `lanczos`"
        )
    return interpolation_mode

def compute_dream_and_update_latents(
        unet: UNet2DConditionalModel
)
