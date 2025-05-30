#!/usr/bin/env python
# coding=utf-8
# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and

import argparse
import time
import copy
import gc
import itertools
import logging
import math
import os
import ssl
os.environ['CURL_CA_BUNDLE'] = ''
os.environ["TOKENIZERS_PARALLELISM"] = "false"
ssl._create_default_https_context = ssl._create_unverified_context
import warnings
warnings.filterwarnings("ignore")
import random
import shutil
import warnings
from contextlib import nullcontext
from pathlib import Path
import json
import numpy as np
import cv2
import torch
from utils.torch_tool import init_torch_npu, is_torch_npu_avaliable
if is_torch_npu_avaliable():
    init_torch_npu()
from diffusers.utils.import_utils import is_torch_npu_available, is_xformers_available

import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import DistributedDataParallelKwargs, ProjectConfiguration, set_seed
from huggingface_hub import create_repo, upload_folder
from huggingface_hub.utils import insecure_hashlib
from PIL import Image
from PIL.ImageOps import exif_transpose
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms.functional import crop, pil_to_tensor
from tqdm.auto import tqdm
from transformers import (
    CLIPTextModelWithProjection,
    CLIPTokenizer,
    PretrainedConfig,
    T5EncoderModel,
    T5TokenizerFast,
    CLIPImageProcessor,
    CLIPVisionModelWithProjection
)
from accelerate import DeepSpeedPlugin
from accelerate.utils import DistributedType
from dataloaders.telebg_sr_loader import build_telebg_sr_dataloader

import diffusers
from diffusers import (
    AutoencoderKL,
    FlowMatchEulerDiscreteScheduler,
    StableDiffusion3Pipeline,
    FlowMatchEulerDiscreteScheduler
)

# from diffusers import SD3Transformer2DModel
from diffusers.optimization import get_scheduler
from diffusers.utils import (
    check_min_version,
    is_wandb_available,
)
from diffusers.utils.hub_utils import load_or_create_model_card, populate_model_card
from diffusers.utils.torch_utils import is_compiled_module
from diffusers.utils import load_image
from torchvision.utils import save_image
from diffusers.training_utils import EMAModel

# from sgm.util import instantiate_from_config
from omegaconf import OmegaConf
import torch.nn as nn
import albumentations

from pipelines.pipeline_stable_diffusion_3_cnext import StableDiffusion3ControlNetPipeline
from models.transformer_sd3_cnext import SD3Transformer2DModel
from models.modules import replace_with_fp32_forwards
from dataloaders.build import build_dataloader


# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.29.0")
logger = get_logger(__name__)

clip_L_32_path = "/data_mount/kong/clip"
clip_G_14_path = "/data_mount/kong/clip-G-14"
clip_image_processor_1 = CLIPImageProcessor(clip_L_32_path)
clip_image_processor_2 = CLIPImageProcessor(clip_G_14_path)

A_PROMPT = "Cinematic, High Contrast, highly detailed, taken using a Canon EOS R camera, hyper detailed photo realistic maximum detail, 32k, Color Grading, ultra HD, extreme meticulous detailing, skin pore detailing, hyper sharpness, perfect without deformations."

def tensor_to_image(x):
    x = torch.clamp(x, -1., 1.)
    x = (x + 1.) / 2. * 255
    x = x.cpu().numpy().astype(np.uint8)
    if x.shape[1] == 3 or x.shape[1] == 1:
        x = np.transpose(x, (0, 2, 3, 1))
    return x


def sliding_windows(h: int, w: int, tile_size: int, tile_stride: int):
    hi_list = list(range(0, h - tile_size + 1, tile_stride))
    if (h - tile_size) % tile_stride != 0:
        hi_list.append(h - tile_size)

    wi_list = list(range(0, w - tile_size + 1, tile_stride))
    if (w - tile_size) % tile_stride != 0:
        wi_list.append(w - tile_size)

    coords = []
    for hi in hi_list:
        for wi in wi_list:
            coords.append((hi, hi + tile_size, wi, wi + tile_size))
    return coords


def tiling_image(image, tile_size=1024, tile_overlap=64):
    w, h = image.size
    image_t = pil_to_tensor(image) # (3, h, w)
    image_tiles = []
    latent_tiles_iterator = sliding_windows(h, w, tile_size, tile_size - tile_overlap)
    for j, (hi, hi_end, wi, wi_end) in enumerate(latent_tiles_iterator):
        image_tile = image_t[..., hi:hi_end, wi:wi_end]
        image_tiles.append(image_tile)
    return image_tiles


def load_text_encoders(class_one, class_two, class_three, args=None):
    text_encoder_one = class_one.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision, variant=args.variant
    )

    text_encoder_two = class_two.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="text_encoder_2", revision=args.revision, variant=args.variant
    )

    if not args.no_t5_encoder and args.use_8bit_t5:
        print('use 8bit t5 encoder')
        assert not is_torch_npu_available(), 'npu not support 8bit'
        from transformers import T5EncoderModel, BitsAndBytesConfig
        quantization_config = BitsAndBytesConfig(load_in_8bit=True)
        text_encoder_three = class_three.from_pretrained(
            args.pretrained_model_name_or_path, 
            subfolder="text_encoder_3", 
            quantization_config=quantization_config
        )
    elif not args.no_t5_encoder:
        text_encoder_three = class_three.from_pretrained(
            args.pretrained_model_name_or_path, 
            subfolder="text_encoder_3", 
            revision=args.revision, variant=args.variant,
        )
    else:
        text_encoder_three = None
    return text_encoder_one, text_encoder_two, text_encoder_three


def compute_density_for_timestep_sampling(
    weighting_scheme: str, batch_size: int, logit_mean: float = None, logit_std: float = None, mode_scale: float = None
):
    """Compute the density for sampling the timesteps when doing SD3 training.

    Courtesy: This was contributed by Rafie Walker in https://github.com/huggingface/diffusers/pull/8528.

    SD3 paper reference: https://arxiv.org/abs/2403.03206v1.
    """
    if weighting_scheme == "logit_normal":
        # See 3.1 in the SD3 paper ($rf/lognorm(0.00,1.00)$).
        u = torch.normal(mean=logit_mean, std=logit_std, size=(batch_size,), device="cpu")
        u = torch.nn.functional.sigmoid(u)
    elif weighting_scheme == "mode":
        u = torch.rand(size=(batch_size,), device="cpu")
        u = 1 - u - mode_scale * (torch.cos(math.pi * u / 2) ** 2 - 1 + u)
    else:
        u = torch.rand(size=(batch_size,), device="cpu")
    return u

def compute_loss_weighting_for_sd3(weighting_scheme: str, sigmas=None):
    """Computes loss weighting scheme for SD3 training.

    Courtesy: This was contributed by Rafie Walker in https://github.com/huggingface/diffusers/pull/8528.

    SD3 paper reference: https://arxiv.org/abs/2403.03206v1.
    """
    if weighting_scheme == "sigma_sqrt":
        weighting = (sigmas**-2.0).float()
    elif weighting_scheme == "cosmap":
        bot = 1 - 2 * sigmas + 2 * sigmas**2
        weighting = 2 / (math.pi * bot)
    else:
        weighting = torch.ones_like(sigmas)
    return weighting

def log_validation(
    pipeline,
    data_loader,
    args,
    accelerator,
    pipeline_args,
    step,
    image_encoders,
    text_encoders,
    tokenizers,
    is_final_validation=False,
    num_images=4,
    sub_dir='val',
):
    logger.info(
        f"Running validation... \n Generating {args.num_validation_images} images with prompt:"
        f" {args.validation_prompt}."
    )
    if not args.use_8bit_t5:
        pipeline = pipeline.to(accelerator.device)
    pipeline.set_progress_bar_config(disable=True)

    # run inference
    generator = torch.Generator(device=accelerator.device).manual_seed(args.seed) if args.seed else None

    dtype = pipeline.transformer.dtype
    autocast_ctx = torch.autocast(accelerator.device.type, dtype=dtype) if not is_final_validation else nullcontext()

    for i, batch in enumerate(data_loader):
        if 'hq' in batch.keys():
            validation_images = tensor_to_image(batch['lq'])
            hq_images = tensor_to_image(batch['hq'])
        else:
            validation_images = tensor_to_image(batch['img_lq'])
            hq_images = tensor_to_image(batch['img'])
        
        validation_prompts = batch['txt']
        batch_global_images = batch['global_image_lq']
        batch_patch_images = batch['local_patch_lq']
        bs = len(validation_prompts)
        tic = time.time()
        for j in range(bs):
            control_image = Image.fromarray(validation_images[j])
            global_images = [ Image.fromarray(batch_global_images[j])]
            patch_images = [Image.fromarray(batch_patch_images[j])]
            validation_prompt = validation_prompts[j]
            hq_image = Image.fromarray(hq_images[j])
            with autocast_ctx:
                prompt_embeds, pooled_prompt_embeds = encode_multi_modal_prompt(
                    image_encoders, global_images, patch_images, text_encoders, tokenizers, validation_prompt, 0.
                )
                image = pipeline(
                    control_image=control_image, 
                    controlnet_conditioning_scale=1.0, 
                    num_inference_steps=28, 
                    guidance_scale=0, 
                    generator=generator,
                    prompt_embeds=prompt_embeds,
                    pooled_prompt_embeds=pooled_prompt_embeds,
                    denoise_vae_use_mode=args.denoise_vae_use_mode
                ).images[0]

            save_dir = os.path.join(args.output_dir, sub_dir)
            save_path = os.path.join(save_dir, f"{step:06d}_{(i*bs + j):06d}.jpg")
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            image.save(save_path)
            control_image.save(save_path.replace('.jpg','_lr.jpg'))
            hq_image.save(save_path.replace('.jpg', '_hr.jpg'))
            with open(save_path.replace('.jpg', '.json'), 'wt') as f:
                json.dump({'caption': validation_prompt}, f)
            f.close()
        
        logger.info(f'Val-{step}, {i}/{len(data_loader)} ... , save in {save_dir}, take {(time.time() - tic) / bs} s')

        if (i + 1) * bs >= num_images:
            break


def import_model_class_from_model_name_or_path(
    pretrained_model_name_or_path: str, revision: str, subfolder: str = "text_encoder"
):
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path, subfolder=subfolder, revision=revision
    )
    model_class = text_encoder_config.architectures[0]
    if model_class == "CLIPTextModelWithProjection":
        from transformers import CLIPTextModelWithProjection

        return CLIPTextModelWithProjection
    elif model_class == "T5EncoderModel":
        from transformers import T5EncoderModel

        return T5EncoderModel
    else:
        raise ValueError(f"{model_class} is not supported.")


def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--pretrained_sd3_tf_model_path",
        type=str,
        default=None,
        required=False,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default=None,
        help="Variant of the model files of the pretrained model identifier from huggingface.co/models, 'e.g.' fp16",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help=(
            "The name of the Dataset (from the HuggingFace hub) containing the training data of instance images (could be your own, possibly private,"
            " dataset). It can also be a path pointing to a local copy of a dataset in your filesystem,"
            " or to a folder containing files that 🤗 Datasets can understand."
        ),
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="The config of the Dataset, leave as None if there's only one config.",
    )
    parser.add_argument(
        "--instance_data_dir",
        type=str,
        default=None,
        help=("A folder containing the training data. "),
    )

    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="The directory where the downloaded models and datasets will be stored.",
    )

    parser.add_argument(
        "--image_column",
        type=str,
        default="image",
        help="The column of the dataset containing the target image. By "
        "default, the standard Image Dataset maps out 'file_name' "
        "to 'image'.",
    )
    parser.add_argument(
        "--caption_column",
        type=str,
        default=None,
        help="The column of the dataset containing the instance prompt for each image",
    )

    parser.add_argument("--repeats", type=int, default=1, help="How many times to repeat the training data.")

    parser.add_argument(
        "--class_data_dir",
        type=str,
        default=None,
        required=False,
        help="A folder containing the training data of class images.",
    )
    parser.add_argument(
        "--instance_prompt",
        type=str,
        default=A_PROMPT,
        help="The prompt with identifier specifying the instance, e.g. 'photo of a TOK dog', 'in the style of TOK'",
    )
    parser.add_argument(
        "--class_prompt",
        type=str,
        default=None,
        help="The prompt to specify images in the same class as provided instance images.",
    )
    parser.add_argument(
        "--validation_prompt",
        type=str,
        default=A_PROMPT,
        help="A prompt that is used during validation to verify that the model is learning.",
    )
    parser.add_argument(
        "--num_validation_images",
        type=int,
        default=4,
        help="Number of images that should be generated during validation with `validation_prompt`.",
    )
    parser.add_argument(
        "--validation_steps",
        type=int,
        default=1000,
        help=(
            "Run dreambooth validation every X epochs. Dreambooth validation consists of running the prompt"
            " `args.validation_prompt` multiple times: `args.num_validation_images`."
        ),
    )
    parser.add_argument(
        "--with_prior_preservation",
        default=False,
        action="store_true",
        help="Flag to add prior preservation loss.",
    )
    parser.add_argument("--prior_loss_weight", type=float, default=1.0, help="The weight of prior preservation loss.")
    parser.add_argument(
        "--num_class_images",
        type=int,
        default=100,
        help=(
            "Minimal class images for prior preservation loss. If there are not enough images already present in"
            " class_data_dir, additional images will be sampled with class_prompt."
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="sd3-dreambooth",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--center_crop",
        default=False,
        action="store_true",
        help=(
            "Whether to center crop the input images to the resolution. If not set, the images will be randomly"
            " cropped. The images will be resized to the resolution first before cropping."
        ),
    )
    parser.add_argument(
        "--random_flip",
        action="store_true",
        help="whether to randomly flip images horizontally",
    )
    parser.add_argument(
        "--train_text_encoder",
        action="store_true",
        help="Whether to train the text encoder. If set, the text encoder should be float32 precision.",
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=4, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument(
        "--sample_batch_size", type=int, default=4, help="Batch size (per device) for sampling images."
    )
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints can be used both as final"
            " checkpoints in case they are better than the last checkpoint, and are also suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=None,
        help=("Max number of checkpoints to store."),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    )

    parser.add_argument(
        "--text_encoder_lr",
        type=float,
        default=5e-6,
        help="Text encoder learning rate to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--lr_num_cycles",
        type=int,
        default=1,
        help="Number of hard resets of the lr in cosine_with_restarts scheduler.",
    )
    parser.add_argument("--lr_power", type=float, default=1.0, help="Power factor of the polynomial scheduler.")
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=4,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument(
        "--weighting_scheme", type=str, default="logit_normal", choices=["sigma_sqrt", "logit_normal", "mode"]
    )
    parser.add_argument("--logit_mean", type=float, default=0.0)
    parser.add_argument("--logit_std", type=float, default=1.0)
    parser.add_argument("--mode_scale", type=float, default=1.29)
    parser.add_argument(
        "--optimizer",
        type=str,
        default="AdamW",
        help=('The optimizer type to use. Choose between ["AdamW", "prodigy"]'),
    )

    parser.add_argument(
        "--use_8bit_adam",
        action="store_true",
        help="Whether or not to use 8-bit Adam from bitsandbytes. Ignored if optimizer is not set to AdamW",
    )

    parser.add_argument(
        "--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam and Prodigy optimizers."
    )
    parser.add_argument(
        "--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam and Prodigy optimizers."
    )
    parser.add_argument(
        "--prodigy_beta3",
        type=float,
        default=None,
        help="coefficients for computing the Prodidy stepsize using running averages. If set to None, "
        "uses the value of square root of beta2. Ignored if optimizer is adamW",
    )
    parser.add_argument("--prodigy_decouple", type=bool, default=True, help="Use AdamW style decoupled weight decay")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-04, help="Weight decay to use for unet params")
    parser.add_argument(
        "--adam_weight_decay_text_encoder", type=float, default=1e-03, help="Weight decay to use for text_encoder"
    )

    parser.add_argument(
        "--adam_epsilon",
        type=float,
        default=1e-08,
        help="Epsilon value for the Adam optimizer and Prodigy optimizers.",
    )

    parser.add_argument(
        "--prodigy_use_bias_correction",
        type=bool,
        default=True,
        help="Turn on Adam's bias correction. True by default. Ignored if optimizer is adamW",
    )
    parser.add_argument(
        "--prodigy_safeguard_warmup",
        type=bool,
        default=True,
        help="Remove lr from the denominator of D estimate to avoid issues during warm-up stage. True by default. "
        "Ignored if optimizer is adamW",
    )
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument("--hub_token", type=str, default=None, help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="The name of the repository to keep in sync with the local `output_dir`.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--prior_generation_precision",
        type=str,
        default=None,
        choices=["no", "fp32", "fp16", "bf16"],
        help=(
            "Choose prior generation precision between fp32, fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to  fp16 if a GPU is available else fp32."
        ),
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    
    parser.add_argument(
        "--dataset_config",
        type=str,
        default='configs/data_100.yaml',
        help="The config of the Dataset",
    )
    
    parser.add_argument(
        "--enable_npu_flash_attention", action="store_true", help="Whether or not to use npu flash attention."
    )
    parser.add_argument(
        "--no_t5_encoder", action="store_true", help="Whether or not to use t5 encoder."
    )
    parser.add_argument(
        "--use_8bit_t5", action="store_true", help="Whether or not to use 8bit t5 encoder."
    )
    parser.add_argument(
        "--num_controlnet_layers", default=6,  type=int, help='number of controlnet layers.'
    )
    parser.add_argument(
        "--use_lr_vae_shift", action="store_true", help='whether use vae shift for lr encoder'
    )

    parser.add_argument(
        "--use_deepspeed", action="store_true", help='whether use deepspeed for training'
    )

    parser.add_argument(
        "--controlnet_use_zero_text", action="store_true", help='whether use zero input for text '
    )
    parser.add_argument("--use_ema", action="store_true", help="Whether to use EMA model.")

    parser.add_argument(
        "--caption_dropout_prob",
        type=float,
        default=0.464,
        help="Randomly change the caption of a sample to a blank string with the given probability.",
    )

    parser.add_argument(
        "--attn_use_fp32",
        default=False,
        action="store_true",
        help="Map dtype of params of attn layers to float32 ",
    )
    parser.add_argument(
        "--only_ft_qkv",
        default=False,
        action="store_true",
        help="Map dtype of params of attn layers to float32 ",
    )

    parser.add_argument(
        "--allow_hf32",
        default=False,
        action="store_true",
    )

    parser.add_argument('--enable_stable_fp32', action='store_true')

    parser.add_argument(
        "--denoise_vae_ckpt",
        type=str,
        default=None,
        help="use to denoise input image",
    )

    parser.add_argument(
        "--loss_scale",
        type=float,
        default=1.,
        help="use to scale the loss",
    )

    parser.add_argument(
        "--denoise_vae_use_mode",
        default=False,
        action="store_true",
        help="Denoise vae use mode() to get latents ",
    )


    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    return args


def prepare_dataset(args, accelerator):
    
    configs = [OmegaConf.load(cfg) for cfg in [args.dataset_config]]
    config = OmegaConf.merge(*configs)
    config.data['params']['batch_size'] = args.train_batch_size
    if 'size' in config.data['params']['train']['params']:
        config.data['params']['train']['params']['size'] = args.resolution
        config.data['params']['validation']['params']['size'] = args.resolution

    data = instantiate_from_config(config.data)
    data.prepare_data()
    data.setup()
    
    # if accelerator.is_main_process:
    #     accelerator.log(config.data)
    #     # accelerator.log("#### Data #####")
    #     for k in data.datasets:
    #         accelerator.log(f"{k}, {data.datasets[k].__class__.__name__}, {len(data.datasets[k])}")

    train_dataset = data.datasets['train']
    train_dataloader = data.train_dataloader()
    val_dataset = data.datasets['validation']
    val_dataloader = data.val_dataloader()

    return train_dataset, train_dataloader, val_dataset, val_dataloader


def prepare_dataset_vcg(args, accelerator):
    cfg = OmegaConf.load(args.dataset_config)
    cfg.dataloader.train.batch_size = args.train_batch_size
    cfg['rank'] = accelerator.local_process_index
    cfg['world_size'] = accelerator.num_processes
    train_dataloader = build_dataloader(cfg, accelerator)
    train_dataset = None
    return train_dataset, train_dataloader


def collate_fn(examples, with_prior_preservation=False):
    pixel_values = [example["instance_images"] for example in examples]
    prompts = [example["instance_prompt"] for example in examples]

    # Concat class and instance examples for prior preservation.
    # We do this to avoid doing two forward passes.
    if with_prior_preservation:
        pixel_values += [example["class_images"] for example in examples]
        prompts += [example["class_prompt"] for example in examples]

    pixel_values = torch.stack(pixel_values)
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()

    batch = {"pixel_values": pixel_values, "prompts": prompts}
    return batch


def _encode_prompt_with_clip_vit(image, patch_image, image_encoder_one, image_encoder_two, device, max_length=77):
    image = Image.fromarray(image) if type(image) is np.ndarray else image
    patch_image = Image.fromarray(patch_image) if type(patch_image) is np.ndarray else patch_image
    clip_patch_image = clip_image_processor_1(images=[patch_image], return_tensors="pt").pixel_values
    
    cls_patch_image_emb_one = image_encoder_one(clip_patch_image.to(device)).last_hidden_state # [1, 50, 768]
    
    cls_patch_image_emb_two = image_encoder_two(clip_patch_image.to(device)).image_embeds # [1, 1280]
    pooled_prompt_embeds = torch.cat([cls_patch_image_emb_one[:,0,:], cls_patch_image_emb_two], dim=-1)
    # pooled_prompt_embeds = pooled_prompt_embeds.unsqueeze(0)
    clip_patch_embeds = torch.nn.functional.pad(
        cls_patch_image_emb_one, (0, pooled_prompt_embeds.shape[-1] - cls_patch_image_emb_one.shape[-1])
    )

    image_tiles = tiling_image(image)
    image_tiles_embs_one_list = []
    image_tiles_embs_two_list = []
    # print(f"image_tiles:{len(image_tiles)}")
    for idx, im_tile in enumerate(image_tiles):
        clip_image_1 = clip_image_processor_1(images=[im_tile], return_tensors="pt", input_data_format="channels_first").pixel_values
        clip_image_2 = clip_image_processor_2(images=[im_tile], return_tensors="pt", input_data_format="channels_first").pixel_values

        image_embeds1 = image_encoder_one(clip_image_1.to(device)).last_hidden_state[:,0,:]
        image_embeds2 = image_encoder_two(clip_image_2.to(device)).image_embeds

        image_tiles_embs_one_list.append(image_embeds1.unsqueeze(0))
        image_tiles_embs_two_list.append(image_embeds2.unsqueeze(0))

    image_tiles_embs_one = torch.cat(image_tiles_embs_one_list, dim=-2)
    image_tiles_embs_two = torch.cat(image_tiles_embs_two_list, dim=-2)
    # print(f"image_tiles_embs_one:{image_tiles_embs_one.shape}, image_tiles_embs_two:{image_tiles_embs_two.shape}")

    clip_prompt_embeds = torch.cat([image_tiles_embs_one, image_tiles_embs_two], dim=-1)
    prompt_embeds = torch.cat([clip_patch_embeds, clip_prompt_embeds], dim=-2)
    batch, seq_len, embeds_size = prompt_embeds.shape
    if seq_len < max_length:
        prompt_embeds = torch.nn.functional.pad(prompt_embeds, (0, 0, 0, max_length - seq_len))
    else:
        prompt_embeds = prompt_embeds[:, :max_length, :]

    # print(f"prompt_embeds:{prompt_embeds.shape}, pooled_prompt_embeds:{pooled_prompt_embeds.shape}")
    return prompt_embeds, pooled_prompt_embeds


def tokenize_prompt(tokenizer, prompt):
    text_inputs = tokenizer(
        prompt,
        padding="max_length",
        max_length=77,
        truncation=True,
        return_tensors="pt",
    )
    text_input_ids = text_inputs.input_ids
    return text_input_ids


def _encode_prompt_with_t5(
    text_encoder,
    tokenizer,
    prompt=None,
    num_images_per_prompt=1,
    device=None,
):
    prompt = [prompt] if isinstance(prompt, str) else prompt
    batch_size = len(prompt)

    text_inputs = tokenizer(
        prompt,
        padding="max_length",
        # max_length=77,
        max_length=256,
        truncation=True,
        add_special_tokens=True,
        return_tensors="pt",
    )
    text_input_ids = text_inputs.input_ids
    prompt_embeds = text_encoder(text_input_ids.to(device))[0]

    dtype = text_encoder.dtype
    prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)

    _, seq_len, _ = prompt_embeds.shape

    # duplicate text embeddings and attention mask for each generation per prompt, using mps friendly method
    prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
    prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

    return prompt_embeds


def _encode_prompt_with_clip(
    text_encoder,
    tokenizer,
    prompt: str,
    device=None,
    num_images_per_prompt: int = 1,
):
    prompt = [prompt] if isinstance(prompt, str) else prompt
    batch_size = len(prompt)

    text_inputs = tokenizer(
        prompt,
        padding="max_length",
        max_length=77,
        truncation=True,
        return_tensors="pt",
    )

    text_input_ids = text_inputs.input_ids
    prompt_embeds = text_encoder(text_input_ids.to(device), output_hidden_states=True)

    pooled_prompt_embeds = prompt_embeds[0]
    prompt_embeds = prompt_embeds.hidden_states[-2]
    prompt_embeds = prompt_embeds.to(dtype=text_encoder.dtype, device=device)

    _, seq_len, _ = prompt_embeds.shape
    # duplicate text embeddings for each generation per prompt, using mps friendly method
    prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
    prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

    return prompt_embeds, pooled_prompt_embeds


def encode_multi_modal_prompt(
    image_encoders,
    global_images,
    patch_images,
    text_encoders,
    tokenizers,
    prompt: str,
    proportion_empty_prompts: float,
    device=None,
    num_images_per_prompt: int = 1,
):
    prompt = [prompt] if isinstance(prompt, str) else prompt

    clip_prompt_embeds_list = []
    clip_pooled_prompt_embeds_list = []
    image_encoder1, image_encoder2 = image_encoders[:2]
    for global_image, patch_image in zip(global_images, patch_images):
        # tiled the image to 1024x1024 pathes and get cls_tokens
        prompt_embeds, pooled_prompt_embeds = _encode_prompt_with_clip_vit(
            global_image, 
            patch_image, 
            image_encoder1, 
            image_encoder2, 
            device=device if device is not None else image_encoder1.device,
            max_length=77
        )

        if random.random() < proportion_empty_prompts:
            prompt_embeds  = torch.zeros(prompt_embeds.shape, device=prompt_embeds.device)
            pooled_prompt_embeds = torch.zeros(pooled_prompt_embeds.shape, device=pooled_prompt_embeds.device)

        clip_prompt_embeds_list.append(prompt_embeds)
        clip_pooled_prompt_embeds_list.append(pooled_prompt_embeds)
    
    clip_prompt_embeds = torch.cat(clip_prompt_embeds_list, dim=0)
    pooled_prompt_embeds = torch.cat(clip_pooled_prompt_embeds_list, dim=0)

    t5_zero_embeds = torch.zeros(len(prompt), 256, 4096).to(device if device is not None else text_encoders[-1].device)
    if text_encoders[-1] is not None:
        t5_prompt_embed = t5_zero_embeds if random.random() < proportion_empty_prompts else _encode_prompt_with_t5(
            text_encoders[-1],
            tokenizers[-1],
            prompt=prompt,
            num_images_per_prompt=num_images_per_prompt,
            device=device if device is not None else text_encoders[-1].device,
        )
    else:
        # print('here, use empty t5')
        t5_prompt_embed = t5_zero_embeds.to(clip_prompt_embeds.dtype)

    clip_prompt_embeds = torch.nn.functional.pad(
        clip_prompt_embeds, (0, t5_prompt_embed.shape[-1] - clip_prompt_embeds.shape[-1])
    )
    prompt_embeds = torch.cat([clip_prompt_embeds, t5_prompt_embed], dim=-2)
    # print(f"last prompt_embeds:{prompt_embeds.shape}, pooled_prompt_embeds:{pooled_prompt_embeds.shape}")
    return prompt_embeds, pooled_prompt_embeds


def main(args):
    # use LayerNorm, GeLu, SiLu always as fp32 mode
    if args.enable_stable_fp32:
        replace_with_fp32_forwards()

    if args.report_to == "wandb" and args.hub_token is not None:
        raise ValueError(
            "You cannot use both --report_to=wandb and --hub_token due to a security risk of exposing your token."
            " Please use `huggingface-cli login` to authenticate with the Hub."
        )

    if torch.backends.mps.is_available() and args.mixed_precision == "bf16":
        # due to pytorch#99272, MPS does not yet support bfloat16.
        raise ValueError(
            "Mixed precision training with bfloat16 is not supported on MPS. Please use fp16 (recommended) or fp32 instead."
        )

    logging_dir = Path(args.output_dir, args.logging_dir)

    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)
    kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
        kwargs_handlers=[kwargs],
    )

    # Disable AMP for MPS.
    if torch.backends.mps.is_available():
        accelerator.native_amp = False

    if args.report_to == "wandb":
        if not is_wandb_available():
            raise ImportError("Make sure to install wandb if you want to use it for logging during training.")

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)


    # Load the tokenizers
    tokenizer_one = CLIPTokenizer.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer",
        revision=args.revision,
    )
    tokenizer_two = CLIPTokenizer.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer_2",
        revision=args.revision,
    )
    tokenizer_three = T5TokenizerFast.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer_3",
        revision=args.revision,
    )

    # import correct text encoder classes
    text_encoder_cls_one = import_model_class_from_model_name_or_path(
        args.pretrained_model_name_or_path, args.revision
    )
    text_encoder_cls_two = import_model_class_from_model_name_or_path(
        args.pretrained_model_name_or_path, args.revision, subfolder="text_encoder_2"
    )
    text_encoder_cls_three = import_model_class_from_model_name_or_path(
        args.pretrained_model_name_or_path, args.revision, subfolder="text_encoder_3"
    )

    clip_image_processor_one = CLIPImageProcessor(clip_L_32_path)
    clip_image_processor_two = CLIPImageProcessor(clip_G_14_path)

    image_encoder_one = CLIPVisionModelWithProjection.from_pretrained(clip_L_32_path).to("npu")
    image_encoder_two = CLIPVisionModelWithProjection.from_pretrained(clip_G_14_path).to("npu")
   

    # Load scheduler and models
    noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="scheduler"
    )
    noise_scheduler_copy = copy.deepcopy(noise_scheduler)
    text_encoder_one, text_encoder_two, text_encoder_three = load_text_encoders(
        text_encoder_cls_one, text_encoder_cls_two, text_encoder_cls_three, args
    )
    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="vae",
        revision=args.revision,
        variant=args.variant,
    )

    denoise_vae = None
    if args.denoise_vae_ckpt is not None and os.path.exists(args.denoise_vae_ckpt):
        logger.info(f"[INFO] load denoise_vae from {args.denoise_vae_ckpt}")
        denoise_vae = AutoencoderKL.from_pretrained(args.denoise_vae_ckpt)

    if args.pretrained_sd3_tf_model_path is None:
        transformer = SD3Transformer2DModel.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="transformer", revision=args.revision, variant=args.variant, low_cpu_mem_usage=False, device_map=None
        )
    else:
        logger.info(f'load transformer form {args.pretrained_sd3_tf_model_path}')
        transformer = SD3Transformer2DModel.from_pretrained(
            args.pretrained_sd3_tf_model_path, revision=args.revision, variant=args.variant, low_cpu_mem_usage=False, device_map=None
        )
        transformer.pos_embed.pos_embed.copy_(torch.load('pos_embed.t5'))
    
    # controlnet = SD3ControlNetModel.from_transformer(transformer, num_layers=args.num_controlnet_layers)

    transformer.requires_grad_(False)
    def check_name(name):
        if not args.only_ft_qkv:
            return True
        if "norm" in name or "control" in name:
            return True
        if "to_out" in name or "to_v" in name or "to_q" in name or "to_k" in name:
            return True
        return False
        
    learable_parameters = []
    for name, params in transformer.named_parameters():
        if check_name(name):
            params.requires_grad_(True)
            if args.attn_use_fp32 and ("to_out" in name or "to_v" in name or "to_q" in name or "to_k" in name):
                params.data = params.to(torch.float32)
            learable_parameters.append(params)
    vae.requires_grad_(False)
    # [31/8/2024] Added by lifan
    if denoise_vae is not None:
        denoise_vae.requires_grad_(False)

    text_encoder_one.requires_grad_(False)
    text_encoder_two.requires_grad_(False)
    if text_encoder_three is not None:
        text_encoder_three.requires_grad_(False)

    # For mixed precision training we cast all non-trainable weights (vae, non-lora text_encoder and non-lora transformer) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    if torch.backends.mps.is_available() and weight_dtype == torch.bfloat16:
        # due to pytorch#99272, MPS does not yet support bfloat16.
        raise ValueError(
            "Mixed precision training with bfloat16 is not supported on MPS. Please use fp16 (recommended) or fp32 instead."
        )

    vae.to(accelerator.device, dtype=torch.float32)
    # [31/8/2024] Added by lifan
    if denoise_vae is not None:
        denoise_vae.to(accelerator.device, dtype=torch.float32)
    text_encoder_one.to(accelerator.device, dtype=weight_dtype)
    text_encoder_two.to(accelerator.device, dtype=weight_dtype)
    if text_encoder_three is not None and not args.use_8bit_t5:
        text_encoder_three.to(accelerator.device, dtype=weight_dtype)

    image_encoder_one.to(accelerator.device, dtype=weight_dtype)
    image_encoder_two.to(accelerator.device, dtype=weight_dtype)

    if args.use_ema:
        # accelerator.log('use ema for transformer')
        ema_transformer = EMAModel(transformer.parameters(), model_cls=SD3Transformer2DModel, model_config=transformer.config)
        ema_transformer.to(accelerator.device)

    if args.enable_npu_flash_attention:
        if is_torch_npu_available():
            logger.info("npu flash attention enabled.")
            transformer.enable_npu_flash_attention()
        else:
            raise ValueError("npu flash attention requires torch_npu extensions and is supported only on npu devices.")

    if args.gradient_checkpointing:
        transformer.enable_gradient_checkpointing()
        # controlnet.enable_gradient_checkpointing()

    def unwrap_model(model):
        model = accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model

    # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
    def save_model_hook(models, weights, output_dir):
        if accelerator.is_main_process:
            if args.use_ema:
                ema_transformer.save_pretrained(os.path.join(output_dir, "cnext_ema"))
            for i, model in enumerate(models):
                if isinstance(unwrap_model(model), SD3Transformer2DModel):
                    unwrap_model(model).save_pretrained(os.path.join(output_dir, "cnext"))
                if weights:
                    weights.pop()

    def load_model_hook(models, input_dir):
        if args.use_ema:
            load_model = EMAModel.from_pretrained(os.path.join(input_dir, "cnext_ema"), SD3Transformer2DModel)
            ema_transformer.load_state_dict(load_model.state_dict())
            ema_transformer.to(accelerator.device)
            del load_model
        for _ in range(len(models)):
            # pop models so that they are not loaded again
            model = models.pop()
            if isinstance(unwrap_model(model), SD3Transformer2DModel):
                load_model = SD3Transformer2DModel.from_pretrained(input_dir, subfolder="cnext")
                model.register_to_config(**load_model.config)
                model.load_state_dict(load_model.state_dict())

            del load_model

    accelerator.register_save_state_pre_hook(save_model_hook)
    accelerator.register_load_state_pre_hook(load_model_hook)

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32 and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
    
    if not args.allow_hf32:
        torch.npu.matmul.allow_hf32 = False
        torch.npu.conv.allow_hf32 = False
    print(f"[INFO] torch.npu.conv.allow_hf32:{torch.npu.conv.allow_hf32}, torch.npu.matmul.allow_hf32:{torch.npu.matmul.allow_hf32}")

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

    # Optimization parameters
    # controlnet_transformer_parameters_with_lr = {"params": controlnet.parameters(), "lr": args.learning_rate}
    cnext_transformer_parameters_with_lr = {"params": learable_parameters, "lr": args.learning_rate}
    params_to_optimize = [cnext_transformer_parameters_with_lr]

    # Optimizer creation
    if not (args.optimizer.lower() == "prodigy" or args.optimizer.lower() == "adamw"):
        logger.warning(
            f"Unsupported choice of optimizer: {args.optimizer}.Supported optimizers include [adamW, prodigy]."
            "Defaulting to adamW"
        )
        args.optimizer = "adamw"

    if args.use_8bit_adam and not args.optimizer.lower() == "adamw":
        logger.warning(
            f"use_8bit_adam is ignored when optimizer is not set to 'AdamW'. Optimizer was "
            f"set to {args.optimizer.lower()}"
        )

    if args.optimizer.lower() == "adamw":
        if args.use_8bit_adam:
            try:
                import bitsandbytes as bnb
            except ImportError:
                raise ImportError(
                    "To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`."
                )

            optimizer_class = bnb.optim.AdamW8bit
        else:
            optimizer_class = torch.optim.AdamW

        optimizer = optimizer_class(
            params_to_optimize,
            betas=(args.adam_beta1, args.adam_beta2),
            weight_decay=args.adam_weight_decay,
            eps=args.adam_epsilon,
        )

    # train_dataset, train_dataloader, val_dataset, val_dataloader = prepare_dataset(args, accelerator)
    # [25/7/2024] Added by lifan
    if "telebg" in args.dataset_config:
        # [16/8/2024] Added by lifan
        train_dataset, train_dataloader = build_telebg_sr_dataloader(args)
    else:
        _, train_dataloader = prepare_dataset_vcg(args, accelerator)
    val_dataset, val_dataloader = None, None
    train_dataloader_unwrap = train_dataloader

    tokenizers = [tokenizer_one, tokenizer_two, tokenizer_three]
    # text_encoders = [text_encoder_one, text_encoder_two, text_encoder_three]
    text_encoders = [text_encoder_three]
    image_encoders = [image_encoder_one, image_encoder_two]

    # def compute_text_embeddings(prompt, text_encoders, tokenizers):
    #     with torch.no_grad():
    #         prompt_embeds, pooled_prompt_embeds = encode_prompt(text_encoders, tokenizers, prompt, args.caption_dropout_prob)
    #         prompt_embeds = prompt_embeds.to(accelerator.device)
    #         pooled_prompt_embeds = pooled_prompt_embeds.to(accelerator.device)
    #     return prompt_embeds, pooled_prompt_embeds

    def compute_image_text_embeddings(global_images, patch_images, image_encoders, prompt, text_encoders, tokenizers):
        with torch.no_grad():
            prompt_embeds, pooled_prompt_embeds = encode_multi_modal_prompt(
                image_encoders, global_images, patch_images, text_encoders, tokenizers, prompt, args.caption_dropout_prob
            )

            prompt_embeds = prompt_embeds.to(accelerator.device)
            pooled_prompt_embeds = pooled_prompt_embeds.to(accelerator.device)
        return prompt_embeds, pooled_prompt_embeds

   
    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
        num_cycles=args.lr_num_cycles,
        power=args.lr_power,
    )

    transformer, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        transformer, optimizer, train_dataloader, lr_scheduler
    )

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        tracker_name = "cnext-sd3"
        accelerator.init_trackers(tracker_name, config=vars(args))

    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    # logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num batches each epoch = {len(train_dataloader)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    logger.info(f"  Loss scale = {args.loss_scale}")
    logger.info(f'  Trainable parameters [cnext] = {(sum([np.prod(p.size()) for p in learable_parameters])):,}')

    global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the mos recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])

            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch

    else:
        initial_global_step = 0

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )

    def get_sigmas(timesteps, n_dim=4, dtype=torch.float32):
        sigmas = noise_scheduler_copy.sigmas.to(device=accelerator.device, dtype=dtype)
        schedule_timesteps = noise_scheduler_copy.timesteps.to(accelerator.device)
        timesteps = timesteps.to(accelerator.device)
        step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]

        sigma = sigmas[step_indices].flatten()
        while len(sigma.shape) < n_dim:
            sigma = sigma.unsqueeze(-1)
        return sigma

    for epoch in range(first_epoch, args.num_train_epochs):
        transformer.train()
        # controlnet.train()

        for step, batch in enumerate(train_dataloader):
            models_to_accumulate = [transformer]
            # models_to_accumulate = [controlnet]

            grad_norm = None
            with accelerator.accumulate(models_to_accumulate):
                if 'hq' in batch.keys():
                    pixel_values = batch["hq"].to(dtype=vae.dtype)
                    conditional_pixel_values = batch["lq"].to(dtype=vae.dtype)
                else:
                    pixel_values = batch["img"].to(dtype=vae.dtype)
                    conditional_pixel_values = batch["img_lq"].to(dtype=vae.dtype)

                prompts = batch["txt"]

                # prompt_embeds, pooled_prompt_embeds = compute_text_embeddings(
                #     prompts, text_encoders, tokenizers
                # )
                prompt_embeds, pooled_prompt_embeds = compute_image_text_embeddings(
                    batch['global_image_lq'], batch['local_patch_lq'], image_encoders, prompts, text_encoders, tokenizers
                )
                 
                # Convert images to latent space
                model_input = vae.encode(pixel_values).latent_dist.sample()
                model_input = (model_input - vae.config.shift_factor) * vae.config.scaling_factor
                model_input = model_input.to(dtype=weight_dtype)

                # Sample noise that we'll add to the latents
                noise = torch.randn_like(model_input)
                bsz = model_input.shape[0]

                # Sample a random timestep for each image
                # indices = torch.randint(0, noise_scheduler_copy.config.num_train_timesteps, (bsz,))
                # timesteps = noise_scheduler_copy.timesteps[indices].to(device=model_input.device)

                # Sample a random timestep for each image
                # for weighting schemes where we sample timesteps non-uniformly
                u = compute_density_for_timestep_sampling(
                    weighting_scheme=args.weighting_scheme,
                    batch_size=bsz,
                    logit_mean=args.logit_mean,
                    logit_std=args.logit_std,
                    mode_scale=args.mode_scale,
                )
                indices = (u * noise_scheduler_copy.config.num_train_timesteps).long()
                timesteps = noise_scheduler_copy.timesteps[indices].to(device=model_input.device)

                # Add noise according to flow matching.
                sigmas = get_sigmas(timesteps, n_dim=model_input.ndim, dtype=model_input.dtype)
                noisy_model_input = sigmas * noise + (1.0 - sigmas) * model_input

                # Prepare LQ condtion
                if denoise_vae is not None:
                    if args.denoise_vae_use_mode:
                        control_image = denoise_vae.encode(conditional_pixel_values).latent_dist.mode()
                    else:
                        control_image = denoise_vae.encode(conditional_pixel_values).latent_dist.sample()
                else:
                    control_image = vae.encode(conditional_pixel_values).latent_dist.sample()
                control_image = (control_image - vae.config.shift_factor) * vae.config.scaling_factor
                control_image = control_image.to(weight_dtype)

                if args.controlnet_use_zero_text:
                    # SD3 tuned with setting uncondition==0
                    embeds_kwargs = dict(
                        encoder_hidden_states=torch.zeros_like(prompt_embeds),
                        pooled_projections=torch.zeros_like(pooled_prompt_embeds),
                    )
                else:
                    # SD3 tuned with setting uncondition==""
                    embeds_kwargs = dict(
                        encoder_hidden_states=prompt_embeds,
                        pooled_projections=pooled_prompt_embeds,
                    )

                model_pred = transformer(
                    hidden_states=noisy_model_input,
                    timestep=timesteps,
                    controlnet_cond=control_image,
                    return_dict=False,
                    **embeds_kwargs
                )[0]

                # Follow: Section 5 of https://arxiv.org/abs/2206.00364.
                # Preconditioning of the model outputs.
                model_pred = model_pred * (-sigmas) + noisy_model_input

                weighting = compute_loss_weighting_for_sd3(weighting_scheme=args.weighting_scheme, sigmas=sigmas)

                target = model_input
                # Compute regular loss.
                loss = torch.mean(
                    (weighting.float() * (model_pred.float() - target.float()) ** 2).reshape(target.shape[0], -1),
                    1,
                )
                loss = loss.mean() * args.loss_scale

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    # params_to_clip = controlnet.parameters()
                    params_to_clip = learable_parameters
                    grad_norm = accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                if args.use_ema:
                    ema_transformer.step(transformer.parameters())

                if accelerator.is_main_process or accelerator.distributed_type == DistributedType.DEEPSPEED:
                    if global_step % args.checkpointing_steps == 0:
                        # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                        if accelerator.is_main_process and args.checkpoints_total_limit is not None:
                            checkpoints = os.listdir(args.output_dir)
                            checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                            # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                            if len(checkpoints) >= args.checkpoints_total_limit:
                                num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                                removing_checkpoints = checkpoints[0:num_to_remove]

                                logger.info(
                                    f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                )
                                logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                                for removing_checkpoint in removing_checkpoints:
                                    removing_checkpoint = os.path.join(args.output_dir, removing_checkpoint)
                                    shutil.rmtree(removing_checkpoint)

                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")

                logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
                if grad_norm is not None:
                    logs.update(grad_norm=accelerator.gather(grad_norm).mean().item())   
                progress_bar.set_postfix(**logs)
                accelerator.log(logs, step=global_step)

            if accelerator.sync_gradients and accelerator.is_main_process:
                # (global_step < 5000 and global_step % 1000 == 0)
                if global_step == 1 or (global_step % args.validation_steps == 0):

                    if args.use_ema:
                        # Store the UNet parameters temporarily and load the EMA parameters to perform inference.
                        ema_transformer.store(transformer.parameters())
                        ema_transformer.copy_to(transformer.parameters())

                    # create pipeline
                    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder='scheduler')
                    pipeline = StableDiffusion3ControlNetPipeline(
                        transformer=accelerator.unwrap_model(transformer),
                        scheduler=scheduler,
                        vae=vae if denoise_vae is None else denoise_vae,
                        text_encoder=text_encoder_one,
                        tokenizer=tokenizer_one,
                        text_encoder_2=text_encoder_two,
                        tokenizer_2=tokenizer_two,
                        text_encoder_3=text_encoder_three,
                        tokenizer_3=tokenizer_three,
                    )
                
                    pipeline_args = {"prompt": args.validation_prompt}
                    log_validation(
                        pipeline=pipeline,
                        data_loader=train_dataloader_unwrap,
                        args=args,
                        accelerator=accelerator,
                        pipeline_args=pipeline_args,
                        step=global_step,
                        sub_dir='train',
                        image_encoders=image_encoders,
                        text_encoders=text_encoders,
                        tokenizers=tokenizers,
                    )
                    if val_dataloader is not None:
                        log_validation(
                            pipeline=pipeline,
                            data_loader=val_dataloader,
                            args=args,
                            accelerator=accelerator,
                            pipeline_args=pipeline_args,
                            step=global_step,
                            sub_dir='val',
                            image_encoders=image_encoders,
                            text_encoders=text_encoders,
                            tokenizers=tokenizers,
                        )

                    del pipeline
                    torch.cuda.empty_cache()

                    if args.use_ema:
                        # Switch back to the original UNet parameters.
                        ema_transformer.restore(transformer.parameters())

                    # accelerator.log('finish validate')                      

            if global_step >= args.max_train_steps:
                break

    accelerator.wait_for_everyone()
    accelerator.end_training()


if __name__ == "__main__":
    args = parse_args()
    main(args)
