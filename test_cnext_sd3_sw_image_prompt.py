
import os.path

import torch
from utils.torch_tool import init_torch_npu, is_torch_npu_avaliable
if is_torch_npu_avaliable():
    init_torch_npu()
# from pipelines.pipeline_stable_diffusion_3_cnext_sw_shift import StableDiffusion3ControlNetPipeline
from pipelines.pipeline_stable_diffusion_3_cnext import StableDiffusion3ControlNetPipeline
from models.transformer_sd3_cnext import SD3Transformer2DModel
from diffusers.utils import load_image
import json
from tqdm import tqdm
import numpy as np
import cv2
import random
from PIL import Image
import functools
# from imgproc import var_center_crop
from torchvision import transforms
from torchvision.utils import save_image
from diffusers import StableDiffusion3Pipeline, AutoencoderKL
from diffusers import FlowMatchEulerDiscreteScheduler
from transformers import CLIPTextModelWithProjection, CLIPVisionModelWithProjection, CLIPTokenizer, T5EncoderModel, T5TokenizerFast
import argparse
from utils.wavelet_color_fix import wavelet_color_fix, adain_color_fix
from accelerate.utils import set_seed
import time
from train_cnext_sd3_image_prompt import (
    _encode_prompt_with_t5, 
    _encode_prompt_with_clip_vit,
    encode_multi_modal_prompt,
    clip_L_32_path,
    clip_G_14_path
)

"""
CUDA_VISIBLE_DEVICES=1 python test_controlnext_sd3.py \
--model_path /data/lwb/projects/picture-quality/sd3control/checkpoint/stable-diffusion-3-medium-diffusers \
--transformer_model_path /data/lwb/projects/picture-quality/sd3control/logs/controlnet_sd3_0708_base_v0/checkpoints/0015000/consolidated.00-of-01.pth \
--test_json_path /data/lwb/projects/picture-quality/sd3control/dataset/plant/test_full.json \
--output_path /data/lwb/projects/picture-quality/sd3control/tmp/test1
"""
def tensor_to_image(x):
    x = torch.clamp(x, -1., 1.)
    x = (x + 1.) / 2. * 255
    x = x.cpu().numpy().astype(np.uint8)
    if x.shape[1] == 3 or x.shape[1] == 1:
        x = np.transpose(x, (0, 2, 3, 1))
    return x
parser = argparse.ArgumentParser()
parser.add_argument("--model_path", type=str, required=True)
parser.add_argument("--transformer_model_path", type=str, required=True)
parser.add_argument("--test_json_path", type=str, required=True)
parser.add_argument("--output_path", type=str, required=True)
parser.add_argument("--data_root", type=str, required=True)
parser.add_argument("--cfg", type=int, default=1)
parser.add_argument("--sampling_steps", type=int, default=28)
parser.add_argument("--suffix", type=str, default=".jpg")
parser.add_argument("--seed", type=int, default=1234)
parser.add_argument("--downsample", default=False, action="store_true",)
parser.add_argument("--infr_4k", default=False, action="store_true",)
parser.add_argument("--infr_2k", default=False, action="store_true",)
parser.add_argument("--denoise_vae_ckpt", type=str, default="")

args = parser.parse_args()
# config
with open(args.test_json_path) as f:
    data_info = json.load(f)
random.shuffle(data_info)

print(f"[INFO] set set_seed({args.seed})")
if args.seed != -1:
    set_seed(args.seed)

# Load model
while not os.path.exists(args.transformer_model_path):
    print("Sleep starting : %s" % time.ctime())
    time.sleep(600)
print("Sleep ending : %s" % time.ctime())

weight_type = torch.bfloat16

print('Loading transformers')
if os.path.isdir(args.transformer_model_path):
    print(f"[INFO] load from {args.transformer_model_path}")
    transformer = SD3Transformer2DModel.from_pretrained(args.transformer_model_path, low_cpu_mem_usage=False, device_map=None, torch_dtype=weight_type)
    transformer.pos_embed.pos_embed.copy_(torch.load('pos_embed.t5'))
else:
    if os.path.splitext(args.transformer_model_path)[1] == '.pth':
        transformer = SD3Transformer2DModel.from_pretrained(args.model_path, subfolder='transformer', low_cpu_mem_usage=False, device_map=None, torch_dtype=weight_type)
        state_dict = torch.load(args.transformer_model_path)
        for name in list(state_dict.keys()):
            if name.startswith("module."):
                state_dict[name.removeprefix("module.")] = state_dict[name].cpu()
                state_dict.pop(name)
        transformer.load_state_dict(state_dict)
    else:
        raise "Unsupport type"

print(f'{sum(p.numel() for p in transformer.parameters()) / 1e9} B')
print(f'{sum(p.numel() for p in transformer.control_conv.parameters()) / 1e6} M')

# pipe = StableDiffusion3Pipeline.from_pretrained(
#     args.model_path,
#     torch_dtype=weight_type
# ).to("cuda")
tokenizer_one = CLIPTokenizer.from_pretrained(args.model_path, subfolder="tokenizer",)
tokenizer_two = CLIPTokenizer.from_pretrained(args.model_path, subfolder="tokenizer_2",)
tokenizer_three = T5TokenizerFast.from_pretrained(args.model_path, subfolder="tokenizer_3",)
tokenizers = [tokenizer_one, tokenizer_two, tokenizer_three] 

# text_encoder_one = None
text_encoder_one = CLIPTextModelWithProjection.from_pretrained(
    args.model_path, subfolder="text_encoder", torch_dtype=weight_type
).to("cuda")

# text_encoder_two = None
text_encoder_two = CLIPTextModelWithProjection.from_pretrained(
    args.model_path, subfolder="text_encoder_2", torch_dtype=weight_type
).to("cuda")

text_encoder_three = T5EncoderModel.from_pretrained(
    args.model_path, subfolder="text_encoder_3", torch_dtype=weight_type
).to("cuda")

text_encoders=[text_encoder_one, text_encoder_two, text_encoder_three]

image_encoder_one = CLIPVisionModelWithProjection.from_pretrained(clip_L_32_path, torch_dtype=weight_type).to("npu")
image_encoder_two = CLIPVisionModelWithProjection.from_pretrained(clip_G_14_path, torch_dtype=weight_type).to("npu")

image_encoders = [image_encoder_one, image_encoder_two]


scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(args.model_path, subfolder="scheduler")

denoise_vae = None
if args.denoise_vae_ckpt is not None and os.path.exists(args.denoise_vae_ckpt):
    print(f"[INFO] load denoise_vae from {args.denoise_vae_ckpt}")
    denoise_vae = AutoencoderKL.from_pretrained(args.denoise_vae_ckpt)
else:
    denoise_vae = AutoencoderKL.from_pretrained(args.model_path, subfolder="vae")

print("time.sleep(5)")
time.sleep(5)
pipe = StableDiffusion3ControlNetPipeline(
    transformer=transformer,
    scheduler=scheduler,
    vae=denoise_vae,
    text_encoder=text_encoder_one,
    tokenizer=tokenizer_one,
    text_encoder_2=text_encoder_two,
    tokenizer_2=tokenizer_two,
    text_encoder_3=text_encoder_three,
    tokenizer_3=tokenizer_three,
).to("cuda")
torch.cuda.empty_cache()
pipe.vae.enable_tiling()
os.makedirs(args.output_path, exist_ok=True)

def keep_ratio_resize(img, long_size=2048):
    w, h = img.size
    aspect_ratio = w / h
    msize = max(h, w)

    if msize == w:
        target_size = (long_size, int(long_size // aspect_ratio))
    else:
        target_size = (int(long_size * aspect_ratio), long_size)

    w_, h_ = target_size
    w_ = int(np.round(w_ / 8.0)) * 8
    h_ = int(np.round(h_ / 8.0)) * 8
    resize_img = img.resize((w_, h_), resample=Image.Resampling.LANCZOS)
    return resize_img

down = 2
print(f"[INFo] num: {len(data_info)}")
for idx, data in tqdm(enumerate(data_info)):
    # test_path = os.path.join(args.output_path, f"item-{idx + 1:04d}"+'.png')
    image_lq_path = data.get("path", data.get("lr_path"))
    prompt = data["prompt"]
    test_path = os.path.join(args.output_path, os.path.basename(image_lq_path)[:-4] + args.suffix)
    if os.path.exists(test_path):
        continue

    image_lq = load_image(os.path.join(args.data_root, image_lq_path))
    print(f"[INFO] image_lq.shape:{image_lq.size}")
    st = time.time()
    w_ori, h_ori = image_lq.size
    image_lq = keep_ratio_resize(image_lq, long_size=1024)
    # if args.infr_4k:
    #     image_lq = keep_ratio_resize(image_lq, long_size=4000)
    # elif args.infr_2k:
    #     image_lq = keep_ratio_resize(image_lq, long_size=2048)
    # elif args.downsample and max(w_ori, h_ori) > 2400:
    #     # image_lq = image_lq.resize((w, h), Image.Resampling.LANCZOS)
    #     image_lq = keep_ratio_resize(image_lq, long_size=2400)
    # else:
    #     w = int(np.round(w_ori / 8.0)) * 8
    #     h = int(np.round(h_ori / 8.0)) * 8
    #     image_lq = image_lq.resize((w, h), Image.Resampling.LANCZOS)

    print(f"[INFO] resize image_lq.shape:{image_lq.size}")
    image_transform = transforms.Compose(
        [
            # transforms.Resize(crop_size_list[0]),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
        ]
    )
    image_lq_t = image_transform(image_lq)
    n_height, n_width = image_lq_t.shape[1:]

    global_images = [image_lq]
    patch_images = [image_lq]
    print(f"[INFO] resize image_lq.shape:{image_lq.size}")
    with torch.cuda.amp.autocast(dtype=weight_type):
        prompt_embeds, pooled_prompt_embeds = encode_multi_modal_prompt(
            image_encoders, global_images, patch_images, text_encoders, tokenizers, prompt, 0.
        )
        # negative_prompt_embeds=torch.zeros_like(prompt_embeds)
        # negative_pooled_prompt_embeds=torch.zeros_like(pooled_prompt_embeds)
        image = pipe(
            # prompt,
            control_image=image_lq_t.unsqueeze(0),
            controlnet_conditioning_scale=1.0,
            num_inference_steps=args.sampling_steps,
            guidance_scale=args.cfg,
            height=n_height,
            width=n_width,
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            # negative_prompt_embeds=negative_prompt_embeds,
            # negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
        ).images[0]
    print(f"[INFO] resize image_lq.shape:{image.size}")
    # to_pil = transforms.ToPILImage()
    # lr_image = to_pil((image_lq_t + 1) / 2)
    
    # crop_size = crop_size_list[0]
    # image = image.resize(crop_size_list[0])
    # lr_image = lr_image.resize(crop_size_list[0])

    # color_fix
    image = wavelet_color_fix(image, image_lq)
    # if args.infr_4k or args.infr_2k:
    #     pass
    # elif args.downsample or w_ori != image.size[0] or h_ori != image.size[1]:
    #     image = image.resize((w_ori, h_ori), Image.Resampling.LANCZOS)
    # print(f"[INFO] processing time:{time.time() - st}")
    # concatenated_image = Image.new("RGB", (crop_size[0]*2, crop_size[1]))
    # concatenated_image.paste(lr_image, (0, 0))
    # concatenated_image.paste(image, (crop_size[0], 0))
    # concatenated_image.save(test_path)
    print(f"[INFO] resize image_lq.shape:{image.size}")
    image.save(test_path)

print('Done.')
