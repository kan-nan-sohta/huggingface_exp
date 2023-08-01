from PIL import Image
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
import torch
from controlnet_aux import HEDdetector, LineartAnimeDetector
from diffusers.utils import load_image


from diffusers import DiffusionPipeline, StableDiffusionPipeline, StableDiffusionImg2ImgPipeline
import torch
import datetime
from PIL import Image
import PIL
import numpy as np
from utils import *
from glob import glob
from tqdm.auto import tqdm
import os


hed = HEDdetector.from_pretrained("lllyasviel/Annotators")

controlnet = [
    ControlNetModel.from_pretrained
    (
        "lllyasviel/sd-controlnet-scribble", 
        torch_dtype=torch.float16, 
        cache_dir="model_weight"
    ),
    ControlNetModel.from_pretrained
    (
        "lllyasviel/control_v11f1e_sd15_tile", 
        torch_dtype=torch.float16, 
        cache_dir="model_weight"
    )
]


pipe = StableDiffusionControlNetPipeline.from_pretrained(
    # "model_weight/blue_pencil-v9", 
    "model_weight/FEIWUmix-2.5D",
    controlnet=controlnet, 
    safety_checker=None, 
    torch_dtype=torch.float16,
    cache_dir="model_weight"
)

pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
pipe.enable_xformers_memory_efficient_attention()
pipe.enable_model_cpu_offload()

name = "koi"
input_dir  = f"movie/frame_cg/{name}"
output_dir = f"movie/frame_anime/{name}"
os.makedirs(output_dir, exist_ok=True)

images_path = sorted(glob(os.path.join(input_dir, "*")))


init_img = load_image(images_path[0])
image = hed(init_img)
image.save(f'e.png')
print(image.size, init_img.resize(size=image.size).size)
image = [image, init_img.resize(size=image.size)]
print(image[0].size, image[1].size)
pre_img = pipe(
    "((red eyes)), ((black hair)), (short hair:1.5), full_body, a cute girl, standing, (simple background:2.0)",
    image, 
    num_inference_steps=200,
    controlnet_conditioning_scale=[0.5, 1.0],
    negative_prompt="(worst quality, bad quality:2.0)"
).images[0]
pre_img.save(f'test.png')

for frame_path in tqdm(images_path):
    if not ".png" in frame_path:
        continue
    frame_num = frame_path.split('/')[-1]
    init_img = load_image(frame_path)
    image = [hed(init_img, scribble=True), pre_img]
    rtn = pipe(
        "((red eyes)), ((black hair)), (short hair:1.5), full_body, a cute girl, standing, (simple background:2.0)",
        image, 
        num_inference_steps=20,
        controlnet_conditioning_scale=[0.5, 1.0],
        negative_prompt="(worst quality, bad quality:2.0)"
    ).images[0]
    pre_img = rtn
    rtn.save(os.path.join(output_dir, frame_num))