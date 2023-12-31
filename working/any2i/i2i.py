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
from PIL import Image
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
import torch
from controlnet_aux import HEDdetector, LineartAnimeDetector, OpenposeDetector
from diffusers.utils import load_image

def preprocess(image):
    w, h = image.size
    w, h = map(lambda x: x - x % 32, (w, h))  # resize to integer multiple of 32
    image = image.resize((w, h), resample=PIL.Image.LANCZOS)
    image = np.array(image).astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    return 2. * image - 1.

def keepAspectResizeSimple(path, size):
    image = Image.open(path)
    width, height = size, size
    ratio = min(width / image.width, height / image.height)
    resize_size = (round(ratio * image.width), round(ratio * image.height))
    resized_image = image.resize(resize_size)

    return resized_image


# pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
#     "model_weight/hakoMayD", 
#     cache_dir="model_weight", 
#     torch_dtype=torch.float16, 
#     custom_pipeline="lpw_stable_diffusion"
# )
# pipe.to("cuda")

pipe = StableDiffusionControlNetPipeline.from_pretrained(
    "model_weight/hakoMayD",
    cache_dir="model_weight", 
    torch_dtype=torch.float16, 
    custom_pipeline="lpw_stable_diffusion"
)
pipe.to("cuda")

base_prompt= ""
base_n_prompt = """
"""
preprocess_1 = LineartAnimeDetector.from_pretrained("lllyasviel/Annotators", cache_dir="model_weight")
preprocess_2 = OpenposeDetector.from_pretrained("lllyasviel/Annotators", cache_dir="model_weight")

# prompt = "((((((frontal face)))))), ((red eyes)), ((black hair)), ((mono tone clothes)), full_body, The girl is in the center of the frame, a cute girl,woman,female, young,20 years old, medium hair, standing"
n_prompt = ""
prompt = ""

def conv_img(img_path, size=512, strength=0.2):
    # init_img = keepAspectResizeSimple(img_path, size).convert('RGB')
    # init_img = init_img.resize((size, size))
    # init_img = preprocess(init_img)[0]
    # print(init_img.shape)
    img = Image.open(img_path).convert("RGB").resize((size, size))
    # init_img_1 = preprocess_1(img)
    # init_img_2 = preprocess_2(img)
    # init_img = Image.composite(init_img_1, init_img_2, Image.new("L", init_img_1.size, 128))
    image = pipe(
        prompt=base_prompt+prompt, 
        negative_prompt=base_n_prompt + n_prompt,
        image=img, strength=strength,
        max_embeddings_multiples=2
    ).images[0]

    return image, img

# name = "koi"
# input_dir  = f"movie/frame_cg/{name}"
# output_dir = f"movie/frame_anime/{name}"
# os.makedirs(output_dir, exist_ok=True)

# images_path = sorted(glob(os.path.join(input_dir, "*")))
path = input("image_path: ")
for i in range(100):
    image, init_img = conv_img(img_path=path, strength=i/100)

    image.save(f'save/test_{i}.png')
    # init_img.save('save/test_progress.png')

# for frame_path in tqdm(images_path):
#     if not ".png" in frame_path:
#         continue
#     frame_num = frame_path.split('/')[-1]
#     image, _ = conv_img(img_path=frame_path)

#     image.save(os.path.join(output_dir, frame_num))
    