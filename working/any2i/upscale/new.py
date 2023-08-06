import requests
from PIL import Image
from io import BytesIO
from diffusers import LDMSuperResolutionPipeline
import torch
import numpy as np
from tqdm.auto import tqdm
import os
from glob import glob

device = "cuda" if torch.cuda.is_available() else "cpu"
model_id = "CompVis/ldm-super-resolution-4x-openimages"

# load model and scheduler
pipeline = LDMSuperResolutionPipeline.from_pretrained(model_id, cache_dir=".cache")
pipeline = pipeline.to(device)

name = "first"
os.makedirs(f"super_frame/{name}", exist_ok = True)
os.makedirs(f"super_frame/{name}_compare", exist_ok = True)
for path in tqdm(sorted(glob(f"origin_frame/{name}/*"))[700:]):
    f_name = path.split("/")[-1]
    # try:
    low_res_img = Image.open(path).convert("RGB")
    low_res_np = np.array(low_res_img)
    fill_size = 200
    low_res_np = low_res_np[int(720/2-fill_size) : int(720/2+fill_size), int(1280/2-fill_size): int(1280/2+fill_size)]
    low_res_img = Image.fromarray(low_res_np)
    low_res_img = low_res_img.resize((fill_size*2, fill_size*2))
    upscaled_image = pipeline(low_res_img, num_inference_steps=100, eta=1).images[0]
    upscaled_image.save(f"super_frame/{name}/{f_name}")
# # run pipeline in inference (sample random noise and denoise)
# upscaled_image = pipeline(low_res_img, num_inference_steps=100, eta=1).images[0]
# # save image
# upscaled_image.save("ldm_generated_image.png")