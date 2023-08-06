import requests
from PIL import Image
from io import BytesIO
from diffusers import StableDiffusionUpscalePipeline
from super_image import PanModel, ImageLoader
import torch
from tqdm.auto import tqdm
from glob import glob
import os
import numpy as np

# load model and scheduler
model_id = "stabilityai/stable-diffusion-x4-upscaler"
# pipeline = StableDiffusionUpscalePipeline.from_pretrained(
#     model_id, revision="fp16", torch_dtype=torch.float16,
#     cache_dir= ".cache"
# )
pipeline = PanModel.from_pretrained('eugenesiow/pan-bam', scale=4, cache_dir= ".cache") 
pipeline = pipeline.to("cuda")

# let's download an  image
name = "first"
os.makedirs(f"super_frame/{name}", exist_ok = True)
os.makedirs(f"super_frame/{name}_compare", exist_ok = True)
for path in tqdm(sorted(glob(f"origin_frame/{name}/*"))[700:]):
    f_name = path.split("/")[-1]
    # try:
    low_res_img = Image.open(path).convert("RGB")
    low_res_np = np.array(low_res_img)
    fill_size = 256
    low_res_np = low_res_np[int(720/2-fill_size) : int(720/2+fill_size), int(1280/2-fill_size): int(1280/2+fill_size)]
    low_res_img = Image.fromarray(low_res_np)
    low_res_img = low_res_img.resize((fill_size*4, fill_size*4))
    prompt = ""

    # upscaled_image = pipeline(prompt=prompt, image=low_res_img, num_inference_steps=75).images[0]
    low_res_img = ImageLoader.load_image(low_res_img).to("cuda")
    upscaled_image = pipeline(low_res_img)
    # print(f"super_frame/{name}/{f_name}")
    # upscaled_image.save(f"super_frame/{name}/{f_name}")
    ImageLoader.save_image(upscaled_image, f"super_frame/{name}/{f_name}")
    ImageLoader.save_compare(low_res_img, upscaled_image, f"super_frame/{name}_compare/{f_name}")
    del upscaled_image
    # except Exception as e:
    #     print(e)