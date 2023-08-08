import os
from PIL import Image
from transformers import pipeline
import numpy as np
import cv2
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
import torch
from diffusers.utils import load_image
from torchvision.transforms import ToTensor
from tqdm.auto import tqdm
from glob import glob
from diffusers.models import AutoencoderKL
transform = ToTensor()

image = load_image("https://huggingface.co/lllyasviel/sd-controlnet-normal/resolve/main/images/toy.png").convert("RGB")

# depth_estimator = pipeline("depth-estimation", model ="Intel/dpt-hybrid-midas" )

print(image)

# image = depth_estimator(image)['predicted_depth'][0]

# print(image)

image = np.array(image)

# print(image)

# image_depth = image.copy()
# image_depth -= np.min(image_depth)
# image_depth /= np.max(image_depth)

# bg_threhold = 0.4

# x = cv2.Sobel(image, cv2.CV_32F, 1, 0, ksize=3)
# x[image_depth < bg_threhold] = 0

# y = cv2.Sobel(image, cv2.CV_32F, 0, 1, ksize=3)
# y[image_depth < bg_threhold] = 0

# z = np.ones_like(x) * np.pi * 2.0

# image = np.stack([x, y, z], axis=2)
# image /= np.sum(image ** 2.0, axis=2, keepdims=True) ** 0.5
# image = (image * 127.5 + 127.5).clip(0, 255).astype(np.uint8)
fill_size = 128
tate = image.shape[0]
yoko = image.shape[1]
image = image[int(tate/2-fill_size) : int(tate/2+fill_size), int(yoko/2-fill_size): int(yoko/2+fill_size)]
image = Image.fromarray(image)

controlnet = ControlNetModel.from_pretrained(
    "fusing/stable-diffusion-v1-5-controlnet-normal", torch_dtype=torch.float16, cache_dir=".cache"
)
vae = AutoencoderKL.from_pretrained("weight/mse840000_klf8anime")
pipe = StableDiffusionControlNetPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5", controlnet=controlnet, safety_checker=None, torch_dtype=torch.float16,vae=vae, cache_dir=".cache"
)


pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)

# Remove if you do not have xformers installed
# see https://huggingface.co/docs/diffusers/v0.13.0/en/optimization/xformers#installing-xformers
# for installation instructions
# pipe.enable_xformers_memory_efficient_attention()

pipe.enable_model_cpu_offload()
image.save('toy_normal.png')
pipe.vae.half()
print(transform(image).to(dtype=torch.float16).shape)
image = transform(image).to(dtype=torch.float16).reshape(1, 3, fill_size*2, fill_size*2)
image /= torch.max(image)
for _ in range(10):  
    image = pipe.vae.forward(image).sample
image = image[0]
print(image.permute(1, 2, 0).cpu().detach().numpy().shape)
image = image.permute(1, 2, 0).cpu().detach().numpy() * 255
image = image.clip(0, 255)
image = Image.fromarray(image.astype('uint8'))

image.save('toy_normal_out.png')
name = "first"
os.makedirs(f"super_frame/{name}", exist_ok = True)
os.makedirs(f"super_frame/{name}_compare", exist_ok = True)
del image
for path in tqdm(sorted(glob(f"origin_frame/{name}/*"))[700:]):
    f_name = path.split("/")[-1]
    # try:
    image = load_image(path).convert("RGB")
    image = np.array(image)
    fill_size = 200
    tate = image.shape[0]
    yoko = image.shape[1]
    image = image[int(tate/2-fill_size) : int(tate/2+fill_size), int(yoko/2-fill_size): int(yoko/2+fill_size)]
    image = Image.fromarray(image).resize((fill_size*2, fill_size*2))
    image = transform(image).to(dtype=torch.float16).reshape(1, 3, fill_size*2, fill_size*2)
    image /= torch.max(image)
    for _ in range(2):  
        image = pipe.vae.forward(image).sample
    image = image[0]
    image = image.permute(1, 2, 0).cpu().detach().numpy() * 255
    image = image.clip(0, 255)
    image = Image.fromarray(image.astype('uint8'))
    image.save(f"super_frame/{name}/{f_name}")
    del image
    # ImageLoader.save_image(upscaled_image, f"super_frame/{name}/{f_name}")
    # ImageLoader.save_compare(low_res_img, upscaled_image, f"super_frame/{name}_compare/{f_name}")
    # del upscaled_image