import sys
import cv2
from glob import glob
from tqdm.auto import tqdm
import os

name = "koi"
pathes = sorted(glob(f"upscale/Real-ESRGAN/results/*"))
# encoder(for mp4)
fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
# output file name, encoder, fps, size(fit to image size)
print(cv2.imread(pathes[0]).shape)
video = cv2.VideoWriter(f'/mnt/c/Users/ctddd/Downloads/{name}.mp4',fourcc, 30, (5120, 2880))

if not video.isOpened():
    print("can't be opened")
    sys.exit()

for i in tqdm(pathes[3000:4000]):
    # hoge0000.png, hoge0001.png,..., hoge0090.png
    img = cv2.resize(cv2.imread(i), (5120, 2880))

    # can't read image, escape
    if img is None:
        print("can't read")
        break

    # add
    video.write(img)

video.release()

print('written')