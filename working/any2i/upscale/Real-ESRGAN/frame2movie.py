import cv2
import os
import numpy as np
from tqdm.auto import tqdm
from glob import glob

# 画像が保存されているディレクトリパス
image_folder = glob('results/*')

# 動画ファイルの名前と拡張子
video_name = 'video.mkv'

# 画像を取得
images = [img for img in image_folder if img.endswith(".png")]

# 画像を名前順にソート
images.sort()

# 画像のリストから最初の画像を読み込みます。
sample_image = cv2.imread(images[0])

# 画像の高さと幅を取得します。
height, width, _ = sample_image.shape

# VideoWriterオブジェクトを作成します。
video = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*'MJPG'), 30, (width,height))

# 各画像を動画に追加します。
for image in tqdm(images):
    video.write(cv2.imread(image))

# 動画ファイルを保存します。
video.release()