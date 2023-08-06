import cv2
import os
from tqdm.auto import tqdm

name = "first"
print(name)
def save_all_frames(video_path, dir_path, basename, ext='jpg'):
    if not os.path.exists(video_path):
        print(f"{video_path} is not exist")
        return
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        return

    os.makedirs(dir_path, exist_ok=True)
    base_path = os.path.join(dir_path, basename)

    digit = len(str(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))))

    n = 0

    while True:
        ret, frame = cap.read()
        print(f"\r{n} {frame.shape}", end="")
        if ret:
            n += 1
            cv2.imwrite('{}{}.{}'.format(base_path, str(n).zfill(digit), ext), frame)
        else:
            return

save_all_frames(f'origin/{name}.mp4', f'origin_frame/{name}/', '', 'png')