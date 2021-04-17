import cv2
import os
import sys
from pathlib import Path


def get_frames(video_path):
    cap= cv2.VideoCapture(video_path)
    p = Path(video_path)
    video_name = p.stem
    target_folder_name = Path('frames')
    i = 0
    out_path = target_folder_name / video_name
    out_path.mkdir(parents=True, exist_ok=True)
    while cap.isOpened():
        ret, frame = cap.read()
        if ret == False:
            break
        print(i)
        scale_factor = 0.25
        frame = cv2.resize(frame, (int(frame.shape[1] * scale_factor), int(frame.shape[0] * scale_factor)), interpolation = cv2.INTER_AREA)
        cv2.imwrite(str(out_path /  f'{i}.jpg'), frame)
        i += 1
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    get_frames(sys.argv[1])