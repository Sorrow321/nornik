import cv2
import os
import sys
from pathlib import Path


def get_frames(video_path):
    cap= cv2.VideoCapture(video_path)
    p = Path(video_path)
    filename = p.stem
    aim_folder_name = 'frames'
    i=0
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret == False:
            break
        cv2.imwrite(os.path.join(aim_folder_name, filename+str(i)+'.png'), frame)
        i += 1
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    get_frames(sys.argv[1])