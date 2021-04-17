import cv2
import numpy as np
from pathlib import Path

p = Path('frames_binarized/F1_1_1_1')
l = p.iterdir()
img=[]
for i in range(1787):
    pa = f'frames_binarized/F1_1_1_1/{i}.jpg'
    print(pa)
    img.append(cv2.imread(pa))

height,width,layers=img[1].shape

fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
video=cv2.VideoWriter('video.avi', fourcc, 30,(width,height))

for j in range(len(img)):
    video.write(img[j])

cv2.destroyAllWindows()
video.release()