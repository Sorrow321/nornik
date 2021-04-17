import cv2
import sys
import numpy as np
import time
from PIL import Image
from kraken import binarization
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
    

def _binarize(np_img):
    #print(np_img)
    mark = time.time()
    np_img_cv = np.array(np_img)#cv2.cvtColor(np.array(np_img), cv2.COLOR_RGB2BGR)
    #mean
    th1 = cv2.adaptiveThreshold(np_img_cv,255,cv2.ADAPTIVE_THRESH_MEAN_C,\
                cv2.THRESH_BINARY,151,1) #11,2
    #gauss
    
    th2 = cv2.adaptiveThreshold(np_img_cv,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
                cv2.THRESH_BINARY,151,1) #11,2
    #otsu
    _,th3 = cv2.threshold(np_img_cv,200,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    #kraken
    #print(time.time() - mark)
    th4 = binarization.nlbin(Image.fromarray(np_img_cv), threshold=0.9999)
    th4 = np.array(th4)
    #sum
    ni = np.ones(np_img.shape)

    mask = ((th2 == 0) | (th3 == 0) | (th4 == 0))
    ni[mask] = 0
    ni[~mask] = 255 
    #print(time.time() - mark)
    return {'M': th1, 'G': th2, 'O': th3, 'S': ni} 

def _get_centers(mask):
    #ci = np.zeros(ni.shape)
    mask = mask.astype(np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    centers = []
    for c in contours:
        M = cv2.moments(c)
        try:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            #ci[cY,cX] = 255
            #centers.append([cY,cX])
            centers.append([cX,cY])
        except:
            pass
    return centers

def process_img(filename):
    img = cv2.imread(str(filename))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    r = _binarize(img)
    img = r['S']
    print(filename.stem)
    cv2.imwrite(f'frames_binarized/F1_1_2_1/{filename.stem}.jpg', img)

if __name__ == '__main__':
    input = Path('frames/F1_1_2_1')
    with ThreadPoolExecutor(max_workers=1) as executor:
        executor.map(process_img, list(input.iterdir()))
    '''
    for i, filename in enumerate(input.iterdir()):  
        print(i)
        img = cv2.imread(str(filename))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        r = _binarize(img)
        cv2.imwrite(f'out/{i}.jpg', r['S'])
    '''