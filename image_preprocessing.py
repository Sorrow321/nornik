import os
from pathlib import Path
from matplotlib import cm
import cv2
from kraken import binarization 
from PIL import Image
from collections import Counter
from sklearn.cluster import KMeans,DBSCAN
import numpy as np
import math
import time
#from matplotlib.colors import RGB2HEX

import skimage
from skimage import io, segmentation, color

from scipy.spatial import Voronoi


#цвета с учетом отблесков и в 4-х регионах (условия освещения и быстрое реагирование)
#количество и размеры по отблескам + ДЕТЕКЦИЯ наиболее крупных пузырей
#ДВИЖЕНИЕ?
#ЛОПАЮТСЯ?
#Предиктивная аналитика по параметру
#Визуализация

#бизнес-смысл, фишки и академическое образование - преимущества

#что-то детектим YOLO

#раскадровка

#визуализация статистики


#PIPELINE:
#1. Video to frames - V
#2. Masking - P
#3. Binarizetions, contours, centers of bubbles, bubbles shapes (with masking) - V
#4. Color clustering on frame (excluding white/bright areas) - V
#5. Movement detection - P
#6. Bubbles stability - P
#7. LSTM - ?
#8. NN binarization (or Fast) - P
#9. Web front - ?
#10. Web back - ?
#11. Bubbles corruption on LSTM - ?

#берем 1 кадр в 5 кадров, потом усредняем снятые значения

#Opt: 
#7. Anomaly detection (Yolo)


'''
По опыту работы с северсталью (на физическом проекте) главное в пене - стабильность
Комменты: попробовали цвета делать красиво - не вышло

Сумма с кракеном - офигенно, но надо его заменить - сеткой?
'''

#1. Video to frames
def get_frames(video_path, do_save = True):
    frames = []
    cap= cv2.VideoCapture(video_path)
    p = Path(video_path)
    if do_save:
        filename = p.stem
        v_i = 0
        while os.path.isdir('frames'+str(v_i)):
            v_i+=1
        aim_folder_name = 'frames'+str(v_i)
        os.mkdir(aim_folder_name)
    i=0
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret == False:
            break
        if do_save:
            cv2.imwrite(os.path.join(aim_folder_name, str(i)+'_'+filename+'.png'),frame)
            i+=1
        frames.append(frame)
    cap.release()
    cv2.destroyAllWindows()
    return frames

#2. Binarizetions, contours, centers of bubbles, bubbles shapes (with masking)
def _put_mask(np_img, np_mask):
    try:
        if np_mask[0][0] != 0 and np_mask[0][0] != 255:
            raise ValueError('Mask should be 1 dim per pixel, containing 0 or 255')
    except:
        raise ValueError('Mask should be 1 dim per pixel')
    if np_img.shape[0] == np_mask.shape[0] and np_img.shape[1] == np_mask.shape[1]:
        try:
            if len(np_img[0][0]) == 3:
                stacked_mask = np.stack((np_mask,)*3, axis=-1)
                np_img = np_img*stacked_mask
        except:
            np_img = np_img*np_mask
        return np_img
    else:
        raise ValueError(f'Shapes of image and mask should be same (at 2 dims): image - {np_img.shape}, mask - {np_mask.shape}')

def _binarize(np_img):
    #print(np_img)
    #mark = time.time()
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

    mask = ((th2 == 0) | (th3 == 0) | (th4==0))
    ni[mask] = 0
    ni[~mask] = 255 
    #print(time.time() - mark)
    return {'M': th1, 'G': th2, 'O': th3, 'K': th4, 'S': ni} 

def _binarizeOLD(np_img):
    #print(np_img)
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
    th4 = binarization.nlbin(Image.fromarray(np_img_cv), threshold=0.9999)
    th4 = np.array(th4)
    #sum
    ni = np.ones(np_img.shape)
    for i2,i3,i4,i in zip(th2,th3,th4,range(len(ni))):
        for p2,p3,p4,p in zip(i2,i3,i4,range(len(i3))):
            if p2 == 0 or p3==0 or p4==0:
                ni[i][p] = 0
            else:
                ni[i][p] = 255
    return {'M': th1, 'G': th2, 'O': th3, 'K': th4, 'S': ni} 

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
            centers.append([cX,cY])
        except:
            pass
    return centers
    
#copy pasted from: https://ipython-books.github.io/145-computing-the-voronoi-diagram-of-a-set-of-points/
def _voronoi_finite_polygons_2d(vor, radius=None):
    """Reconstruct infinite Voronoi regions in a
    2D diagram to finite regions.
    """
    if vor.points.shape[1] != 2:
        raise ValueError("Requires 2D input")
    new_regions = []
    new_vertices = vor.vertices.tolist()
    center = vor.points.mean(axis=0)
    if radius is None:
        radius = vor.points.ptp().max()
    # Construct a map containing all ridges for a
    # given point
    all_ridges = {}
    for (p1, p2), (v1, v2) in zip(vor.ridge_points,
                                  vor.ridge_vertices):
        all_ridges.setdefault(
            p1, []).append((p2, v1, v2))
        all_ridges.setdefault(
            p2, []).append((p1, v1, v2))
    # Reconstruct infinite regions
    for p1, region in enumerate(vor.point_region):
        vertices = vor.regions[region]
        if all(v >= 0 for v in vertices):
            # finite region
            new_regions.append(vertices)
            continue
        # reconstruct a non-finite region
        ridges = all_ridges[p1]
        new_region = [v for v in vertices if v >= 0]
        for p2, v1, v2 in ridges:
            if v2 < 0:
                v1, v2 = v2, v1
            if v1 >= 0:
                # finite ridge: already in the region
                continue
            # Compute the missing endpoint of an
            # infinite ridge
            t = vor.points[p2] - \
                vor.points[p1]  # tangent
            t /= np.linalg.norm(t)
            n = np.array([-t[1], t[0]])  # normal
            midpoint = vor.points[[p1, p2]]. \
                mean(axis=0)
            direction = np.sign(
                np.dot(midpoint - center, n)) * n
            far_point = vor.vertices[v2] + \
                direction * radius
            new_region.append(len(new_vertices))
            new_vertices.append(far_point.tolist())
        # Sort region counterclockwise.
        vs = np.asarray([new_vertices[v]
                         for v in new_region])
        c = vs.mean(axis=0)
        angles = np.arctan2(
            vs[:, 1] - c[1], vs[:, 0] - c[0])
        new_region = np.array(new_region)[
            np.argsort(angles)]
        new_regions.append(new_region.tolist())
    return new_regions, np.asarray(new_vertices)

#https://en.wikipedia.org/wiki/Shoelace_formula
def _poly_area(x,y):
    return 0.5*np.abs(np.dot(x,np.roll(y,1))-np.dot(y,np.roll(x,1)))

#do in RGB
def _get_color_schema(img, mask1, mask2=None, do_in_RGB=False, create_test_image = False):
    img = _put_mask(img, cv2.bitwise_not(mask1))
    if mask2 is not None:
        img = _put_mask(img, mask2)
    if create_test_image:
        cv2.imwrite('masks_test.png', img)
    if do_in_RGB:
        #pixels_colors = img[img != np.array([0,0,0])]
        pixels_colors = img[ (img[:, :, 0] != 0) | (img[:, :, 1] != 0) | (img[:, :, 2] != 0) ]
        return {'colors_var': np.var(pixels_colors, axis=0), 'colors_mean': np.mean(pixels_colors, axis=0)}
    else:
        pixels_colors = img.flatten()
        pixels_colors = pixels_colors[pixels_colors != 0]    
        return {'colors_var': np.var(pixels_colors), 'colors_mean': np.mean(pixels_colors)}

def _do_Voronoi(centers, create_test_image = False, test_img_shape = None):
    vor = Voronoi(centers)
    regions, vertices = _voronoi_finite_polygons_2d(vor)
    #print(time.time() - x)
    cells = [vertices[region] for region in regions]
    #check if positive and allowed (based on mask)
    pos_cells = []
    for cell in cells:
        if all([False if (c[0]<0 or c[1]<0) else True for c in cell]):
            pos_cells.append(cell)
    areas = []
    for point in pos_cells:
        pts = point.astype(int)
        x = [x[0] for x in pts]
        y = [x[1] for x in pts]
        areas.append(_poly_area(x,y))
    if create_test_image:
        ni = np.zeros(test_img_shape)
        for point in pos_cells:
            pts = point.astype(int)
            cv2.polylines(ni,[pts],True,(255))
            x = [x[0] for x in pts]
            y = [x[1] for x in pts]
        cv2.imwrite('test1.png', ni)
    return areas

def preprocess(img_path=None, mask = None, check_time = False, resize = None, input_numpy_img = None):
    #reshape all? - test it - 800x600
    im = None
    if input_numpy_img is not None:
        im = input_numpy_img
        img = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    elif img_path is not None:
        if Path(img_path).name.rsplit('.',maxsplit=1)[1] in ['png', 'jpg', 'jpeg']:
            im = cv2.imread(img_path)
            img = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        else:
            return None
    #if Path(img_path).name.rsplit('.',maxsplit=1)[1] in ['png', 'jpg', 'jpeg']:
    if im is not None:
        if check_time:
            time_mark = time.time()
        #img = cv2.imread(img_path, 0)
        #im = cv2.imread(img_path)
        if resize is not None:
            img = cv2.resize(img, resize, interpolation = cv2.INTER_AREA)
        if mask is not None:
            img = _put_mask(img, mask)

        binarizations = _binarize(img)

        centers = _get_centers(binarizations['S'])

        #area getting algorithm (don't forget about the mask) - test with no corrections

        areas = _do_Voronoi(centers)

        #colors detecting
        #'G' or 'M' for colors detection

        colors = _get_color_schema(im,binarizations['G'], mask, do_in_RGB = True)
        if check_time:
            print(time.time() - time_mark)
        return {'areas': {'areas_mean': np.mean(np.array(areas)), 'areas_var':np.var(np.array(areas))}, 'colors':colors, 'binarizations':binarizations, 'centers':centers}#may be class?
    else:
        return None


if __name__ == '__main__':
    time_mark = time.time()
    points = preprocess(img_path='/Users/romansmirnov/eclipse-workspace/Hackatons/Nornikel/images/Снимок экрана 2021-04-14 в 18.35.30.png')
    print(points['areas'])
    print(points['colors'])
    print(time.time() - time_mark)