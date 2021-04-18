#stream_processing.py

import cv2
import numpy as np
from image_preprocessing import _binarize, _get_centers
import math
import time

#vector distance p0 and p1 - vectors
def _get_distances(p0,p1):
    vec = [[o1[0]-o0[0], o1[1]-o0[1]] for o0,o1 in zip(p0,p1)]
    return [math.sqrt((v[0])**2+(v[1])**2) for v in vec], vec

#create numpy from zeros with only white (255) centers of bubbles 
def _img_from_centers(shape, centers):
    z = np.zeros(shape)
    for x in centers:
        z[x[1],x[0]] = 255 
    return z

#main loop
def calc_stream_params(np_img1_bin_mask, np_img2_bin_mask, first_centers, second_centers = None, auto_best = True, use_bin_mask = True): 
    '''
    Auto_best should be False for better accuracy

    np_img1_bin_mask - first image binarized
    np_img2_bin_mask - second image binarized
    first_centers - centers of first image
    second_centers - centers of second image (needs to be defined if use_bin_mask = False)
    auto_best - to use auto mode for finding points for tracking or not (recommendation: use False)
    use_bin_mask - to use binarization masks or just centers (recommendation: use True)
    '''
    #use_bin_mask must be true!
    lk_params = dict( winSize  = (15,15),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    feature_params = dict( maxCorners = 100,
                        qualityLevel = 0.3,
                        minDistance = 7,
                        blockSize = 7 )

    if auto_best:
        p0 = cv2.goodFeaturesToTrack(np_img1_bin_mask.astype(np.uint8), mask = None, **feature_params)
    else:
        p0 = np.expand_dims(first_centers, axis=1).astype(np.float32)
    if use_bin_mask:
        first_frame = np_img1_bin_mask.astype(np.uint8)
        second_frame = np_img2_bin_mask.astype(np.uint8)
    else:
        first_frame = _img_from_centers(np_img1_bin_mask.shape, first_centers).astype(np.uint8)
        second_frame = _img_from_centers(np_img2_bin_mask.shape, second_centers).astype(np.uint8)

    p1, st, err = cv2.calcOpticalFlowPyrLK(first_frame, second_frame, p0, None, **lk_params)
    good_new = p1[st == 1]
    good_old = p0[st == 1]

    distance, vectors = _get_distances(good_old, good_new)

    sum_vec = np.sum(vectors, axis = 0)
    sum_vec_lenth = math.sqrt(sum_vec[0]**2+sum_vec[1]**2)
    sum_vec_angle = np.arccos(sum_vec[0] / sum_vec_lenth)

    return {'mean_distance': np.mean(np.array(distance)), 
            'var_distance': np.var(np.array(distance)), 
            'sum_vec' : {'destination' : sum_vec, 'length' : np.array(math.sqrt(sum_vec[0]**2+sum_vec[1]**2)), 'angle': np.array(sum_vec_angle)}}

#суммарный вектор - вектор направления
    
if __name__ == '__main__':
    img1_path = '/Nornikel/frames0/11_F1_1_1_1.png'
    img2_path = '/Nornikel/frames0/13_F1_1_1_1.png'
    img3_path = '/Nornikel/frames0/15_F1_1_1_1.png'

    im1 = cv2.imread(img1_path, 0)
    im2 = cv2.imread(img2_path, 0)
    im3 = cv2.imread(img3_path, 0)

    b1 = _binarize(im1)['S']
    b2 = _binarize(im2)['S']
    b3 = _binarize(im3)['S']

    c1 = _get_centers(b1)
    c2 = _get_centers(b2)
    c3 = _get_centers(b3)
    time_mark = time.time()
    print(calc_stream_params(b1, b3, c1, auto_best = False, use_bin_mask = True))
    print(time.time() - time_mark)

