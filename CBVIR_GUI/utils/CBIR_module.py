import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import time
import os
from pathlib import Path


def feat_extr(img_path):
    #start = time.time()
    img = cv.imread(img_path, cv.IMREAD_GRAYSCALE)
    
    # Initiate sift detector
    sift = cv.SIFT_create(
        edgeThreshold = 15,
        contrastThreshold = 0.06,
        nfeatures = 2000
    )
    
    scaled_width = 800
    h_img = int(scaled_width * img.shape[0]/img.shape[1])
    img = cv.resize(img,[scaled_width, h_img], interpolation= cv.INTER_AREA)
    kp, des = sift.detectAndCompute(img, None)
    #print(time.time()-start)
    return des

def sift_match(des1, des2):
    #start = time.time()
    if (des1 is None) or (des2 is None):
        return 0
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = {}   # or pass empty dictionary
    flann = cv.FlannBasedMatcher(index_params,search_params)
    matches = flann.knnMatch(des1,des2,k=2)
    # Need to draw only good matches, so create a mask
    # matchesMask = [[0,0] for i in range(len(matches))]
    # ratio test as per Lowe's paper
    t = 0 # count matching points
    for i,(m,n) in enumerate(matches):
        if m.distance < 0.7*n.distance:
            t += 1
    #print(time.time()-start)
    return t


def ranking(T, thr):
    sort_idx = np.argsort(-T);
    T_sorted = T[sort_idx]
    t_max = T_sorted[0]
    c = 0
    # print(type(T))
    for i in range(len(T)):
        if T_sorted[i] < thr : # or t_max/T_sorted[i] > 3 score too small, not valid
            break
        c += 1

    return T_sorted[0:c], sort_idx[0:c]


def search(query_path, gal_path, result_path):
    start_time = time.time()
    frames_name = []
    result_frames = []
    T = []

    des1 = feat_extr(query_path)
    for kf_path in os.listdir(gal_path):
        if kf_path[0] == '.':
            continue
        des2 = feat_extr(gal_path+'/'+kf_path)
        t = sift_match(des1, des2)
        frames_name.append(kf_path)
        T.append(t)
    T = np.array(T)
    # print(T)
    exe_time = time.time() - start_time

    T_res, Idx = ranking(T, thr=30)
    for i in Idx:
        result_frames.append(frames_name[i])
        # print(frames_name[i])
    
    query_name = Path(query_path).stem
    txt_path = f"result_query_{query_name}.txt"
    final_txt_path = Path(result_path) / txt_path
    np.savetxt(final_txt_path, result_frames, fmt='%s')
    return final_txt_path, exe_time


if __name__ == "__main__":
    query_path = "C:/Users/Nina/Desktop/CBVIR_GUI/dataset/Battuta_1/query_jpg/1.jpg"
    gal_path = "C:/Users/Nina/Desktop/CBVIR_GUI/results/keyframes_2"
    search(query_path, gal_path, gal_path)