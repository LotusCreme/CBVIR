import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import time
import os
from PIL import Image
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import svds
from operator import itemgetter
from collections import defaultdict


# functions

def frame_to_hist(frame_rgb):
    # Dividing frame into blocks
    height, width, channels = frame_rgb.shape
    if height % 3 == 0:
        h_block = int(height/3)
    else:
        h_block = int(height/3) + 1

    if width % 3 == 0:
        w_block = int(width/3)
    else:
        w_block = int(width/3) + 1
    h = 0
    w = 0
    feature_vector = []
    for i in range(1, 4):
        h_window = h_block*i
        for j in range(1, 4):
            frame = frame_rgb[h: h_window, w: w_block*j, :]
            hist = cv2.calcHist(frame, [0, 1, 2], None, [6, 6, 6], [0, 256, 0, 256, 0, 256]) 
            hist1 = hist.flatten()  # flatten the hist to one-dimensinal vector 
            feature_vector += list(hist1)  # concatenating the block features
            w = w_block*j
        h = h_block*i
        w = 0
    return feature_vector

def clustering(de_feat_mat, dsample_rate, similarity_threshold, cluster_threshold):
    # dynamic clustering of projected frame histograms to find which all frames are similar i.e make shots
    cluster_set = dict() # to store frames in respective cluster
    for i in range(de_feat_mat.shape[0]):
        cluster_set[i] = np.empty((0,63), int)

    # adding first two projected frames in first cluster i.e Initializaton    
    cluster_set[0] = np.vstack((cluster_set[0], de_feat_mat[0]))   
    cluster_set[0] = np.vstack((cluster_set[0], de_feat_mat[1]))

    centroid_set = dict() # to store centroids of each cluster
    for i in range(de_feat_mat.shape[0]):
        centroid_set[i] = np.empty((0,63), int)
    # finding centroid of centroid_set[0] cluster
    centroid_set[0] = np.mean(cluster_set[0], axis=0) 

    count = 0
    for i in range(2, de_feat_mat.shape[0]):
        similarity2 = np.dot(de_feat_mat[i], centroid_set[count])**2/(np.dot(de_feat_mat[i],de_feat_mat[i])*np.dot(centroid_set[count], centroid_set[count]) ) # cosine similarity
        if similarity2 < similarity_threshold:
            count += 1
            cluster_set[count] = np.vstack((cluster_set[count], de_feat_mat[i]))
            centroid_set[count] = np.mean(cluster_set[count], axis=0)
        else:  # if they are similar then assign this data point to last cluster formed and update the centroid of the cluster
            cluster_set[count] = np.vstack((cluster_set[count], de_feat_mat[i])) 
            centroid_set[count] = np.mean(cluster_set[count], axis=0)

    # print(f'One feature matrix is {type(de_feat_mat)} with shape {de_feat_mat.shape}')

    num = []  # find the number of data points in each cluster formed.
    for i in range(de_feat_mat.shape[0]):
        num.append(cluster_set[i].shape[0])

    last = num.index(0)  # where we find 0 in b indicates that all required clusters have been formed, so we can delete these from C
    size_clu = num[:last]  # The size of each valid cluster.
    res = [idx for idx, val in enumerate(size_clu) if val >= cluster_threshold/dsample_rate] 

    new_label = cluster_set  # Label each cluster, making it easier to identify frames in each cluster
    for i in range(last):
        p = np.repeat(i, size_clu[i]).reshape(size_clu[i], 1)
        new_label[i] = np.hstack((new_label[i], p))

    full_clu = np.empty((0, 64), int) 
    for i in range(last):
        full_clu = np.vstack((full_clu, new_label[i]))

    # print('sub_res is ', type(full_clu), len(full_clu))
    # print('res is ', type(res), len(res), res)
    return full_clu, res

def get_keyframe_idx(feature_mat, dsample_rate, similarity_threshold,cluster_threshold):
    if feature_mat.shape[0] < 100:
        return [feature_mat.shape[0]-1]
    
    feature_mat = csc_matrix(feature_mat.transpose(), dtype=float) # each column of A is a feature vector
    # top 63 singular values from 76082 to 508
    # implementing SVD
    u, s, vt = svds(feature_mat, k = 63)
    vt = vt.transpose()
    de_feat_mat = vt @ np.diag(s) # each row of de_feat_mat is a 63 dimensional vector

    full_clu, res = clustering(de_feat_mat, dsample_rate, similarity_threshold,cluster_threshold)
    colnames = []
    for i in range(1, 65):
        col_name = "v" + str(i)
        colnames += [col_name]

    df = pd.DataFrame(full_clu, columns = colnames)
    df['v64'] = df['v64'].astype(int)  # converting the cluster level from float type to integer type
    df1 = df[df.v64.isin(res)]
    KF = df1.groupby('v64').tail(1)['v64']
    KF = KF.index - 1

    return KF


def remove_blackbar(frame_img):
    img = frame_img
    # img_copy = img.copy()

    # convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # invert gray image
    gray = 255 - gray
    # gaussian blur
    blur = cv2.GaussianBlur(gray, (3,3), 0)
    # threshold
    thresh = cv2.threshold(blur, 236, 255, cv2.THRESH_BINARY)[1]
    # apply close and open morphology to fill tiny black and white holes
    kernel = np.ones((5,5), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    # invert thresh
    thresh = 255 - thresh
    img_cpy = img.copy()
    # get contours (presumably just one around the nonzero pixels)
    # then crop it to bounding rectangle
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # print(np.size(contours))

    sum = []
    extreme = []
    for num_con in range(len(contours)):
        contour = contours[num_con]
        contour = [v[0].tolist() for v in contour]
        contour = np.array(contour)
        max0 = max(contour, key=itemgetter(0))  # get the max of x or y and corresponding y or x that's bound to it
        max1 = max(contour, key=itemgetter(1))
        min0 = min(contour, key=itemgetter(0))
        min1 = min(contour, key=itemgetter(1))
        dx = max0[0] - min0[0]
        dy = max1[1] - min1[1]
        extreme.append((min0[0], min1[1], max0[0], max1[1]))  # min x, y / max x, y 
        sum.append(dx + dy)

    max_value = max(sum)
    max_index = sum.index(max_value)
    contour_elect = max_index  # choose the contour with the largest area
    print('Contour index', contour_elect)
    print('Corner index:', extreme)

    # crop = img_copy[extreme[contour_elect][1]:extreme[contour_elect][3], extreme[contour_elect][0]: extreme[contour_elect][2]]
    # img = cv2.drawContours(img, contours, -1, (0,255,0), 5)  # img is 3-channel

    # cv2.imshow('img with contours', img)
    # cv2.imshow("THRESH", thresh)
    # cv2.imshow("CROP", crop)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    return extreme[contour_elect][0], extreme[contour_elect][2], extreme[contour_elect][1], extreme[contour_elect][3]



def KFE(video_path, save_to):
    BACKGROUND_REMOVAL = False
    dsample_rate = 10  # take a frame per dsample_rate frames
    clip_period = 60  # second, need to be greater than 120*dsample_rate/fps ~= 4*dsample_rate (svd 63 dimension constraint)
    similarity_threshold = 0.95  # squared
    CLUSTER_THRESHOLD = 20

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    clip_length = int(clip_period * fps)

    count = 0  # temporary counter (real time reading frame's index for each clip, total frames)
    count_p = 0  # temporary counter (index of processed frames)
    clip_idx = -1
    frames_set = dict()
    feature_mat_set = dict()
    if cap.isOpened() is not True:
        raise NameError('Video path problem or cannot be opened.')

    start_time = time.time()
    ret, frame = cap.read()
    x_min = 0; y_min = 0
    x_max, y_max, channels = frame.shape

    if BACKGROUND_REMOVAL:
        for i in range(30):
            ret, frame = cap.read() 
        y_min, y_max, x_min, x_max = remove_blackbar(frame)
    
    cap.release()

    cap = cv2.VideoCapture(video_path) # read the file the second time
    while cap.isOpened():
        # Read the video file.
        ret, frame = cap.read()  # ret is the flag of reading result (True, False)

        # If we got frames.
        if ret is True:
            if count % clip_length == 0:  # New clip
                clip_idx += 1
                # to store all frames (after down-sampling) of the current clip (matrices)
                frames_set[clip_idx] = dict()
                # to store all feature vectors (after downsampling) of the current clip
                feature_mat_set[clip_idx] = np.empty((clip_length//dsample_rate + 1, 1944), int)
                if clip_idx > 0:
                    # the clip space preserved in last step was rough, remove the previous clip unused space
                    feature_mat_set[clip_idx-1] = feature_mat_set[clip_idx-1][:count_p]
                count = 0
                count_p = 0
            if count % dsample_rate == 0:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # since cv reads frame in bgr order so rearraning to get frames in rgb order
                frames_set[clip_idx][count_p] = frame_rgb   # storing each frame (array) to frames
                frame_copy = frame_rgb.copy()
                frame_rgb_crop = frame_copy[y_min: y_max, x_min: x_max]
                feature_vector = frame_to_hist(frame_rgb_crop)
                feature_mat_set[clip_idx][count_p] = feature_vector  # appending each one-dimensinal vector to generate a Nx1944 matrix
                count_p += 1
            count += 1
        else:
            feature_mat_set[clip_idx] = feature_mat_set[clip_idx][:count_p]
            break
    # print("--- %.2f seconds in computing color histograms ---" % (time.time() - start_time))
    
    # start_time = time.time()
    # get key frames for each clip
    num_of_clips = clip_idx + 1
    kf_idx_set = dict() # to store the index of key frames for each clip
    for i in range(num_of_clips):
        frames = frames_set[i]
        feature_mat = feature_mat_set[i]
        kf_idx_set[i] = get_keyframe_idx(feature_mat, dsample_rate, similarity_threshold, CLUSTER_THRESHOLD)  # keyframe extraction excution

    # removing repeated keyframes between the gap of consecutive clips
    for i in range(num_of_clips-1):
        feature1 = feature_mat_set[i][kf_idx_set[i][-1]] 
        feature2 = feature_mat_set[i+1][kf_idx_set[i+1][0]]
        # Cosine similarity squared
        similarity2 = np.dot(feature1, feature2)**2/( np.dot(feature1,feature1)  * np.dot(feature2, feature2) )
        if similarity2 > similarity_threshold:  # remove the last keyframe index of the first clip
            kf_idx_set[i] = np.delete(kf_idx_set[i],-1)
    # print('keyframes extracted!')
    # print("--- %.2f seconds in extracting keyframes ---" % (time.time() - start_time))
    
    # start_time = time.time()
    # save the key frames to save_to
    if os.path.exists(save_to):
        i = 1
        while os.path.exists(save_to+'_'+str(i)):
            i += 1
        os.mkdir(save_to+'_'+str(i))
        save_to = save_to + '_' + str(i)
    else:
        os.mkdir(save_to)

    # output the frames in jpg format
    num_kf = 0
    for i in range(num_of_clips):
        for kf_idx in kf_idx_set[i]:
            num_kf += 1
            time_in_sec = int( i*clip_period + kf_idx*dsample_rate/fps )
            time_minute = time_in_sec // 60
            time_second = time_in_sec % 60
            # print('Key frame at : %d:%2d' % (time_minute, time_second))
            frame_rgb = cv2.cvtColor(frames_set[i][kf_idx], cv2.COLOR_RGB2BGR)
            time_chr = str(time_minute) + '_' + str(time_second)
            file_name = 'time' + time_chr + '.jpg'
            cv2.imwrite(os.path.join(save_to, file_name), frame_rgb)

    print("--- saved %d key frames to %s ---" % (num_kf, save_to))
    # print("--- %.2f seconds in saving keyframes ---" % (time.time() - start_time))
    exe_time = time.time() - start_time
    print("--- %.2f seconds in saving keyframes ---" %exe_time)
    return exe_time, save_to

