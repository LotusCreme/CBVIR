import numpy as np
import pandas as pd
import time
import cv2
import os

from scipy.sparse import csc_matrix
from scipy.sparse.linalg import svds
from operator import itemgetter
from collections import defaultdict
from sklearn.cluster import KMeans



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


def get_kf_idx(feature_mat, dsample_rate, simi_thr, cluster_thr, method='dynamic', K=15):
    if feature_mat.shape[0] < 100:
        return [feature_mat.shape[0]-1]

    feature_mat = csc_matrix(feature_mat.transpose(), dtype=float)
    # implementing SVD
    u, s, vt = svds(feature_mat, k=60)
    vt = vt.transpose()
    f_mat = vt @ np.diag(s)

    if method == 'dynamic':
        cluster_set = dict()
        for i in range(f_mat.shape[0]):
            cluster_set[i] = np.empty((0, f_mat.shape[1]), int)

        # initialize the cluster
        cluster_set[0] = np.vstack((cluster_set[0], f_mat[0]))
        cluster_set[0] = np.vstack((cluster_set[0], f_mat[1]))

        centroid_set = dict()  # to store centroids of each cluster
        for i in range(f_mat.shape[0]):
            centroid_set[i] = np.empty((0, f_mat.shape[1]), int)
        # finding centroid of centroid_set[0] cluster
        centroid_set[0] = np.mean(cluster_set[0], axis=0)

        count = 0
        for i in range(2, f_mat.shape[0]):
            similarity2 = np.dot(f_mat[i], centroid_set[count])**2/(np.dot(f_mat[i], f_mat[i])*np.dot(centroid_set[count], centroid_set[count]))
            if similarity2 < simi_thr:
                count += 1
                cluster_set[count] = np.vstack((cluster_set[count], f_mat[i]))
                centroid_set[count] = np.mean(cluster_set[count], axis=0)
            else:
                cluster_set[count] = np.vstack((cluster_set[count], f_mat[i]))
                centroid_set[count] = np.mean(cluster_set[count], axis=0)

        num = []  # find the number of data points in each cluster formed.
        for i in range(f_mat.shape[0]):
            num.append(cluster_set[i].shape[0])

        KF_idx = []
        KF_vec = []
        i = 0
        s = 0
        while num[i] != 0:
            if num[i] >= cluster_thr/dsample_rate:
                new_KF_idx = s + (num[i]+1)//2 - 1  # ceiling
                # new_KF_idx = s + num[i] - 1  # ceiling (the second last frame)
                KF_idx.append(new_KF_idx)  # python idx start from 0
                KF_vec.append(f_mat[new_KF_idx])
            s += num[i]
            i += 1
        KF_vec = np.array(KF_vec)

    elif method == 'Kmeans':
        # print("using KMeans clustering, K=", K)
        kmeans = KMeans(n_clusters=K, random_state=42)
        clusters = kmeans.fit_predict(f_mat)

        # Create a dictionary to store the image indices for each cluster
        cluster_indices = {}
        for i, cluster_label in enumerate(clusters):
            if cluster_label not in cluster_indices:
                cluster_indices[cluster_label] = []
            cluster_indices[cluster_label].append(i)

        KF_idx = []
        KF_vec = []
        i = 0
        s = 0
        for i in range(K):
            if len(cluster_indices[i]) >= 2:
                new_KF_idx = cluster_indices[i][(len(cluster_indices[i])+1)//2]  # ceiling
                # new_KF_idx = s + num[i] - 1  # ceiling (the second last frame)
                KF_idx.append(new_KF_idx)  # python idx start from 0
                KF_vec.append(f_mat[new_KF_idx])
        KF_vec = np.array(KF_vec)

    return KF_idx


def remove_blackbar(frame_img):
    img = frame_img
    # img_copy = img.copy()

    # convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # invert gray image
    gray = 255 - gray
    # gaussian blur
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    # threshold
    thresh = cv2.threshold(blur, 236, 255, cv2.THRESH_BINARY)[1]
    # apply close and open morphology to fill tiny black and white holes
    kernel = np.ones((5, 5), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    # invert thresh
    thresh = 255 - thresh
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

    return extreme[contour_elect][0], extreme[contour_elect][2], extreme[contour_elect][1], extreme[contour_elect][3]


def full_KFE_onetime(video_path, save_to, dsample_rate, clip_period, similarity_threshold, CLUSTER_THRESHOLD, BACKGROUND_REMOVAL, clu_method='dynamic', K=15):

    # capture video in clips and compute color histograms as feature matrices
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
    x_min = 0
    y_min = 0
    x_max, y_max, channels = frame.shape

    if BACKGROUND_REMOVAL:
        for i in range(30):
            ret, frame = cap.read()
        y_min, y_max, x_min, x_max = remove_blackbar(frame)

    cap.release()

    cap = cv2.VideoCapture(video_path)  # read the file the second time
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
    time_ch = time.time() - start_time
    print("--- %.2f seconds in computing color histograms ---" % (time_ch))

    start_time = time.time()
    # get key frames for each clip
    num_of_clips = clip_idx + 1
    kf_idx_set = dict()  # to store the index of key frames for each clip
    for i in range(num_of_clips):
        feature_mat = feature_mat_set[i]
        kf_idx_set[i] = get_kf_idx(feature_mat, dsample_rate, similarity_threshold, CLUSTER_THRESHOLD, method=clu_method, K=K)  # keyframe extraction excution
    time_clu = time.time() - start_time

    # removing repeated keyframes between the gap of consecutive clips
    for i in range(num_of_clips-1):
        feature1 = feature_mat_set[i][kf_idx_set[i][-1]]
        feature2 = feature_mat_set[i+1][kf_idx_set[i+1][0]]
        # Cosine similarity squared
        similarity2 = np.dot(feature1, feature2)**2/(np.dot(feature1, feature1) * np.dot(feature2, feature2))
        if similarity2 > similarity_threshold:  # remove the last keyframe index of the first clip
            kf_idx_set[i] = np.delete(kf_idx_set[i], -1)
    # print('keyframes extracted!')

    start_time = time.time()
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
            time_in_sec = int(i*clip_period + kf_idx*dsample_rate/fps)
            time_minute = time_in_sec // 60
            time_second = time_in_sec % 60
            # print('Key frame at : %d:%2d' % (time_minute, time_second))
            frame_rgb = cv2.cvtColor(frames_set[i][kf_idx], cv2.COLOR_RGB2BGR)
            time_chr = str(time_minute) + '_' + str(time_second)
            file_name = 'time' + time_chr + '.jpg'
            cv2.imwrite(os.path.join(save_to, file_name), frame_rgb)
    time_save = time.time() - start_time
    print("--- saved %d key frames to %s ---" % (num_kf, save_to))
    print("--- %.2f seconds in saving keyframes ---" % (time_save))

    return time_ch, time_clu, time_save, num_kf


def KFE_acc_check(numbers, intervals):
    # Create a defaultdict to store the counts for each interval
    interval_counts = defaultdict(int)

    # Iterate through the numbers
    for number in numbers:
        # Iterate through the intervals
        for interval in intervals:
            # Check if the number is within the interval
            if interval[0] <= number <= interval[1]:
                # If it is, increment the count for this interval
                interval_counts[interval] += 1
                # We can break out of the loop here, since we only want to count
                # the number once, even if it is contained in multiple intervals
                break
    return interval_counts
