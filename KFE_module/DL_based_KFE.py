# -*- coding: utf-8 -*-

import numpy as np
import time
import cv2
import os
import pandas as pd
from utils.vgg import VGGNetFeat
from utils.resnet import ResNetFeat
from utils.hog import HOG


def read_video_DL(video_path):
    cap = cv2.VideoCapture(video_path)
    global fps
    fps = cap.get(cv2.CAP_PROP_FPS)

    frame_set = dict()
    count = 0
    count_p = 0
    if cap.isOpened() is not True:
        raise NameError('Video path problem or cannot be opened.')

    while cap.isOpened():
        # Read the video file.
        ret, frame = cap.read()

        if ret is True:
            if count % dsample_rate == 0:
                # frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)    #since cv reads frame in bgr order so rearraning to get frames in rgb order
                # frame_set_rgb[count_p] = frame_rgb
                frame_set[count_p] = frame     # storing each frame (array) to D , so that we can identify key frames later
                count_p += 1
            count += 1
        else:
            break

    return frame_set


def extract_high_lev_f(frame_set):
    samples, init_time = method.make_samples(frame_set, video_path)
    f_mat = np.vstack([d['hist'] for d in samples])

    return f_mat, init_time


def clustering(f_mat, clu_min_thres=2):
    # dynamic clustering of projected frame histograms to find shots
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
        if similarity2 < similarity_threshold:
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
        if num[i] >= clu_min_thres:
            new_KF_idx = s + (num[i]+1)//2 - 1  # ceiling
            # new_KF_idx = s + num[i] - 1  # ceiling (the second last frame)
            KF_idx.append(new_KF_idx)  # python idx start from 0
            KF_vec.append(f_mat[new_KF_idx])
        s += num[i]
        i += 1
    KF_vec = np.array(KF_vec)
    return KF_vec, KF_idx


def save_KFs(KF_idx, save_to, frame_set):
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
    for kf_idx in KF_idx:
        num_kf += 1
        time_in_sec = int(kf_idx*dsample_rate/fps)
        time_minute = time_in_sec // 60
        time_second = time_in_sec % 60
        # print('Key frame at : %d:%2d' % (time_minute, time_second))
        # frame_rgb = cv2.cvtColor(frame_set[kf_idx], cv2.COLOR_RGB2BGR)
        time_chr = str(time_minute) + '_' + str(time_second)
        file_name = 'time' + time_chr + '.jpg'
        cv2.imwrite(os.path.join(save_to, file_name), frame_set[kf_idx])

    save_time = time.time() - start_time
    print("--- saved %d key frames to %s ---" % (num_kf, save_to))
    print("--- %.2f seconds in saving keyframes ---" % (save_time))

    return num_kf, save_time


def save_performance_results(execution_time, initiation_times, num_kfs, save_result_to):
    dict_list = [execution_time, initiation_times, num_kfs]
    df = pd.DataFrame(dict_list)
    output_path = os.path.join(save_result_to, 'vgg_timeperformanceB6.xlsx')

    with pd.ExcelWriter(output_path, engine='xlsxwriter') as writer:
        df.to_excel(writer, sheet_name='Sheet1', index=False)
    print('The output sheet is saved to', output_path)
    print('finished')


##########################################
##########################################


global method, similarity_threshold, dsample_rate
method = ResNetFeat()
similarity_threshold = 0.98
dsample_rate = 15

global_path = ".../KFE_deep_learning"
data_path = os.path.join(global_path, "dataset")  #
iteration_times = {}
initiation_times = {}
num_kfs = {}

for folderpath in os.listdir(data_path):
    try:
        start_time = time.time()
        video_path = os.path.join(data_path, folderpath, folderpath) + '.mp4'
        save_to = os.path.join(data_path, folderpath, 'keyframes')
        print(video_path)
        # main process start
        frame_set = read_video_DL(video_path)
        f_mat, init_time = extract_high_lev_f(frame_set)
        print(f_mat.shape)
        # print(len(f_mat), init_time)
        # full_clu, res = clustering_DL(f_mat)
        # KFs = get_clustered_KFs(full_clu, res)
        KF_vec, KFs = clustering(f_mat)
        num_kf, save_time = save_KFs(KFs, save_to, frame_set)
        # main process end
        end_time = time.time()
        execution_time = end_time - start_time

        iteration_times[folderpath] = execution_time
        initiation_times[folderpath] = init_time
        print(f'--- the overall execution time of {folderpath} is %.2f ---' % (iteration_times[folderpath]))
        num_kfs[folderpath] = num_kf
        print(f"=== {folderpath} is done. ===")

    except ValueError:
        print(f"Error at {folderpath}")

save_result_to = os.path.join(global_path, "results", "time_performance")
save_performance_results(iteration_times, initiation_times, num_kfs, save_result_to)
