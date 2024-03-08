import torch
import torch.nn as nn
import numpy as np
import time
import cv2
import os
import pandas as pd

from torch.autograd import Variable
from torchvision import models
from torchvision.models.resnet import Bottleneck, BasicBlock, ResNet
import torch.utils.model_zoo as model_zoo
from tqdm import tqdm
from torchvision.models.vgg import VGG
from utils.vgg2 import VGGNetFeat
from utils.resnet import ResNetFeat


#########################################
#########################################


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


def clustering_DL(f_mat):
    # dynamic clustering of projected frame histograms to find which all frames are similar i.e make shots
    cluster_set = dict()  # to store frames in respective cluster
    for i in range(f_mat.shape[0]):
        cluster_set[i] = np.empty((0, 512), int)

    # adding first two projected frames in first cluster i.e Initializaton        
    cluster_set[0] = np.vstack((cluster_set[0], f_mat[0]))     
    cluster_set[0] = np.vstack((cluster_set[0], f_mat[1]))

    centroid_set = dict()  # to store centroids of each cluster
    for i in range(f_mat.shape[0]):
        centroid_set[i] = np.empty((0, 512), int)
    # finding centroid of centroid_set[0] cluster
    centroid_set[0] = np.mean(cluster_set[0], axis=0) 

    count = 0
    for i in range(2, f_mat.shape[0]):
        similarity2 = np.dot(f_mat[i], centroid_set[count])**2/(np.dot(f_mat[i],f_mat[i])*np.dot(centroid_set[count], centroid_set[count]) ) # cosine similarity
        if similarity2 < similarity_threshold:
            count += 1
            cluster_set[count] = np.vstack((cluster_set[count], f_mat[i]))
            centroid_set[count] = np.mean(cluster_set[count], axis=0)
        else:    # if they are similar then assign this data point to last cluster formed and update the centroid of the cluster
            cluster_set[count] = np.vstack((cluster_set[count], f_mat[i])) 
            centroid_set[count] = np.mean(cluster_set[count], axis=0)

    num = []    # find the number of data points in each cluster formed.
    for i in range(f_mat.shape[0]):
        num.append(cluster_set[i].shape[0])

    last = num.index(0)    # where we find 0 in b indicates that all required clusters have been formed, so we can delete these from C
    size_clu = num[:last]    # The size of each valid cluster.
    res = [idx for idx, val in enumerate(size_clu) if val >= 2] 

    new_label = cluster_set    # Label each cluster, making it easier to identify frames in each cluster
    for i in range(last):
        p = np.repeat(i, size_clu[i]).reshape(size_clu[i], 1)
        new_label[i] = np.hstack((new_label[i], p))

    full_clu = np.empty((0, 513), int) 
    for i in range(last):
        full_clu = np.vstack((full_clu, new_label[i]))

    # print('full_clu is ', type(full_clu), len(full_clu))
    # print('res is ', type(res), len(res), res)

    return full_clu, res


def get_clustered_KFs(full_clusters, residual):
    colnames = []
    for i in range(1, 514):
        col_name = "v" + str(i)
        colnames += [col_name]

    df = pd.DataFrame(full_clusters, columns=colnames)
    df['v513'] = df['v513'].astype(int)    # converting the cluster level from float type to integer type
    df1 = df[df.v513.isin(residual)]
    KF = df1.groupby('v513').tail(1)['v513']
    KF = KF.index - 1

    # print('Number of keyframes is ', len(KF))
    # print('keyframes are ', KF)

    return KF


def save_KFs(KF, save_to, frame_set):
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
    for kf_idx in KF:
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


def save_performance_results(execution_time, initiation_times, num_kfs):
    dict_list = [execution_time, initiation_times, num_kfs]
    df = pd.DataFrame(dict_list)
    output_path = os.path.join(global_path, 'output.xlsx')
    with pd.ExcelWriter(output_path, engine='xlsxwriter') as writer:
        df.to_excel(writer, sheet_name='Sheet1', index=False)
    print('The output sheet is saved to', output_path)
    print('finished')
