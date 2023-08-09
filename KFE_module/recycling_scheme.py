import torch
import numpy as np
import time
import cv2
import os
import pandas as pd
import scipy

from torchvision.models.resnet import Bottleneck, BasicBlock, ResNet
import torch.utils.model_zoo as model_zoo
from tqdm import tqdm


class VideoReader:
    def __init__(self):
        self.__vd_path = ''
        self.__frame_set = None
        self.__fps = 0
        self.__dn_rate = 0
        self.__status = 0
        self.__count = 0
        self.__count_p = 0

    def read(self, video_path, dn_rate=1):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            self.__status = 0
            print(f'Fail to open the video or video does not exist! Was trying to open {video_path}')
            return
        # video is successfully read
        self.__status = 1
        self.__vd_path = video_path
        self.__fps = cap.get(cv2.CAP_PROP_FPS)
        self.__dn_rate = dn_rate
        self.__frame_set = []
        while cap.isOpened():
            # Read the video file.
            ret, frame = cap.read()
            if ret:
                if (self.__count % self.__dn_rate == 0):
                    self.__frame_set.append(frame)   # storing each frame (array) to D , so that we can identify key frames later
                    self.__count_p += 1
                self.__count += 1
            else:
                break

    def __check_status(self):
        if not self.__status:
            raise RuntimeError('Video is not read')
            return False
        return True

    def get_fps(self):
        if self.__check_status():
            return self.__fps
        return

    def get_frame_set(self):
        if self.__check_status():
            return self.__frame_set
        return

    def get_video_path(self):
        if self.__check_status():
            return self.__vd_path
        return

    def get_num_frames(self):
        if self.__check_status():
            return self.__count_p
        return

    def get_frame_by_idx(self, idx):
        if idx >= self.get_num_frames() or idx <= 0:
            raise RuntimeError('Idx out of range')
        if self.__check_status():
            return self.__frame_set[idx]
        return


class ImageReader:
    def __init__(self):
        self.__im_path = ''
        self.__img = None
        self.__width = 0
        self.__height = 0
        self.__channel = 0
        self.__status = 0

    def read(self, im_path):
        img = cv2.imread(im_path)
        if img is None:
            self.__status = 0
            print(f'Fail to open the image or image does not exist! Was trying to open {im_path}')
            return
        # Image is successfully read
        self.__status = 1
        self.__im_path = im_path
        self.__img = img
        self.__height = img.shape[0]
        self.__width = img.shape[1]
        self.__channel = img.shape[2]

    def __check_status(self):
        if not self.__status:
            raise RuntimeError('Image is not read')
            return False
        return True

    def get_image(self):
        if self.__check_status():
            return self.__img
        return


def clustering(f_mat, similarity_threshold=0.95, clu_min_thres=2):
    cluster_set = dict()  # to store frames in respective cluster
    for i in range(f_mat.shape[0]):
        cluster_set[i] = np.empty((0, f_mat.shape[1]), int)

    cluster_set[0] = np.vstack((cluster_set[0], f_mat[0]))
    cluster_set[0] = np.vstack((cluster_set[0], f_mat[1]))

    centroid_set = dict()  # to store centroids of each cluster
    for i in range(f_mat.shape[0]):
        centroid_set[i] = np.empty((0, f_mat.shape[1]), int)
    # finding centroid of centroid_set[0] cluster
    centroid_set[0] = np.mean(cluster_set[0], axis=0)
    count = 0
    for i in range(2, f_mat.shape[0]):
        similarity2 = np.dot(f_mat[i], centroid_set[count])**2/(np.dot(f_mat[i], f_mat[i])*np.dot(centroid_set[count], centroid_set[count]))  # cosine similarity
        if similarity2 < similarity_threshold:
            count += 1
            cluster_set[count] = np.vstack((cluster_set[count], f_mat[i]))
            centroid_set[count] = np.mean(cluster_set[count], axis=0)
        else:  # if they are similar then assign this data point to last cluster formed and update the centroid of the cluster
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
            new_KF_idx = s + (num[i]+1)//2 - 1  # middle
            # new_KF_idx = s + num[i] - 1  # second last
            KF_idx.append(new_KF_idx)  # python idx start from 0
            KF_vec.append(f_mat[new_KF_idx])
        s += num[i]
        i += 1
    KF_vec = np.array(KF_vec)
    return KF_vec, KF_idx


# configs
RES_model  = 'resnet18'  # model type
pick_layer = 'avg'        # extract feature of this layer

use_gpu = torch.cuda.is_available()
means = np.array([103.939, 116.779, 123.68]) / 255. # mean of three channels in the order of BGR

# from https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


class ResidualNet(ResNet):
    def __init__(self, model=RES_model, pretrained=True):
        if model == "resnet18":
            super().__init__(BasicBlock, [2, 2, 2, 2], 1000)
            if pretrained:
                self.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
        elif model == "resnet34":
            super().__init__(BasicBlock, [3, 4, 6, 3], 1000)
            if pretrained:
                self.load_state_dict(model_zoo.load_url(model_urls['resnet34']))
        elif model == "resnet50":
            super().__init__(Bottleneck, [3, 4, 6, 3], 1000)
            if pretrained:
                self.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
        elif model == "resnet101":
            super().__init__(Bottleneck, [3, 4, 23, 3], 1000)
            if pretrained:
                self.load_state_dict(model_zoo.load_url(model_urls['resnet101']))
        elif model == "resnet152":
            super().__init__(Bottleneck, [3, 8, 36, 3], 1000)
            if pretrained:
                self.load_state_dict(model_zoo.load_url(model_urls['resnet152']))

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)  # x after layer4, shape = N * 512 * H/32 * W/32
        max_pool = torch.nn.MaxPool2d((x.size(-2), x.size(-1)), stride=(x.size(-2), x.size(-1)), padding=0, ceil_mode=False)
        Max = max_pool(x)  # avg.size = N * 512 * 1 * 1
        Max = Max.view(Max.size(0), -1)  # avg.size = N * 512
        avg_pool = torch.nn.AvgPool2d((x.size(-2), x.size(-1)), stride=(x.size(-2), x.size(-1)), padding=0, ceil_mode=False, count_include_pad=True)
        avg = avg_pool(x)  # avg.size = N * 512 * 1 * 1
        avg = avg.view(avg.size(0), -1)  # avg.size = N * 512
        fc = self.fc(avg)  # fc.size = N * 1000
        output = {
            'max': Max,
            'avg': avg,
            'fc' : fc
        }
        return output


class ResNetFeat(object):
    def make_samples(self, frame_set):
        start_time = time.time()
        res_model = ResidualNet(model=RES_model)
        res_model.eval()
        if use_gpu:
            res_model = res_model.cuda()
        init_time = time.time() - start_time
        samples = np.zeros((len(frame_set), 512))
        idx = 0
        for img in tqdm(frame_set):
            img = np.transpose(img, (2, 0, 1)) / 255.0
            img[0] -= means[0]
            img[1] -= means[1]
            img[2] -= means[2]
            img = np.expand_dims(img, axis=0)
            if use_gpu:
                inputs = torch.autograd.Variable(torch.from_numpy(img).cuda().float())
            else:
                inputs = torch.autograd.Variable(torch.from_numpy(img).float())
            d_hist = res_model(inputs)[pick_layer]
            d_hist = d_hist.data.cpu().numpy().flatten()
            d_hist /= np.sum(d_hist)  # normalize
            samples[idx, :] = d_hist
            idx += 1
        return samples, init_time


def save_KFs(KF, save_to, frame_set, fps):
    # save the key frames to save_to
    if os.path.exists(save_to):
        i = 1
        while os.path.exists(save_to + '_' + str(i)):
            i += 1
        save_to = save_to + '_' + str(i)
        os.mkdir(save_to)
    else:
        os.mkdir(save_to)

    # output the frames in jpg format
    num_kf = 0
    for kf_idx in KF:
        num_kf += 1
        time_chr = frame_idx_to_time(kf_idx, fps)
        file_name = 'time' + time_chr + '.jpg'
        cv2.imwrite(os.path.join(save_to, file_name), frame_set[kf_idx])

    print("--- saved %d key frames to %s ---" % (num_kf, save_to))
#     print("--- %.2f seconds in saving keyframes ---" % (save_time))


def frame_idx_to_time(frame_idx, fps, split='_'):
    time_in_sec = int(frame_idx*dsample_rate/fps)
    time_minute = time_in_sec // 60
    time_second = time_in_sec % 60
    time_chr = str(time_minute) + split + "{:02d}".format(time_second)
    return time_chr


def save_performance_results(iteration_times, initiation_times, num_kfs):
    output_path = 'recycling_output.xlsx'
    df = pd.DataFrame([iteration_times, initiation_times, num_kfs])
    with pd.ExcelWriter(output_path, engine='xlsxwriter') as writer:
        df.to_excel(writer, sheet_name='Sheet1', index=False)
    print('The output sheet is saved to', output_path)
    print('finished')


def save_ranks(idx_frames, time_frames, q_name, output_path):
    df1 = pd.DataFrame(idx_frames, columns=q_name)
    df2 = pd.DataFrame(time_frames, columns=q_name)
    with pd.ExcelWriter(output_path, engine='xlsxwriter') as writer:
        df1.to_excel(writer, sheet_name='Index', index=False)
        df2.to_excel(writer, sheet_name='Time', index=False)


#################################################
################### START #######################
#################################################

method = ResNetFeat()
similarity_threshold = 0.98
dsample_rate = 15
global_path = ".../KFE_deep_learning/dataset"
iteration_times = {}
initiation_times = {}
num_kfs = {}

for folderpath in os.listdir(global_path):
    if folderpath[0] == '.':
        continue
    try:
        vr = VideoReader()
        ir = ImageReader()
        start_time = time.time()
        video_path = os.path.join(global_path, folderpath, folderpath) + '.mp4'
        save_to = os.path.join(global_path, folderpath, 'keyframes')  #
        # print(video_path)
        # main process start

        # record query_name
        q_name = []

        # read video
        vr.read(video_path, dsample_rate)
        frame_set = vr.get_frame_set()  # get all frames
        fps = vr.get_fps()

        # read queries
        for image_path in os.listdir(os.path.join(global_path, folderpath)):
            if (image_path.endswith(".png") or image_path.endswith(".jpg") or image_path.endswith(".jpeg")):
                ir.read(os.path.join(global_path, folderpath, image_path))
                qu_img = ir.get_image()
                frame_set.append(qu_img)
                q_name.append(image_path.rsplit('.')[0])

        # get all features, including video frames and query images, with given method
        f_mat, init_time = method.make_samples(frame_set)
        # print(f'All features extracted for Video {folderpath}')

        num_vd_frames = vr.get_num_frames()
        f_mat_video = f_mat[0:num_vd_frames, :]
        f_mat_query = f_mat[num_vd_frames:, :]

        # clustering, getting all key frames
        KF_vec, KF_idx = clustering(f_mat_video, similarity_threshold)
        # print(f'Keyframes found for Video {folderpath}')

        # comparing keyframe features with query features
        # construcing KDTree
        kd = scipy.spatial.KDTree(KF_vec)
        distance, idx = kd.query(f_mat_query, min(20, len(KF_idx)))
        print("Distances got")
        # idx is the indices within keyframes, next we convert them to indices of frames
        idx_frames = idx.transpose()
        for i in range(idx_frames.shape[0]):
            for j in range(idx_frames.shape[1]):
                idx_frames[i][j] = KF_idx[idx_frames[i][j]]
        execution_time = time.time() - start_time

        # saving keyframes
        save_KFs(KF_idx, save_to, frame_set[0:num_vd_frames], fps)

        # saving ranks
        time_frames = []
        for i in range(idx_frames.shape[0]):
            t_row = []
            for j in range(idx_frames.shape[1]):
                t_row.append(frame_idx_to_time(idx_frames[i][j], fps, split=':'))
            time_frames.append(t_row)

        ranks_path = os.path.join(global_path, folderpath, 'ranks.xlsx')
        print("rank path is", ranks_path)
        save_ranks(idx_frames, time_frames, q_name, ranks_path)
        print("All ranks saved.")

        iteration_times[folderpath] = execution_time
        initiation_times[folderpath] = init_time
        print(f'--- the overall execution time of {folderpath} is %.2f ---' % (iteration_times[folderpath]))
        num_kfs[folderpath] = len(KF_idx)
        save_performance_results(iteration_times, initiation_times, num_kfs)

        print(f"=== {folderpath} is done. ===")

    except ValueError:
        print(f"Error at {folderpath}")

