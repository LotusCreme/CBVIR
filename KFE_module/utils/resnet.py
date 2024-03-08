import numpy as np
import time
import os
from tqdm import tqdm

import torch
import torch.utils.model_zoo as model_zoo
from torchvision.models.resnet import Bottleneck, BasicBlock, ResNet


# ResNet model configuration
RES_model = 'resnet18'
pick_layer = 'avg'

use_gpu = torch.cuda.is_available()
means = np.array([103.939, 116.779, 123.68]) / 255.  # mean of 3 channels (BGR)

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
            'fc': fc
        }
        return output


class ResNetFeat(object):
    def make_samples(self, frame_set, video_path):
        start_time = time.time()
        res_model = ResidualNet(model=RES_model)
        res_model.eval()
        if use_gpu:
            res_model = res_model.cuda()
        global init_time
        init_time = time.time() - start_time
        print(f'ResNet initiation cost {init_time} seconds')

        video_name = os.path.basename(video_path)
        print("--- extracting features of all frames..., video_index=%s, model=%s, layer=%s ---" % (video_name, RES_model, pick_layer))

        # features
        samples = []
        for idx in tqdm(frame_set):
            img = frame_set[idx]
            img = np.transpose(img, (2, 0, 1)) / 255.
            img[0] -= means[0]  # reduce B's mean
            img[1] -= means[1]  # reduce G's mean
            img[2] -= means[2]  # reduce R's mean
            img = np.expand_dims(img, axis=0)
            try:
                if use_gpu:
                    inputs = torch.autograd.Variable(torch.from_numpy(img).cuda().float())
                else:
                    inputs = torch.autograd.Variable(torch.from_numpy(img).float())
                d_hist = res_model(inputs)[pick_layer]
                d_hist = d_hist.data.cpu().numpy().flatten()
                d_hist /= np.sum(d_hist)  # normalize
                samples.append({
                                'img_idx': idx,
                                'hist': d_hist
                                })

            except ValueError as e:
                print("Wrong", e)
                raise

        return samples, init_time
