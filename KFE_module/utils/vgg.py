import numpy as np
import os
import time
from tqdm import tqdm

import torch
import torch.nn as nn
from torchvision import models
from torch.autograd import Variable
from torchvision.models.vgg import VGG

# VGG model configuration
VGG_model = 'vgg16'
pick_layer = 'avg'

use_gpu = torch.cuda.is_available()
means = np.array([103.939, 116.779, 123.68]) / 255.  # mean of 3 color channels (BGR)

ranges = {
    'vgg11': ((0, 3), (3, 6), (6, 11), (11, 16), (16, 21)),
    'vgg13': ((0, 5), (5, 10), (10, 15), (15, 20), (20, 25)),
    'vgg16': ((0, 5), (5, 10), (10, 17), (17, 24), (24, 31)),
    'vgg19': ((0, 5), (5, 10), (10, 19), (19, 28), (28, 37))
}


# from https://github.com/pytorch/vision/blob/master/torchvision/models/vgg.py
cfg = {
    'vgg11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'vgg19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


class VGGNet(VGG):
    def __init__(self, weights=True, model=VGG_model, requires_grad=False, remove_fc=False, show_params=False):
        super().__init__(make_layers(cfg[model]))
        self.ranges = ranges[model]
        self.fc_ranges = ((0, 2), (2, 5), (5, 7))

        if weights:
            exec("self.load_state_dict(models.%s(weights=True).state_dict())" % model)

        if not requires_grad:
            for param in super().parameters():
                param.requires_grad = False

        if remove_fc:  # delete redundant fully-connected layer params, can save memory
            del self.classifier

        if show_params:
            for name, param in self.named_parameters():
                print(name, param.size())

    def forward(self, x):
        output = {}

        x = self.features(x)

        avg_pool = torch.nn.AvgPool2d((x.size(-2), x.size(-1)), stride=(
            x.size(-2), x.size(-1)), padding=0, ceil_mode=False, count_include_pad=True)
        avg = avg_pool(x)  # avg.size = N * 512 * 1 * 1
        avg = avg.view(avg.size(0), -1)  # avg.size = N * 512
        output['avg'] = avg

        x = x.view(x.size(0), -1)  # flatten()
        dims = x.size(1)
        if dims >= 25088:
            x = x[:, :25088]
            for idx in range(len(self.fc_ranges)):
                for layer in range(self.fc_ranges[idx][0], self.fc_ranges[idx][1]):
                    x = self.classifier[layer](x)
                output["fc%d" % (idx+1)] = x
        else:
            w = self.classifier[0].weight[:, :dims]
            b = self.classifier[0].bias
            x = torch.matmul(x, w.t()) + b
            x = self.classifier[1](x)
            output["fc1"] = x
            for idx in range(1, len(self.fc_ranges)):
                for layer in range(self.fc_ranges[idx][0], self.fc_ranges[idx][1]):
                    x = self.classifier[layer](x)
                output["fc%d" % (idx+1)] = x

        return output


class VGGNetFeat(object):

    def make_samples(self, frame_set, video_path):
        start_time = time.time()
        vgg_model = VGGNet(requires_grad=False, model=VGG_model)
        vgg_model.eval()
        if use_gpu:
            vgg_model = vgg_model.cuda()
        global init_time
        init_time = time.time() - start_time
        print(f'VGG model initiation cost {init_time} seconds')

        video_name = os.path.basename(video_path)
        print("--- extracting features of all frames..., video_index=%s, model=%s, layer=%s ---" %
              (video_name, VGG_model, pick_layer))
        samples = []
        for idx in tqdm(frame_set):
            # print(f'Extracting frame {idx}\'s feature')
            img = frame_set[idx]
            img = np.transpose(img, (2, 0, 1)) / 255.
            img[0] -= means[0]  # reduce B's mean
            img[1] -= means[1]  # reduce G's mean
            img[2] -= means[2]  # reduce R's mean
            img = np.expand_dims(img, axis=0)
            try:
                if use_gpu:
                    inputs = torch.autograd.Variable(
                        torch.from_numpy(img).cuda().float())
                else:
                    inputs = torch.autograd.Variable(
                        torch.from_numpy(img).float())
                d_hist = vgg_model(inputs)[pick_layer]
                d_hist = np.sum(d_hist.data.cpu().numpy(), axis=0)
                d_hist /= np.sum(d_hist)  # normalize
                samples.append({
                    'img_idx':  idx,
                    'hist': d_hist
                })        
            except:
                pass
        # cPickle.dump(samples, open(os.path.join(cache_dir, sample_cache), "wb", True))

        return samples, init_time
