import numpy as np
import imageio
import os
import time
from tqdm import tqdm
from skimage.feature import hog
from skimage import color

# cell size: the dimensions of these cells
# block size: dimentions of blocks which are formed by grouping cells after histogram computation
# block stride: thestep size or overlap between neighboring blocks

n_bin    = 10  # the number of orientation bins or divisions
n_slice  = 6  # the number of image slices, 6 equal-sized vertical slices
n_orient = 8  # the number of grandient orientations considered for constructing the histogram
p_p_c    = (2, 2)  # pixel per cell
c_p_b    = (1, 1)  # cell per block
h_type   = 'region'  # computer over a region of interest instead of global


class HOG(object):
    def histogram(self, input, n_bin=n_bin, type=h_type, n_slice=n_slice, normalize=True):
        if isinstance(input, np.ndarray):# examinate input type
            img = input.copy()
        else:
            print("img is not nd.array")
            img = imageio.imread(input, pilmode='RGB')
        height, width, channel = img.shape

        if type == 'global':
            hist = self._HOG(img, n_bin)

        elif type == 'region':
            hist = np.zeros((n_slice, n_slice, n_bin))
            h_silce = np.around(np.linspace(0, height, n_slice+1, endpoint=True)).astype(int)
            w_slice = np.around(np.linspace(0, width, n_slice+1, endpoint=True)).astype(int)

            for hs in range(len(h_silce)-1):
                for ws in range(len(w_slice)-1):
                    img_r = img[h_silce[hs]:h_silce[hs+1], w_slice[ws]:w_slice[ws+1]]  # slice img to regions
                    hist[hs][ws] = self._HOG(img_r, n_bin)

        if normalize:
            hist /= np.sum(hist)

        return hist.flatten()

    def _HOG(self, img, n_bin, normalize=True):
        image = color.rgb2gray(img)
        fd = hog(image, orientations=n_orient, pixels_per_cell=p_p_c, cells_per_block=c_p_b)
        bins = np.linspace(0, np.max(fd), n_bin+1, endpoint=True)
        hist, _ = np.histogram(fd, bins=bins)

        if normalize:
            hist = np.array(hist) / np.sum(hist)

        return hist

    def make_samples(self, frame_set, video_path):
        start_time = time.time()
        video_name = os.path.basename(video_path)
        print("Counting histogram..., video_index=%s, model=HOG" % (video_name))

        samples = []
        for idx in tqdm(frame_set):
            img = frame_set[idx]
            hist = self.histogram(img, type=h_type, n_slice=n_slice)
            samples.append({
                            'img_idx': idx, 
                            'hist': hist
                            })
            exe_time = time.time() - start_time
        return samples, exe_time