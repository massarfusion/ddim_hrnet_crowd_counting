# Copyright 2021 Tencent

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================
import os
import random
import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms.functional as F
import h5py
import cv2
import glob
from time import sleep

from scipy.ndimage import gaussian_filter
from torch import nn
from scipy.io import loadmat


class Shanghaitech(Dataset):
    def __init__(self, root_path, mode="train", transform=None):
        """
        Args:
            root_path:  [Access definition place to have best view experience]
                [root_path]
                    ├─test_data
                    │  ├─ground-truth
                    │  │      GT_IMG_1.mat
                    │  │      GT_IMG_10.mat
                    │  │      GT_IMG_100.mat
                    │  │      ...
                    │  └─images
                    │          IMG_1.jpg
                    │          IMG_10.jpg
                    │          ...
                    └─train_data
                        ├─ground-truth
                        │      GT_IMG_1.mat
                        │      GT_IMG_10.mat
                        │      ...
                        └─images
                                IMG_1.jpg
                                ...
            
            mode: either "train" or "test", captial letter or not dont matter
            
            transform: it is Shanghai tech dataset, copy some commonly used transforms combination and here we are.
            
        """
        if mode.lower() != "train" or mode.lower() != "test":
            mode = "train"
            print("WARNING, dataset mode set to default training set mode")
        mode = mode.lower()
        img_list = sorted(glob.glob(os.path.join(root_path, mode + '_data', 'images', '*.jpg')))
        self.nSamples = len(img_list)
        self.lines = img_list
        self.transform = transform
        
        kernel_size_list, sigma_list = get_kernel_and_sigma_list()
        self.kernel_list = []
        self.kernel_list = [create_density_kernel(kernel_size_list[index], sigma_list[index]) for index in range(len(sigma_list))]  # a bunch of kernals, wrapped in conv2d , re-written conv2d weight matrix
        self.normalizer = [kernel.max() for kernel in self.kernel_list]
        self.kernel_list = [GaussianKernel(kernel, 'cpu') for kernel in self.kernel_list]
    
    def __len__(self):
        return self.nSamples
    
    def __getitem__(self, index):
        assert index <= len(self), 'index range error'
        # get the image path
        img_path = self.lines[index]
        
        # img, target = load_data(img_path)
        
        # load the images and locations
        img = Image.open(img_path).convert('RGB')
        # image = np.asarray(image).astype(np.uint8)
        file = img_path.replace('images', 'ground-truth').replace('IMG', 'GT_IMG').replace('jpg', 'mat')
        locations = loadmat(file)['image_info'][0][0]['location'][0][0]
        # create dot map DONE
        density = create_dot_map(locations, np.asarray(img).astype(np.uint8).shape)
        density = torch.tensor(density)
        
        density = density.unsqueeze(0).unsqueeze(0)
        density_maps = [kernel(density) for kernel in self.kernel_list]
        density = torch.stack(density_maps).detach()  # B,H,W
        
        # perform data augumentation
        if self.transform is not None:
            img = self.transform(img)
        
        # img = torch.Tensor(img)
        # target = torch.Tensor(density)
        
        return density, img


def get_kernel_and_sigma_list():
    kernel_list = [3]
    sigma_list = [0.5]
    
    return kernel_list, sigma_list


class GaussianKernel(nn.Module):
    
    def __init__(self, kernel_weights, device):
        super().__init__()
        self.kernel = nn.Conv2d(1, 1, kernel_weights.shape, bias=False, padding=kernel_weights.shape[0] // 2)
        kernel_weights = torch.tensor(kernel_weights).unsqueeze(0).unsqueeze(0)
        with torch.no_grad():
            self.kernel.weight = nn.Parameter(kernel_weights)
    
    def forward(self, density):
        return self.kernel(density).squeeze()


def create_density_kernel(kernel_size, sigma):
    kernel = np.zeros((kernel_size, kernel_size))
    mid_point = kernel_size // 2
    kernel[mid_point, mid_point] = 1
    kernel = gaussian_filter(kernel, sigma=sigma)
    
    return kernel


def create_dot_map(locations, image_size):
    density = np.zeros(image_size[:-1])
    for x, y in locations:
        x, y = int(x), int(y)
        density[y, x] = 1.
    
    return density

# def load_data(img_path):
#     # get the path of the ground truth
#     gt_path = img_path.replace('.jpg', '_sigma4.h5').replace('images', 'ground_truth')
#     # open the image
#     img = Image.open(img_path).convert('RGB')
#     # load the ground truth
#     while True:
#         try:
#             gt_file = h5py.File(gt_path)
#             break
#         except:
#             sleep(2)
#     target = np.asarray(gt_file['density'])
#
#     return img, target
