import os
import torch
import glob
from PIL import Image
import torch.utils.data as data
import torchvision.transforms as transforms
import numpy as np
import random
from pix2pixHD.data.base_dataset import *

def CreateDataLoader(opt):
    data_loader = CustomDatasetDataLoader()
    print(data_loader.name())
    data_loader.initialize(opt)
    return data_loader

class CustomDatasetDataLoader():
    def __init__(self):
        pass

    def name(self):
        return 'CustomDatasetDataLoader'

    def initialize(self, opt):
        self.opt = opt
        self.dataset = CustomDataset()
        self.dataset.initialize(opt)
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=opt.batchSize,
            shuffle=not opt.serial_batches,
            num_workers=int(opt.nThreads))

    def load_data(self):
        return self.dataloader

    def __len__(self):
        return min(len(self.dataset), self.opt.max_dataset_size)

class CustomDataset():
    def __init__(self):
        pass

    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot    

        ### poses
        self.pose_dir = os.path.join(opt.dataroot, 'poses')
        self.pose_paths = sorted(glob.glob(self.pose_dir + "/*.jpg"), key=lambda x: int(x.split('/')[-1].split('_')[0][5:]))

        ### input B (real images)
        if opt.isTrain:
            self.gt_dir = os.path.join(opt.dataroot, 'frames')  
            self.gt_paths = sorted(glob.glob(self.gt_dir + "/*.jpg"), key=lambda x: int(x.split('/')[-1].split('.')[0][5:]))
            assert(len(self.gt_paths) == len(self.pose_paths))

        self.dataset_size = len(self.pose_paths) - 1

    def __getitem__(self, index):
        assert(index < self.dataset_size)
        ### poses
        pose1_path = self.pose_paths[index]
        pose2_path = self.pose_paths[index+1]
        pose1 = Image.open(pose1_path)
        pose2 = Image.open(pose2_path)
        params = get_params(self.opt, pose1.size)

        transform_pose = get_transform(self.opt, params, method=Image.NEAREST, normalize=False)
        pose1_tensor = transform_pose(pose1) * 255.0
        pose2_tensor = transform_pose(pose2) * 255.0

        gt1_tensor = 0
        gt2_tensor = 0
        ### ground truth images
        if self.opt.isTrain:
            gt1_path = self.gt_paths[index]
            gt2_path = self.gt_paths[index+1] 
            gt1 = Image.open(gt1_path).convert('RGB')
            gt2 = Image.open(gt2_path).convert('RGB')

            params = get_params(self.opt, gt1.size)
            transform_gt = get_transform(self.opt, params)      
            gt1_tensor = transform_gt(gt1)
            gt2_tensor = transform_gt(gt2)             

        input_dict = {'label': torch.cat((pose1_tensor, pose2_tensor), dim=0), 'image': torch.cat((gt1_tensor, gt2_tensor), dim=0), 'path': pose1_path}

        return input_dict

    def __len__(self):
        return self.dataset_size // self.opt.batchSize * self.opt.batchSize

    def name(self):
        return 'CustomDataset'

