import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from image_augment import *

class RoomDataset(Dataset):
    def __init__(self, file_path, train=True, augment=False):
        self.file_path = file_path
        self.augment = augment
        self.train = train

        self.img_list = [f for f in os.listdir(file_path) if 'image' in f]
        self.label_list = [f.replace('image', 'room') for f in self.img_list]

    def __len__(self):
        return len(self.img_list)

    # from ndarray to pytorch tensor
    def _to_tensor(self, array, is_label=False):
        assert (isinstance(array, np.ndarray))
        tensor = torch.from_numpy(array)
        return tensor.long() if is_label else tensor.float()

    def __getitem__(self, index):
        
        image = np.load(os.path.join(self.file_path, self.img_list[index])).transpose(1,2,0)
        label = np.load(os.path.join(self.file_path, self.label_list[index]))

        height, width = label.shape

        if self.train and self.augment:
          # random rotations
          random_rotation

          # random h-flips

          # random v-flips

          # random crops

        return self._to_tensor(image).permute(2,0,1), self._to_tensor(label, is_label=True)
