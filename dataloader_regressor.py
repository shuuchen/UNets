import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.transforms import functional as F
from torchvision import transforms
from image_augment_pairs import *

class RoomDataset(Dataset):
    def __init__(self, file_path, train=True, augment=False):
        self.file_path = file_path
        self.augment = augment
        self.train = train

        self.img_list = []
        self.label_list = []
        
        with open(file_path) as f:
            self.list = f.readlines()
        f.close()
        
        self.list = [l[:-1] for l in self.list]
        
        self.img_dir = '../../data/image'
        self.label_dir = '../../data/height_arr'

    def __len__(self):
        return len(self.list)

    # convert PIL image to ndarray
    def _pil2np(self, img):
        if isinstance(img, Image.Image):
            img = np.asarray(img)
        return img

    # convert ndarray to PIL image
    def _np2pil(self, img):
        if isinstance(img, np.ndarray):
            if img.dtype != np.uint8:
                img = img.astype(np.uint8)
            img = F.to_pil_image(img)
        return img
    
    def _to_tensor(self, array):
        assert (isinstance(array, np.ndarray))
        # handle numpy array
        try:
            tensor = torch.from_numpy(array).permute(2, 0, 1)
        except:
            tensor = torch.from_numpy(np.expand_dims(array, axis=2)).permute(2, 0, 1)
        # put it from HWC to CHW format
        return tensor.float()

    def __getitem__(self, index):
        
        image = cv2.imread(os.path.join(self.img_dir, self.list[index] + '.png'))
        label = np.load(os.path.join(self.label_dir, self.list[index] + '.npy'))

        height, width = label.shape
        
        if self.train and self.augment:
            
          # random rotations
          random_rotation(image, label)

          # random h-flips
          horizontal_flip(image, label)

          # random v-flips
          vertical_flip(image, label)

          # random crops
          #scale_augmentation(image, label)

        return self._to_tensor(image), self._to_tensor(label)
