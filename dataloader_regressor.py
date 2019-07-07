import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.transforms import functional as F
from torchvision import transforms

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
        ch_label = 1

        if self.train and self.augment:
          # random rotations
          if np.random.randint(2) == 0:
              ang = np.random.choice([90, -90])
              image = np.dstack([F.rotate(_np2pil(image[:, :, i]), ang) for i in range(3)])
              label = np.dstack([F.rotate(_np2pil(label[:, :, i]), ang) for i in range(ch_label)])

          # random h-flips
          if np.random.randint(2) == 0:
              image = np.dstack([F.hflip(_np2pil(image[:, :, i])) for i in range(3)])
              label = np.dstack([F.hflip(_np2pil(label[:, :, i])) for i in range(ch_label)])

          # random v-flips
          if np.random.randint(2) == 0:
              image = np.dstack([F.vflip(_np2pil(image[:, :, i])) for i in range(3)])
              label = np.dstack([F.vflip(_np2pil(label[:, :, i])) for i in range(ch_label)])

          # random crops
          if np.random.randint(2) == 0:
              i, j, h, w = transforms.RandomCrop.get_params(_np2pil(label), output_size=(height//2, width//2))
              image = np.dstack([F.resized_crop(_np2pil(image[:, :, ii]), i, j, h, w, (height, width)) for ii in range(3)])
              label = np.dstack([F.resized_crop(_np2pil(label[:, :, ii]), i, j, h, w, (height, width)) for ii in range(ch_label)])

        return self._to_tensor(image), self._to_tensor(label)
