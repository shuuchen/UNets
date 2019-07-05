import os
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
        for img in [f for f in os.listdir(self.file_path) if 'image' in f]:
            self.img_list += [img]
            self.label_list += [img.replace('image', 'room')]

    def __len__(self):
        return len(self.img_list)

    # convert PIL image to ndarray
    def _pil2np(img):
        if isinstance(img, Image.Image):
            img = np.asarray(img)
        return img

    # convert ndarray to PIL image
    def _np2pil(img):
        if isinstance(img, np.ndarray):
            if img.dtype != np.uint8:
                img = img.astype(np.uint8)
            img = F.to_pil_image(img)
        return img

    def __getitem__(self, index):
        
        image = np.load(os.path.join(self.file_path, self.img_list[index]))
        label = np.load(os.path.join(self.file_path, self.label_list[index]))

        height, width = image.shape[1:]
        ch_label = label[2]

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

        return image, label
