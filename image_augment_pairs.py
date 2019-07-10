'''
This is for image, label pair augmentation.

The image, label should be augmented equally. And the data type and distribution
should not change before and after augmenatation.

Don't use scipy.misc.imresize to resize an image, cause it will convert data type 
from float32 to unit8, use cv2 instead. Check the data distribution before and after
augmentation.
'''
import numpy as np
import cv2
from scipy.ndimage.interpolation import rotate


def subtract(image):
    image = image / 255
    return image

def normalize(image):
    return image

def center_crop(image, crop_size):
    h, w, _ = image.shape
    top = (h - crop_size[0]) // 2
    left = (w - crop_size[1]) // 2
    bottom = top + crop_size[0]
    right = left + crop_size[1]
    image = image[top:bottom, left:right, :]
    return image


def random_crop(image, label, crop_size, rate=0.5, resized=False):
    if np.random.rand() < rate or resized:
      h, w, _ = image.shape
      top = np.random.randint(0, h - crop_size[0])
      left = np.random.randint(0, w - crop_size[1])
      bottom = top + crop_size[0]
      right = left + crop_size[1]
      image = image[top:bottom, left:right, :]
      label = label[top:bottom, left:right]
    return image, label


def horizontal_flip(image, label, rate=0.5):
    if np.random.rand() < rate:
        image = image[:, ::-1, :]
        label = label[:, ::-1]
    return image, label


def vertical_flip(image, label, rate=0.5):
    if np.random.rand() < rate:
        image = image[::-1, :, :]
        label = label[::-1, :]
    return image, label


def scale_augmentation(image, label, scale_range=(256, 400), crop_size=(256, 256), rate=0.5):
    if np.random.rand() < rate:
      scale_size = np.random.randint(*scale_range)
      image = cv2.resize(image, (scale_size, scale_size))
      label = cv2.resize(label, (scale_size, scale_size))
      image, label = random_crop(image, label, crop_size, resized=True)
    return image, label


def random_rotation(image, label, angles=[90, -90], rate=0.5):
    if np.random.rand() < rate:
      h, w, _ = image.shape
      angle = np.random.choice(angles)
      image = rotate(image, angle)
      label = rotate(label, angle)
      image = cv2.resize(image, (h, w))
      label = cv2.resize(label, (h, w))
    return image, label

# only for input image
def cutout(image_origin, mask_size, mask_value='mean', rate=0.5):
    image = np.copy(image_origin)
    if np.random.rand() < rate:
      if mask_value == 'mean':
          mask_value = image.mean()
      elif mask_value == 'random':
          mask_value = np.random.randint(0, 256)

      h, w, _ = image.shape
      top = np.random.randint(0 - mask_size // 2, h - mask_size)
      left = np.random.randint(0 - mask_size // 2, w - mask_size)
      bottom = top + mask_size
      right = left + mask_size
      if top < 0:
          top = 0
      if left < 0:
          left = 0
      image[top:bottom, left:right, :].fill(mask_value)
    return image


# only for input image
def random_erasing(image_origin, p=0.5, s=(0.02, 0.4), r=(0.3, 3), mask_value='random', rate=0.5):
    image = np.copy(image_origin)
    if np.random.rand() < rate:
      if np.random.rand() > p:
          return image
      if mask_value == 'mean':
          mask_value = image.mean()
      elif mask_value == 'random':
          mask_value = np.random.randint(0, 256)

      h, w, _ = image.shape
      mask_area = np.random.randint(h * w * s[0], h * w * s[1])
      mask_aspect_ratio = np.random.rand() * r[1] + r[0]
      mask_height = int(np.sqrt(mask_area / mask_aspect_ratio))
      if mask_height > h - 1:
          mask_height = h - 1
      mask_width = int(mask_aspect_ratio * mask_height)
      if mask_width > w - 1:
          mask_width = w - 1

      top = np.random.randint(0, h - mask_height)
      left = np.random.randint(0, w - mask_width)
      bottom = top + mask_height
      right = left + mask_width
      image[top:bottom, left:right, :].fill(mask_value)
    return image

