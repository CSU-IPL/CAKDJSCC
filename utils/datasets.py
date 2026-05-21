from torch.utils.data import Dataset
from PIL import Image
# import cv2
import os
import numpy as np
from glob import glob
from torchvision import transforms, datasets
from torch.utils.data.dataset import Dataset
import torch
import math
import torch.utils.data as data

NUM_DATASET_WORKERS = 8
SCALE_MIN = 0.75
SCALE_MAX = 0.95


class HR_image(Dataset):
    files = {"train": "train", "test": "test", "val": "validation"}

    def __init__(self, img_size, data_dir, transform_=None):
        self.imgs = []

        self.imgs += glob(os.path.join(data_dir, '*.jpg'))
        self.imgs += glob(os.path.join(data_dir, '*.png'))
        self.imgs += glob(os.path.join(data_dir, '*.JPEG'))
        self.im_height = self.im_width = img_size
        self.crop_size = self.im_height
        self.image_dims = (3, self.im_height, self.im_width)
        self.transform = self._transforms()
        self.transform_ = transform_

    def _transforms(self, ):
        """
        Up(down)scale and randomly crop to `crop_size` x `crop_size`
        """
        transforms_list = [
            transforms.RandomCrop((self.im_height, self.im_width)),
            transforms.ToTensor()]

        return transforms.Compose(transforms_list)

    def __getitem__(self, idx):
        img_path = self.imgs[idx]
        img = Image.open(img_path)
        img = img.convert('RGB')
        sample = self.transform(img)
        if self.transform_:
            aug_sample = self.transform_(img)
            return sample, aug_sample
        else:
            return sample

    def __len__(self):
        return len(self.imgs)


class Datasets(Dataset):
    def __init__(self, data_dir):
        self.imgs = []

        self.imgs += glob(os.path.join(data_dir, '*.jpg'))
        self.imgs += glob(os.path.join(data_dir, '*.png'))
        self.imgs += glob(os.path.join(data_dir, '*.JPEG'))
        self.imgs.sort()

    def __getitem__(self, item):
        image_ori = self.imgs[item]
        image = Image.open(image_ori).convert('RGB')
        self.im_height, self.im_width = image.size
        if self.im_height % 128 != 0 or self.im_width % 128 != 0:
            self.im_height = self.im_height - self.im_height % 128
            self.im_width = self.im_width - self.im_width % 128
        self.transform = transforms.Compose([
            transforms.CenterCrop((self.im_width, self.im_height)),
            transforms.ToTensor()])
        img = self.transform(image)
        return img

    def __len__(self):
        return len(self.imgs)


def worker_init_fn_seed(worker_id):
    seed = 10
    seed += worker_id
    np.random.seed(seed)


def get_loader(args, num_workers=None, transform_=None):
    train_dataset = HR_image(args.input_size, args.train_data_path, transform_)
    test_dataset = Datasets(args.test_data_path)

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               num_workers=NUM_DATASET_WORKERS if num_workers is None else num_workers,
                                               pin_memory=True,
                                               batch_size=args.batch_size,
                                               worker_init_fn=worker_init_fn_seed,
                                               shuffle=True,
                                               drop_last=True)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=1,
                                              shuffle=False)

    return train_loader, test_loader
