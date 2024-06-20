import os

import cv2
import numpy as np
from tqdm.auto import tqdm
import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data import WeightedRandomSampler

from config import CONFIG
from utils import load_dicom
import torchvision.transforms as transforms
import albumentations as A
tqdm.pandas()


class CustomDatasetInference(Dataset):
    def __init__(self, dataframe, size, label_name='target'):
        height, width = size
        self.dataframe = dataframe

        # self.transform = transforms.Compose([
        #     transforms.Lambda(lambda x: (x * 255).astype(np.uint8)),  # Convert back to uint8 for PIL
        #     transforms.ToPILImage(),
        #     transforms.Resize((height, width)),
        #     transforms.Grayscale(num_output_channels=3),
        #     transforms.ToTensor(),
        # ])
        self.transform = A.Compose([
            A.Resize(height, width),
            A.Normalize(mean=0.5, std=0.5),
            A.ToRGB()
        ])

        self.label = dataframe.loc[:, label_name]

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        image_path = self.dataframe['image_path'][index]
        image = load_dicom(image_path)  # Define this function to load your DICOM images
        target = self.dataframe['target'][index]

        if self.transform:
            # image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            image = self.transform(image=image)['image']
            # print("valid", image.shape)
            image = image.transpose(2, 0, 1).astype(np.float32) / 255.

        return image, torch.tensor(target).float()

    def get_labels(self):
        return self.label


class CustomDataset(Dataset):
    def __init__(self, dataframe, size, label_name='target'):
        height, width = size
        self.dataframe = dataframe
        # self.transform = transforms.Compose([
        #     transforms.Lambda(lambda x: (x * 255).astype(np.uint8)),  # Convert back to uint8 for PIL
        #     transforms.ToPILImage(),
        #     transforms.Resize((height, width)),
        #     transforms.Grayscale(num_output_channels=3),
        #     transforms.RandomHorizontalFlip(p=0.5),
        #     transforms.RandomRotation(degrees=(0, 30)),
        #     transforms.ToTensor(),
        # ])
        AUG_PROB = CONFIG["AUG_PROB"]
        self.transform = A.Compose([
            A.RandomBrightnessContrast(brightness_limit=(-0.2, 0.2), contrast_limit=(-0.2, 0.2), p=AUG_PROB),
            A.OneOf([
                A.MotionBlur(blur_limit=5),
                A.MedianBlur(blur_limit=5),
                A.GaussianBlur(blur_limit=5),
                A.GaussNoise(var_limit=(5.0, 30.0)),
            ], p=AUG_PROB),

            A.OneOf([
                A.OpticalDistortion(distort_limit=1.0),
                A.GridDistortion(num_steps=5, distort_limit=1.),
                A.ElasticTransform(alpha=3),
            ], p=AUG_PROB),

            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, border_mode=0, p=AUG_PROB),
            A.Resize(height, width),
            A.CoarseDropout(max_holes=16, max_height=64, max_width=64, min_holes=1, min_height=8, min_width=8,
                            p=AUG_PROB),
            A.Normalize(mean=0.5, std=0.5),
            A.ToRGB()
        ])
        self.label = dataframe.loc[:, label_name]

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        image_path = self.dataframe['image_path'][index]
        image = load_dicom(image_path)  # Define this function to load your DICOM images
        target = self.dataframe['target'][index]

        if self.transform:
            image = self.transform(image=image)['image']
            image = image.transpose(2, 0, 1).astype(np.float32) / 255.

        return image, torch.tensor(target).float()

    def get_labels(self):
        return self.label


def get_train_dataloaders(data, cfg):
    img_size = cfg['img_size']
    size = (img_size, img_size)
    ds = CustomDataset(data, size)
    labels = ds.get_labels()
    class_weights = torch.tensor([1, 2, 4])
    samples_weights = class_weights[labels]
    #         print(class_weights)
    sampler = WeightedRandomSampler(weights=samples_weights,
                                    num_samples=len(samples_weights),
                                    replacement=True)

    dls = DataLoader(ds,
                     batch_size=cfg['batch_size'],
                     sampler=sampler,
                     num_workers=os.cpu_count(),
                     drop_last=True,
                     pin_memory=True)

    return dls


def get_test_dataloaders(data, cfg):
    img_size = cfg['img_size']
    size = (img_size, img_size)

    ds = CustomDatasetInference(data, size)
    dls = DataLoader(
        ds,
        batch_size=2 * cfg['batch_size'],
        shuffle=False,
        num_workers=os.cpu_count(),
        drop_last=False,
        pin_memory=True
    )
    return dls
