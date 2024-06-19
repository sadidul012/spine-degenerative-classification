import os

import cv2
import numpy as np
from tqdm.auto import tqdm
import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data import WeightedRandomSampler
from utils import load_dicom
import torchvision.transforms as transforms

tqdm.pandas()


class CustomDatasetInference(Dataset):
    def __init__(self, dataframe, size, label_name='target'):
        height, width = size
        self.dataframe = dataframe
        self.transform = transforms.Compose([
            transforms.Lambda(lambda x: (x * 255).astype(np.uint8)),  # Convert back to uint8 for PIL
            transforms.ToPILImage(),
            transforms.Resize((height, width)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
        ])
        self.label = dataframe.loc[:, label_name]

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        image_path = self.dataframe['image_path'][index]
        image = load_dicom(image_path)  # Define this function to load your DICOM images
        target = self.dataframe['target'][index]

        if self.transform:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            image = self.transform(image)
            # print(image.shape)
            # image = image.transpose(2, 0, 1).astype(np.float32) / 255.

        return image, torch.tensor(target).float()

    def get_labels(self):
        return self.label


class CustomDataset(Dataset):
    def __init__(self, dataframe, size, label_name='target'):
        height, width = size
        self.dataframe = dataframe
        self.transform = transforms.Compose([
            transforms.Lambda(lambda x: (x * 255).astype(np.uint8)),  # Convert back to uint8 for PIL
            transforms.ToPILImage(),
            transforms.Resize((height, width)),
            transforms.Grayscale(num_output_channels=3),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=(0, 30)),
            transforms.ToTensor(),
        ])
        self.label = dataframe.loc[:, label_name]

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        image_path = self.dataframe['image_path'][index]
        image = load_dicom(image_path)  # Define this function to load your DICOM images
        target = self.dataframe['target'][index]

        if self.transform:
            image = self.transform(image)

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
