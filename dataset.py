import os
from tqdm.auto import tqdm
import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data import WeightedRandomSampler
from utils import load_dicom, get_transforms

tqdm.pandas()


class CustomDatasetInference(Dataset):
    def __init__(self, dataframe, transform=None, label_name='target'):
        self.dataframe = dataframe
        self.transform = transform
        self.label = dataframe.loc[:, label_name]

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        image_path = self.dataframe['image_path'][index]
        image = load_dicom(image_path)  # Define this function to load your DICOM images
        target = self.dataframe['target'][index]

        if self.transform:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            image = self.transform(image=image)['image']
            image = image.transpose(2, 0, 1).astype(np.float32) / 255.

        return image, torch.tensor(target).float()

    def get_labels(self):
        return self.label


class CustomDataset(Dataset):
    def __init__(self, dataframe, transform=None, label_name='target'):
        self.dataframe = dataframe
        self.transform = transform
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


def get_dataloaders(data, cfg, split="train"):
    img_size = cfg['img_size']
    height, width = img_size, img_size
    tsfm = get_transforms(height=height, width=width)
    if split == 'train':
        tr_tsfm = tsfm['train']
        ds = CustomDataset(data, transform=tr_tsfm)
        labels = ds.get_labels()
        #         class_weights = torch.tensor(compute_class_weight(class_weight="balanced", classes=np.unique(labels), y=labels))
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

    elif split == 'valid' or split == 'test':
        eval_tsfm = tsfm['eval']
        ds = CustomDataset(data, transform=eval_tsfm)
        dls = DataLoader(
            ds,
            batch_size=2 * cfg['batch_size'],
            shuffle=False,
            num_workers=os.cpu_count(),
            drop_last=False,
            pin_memory=True
        )
    else:
        raise Exception("Split should be 'train' or 'valid' or 'test'!!!")
    return dls
