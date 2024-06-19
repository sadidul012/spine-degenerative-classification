import os
from tqdm.auto import tqdm
import pandas as pd
import numpy as np
import torch
import cv2
import warnings

from pathlib import Path
import random
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import pydicom
import torch.nn as nn
import timm

# replace this parts with the scripts
CONFIG = dict(
    project_name="PL-RSNA-2024-Lumbar-Spine-Classification",
    weights_path="data/efficientnet_b0-0.pth",
    # data_path="/kaggle/input/rsna-2024-lumbar-spine-degenerative-classification",
    data_path="/mnt/Cache/rsna-2024-lumbar-spine-degenerative-classification",
    artifact_name="rsnaEffNetModel",
    load_kernel=None,
    load_last=True,
    n_folds=5,
    backbone="efficientnet_b0",  # tf_efficientnetv2_s_in21ft1k
    # backbone="efficientnet_b0.ra_in1k",  # tf_efficientnetv2_s_in21ft1k
    img_size=384,
    n_slice_per_c=16,
    in_chans=1,

    drop_rate=0.,
    drop_rate_last=0.3,
    drop_path_rate=0.,
    p_mixup=0.5,
    p_rand_order_v1=0.2,
    lr=1e-5,

    out_dim=3,
    epochs=2,
    batch_size=8,
    device=torch.device("cuda") if torch.cuda.is_available() else "cpu",
    seed=2024,

    patience=7,
)
DATA_PATH = Path(CONFIG["data_path"])


def seeding(SEED):
    np.random.seed(SEED)
    random.seed(SEED)
    os.environ['PYTHONHASHSEED'] = str(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    #     os.environ['TF_CUDNN_DETERMINISTIC'] = str(SEED)
    #     tf.random.set_seed(SEED)
    #     keras.utils.set_random_seed(seed=SEED)
    print('seeding done!!!')


def get_image_paths(row, base_path=None):
    if base_path is None:
        base_path = f"{str(DATA_PATH)}/train_images/"
    series_path = os.path.join(base_path, str(row['study_id']), str(row['series_id']))
    if os.path.exists(series_path):
        return [
            os.path.join(series_path, f) for f in os.listdir(series_path) if
            os.path.isfile(os.path.join(series_path, f))
        ]
    return []


def load_dicom(path):
    dicom = pydicom.read_file(path)
    data = dicom.pixel_array
    data = data - np.min(data)
    if np.max(data) != 0:
        data = data / np.max(data)
    data = (data * 255).astype(np.uint8)
    return data


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


class TimmModel(nn.Module):
    def __init__(self, backbone, pretrained=False):
        super(TimmModel, self).__init__()

        self.encoder = timm.create_model(
            backbone,
            num_classes=CONFIG["out_dim"],
            features_only=False,
            drop_rate=CONFIG["drop_rate"],
            drop_path_rate=CONFIG["drop_path_rate"],
            pretrained=pretrained
        )
        hdim = 1
        if 'efficient' in backbone:
            hdim = self.encoder.conv_head.out_channels
            self.encoder.classifier = nn.Identity()
        elif 'convnext' in backbone:
            hdim = self.encoder.head.fc.in_features
            self.encoder.head.fc = nn.Identity()

        self.lstm = nn.LSTM(hdim, 256, num_layers=2, dropout=CONFIG["drop_rate"], bidirectional=True, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.Dropout(CONFIG["drop_rate_last"]),
            nn.LeakyReLU(0.1),
            nn.Linear(256, CONFIG["out_dim"]),
        )

    def forward(self, x):
        feat = self.encoder(x)
        feat, _ = self.lstm(feat)
        feat = self.head(feat)
        return feat

    def save_state_dict(self, path):
        torch.save(self.state_dict(), path)

    def load_state_dict_from_path(self, path):
        weights = torch.load(path)
        self.load_state_dict(weights)


warnings.filterwarnings("ignore")
cv2.setNumThreads(0)
tqdm.pandas()

seeding(CONFIG['seed'])
os.listdir(DATA_PATH)
sample_df = pd.read_csv(DATA_PATH / "sample_submission.csv")
test_desc = pd.read_csv(DATA_PATH / "test_series_descriptions.csv")
# define the base path for test images
base_path = f"{str(DATA_PATH)}/test_images"

# Mapping of series_description to conditions
condition_mapping = {
    'Sagittal T1': {'left': 'left_neural_foraminal_narrowing', 'right': 'right_neural_foraminal_narrowing'},
    'Axial T2': {'left': 'left_subarticular_stenosis', 'right': 'right_subarticular_stenosis'},
    'Sagittal T2/STIR': 'spinal_canal_stenosis'
}

# Create a list to store the expanded rows
expanded_rows = []

# Expand the dataframe by adding new rows for each file path
for index, row in test_desc.iterrows():
    image_paths = get_image_paths(row, base_path)
    conditions = condition_mapping.get(row['series_description'], {})
    if isinstance(conditions, str):  # Single condition
        conditions = {'left': conditions, 'right': conditions}
    for side, condition in conditions.items():
        for image_path in image_paths:
            expanded_rows.append({
                'study_id': row['study_id'],
                'series_id': row['series_id'],
                'series_description': row['series_description'],
                'image_path': image_path,
                'condition': condition,
                'row_id': f"{row['study_id']}_{condition}"
            })

# Create a new dataframe from the expanded rows
expanded_test_desc = pd.DataFrame(expanded_rows)
# print(expanded_test_desc.to_string())
test_data = expanded_test_desc.copy()
test_data['target'] = 0
test_data.head()
label2id = {"Normal/Mild": 0, "Moderate": 1, "Severe": 2}
id2label = {v: k for k, v in label2id.items()}

FLIPS = [None, [-1], [-2], [-2, -1]]


def inference_loop(model, loader):
    model.to(CONFIG["device"])
    model.eval()
    preds = np.empty((0, 3))
    with torch.no_grad():
        for batch in tqdm(loader):
            images, labels = batch
            images = images.to(CONFIG["device"], non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                #                 logits = model(images.to(torch.float32))
                logits = model(images)
                #                 logits = logits.mean(axis=1).softmax(dim=-1)
                logits = logits.softmax(dim=-1)
                preds = np.concatenate([preds, logits.detach().cpu().numpy()])
    np.save('preds.npy', preds)


def tta_inference_loop(model, loader):
    model.to(CONFIG["device"])
    model.eval()
    preds = np.empty((0, 3))
    with torch.no_grad():
        for batch in tqdm(loader):
            images, labels = batch
            images = images.to(CONFIG["device"], non_blocking=True)
            pred_tta = []
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                for f in FLIPS:
                    logits = model(torch.flip(images, f) if f is not None else images)
                    logits = logits.softmax(dim=-1)
                    pred_tta.append(logits.detach().cpu().numpy())
                #                 preds = np.concatenate([preds, logits.detach().cpu().numpy()])
                preds = np.concatenate([preds, np.mean(pred_tta, 0)])
    np.save('preds.npy', preds)


#     return preds


weights_path = CONFIG["weights_path"]
weights = torch.load(weights_path, map_location=torch.device("cpu"))
model = TimmModel(backbone=CONFIG["backbone"], pretrained=False)
model.load_state_dict(weights)

dls = get_test_dataloaders(test_data, CONFIG)
# inference_loop(model, dls)
tta_inference_loop(model, dls)
# _ = Parallel(n_jobs=mp.cpu_count())(
#     delayed(inference_loop(model, dls))
# )

preds = np.load('preds.npy')
levels = ['l1_l2', 'l2_l3', 'l3_l4', 'l4_l5', 'l5_s1']


# Function to update row_id with levels
def update_row_id(row, levels):
    level = levels[row.name % len(levels)]
    return f"{row['study_id']}_{row['condition']}_{level}"


# print(expanded_test_desc.to_string())
# Update row_id in expanded_test_desc to include levels
expanded_test_desc['row_id'] = expanded_test_desc.apply(lambda row: update_row_id(row, levels), axis=1)
expanded_test_desc[["normal_mild", "moderate", "severe"]] = preds

final_df = expanded_test_desc[["row_id", "normal_mild", "moderate", "severe"]]

target_cols = sample_df.columns.tolist()
final_df = final_df.groupby('row_id').sum().reset_index()
# normalize the columns
final_df[target_cols[1:]] = final_df[target_cols[1:]].div(final_df[target_cols[1:]].sum(axis=1), axis=0)
final_df[target_cols].to_csv('submission.csv', index=False)
print(pd.read_csv('submission.csv').to_string())
