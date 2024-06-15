import os
from tqdm.auto import tqdm
import pandas as pd
import numpy as np
import torch
import cv2
import warnings

# replace this parts with the scripts
from config import CONFIG
from utils import get_image_paths, seeding, DATA_PATH
from dataset import get_dataloaders
from models import TimmModel

warnings.filterwarnings("ignore")
cv2.setNumThreads(0)
tqdm.pandas()

seeding(CONFIG['seed'])
os.listdir(DATA_PATH)
sample_df = pd.read_csv(DATA_PATH / "sample_submission.csv")
test_desc = pd.read_csv(DATA_PATH / "test_series_descriptions.csv")
train_desc = pd.read_csv(DATA_PATH / "train_series_descriptions.csv")
train_main = pd.read_csv(DATA_PATH / "train.csv")
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
    # print(index, row)
    image_paths = get_image_paths(row, base_path)
    conditions = condition_mapping.get(row['series_description'], {})
    if isinstance(conditions, str):  # Single condition
        conditions = {'left': conditions, 'right': conditions}
    for side, condition in conditions.items():
        print("side", side, condition, image_paths)
        for image_path in image_paths:
            print(image_path)
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
print(expanded_test_desc)
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

dls = get_dataloaders(test_data, CONFIG, split="test")
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


print(expanded_test_desc)
# Update row_id in expanded_test_desc to include levels
expanded_test_desc['row_id'] = expanded_test_desc.apply(lambda row: update_row_id(row, levels), axis=1)
expanded_test_desc[["normal_mild", "moderate", "severe"]] = preds

final_df = expanded_test_desc[["row_id", "normal_mild", "moderate", "severe"]]

target_cols = sample_df.columns.tolist()
final_df = final_df.groupby('row_id').sum().reset_index()
# normalize the columns
final_df[target_cols[1:]] = final_df[target_cols[1:]].div(final_df[target_cols[1:]].sum(axis=1), axis=0)
final_df[target_cols].to_csv('submission.csv', index=False)
print(pd.read_csv('submission.csv'))
