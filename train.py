import os

import torch
from tqdm.auto import tqdm
import pandas as pd
from sklearn import model_selection
import lightning as pl
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from config import CONFIG
from utils import get_image_paths, reshape_row, seeding, check_image_exists, check_series_id, check_study_id, DATA_PATH
from models import LumbarLightningModel
from dataset import get_train_dataloaders, get_test_dataloaders

import cv2
import warnings

cv2.setNumThreads(0)
warnings.filterwarnings("ignore")
tqdm.pandas()

os.listdir(DATA_PATH)

seeding(CONFIG['seed'])
train_main = pd.read_csv(DATA_PATH / "train.csv")
train_desc = pd.read_csv(DATA_PATH / "train_series_descriptions.csv")
train_label_coordinates = pd.read_csv(DATA_PATH / "train_label_coordinates.csv")

# Mapping of series_description to conditions
condition_mapping = {
    'Sagittal T1': {'left': 'left_neural_foraminal_narrowing', 'right': 'right_neural_foraminal_narrowing'},
    'Axial T2': {'left': 'left_subarticular_stenosis', 'right': 'right_subarticular_stenosis'},
    'Sagittal T2/STIR': 'spinal_canal_stenosis'
}

# Create a list to store the expanded rows
expanded_rows = []

# Expand the dataframe by adding new rows for each file path
for index, row in tqdm(train_desc.iterrows(), total=len(train_desc)):
    image_paths = get_image_paths(row)
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
expanded_train_desc = pd.DataFrame(expanded_rows)
# Reshape the DataFrame for all rows
new_train_df = pd.concat([reshape_row(row) for _, row in train_main.iterrows()], ignore_index=True)

# Merge the dataframes on the common columns
merged_df = pd.merge(new_train_df, train_label_coordinates, on=['study_id', 'condition', 'level'], how='inner')
final_merged_df = pd.merge(merged_df, train_desc, on=['series_id', 'study_id'], how='inner')

# Create the row_id column
final_merged_df['row_id'] = (
        final_merged_df['study_id'].astype(str) + '_' +
        final_merged_df['condition'].str.lower().str.replace(' ', '_') + '_' +
        final_merged_df['level'].str.lower().str.replace('/', '_')
)

# Create the image_path column
final_merged_df['image_path'] = (
        f'{str(DATA_PATH)}/train_images/' +
        final_merged_df['study_id'].astype(str) + '/' +
        final_merged_df['series_id'].astype(str) + '/' +
        final_merged_df['instance_number'].astype(str) + '.dcm'
)

final_merged_df['severity'] = final_merged_df['severity'].map(
    {'Normal/Mild': 'normal_mild', 'Moderate': 'moderate', 'Severe': 'severe'}
)

train_data = final_merged_df.copy()

# Apply the functions to the train_data dataframe
train_data['study_id_exists'] = train_data.progress_apply(check_study_id, axis=1)
train_data['series_id_exists'] = train_data.progress_apply(check_series_id, axis=1)
train_data['image_exists'] = train_data.progress_apply(check_image_exists, axis=1)

# Filter train_data
train_data = train_data[(train_data['study_id_exists']) & (train_data['series_id_exists']) & (train_data['image_exists'])]

label2id = {v: i for i, v in enumerate(train_data['severity'].unique())}
id2label = {v: k for k, v in label2id.items()}
train_data['target'] = train_data['severity'].map(label2id)
train_data = train_data.dropna(subset=['severity']).reset_index(drop=True)

series2id = {v: i for i, v in enumerate(train_data['series_description'].unique().tolist())}
id2series = {v: k for k, v in series2id.items()}
train_data['series2id'] = train_desc['series_description'].map(series2id)
train_data = train_data.dropna(subset=['series2id']).reset_index(drop=True)

train_data, test_data = model_selection.train_test_split(train_data, test_size=0.05)
train_data = train_data.reset_index(drop=True)
test_data = test_data.reset_index(drop=True)

kfold = model_selection.StratifiedKFold(n_splits=CONFIG["n_folds"], shuffle=True, random_state=CONFIG["seed"])
x = train_data.index.values
y = train_data['target'].values.astype(int)
# g = train_data['series2id'].values.astype(int)

train_data['fold'] = -1
for fold, (tr_idx, val_idx) in enumerate(kfold.split(x, y)):
    train_data.loc[val_idx, 'fold'] = fold

train = True

for fold in range(CONFIG["n_folds"]):
    train_ds = train_data[train_data['fold'] != fold].reset_index(drop=True)
    valid_ds = train_data[train_data['fold'] == fold].reset_index(drop=True)

    train_loader = get_train_dataloaders(train_ds, CONFIG)
    valid_loader = get_test_dataloaders(valid_ds, CONFIG)
    logger = TensorBoardLogger(
        save_dir="data/logs",
        name=f"folds",
        version=f"fold-{fold}-{CONFIG["backbone"]}"
    )

    callbacks = [
        # ModelCheckpoint(
        #     dirpath="data/model_checkpoint",
        #     save_weights_only=True,
        #     mode="min",
        #     monitor="valid_loss"
        # ),
        LearningRateMonitor("epoch"),
        # EarlyStopping(monitor="valid_loss", min_delta=0.0, patience=CONFIG['patience'], verbose=False, mode="min"),
    ]

    net = LumbarLightningModel()
    if not train:
        net.load(CONFIG["weights_path"])

    trainer = pl.Trainer(
        accelerator="gpu",
        devices=1,
        precision="16-mixed",
        max_epochs=CONFIG['epochs'],
        logger=logger,
        callbacks=callbacks,
        default_root_dir=os.getcwd(),
        enable_progress_bar=False
    )

    if train:
        trainer.fit(net, train_dataloaders=train_loader, val_dataloaders=valid_loader)
        trainer.model.save(f"data/{CONFIG["backbone"]}-{fold}.pth")

    print(f"testing, fold {fold}")
    test_loader = get_test_dataloaders(test_data, CONFIG)
    print(test_data.head().to_string())
    batch_images, batch_labels = next(test_loader.__iter__())
    print(batch_images.shape, batch_labels.shape)
    trainer.test(net, test_loader)
