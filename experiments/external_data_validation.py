from config import CONFIG
from cbam import CBAM
from tripletattention import TripletAttention, AttentionGate, ZPool, BasicConv
from dataloader import make_loaders_for_fold
from feature import num_cols, cat_cols, non_feature_cols, get_train_file_path, prepare_features
from isicdataset import ISICDataset
from isicmodel import ISICModel
from training import run_experiment_multi_seed
from transform import data_augm
from utils import set_seed, custom_metric

import pandas as pd
import glob
from sklearn.metrics import roc_auc_score
import os
import joblib
import torch
import numpy as np
from torch.utils.data import DataLoader


ROOT_DIR = "/workspace/derm12345/"
TRAIN_DIR = f'{ROOT_DIR}/images'
train_images = sorted(glob.glob(f"{TRAIN_DIR}/*.jpg"))

df = pd.read_csv(f"{ROOT_DIR}/metadata.csv")
df["target"] = df["diagnosis_1"]
df = df[df["target"] != "Indeterminate"]
df["target"] = df["target"].replace({
    "Malignant": 1,
    "Benign": 0
})
df["target"].value_counts()


df_positive = df[df["target"] == 1].reset_index(drop=True)
df_negative = df[df["target"] == 0].reset_index(drop=True)

df['file_path'] = df['isic_id'].apply(
    lambda x: get_train_file_path(x, TRAIN_DIR)
)

df = df[ df["file_path"].isin(train_images) ].reset_index(drop=True)


dataset = ISICDataset(
    df.reset_index(drop=True),
    transforms=data_augm["valid"],
    balanced= False,
    meta_cols=None
)

valid_loader = DataLoader(
    dataset,
    batch_size=CONFIG["valid_batch_size"],
    num_workers=4,
    shuffle=False,
    drop_last= False
)


model_path = "/workspace/TFE-2026/triplet_cnn_final.joblib"
y_true = df["target"].values.astype(float)
device = CONFIG["device"]

model = joblib.load(model_path)
model.to(device)

model.eval()

preds = []

with torch.no_grad():
    for batch in valid_loader:
        images = batch["image"]
        images = images.to(device, dtype=torch.float)
        logits = model(images)                     
        probas = torch.sigmoid(logits).squeeze(1)  
        preds.append(probas.detach().cpu().numpy())

y_pred = np.concatenate(preds, axis=0)

auroc = float(roc_auc_score(y_true, y_pred))
pauc = custom_metric(y_pred, y_true)

print(f"AUROC = {auroc:.4f}")
print(f"pAUC  = {pauc:.4f}")
