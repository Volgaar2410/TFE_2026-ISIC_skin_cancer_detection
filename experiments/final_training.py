from config import CONFIG
from cbam import CBAM
from tripletattention import TripletAttention
from dataloader import make_loaders_for_fold
from feature import num_cols, cat_cols, non_feature_cols, get_train_file_path, prepare_features
from isicdataset import ISICDataset
from isicmodel import ISICModel
from training import run_experiment_multi_seed, build_tabular_data,train_one_epoch
from transform import data_augm
from utils import set_seed
from gbmodel import params_cb, params_lgb, params_xgb

import pandas as pd
import glob
from isicdataset import ISICDataset
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
import torch
import joblib
from collections import defaultdict
import gc
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier



def make_loaders(df_train, data_transforms, balanced, meta_cols=None):
    train_dataset = ISICDataset(
        df_train,
        transforms=data_transforms["train"],
        balanced=balanced,
        meta_cols = meta_cols
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=CONFIG["train_batch_size"],
        num_workers=4,
        shuffle=True,
        pin_memory=True
    )

    return train_loader

def run_training(model, optimizer, scheduler, device,
                 train_loader,num_epochs):

    history = defaultdict(list)

    for epoch in range(1, num_epochs + 1):
        gc.collect()

        train_loss, train_pauc, train_auroc = train_one_epoch(
            model, optimizer, scheduler, dataloader=train_loader,
            device=device
        )

        history['Train Loss'].append(train_loss)
        history['Train pAUC'].append(train_pauc)
        history['Train AUROC'].append(train_auroc)

    return model, history


def run_final_training_multimodal(
    df,
    df_subset,
    data_transforms,
    pretrained,
    replace_groups,
    attention_type,
    meta_cols,
    backbone_ckpt_path=None,
    gbm_type="xgb",
):
    
    train_loader_subset = make_loaders(
        df_subset, data_transforms=data_transforms, meta_cols=None, balanced=False
    )

    train_loader = make_loaders(
        df, data_transforms=data_transforms, meta_cols=meta_cols, balanced=False
    )

    
    model = ISICModel(
        model_name=CONFIG["model_name"],
        pretrained=pretrained,
        replace_groups=replace_groups,
        attention_type=attention_type,
        backbone_ckpt_path=backbone_ckpt_path,
        meta_cnn_integration=None,
        meta_cols=meta_cols,
        num_classes=1,
    ).to(CONFIG["device"])

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=CONFIG["learning_rate"],
        weight_decay=CONFIG["weight_decay"],
    )
    
    scheduler = lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=CONFIG["T_max"], eta_min=CONFIG["min_lr"]
        )

    model, history = run_training(
        model,
        optimizer,
        scheduler,
        device=CONFIG["device"],
        train_loader=train_loader_subset,
        num_epochs=CONFIG["stop_epoch"]
    )
    
    model.eval()
    
    X_tab, y_tab = build_tabular_data(model, train_loader)
    print(X_tab.shape)

    if gbm_type == "xgb":
        gb_model = XGBClassifier(**params_xgb)
    elif gbm_type == "lgb":
        gb_model = LGBMClassifier(**params_lgb)
    elif gbm_type == "cb":
        gb_model = CatBoostClassifier(**params_cb)
        
    gb_model.fit(X_tab, y_tab)

    return model, gb_model, history


ROOT_DIR = "/workspace/isic-2024-challenge"
TRAIN_DIR = f'{ROOT_DIR}/train-image/image'
train_images = sorted(glob.glob(f"{TRAIN_DIR}/*.jpg"))

df = pd.read_csv(f"{ROOT_DIR}/train-metadata.csv")

df_positive = df[df["target"] == 1].reset_index(drop=True)
df_negative = df[df["target"] == 0].reset_index(drop=True)
 

df['file_path'] = df['isic_id'].apply(
    lambda x: get_train_file_path(x, TRAIN_DIR)
)
df = df[ df["file_path"].isin(train_images) ].reset_index(drop=True)

df_subset = pd.concat([df_positive, df_negative.iloc[:df_positive.shape[0]*CONFIG["undersample_neg_pos_ratio"], :]]) 

df_subset['file_path'] = df_subset['isic_id'].apply(
    lambda x: get_train_file_path(x, TRAIN_DIR)
)
df_subset = df_subset[ df_subset["file_path"].isin(train_images) ].reset_index(drop=True)

df, feature_cols = prepare_features(df)
df_subset, _ = prepare_features(df_subset)

CONFIG["meta_cols"] = feature_cols


CONFIG['T_max'] = df_subset.shape[0] * CONFIG['epochs'] // CONFIG['train_batch_size']

CONFIG["stop_epoch"] = 60
TRIPLET_BACKBONE = "/workspace/effb0_tripletattention_byol_ema_v2.pth"
BASELINE_BACKBONE_BYOL = "/workspace/effb0_byol_ema_v2.pth"
replace_groups = [True, True, True, False]
attention_type = "triplet"


model_cnn, model_gbm, history = run_final_training_multimodal(
    df=df,
    df_subset=df_subset,
    data_transforms=data_augm,
    pretrained=False,
    replace_groups=replace_groups,
    attention_type=attention_type,
    backbone_ckpt_path=TRIPLET_BACKBONE,
    gbm_type="cb",
    meta_cols=feature_cols,
)

cnn_path = "triplet_cnn_final.joblib"
joblib.dump(model_cnn, cnn_path)

gbm_path = "triplet_gbm_final.joblib"
joblib.dump(model_gbm, gbm_path)


