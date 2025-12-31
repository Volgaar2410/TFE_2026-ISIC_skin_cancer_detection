from config import CONFIG
from cbam import CBAM
from tripletattention import TripletAttention
from dataloader import make_loaders_for_fold
from feature import num_cols, cat_cols, non_feature_cols, get_train_file_path, prepare_features
from isicdataset import ISICDataset
from isicmodel import ISICModel
from training import run_experiment_multi_seed
from transform import data_augm
from utils import set_seed

import pandas as pd
import glob


ROOT_DIR = "/workspace/isic-2024-challenge"
TRAIN_DIR = f'{ROOT_DIR}/train-image/image'
train_images = sorted(glob.glob(f"{TRAIN_DIR}/*.jpg"))

df = pd.read_csv(f"{ROOT_DIR}/train-metadata.csv")


df_positive = df[df["target"] == 1].reset_index(drop=True)
df_negative = df[df["target"] == 0].reset_index(drop=True)

df = pd.concat([df_positive, df_negative.iloc[:df_positive.shape[0]*CONFIG["undersample_neg_pos_ratio"], :]])  

df['file_path'] = df['isic_id'].apply(
    lambda x: get_train_file_path(x, TRAIN_DIR)
)

df = df[ df["file_path"].isin(train_images) ].reset_index(drop=True)

CONFIG['T_max'] = df.shape[0] * (CONFIG["n_fold"]-1) * CONFIG['epochs'] // CONFIG['train_batch_size'] // CONFIG["n_fold"]



CONFIG["stop_epoch"] = 70
replace_groups = [True, True, True, False]
TRIPLET_BACKBONE_BYOL = "/workspace/effb0_tripletattention_byol_ema_v2.pth"
BASELINE_BACKBONE_BYOL = "/workspace/effb0_byol_ema_v2.pth"

#efficientnet-B0 baseline no pretraining
run_experiment_multi_seed(
    df=df,
    attention_type=None,
    replace_groups=None,
    backbone_ckpt_path=None,
    pretrained=False,
    gbm_type=None,
    meta_cnn_integration = None,
    meta_cols = None,
)

#efficientnet-B0 cbam no pretraining
run_experiment_multi_seed(
    df=df,
    attention_type="cbam",
    replace_groups=replace_groups,
    backbone_ckpt_path=None,
    pretrained=False,
    gbm_type=None,
    meta_cnn_integration = None,
    meta_cols = None,
)

#efficientnet-B0 tripletattention no pretraining
run_experiment_multi_seed(
    df=df,
    attention_type="triplet",
    replace_groups=replace_groups,
    backbone_ckpt_path=TRIPLET_BACKBONE_BYOL,
    pretrained=False,
    gbm_type=None,
    meta_cnn_integration = None,
    meta_cols = None,
)

CONFIG["stop_epoch"] = 60

#efficientnet-B0 baseline byol pretraining
run_experiment_multi_seed(
    df=df,
    attention_type=None,
    replace_groups=None,
    backbone_ckpt_path=BASELINE_BACKBONE_BYOL,
    pretrained=False,
    gbm_type=None,
    meta_cnn_integration = None,
    meta_cols = None,
)

#efficientnet-B0 tripletattention byol pretraining
run_experiment_multi_seed(
    df=df,
    attention_type="triplet",
    replace_groups=replace_groups,
    backbone_ckpt_path=TRIPLET_BACKBONE_BYOL,
    pretrained=False,
    gbm_type=None,
    meta_cnn_integration = None,
    meta_cols = None,
)

CONFIG["stop_epoch"] = 20

#efficientnet-B0 baseline imagenet1k pretraining
run_experiment_multi_seed(
    df=df,
    attention_type=None,
    replace_groups=None,
    backbone_ckpt_path=None,
    pretrained=True,
    gbm_type=None,
    meta_cnn_integration = None,
    meta_cols = None,
)