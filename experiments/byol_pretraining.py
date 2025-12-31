from tripletattention import TripletAttention
from dataloader import make_loaders_for_fold
from feature import get_train_file_path
from isicmodel import replace_se_with_new_attention
from utils import set_seed

import os
import time
import copy
import glob
import cv2

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback

import timm

from lightly.loss import NegativeCosineSimilarity
from lightly.models.modules import BYOLPredictionHead, BYOLProjectionHead
from lightly.models.utils import deactivate_requires_grad, update_momentum
from lightly.transforms.byol_transform import BYOLTransform,BYOLView1Transform, BYOLView2Transform
from lightly.utils.scheduler import cosine_schedule


ROOT_DIR = "/workspace"
TRAIN_DIR = f'{ROOT_DIR}/all_images_224_unique_by_name_dir'

IMAGE_SIZE  = 224
BATCH_SIZE  = 128
EPOCHS      = 1000
NUM_WORKERS = 8
LR          = 0.1    
WD          = 1e-4
BACKBONE_OUT_DIM = 1280  # efficientnetb0

set_seed(42)


train_images = sorted(glob.glob(f"{TRAIN_DIR}/*.jpg"))
df = pd.read_csv(f"{ROOT_DIR}/merged_patient_isic.csv")

df['file_path'] = df['isic_id'].apply(
    lambda x: get_train_file_path(x, TRAIN_DIR)
)
df = df[ df["file_path"].isin(train_images) ].reset_index(drop=True)


def make_backbone_triplet(replace_groups, attention_type):
    backbone = timm.create_model('efficientnet_b0', pretrained=False, num_classes=0, global_pool='')
    replace_se_with_new_attention(backbone, replace_groups, attention_type)  

    encoder = nn.Sequential(
        backbone,
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(start_dim=1)
    )
    return encoder


class ISICDataset(Dataset):
    def __init__(self, df, transform):
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.path_col = "file_path"

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        path = self.df.iloc[idx][self.path_col]
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        views = self.transform(img)           
        return (views, 0)                      
    
# Adapted from the Lightly:
# https://docs.lightly.ai/self-supervised-learning/byol.html
class BYOL_Lightly(pl.LightningModule):
    def __init__(self, max_epochs: int = EPOCHS):
        super().__init__()
        self.encoder    = make_backbone_triplet(replace_groups=REPLACE_GROUPS, attention_type=ATTENTION_TYPE)
        self.projector  = BYOLProjectionHead(BACKBONE_OUT_DIM, 4096, 256)  
        self.predictor  = BYOLPredictionHead(256, 4096, 256)            

        self.encoder_m   = copy.deepcopy(self.encoder)
        self.projector_m = copy.deepcopy(self.projector)
        deactivate_requires_grad(self.encoder_m)
        deactivate_requires_grad(self.projector_m)

        self.criterion = NegativeCosineSimilarity()
        self.max_epochs = max_epochs

    @torch.no_grad()
    def forward_momentum(self, x):
        y = self.encoder_m(x)
        z = self.projector_m(y)
        return z.detach()

    def forward(self, x):
        y = self.encoder(x)
        z = self.projector(y)
        p = self.predictor(z)
        return p

    def training_step(self, batch, batch_idx):
        momentum = cosine_schedule(self.current_epoch, self.max_epochs, 0.996, 1.0)
        update_momentum(self.encoder,   self.encoder_m,   m=momentum)
        update_momentum(self.projector, self.projector_m, m=momentum)
    
        (x0, x1) = batch[0]
        p0 = self.forward(x0)
        z0 = self.forward_momentum(x0)
        p1 = self.forward(x1)
        z1 = self.forward_momentum(x1)
    
        loss = 0.5 * (self.criterion(p0, z1) + self.criterion(p1, z0))
        return loss
    
    def configure_optimizers(self):
        optim = torch.optim.SGD(self.parameters(), lr=LR, momentum=0.9, weight_decay=WD, nesterov=True)
        return optim

class EpochTimerCallback(pl.Callback):
    def on_train_epoch_start(self, trainer, pl_module):
        self.epoch_start_time = time.time()

    def on_train_epoch_end(self, trainer, pl_module):
        elapsed = time.time() - self.epoch_start_time
        current_epoch = trainer.current_epoch
        print(f"Epoch {current_epoch} duration : {elapsed:.2f} s")


transform = BYOLTransform(
    view_1_transform=BYOLView1Transform(input_size=IMAGE_SIZE),
    view_2_transform=BYOLView2Transform(input_size=IMAGE_SIZE),
)

ds = ISICDataset(df, transform=transform) 
loader = DataLoader(
    ds,
    batch_size=BATCH_SIZE,
    shuffle=True,
    drop_last=True,
    num_workers=NUM_WORKERS,
)

loader = DataLoader(
    ds,
    batch_size=BATCH_SIZE,
    shuffle=True,
    drop_last=True,
    num_workers=NUM_WORKERS,
)

ATTENTION_TYPE = "triplet"
REPLACE_GROUPS = [True, True, True, False]
model = BYOL_Lightly(max_epochs=EPOCHS)

trainer = pl.Trainer(
    accelerator="gpu" if torch.cuda.is_available() else "cpu",
    devices=1,
    max_epochs=EPOCHS,
    limit_train_batches=0.1,
    callbacks=[EpochTimerCallback()],
)


trainer.fit(model=model, train_dataloaders=loader)

os.makedirs("backbones_pretrained", exist_ok=True)
online_backbone   = model.encoder[0]    
momentum_backbone = model.encoder_m[0]

torch.save(online_backbone.state_dict(),   "backbones_pretrained/effb0_tripletattention_byol_online_v2.pth")
torch.save(momentum_backbone.state_dict(), "backbones_pretrained/effb0_tripletattention_byol_ema_v2.pth")
