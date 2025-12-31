import gc
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
from torch.optim import lr_scheduler
from sklearn.model_selection import StratifiedGroupKFold

from config import CONFIG
from utils import criterion, custom_metric
from dataloader import make_loaders_for_fold
from isicmodel import ISICModel
from utils import set_seed
from transform import data_augm

from torcheval.metrics.functional import binary_auroc
from gbmodel import params_xgb, params_lgb, params_cb
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier



def train_one_epoch(model, optimizer, scheduler, dataloader, device):
    model.train()
    dataset_size = 0
    running_loss = 0.0

    all_outputs = []
    all_targets = []

    for data in dataloader:
        images = data['image'].to(device, dtype=torch.float)
        targets = data['target'].to(device, dtype=torch.float).view(-1)
        
        metadata = data.get("meta", None)
        if metadata is not None:
            metadata = metadata.to(device, dtype=torch.float)

        batch_size = images.size(0)

        outputs = model(images, meta=metadata).squeeze(1)
        loss = criterion(outputs, targets)
        loss.backward()

        optimizer.step()
        optimizer.zero_grad()
        if scheduler is not None:
            scheduler.step()

        probs = torch.sigmoid(outputs)

        running_loss += loss.item() * batch_size
        dataset_size += batch_size

        all_outputs.append(probs.detach().cpu())
        all_targets.append(targets.detach().cpu())

    epoch_loss = running_loss / dataset_size

    all_outputs = torch.cat(all_outputs)
    all_targets = torch.cat(all_targets)

    epoch_auroc = binary_auroc(all_outputs, all_targets).item()
    epoch_pAUC = custom_metric(all_outputs.numpy(), all_targets.numpy())

    gc.collect()
    return epoch_loss, epoch_pAUC, epoch_auroc


@torch.inference_mode()
def valid_one_epoch(model, dataloader, device):
    model.eval()
    
    dataset_size = 0
    running_loss = 0.0

    all_outputs = []
    all_targets = []  

    for data in dataloader:        
        images = data['image'].to(device, dtype=torch.float)
        targets = data['target'].to(device, dtype=torch.float).view(-1) 
        
        metadata = data.get("meta", None)
        if metadata is not None:
            metadata = metadata.to(device, dtype=torch.float)

        batch_size = images.size(0)

        outputs = model(images, meta=metadata).squeeze(1)  
        loss = criterion(outputs, targets)

        probs = torch.sigmoid(outputs)

        running_loss += loss.item() * batch_size
        dataset_size += batch_size

        all_outputs.append(probs.detach().cpu())
        all_targets.append(targets.detach().cpu())

    epoch_loss = running_loss / dataset_size

    all_outputs = torch.cat(all_outputs)       
    all_targets = torch.cat(all_targets)        

    epoch_auroc = binary_auroc(all_outputs, all_targets).item()

    epoch_pAUC = custom_metric(
        all_outputs.numpy(),
        all_targets.numpy()
    )

    gc.collect()

    return epoch_loss, epoch_pAUC, epoch_auroc



def run_training(model, optimizer, scheduler, device, num_epochs,
                 train_loader, valid_loader):

    history = defaultdict(list)

    for epoch in range(1, num_epochs + 1):
        gc.collect()

        train_loss, train_pauc, train_auroc = train_one_epoch(
            model, optimizer, scheduler, dataloader=train_loader,
            device=device
        )

        val_loss, val_pauc, val_auroc = valid_one_epoch(
            model, valid_loader, device=device
        )

        history['Train Loss'].append(train_loss)
        history['Train pAUC'].append(train_pauc)
        history['Train AUROC'].append(train_auroc)

        history['Valid Loss'].append(val_loss)
        history['Valid pAUC'].append(val_pauc)
        history['Valid AUROC'].append(val_auroc)

    return model, history


def build_tabular_data(model, loader):
    all_feats = []
    all_targets = []

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(CONFIG["device"])
            targets = batch["target"].cpu().numpy()
            meta_np = batch["meta"].cpu().numpy()


            logits = model(images)

            probs = torch.sigmoid(logits).detach().cpu().numpy().reshape(-1, 1)
            
            feats = np.concatenate([meta_np, probs], axis=1)

            all_feats.append(feats)
            all_targets.append(targets)

    X = np.concatenate(all_feats, axis=0)
    y = np.concatenate(all_targets, axis=0)
    return X, y


def run_training_cv_5fold(
    df,
    data_transforms,
    pretrained,
    replace_groups,
    attention_type,
    meta_cnn_integration,
    meta_cols,
    backbone_ckpt_path=None,
    gbm_type=None,
):
    sgkf_outer = StratifiedGroupKFold(
        n_splits=CONFIG["n_fold"], shuffle=True, random_state=CONFIG["seed"]
    )
    y = df["target"].to_numpy()
    groups = df["patient_id"].to_numpy()

    fold_pauc = []
    fold_auroc = []

    gbm_fold_pauc = []
    gbm_fold_auroc = []
    tabular_valid_scores_per_fold = []

    for fold, (train_idx, valid_idx) in enumerate(
        sgkf_outer.split(X=np.zeros(len(df)), y=y, groups=groups)
    ):
        train_loader, valid_loader = make_loaders_for_fold(
            df, train_idx, valid_idx, data_transforms, CONFIG["batch_balanced"], meta_cols=meta_cols
        )

        model = ISICModel(
            model_name=CONFIG["model_name"],
            pretrained=pretrained,
            replace_groups=replace_groups,
            attention_type=attention_type,
            backbone_ckpt_path=backbone_ckpt_path,
            meta_cnn_integration=meta_cnn_integration,
            meta_cols=meta_cols,
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
            num_epochs=CONFIG["stop_epoch"],
            train_loader=train_loader,
            valid_loader=valid_loader,
        )
        df_hist = pd.DataFrame.from_dict(history)
        df_hist.to_csv(f"history_{CONFIG['model_name']}_fold{fold}.csv", index=False)

        val_pauc = float(history["Valid pAUC"][-1])
        val_auroc = float(history["Valid AUROC"][-1])

        fold_pauc.append(val_pauc)
        fold_auroc.append(val_auroc)

        print(f"[FOLD {fold}] CNN validation pAUC : {val_pauc:.6f}")
        print(f"[FOLD {fold}] CNN validation AUROC: {val_auroc:.6f}")

        if gbm_type is not None:
            if gbm_type == "xgb":
                gb_model = XGBClassifier(**params_xgb)
            elif gbm_type == "lgb":
                gb_model = LGBMClassifier(**params_lgb)
            elif gbm_type == "cb":
                gb_model = CatBoostClassifier(**params_cb)

            X_train_tab, y_train_tab = build_tabular_data(model, train_loader)
            X_valid_tab, y_valid_tab = build_tabular_data(model, valid_loader)

            gb_model.fit(X_train_tab, y_train_tab)

            valid_scores = gb_model.predict_proba(X_valid_tab)[:, 1]
            tabular_valid_scores_per_fold.append(valid_scores)

            gbm_pauc = float(custom_metric(valid_scores, y_valid_tab))
            gbm_auroc = float(
                binary_auroc(torch.tensor(valid_scores), torch.tensor(y_valid_tab)).item()
            )

            gbm_fold_pauc.append(gbm_pauc)
            gbm_fold_auroc.append(gbm_auroc)

            print(f"[FOLD {fold}] {gbm_type} validation pAUC : {gbm_pauc:.6f}")
            print(f"[FOLD {fold}] {gbm_type} validation AUROC: {gbm_auroc:.6f}")

            del gb_model

        del model, optimizer, scheduler, train_loader, valid_loader
        gc.collect()


    print("\nCross-validation results (CNN)")
    print(f"pAUC (mean ± std) : {np.mean(fold_pauc):.6f} ± {np.std(fold_pauc):.6f}")
    print(f"AUROC (mean ± std): {np.mean(fold_auroc):.6f} ± {np.std(fold_auroc):.6f}")

    if gbm_type is not None:
        print(f"\nCross-validation results ({gbm_type} on meta + CNN)")
        print(f"pAUC (mean ± std) : {np.mean(gbm_fold_pauc):.6f} ± {np.std(gbm_fold_pauc):.6f}")
        print(f"AUROC (mean ± std): {np.mean(gbm_fold_auroc):.6f} ± {np.std(gbm_fold_auroc):.6f}")
    
    summary = {
        "cnn": {
            "pAUC_mean": float(np.mean(fold_pauc)),
            "pAUC_std": float(np.std(fold_pauc)),
            "AUROC_mean": float(np.mean(fold_auroc)),
            "AUROC_std": float(np.std(fold_auroc)),
        }
    }
    
    if gbm_type is not None:
        summary["gbm"] = {
            "model": gbm_type,
            "pAUC_mean": float(np.mean(gbm_fold_pauc)),
            "pAUC_std": float(np.std(gbm_fold_pauc)),
            "AUROC_mean": float(np.mean(gbm_fold_auroc)),
            "AUROC_std": float(np.std(gbm_fold_auroc)),
        }

    return summary

def run_experiment_multi_seed(
    df,
    attention_type,
    replace_groups,
    meta_cnn_integration,
    meta_cols,
    pretrained=False,
    gbm_type=None,
    backbone_ckpt_path=None,
):

    cnn_aurocs, cnn_paucs = [], []
    gbm_aurocs, gbm_paucs = [], []

    for seed in range(42, 47):
        CONFIG["seed"] = seed
        set_seed(seed)

        print(f"run seed = {seed}")

        summary = run_training_cv_5fold(
            df=df,
            data_transforms=data_augm,
            pretrained=pretrained,
            replace_groups=replace_groups,
            attention_type=attention_type,
            backbone_ckpt_path=backbone_ckpt_path,
            gbm_type=gbm_type,
            meta_cnn_integration=meta_cnn_integration,
            meta_cols=meta_cols,
        )

        cnn_aurocs.append(summary["cnn"]["AUROC_mean"])
        cnn_paucs.append(summary["cnn"]["pAUC_mean"])

        if gbm_type is not None and "gbm" in summary:
            gbm_aurocs.append(summary["gbm"]["AUROC_mean"])
            gbm_paucs.append(summary["gbm"]["pAUC_mean"])

    print("\nResults across seeds")
    print(f"CNN  pAUC  : {float(np.mean(cnn_paucs)):.4f} ± {float(np.std(cnn_paucs)):.4f}")
    print(f"CNN  AUROC : {float(np.mean(cnn_aurocs)):.4f} ± {float(np.std(cnn_aurocs)):.4f}")

    if gbm_type is not None:
        print(f"{gbm_type} pAUC  : {float(np.mean(gbm_paucs)):.4f} ± {float(np.std(gbm_paucs)):.4f}")
        print(f"{gbm_type} AUROC : {float(np.mean(gbm_aurocs)):.4f} ± {float(np.std(gbm_aurocs)):.4f}")

    

