from config import CONFIG
from cbam import CBAM
from tripletattention import TripletAttention
from dataloader import make_loaders_for_fold
from feature import num_cols, cat_cols, non_feature_cols, prepare_features
from isicdataset import ISICDataset
from isicmodel import ISICModel
from training import run_experiment_multi_seed
from transform import data_augm
from utils import set_seed

import optuna
import numpy as np
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import log_loss
import pandas as pd
import glob


ROOT_DIR = "/workspace/isic-2024-challenge"

df = pd.read_csv(f"{ROOT_DIR}/train-metadata.csv")

df, feature_cols = prepare_features(df)

target = "target"
group_col = "patient_id"

X = df[feature_cols].values
y = df[target].values
groups = df[group_col].values



def run_cv_logloss(trial, model_cls, params, model_name: str):
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    oof_losses = []

    for train_idx, valid_idx in sgkf.split(X, y, groups):
        X_tr, X_va = X[train_idx], X[valid_idx]
        y_tr, y_va = y[train_idx], y[valid_idx]

        model = model_cls(**params)
        model.fit(X_tr, y_tr)

        proba_va = model.predict_proba(X_va)[:, 1]
        oof_losses.append(log_loss(y_va, proba_va))

    mean_loss = float(np.mean(oof_losses))
    print(f"{model_name} trial={trial.number} logloss={mean_loss:.6f}")
    return mean_loss


def objective_xgb(trial):
    params_xgb = {
    "n_estimators": trial.suggest_int("n_estimators", 200, 800),
    "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
    "max_depth": trial.suggest_int("max_depth", 3, 10),
    "subsample": trial.suggest_float("subsample", 0.6, 1.0),
    "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
    "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 10.0),
    "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "random_state": 42,
    }
    return run_cv_logloss(trial, xgb.XGBClassifier, params_xgb, "XGB")


def objective_lgb(trial):
    params_lgb = {
    "n_estimators": trial.suggest_int("n_estimators", 200, 800),
    "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
    "num_leaves": trial.suggest_int("num_leaves", 16, 256),
    "max_depth": trial.suggest_int("max_depth", -1, 12),
    "min_child_samples": trial.suggest_int("min_child_samples", 10, 100),
    "subsample": trial.suggest_float("subsample", 0.6, 1.0),
    "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
    "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
    "objective": "binary",
    "metric": "binary_logloss",
    "random_state": 42,
    "verbosity": -1,
    }
    return run_cv_logloss(trial, lgb.LGBMClassifier, params_lgb, "LightGBM")


def objective_cb(trial):
    params_cb = {
    "iterations": trial.suggest_int("iterations", 200, 1000),
    "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
    "depth": trial.suggest_int("depth", 3, 10),
    "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 10.0),
    "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 5.0),
    "loss_function": "Logloss",
    "eval_metric": "Logloss",
    "verbose": False,
    "random_seed": 42,
    "allow_writing_files": False,
    }
    return run_cv_logloss(trial, CatBoostClassifier, params_cb, "CatBoost")

def log_best(study, name, path):
    with open(path, "a") as f:
        f.write(f"{name}\n")
        f.write(f"value {study.best_value}\n")
        for k, v in study.best_params.items():
            f.write(f"{k} {v}\n")
        f.write("\n")


N_TRIALS = 1

LOG = "optuna.txt"

study_xgb = optuna.create_study(direction="minimize")
study_xgb.optimize(objective_xgb, n_trials=N_TRIALS)
log_best(study_xgb, "xgboost", LOG)

study_lgb = optuna.create_study(direction="minimize")
study_lgb.optimize(objective_lgb, n_trials=N_TRIALS)
log_best(study_lgb, "lightgbm", LOG)

study_cb = optuna.create_study(direction="minimize")
study_cb.optimize(objective_cb, n_trials=N_TRIALS)
log_best(study_cb, "catboost", LOG)


