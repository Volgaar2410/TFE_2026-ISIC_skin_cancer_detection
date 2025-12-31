import os
import random
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False



def custom_metric(y_hat, y_true):
    min_tpr = 0.80
    v_gt = abs(y_true-1)
    v_pred = np.array([1.0 - x for x in y_hat])
    max_fpr = abs(1-min_tpr)
    partial_auc_scaled = roc_auc_score(v_gt, v_pred, max_fpr=max_fpr)
    partial_auc = 0.5 * max_fpr**2 + (max_fpr - 0.5 * max_fpr**2) / (1.0 - 0.5) * (partial_auc_scaled - 0.5)
    return partial_auc

def criterion(outputs, targets):
    return nn.BCEWithLogitsLoss()(outputs, targets)
