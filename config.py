import torch


CONFIG = {
    "undersample_neg_pos_ratio" : 20,
    "batch_balanced" : False,
    "seed": 42,
    "epochs": 100,
    "stop_epoch":60,
    "img_size": 224,
    "model_name": "efficientnet_b0",
    "meta_cnn_integration" : None,
    "meta_cols" : None,
    "train_batch_size": 256,
    "valid_batch_size": 512,
    "learning_rate": 1e-3,
    "min_lr": 1e-6,
    "T_max": None,
    "weight_decay": 1e-5,
    "n_fold": 5,
    "device": torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
}