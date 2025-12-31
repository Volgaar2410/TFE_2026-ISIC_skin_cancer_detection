from isicdataset import ISICDataset
from torch.utils.data import DataLoader
from config import CONFIG

def make_loaders_for_fold(df, train_idx, valid_idx, data_transforms, balanced, meta_cols=None):
    df_train = df.iloc[train_idx].reset_index(drop=True)
    df_valid = df.iloc[valid_idx].reset_index(drop=True)        

    train_dataset = ISICDataset(
        df_train,
        transforms=data_transforms["train"],
        balanced=balanced,
        meta_cols = meta_cols
    )
    valid_dataset = ISICDataset(
        df_valid,
        transforms=data_transforms["valid"],
        balanced= False,
        meta_cols = meta_cols
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=CONFIG["train_batch_size"],
        num_workers=4,
        shuffle=True,
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=CONFIG["valid_batch_size"],
        num_workers=4,
        shuffle=False,
    )

    return train_loader, valid_loader