import numpy as np
import cv2
import torch

from torch.utils.data import Dataset

class ISICDataset(Dataset):
    def __init__(
        self,
        df,
        transforms=None,
        balanced=False,
        batch_ratio=1/21,
        seed=42,
        meta_cols=None,
    ):
        self.transforms = transforms
        self.balanced = balanced and (df["target"].eq(0).any() and df["target"].eq(1).any())
        self.batch_ratio = float(batch_ratio)
        self.df = df.reset_index(drop=True)
        self.df_pos = self.df[self.df["target"] == 1].reset_index(drop=True)
        self.df_neg = self.df[self.df["target"] == 0].reset_index(drop=True)
        self.rng = np.random.default_rng(seed)
        self.meta_cols = meta_cols

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        if self.balanced:
            if self.rng.random() < self.batch_ratio:
                sample = self.df_pos.iloc[index % len(self.df_pos)]
            else:
                sample = self.df_neg.iloc[index % len(self.df_neg)]
        else:
            sample = self.df.iloc[index]

        img = cv2.imread(sample["file_path"])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
        if self.transforms is not None:
            img = self.transforms(image=img)["image"]
            
        meta = None
        if self.meta_cols is not None:
            meta_vals = sample[self.meta_cols].values.astype(np.float32)
            meta = torch.from_numpy(meta_vals)

        target = int(sample["target"])

        out = {
            "image": img,
            "target": target,
        }
        if meta is not None:
            out["meta"] = meta

        return out