import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import GroupKFold
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
from src.config import Config

# Disable OpenCV multi-threading for Windows process stability
cv2.setNumThreads(0)


def get_driver_splits(csv_path: str, n_splits: int = 5, seed: int = 42):
    df = pd.read_csv(csv_path)

    df["img_path"] = df.apply(
        lambda x: os.path.join(Config.TRAIN_DIR, x["classname"], x["img"]),
        axis=1,
    )

    df["label"] = df["classname"].str.replace("c", "").astype(int)

    gkf = GroupKFold(n_splits=n_splits)
    df["fold"] = -1

    for fold, (train_idx, val_idx) in enumerate(
        gkf.split(df, groups=df["subject"])
    ):
        df.loc[val_idx, "fold"] = fold

    return df


def get_transforms():
    train_transform = A.Compose([
        A.Resize(Config.IMG_SIZE[0], Config.IMG_SIZE[1]),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.5),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, p=0.5),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])

    val_transform = A.Compose([
        A.Resize(Config.IMG_SIZE[0], Config.IMG_SIZE[1]),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])

    return train_transform, val_transform


class DriverDataset(Dataset):
    def __init__(self, df: pd.DataFrame, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = self.df.loc[idx, "img_path"]
        label = self.df.loc[idx, "label"]

        image = cv2.imread(img_path)
        if image is None:
            raise FileNotFoundError(f"Image not found at {img_path}")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]

        return image, label


def create_dataloaders(df: pd.DataFrame, val_fold: int = 0):
    train_transform, val_transform = get_transforms()

    train_df = df[df["fold"] != val_fold].reset_index(drop=True)
    val_df = df[df["fold"] == val_fold].reset_index(drop=True)

    train_dataset = DriverDataset(train_df, transform=train_transform)
    val_dataset = DriverDataset(val_df, transform=val_transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )

    return train_loader, val_loader