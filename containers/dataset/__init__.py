import pandas as pd
from numpy import random
from typing import Tuple
from torch.utils.data import random_split, DataLoader

from .digitdataset import DigitDataset
from .transform import get_transforms

def build_dataloader(
    root: str="data/train.csv",
    ratio: float=0.8,
    batch_size: int=64
) -> Tuple[DataLoader, DataLoader]:
    df = pd.read_csv(root).sample(frac=1)
    partition = int(len(df) * ratio)
    df_train, df_val = df.iloc[: partition, :], df.iloc[partition: , :]

    train_dataset = DigitDataset(
        df=df_train,
        transforms=get_transforms()["train"]
    )
    val_dataset = DigitDataset(
        df=df_val,
        transforms=get_transforms()["val"]
    )

    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True
    )
    val_dataloader = DataLoader(
        dataset=val_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=True
    )

    return train_dataloader, val_dataloader
