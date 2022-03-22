from typing import Tuple
from torch.utils.data import random_split, DataLoader

from .digitdataset import DigitDataset
from .transform import get_transforms

def build_dataloader(
    root: str="data/train.csv",
    ratio: float=0.8,
    batch_size: int=64
) -> Tuple[DataLoader, DataLoader]:
    dataset = DigitDataset(root=root, transforms=get_transforms())

    len_data = len(dataset)
    len_train = int(len_data * ratio)
    len_val = len_data - len_train

    train_dataset, val_dataset = random_split(
        dataset=dataset,
        lengths=[len_train, len_val]
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
