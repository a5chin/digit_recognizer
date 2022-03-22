import numpy as np
import pandas as pd
from typing import Optional
from torch.utils.data import Dataset
from torchvision.transforms import Compose
from pathlib import Path


class DigitDataset(Dataset):
    def __init__(self, root: str="data/train.csv", transforms: Optional[Compose]=None) -> None:
        super().__init__()
        self.root = Path(root)
        self.df = DigitDataset._read_csv(root=root)
        self.transforms = transforms

    def __getitem__(self, index: int):
        label = self.df.iat[index, 0]

        image = self.df.iloc[index, 1: ].values.astype(np.float32)
        image = image.reshape(28, 28)
        image = self.transforms(image)

        return image, label

    def __len__(self):
        return len(self.df)

    @staticmethod
    def _read_csv(root: str) -> pd.DataFrame:
        return pd.read_csv(root)
