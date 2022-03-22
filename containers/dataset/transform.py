from typing import Dict
from torchvision import transforms


def get_transforms() -> Dict[str, transforms.Compose]:
    transform = {
        "train": transforms.Compose([
            transforms.ToTensor(),
            transforms.RandomCrop(size=20),
            transforms.Resize(size=28),
            transforms.RandomRotation(degrees=20),
            transforms.Normalize(mean=(0.5,), std=(0.5,))
        ]),
        "val": transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5,), std=(0.5,))
        ]),
    }

    return transform
