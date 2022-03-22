from torchvision import transforms


def get_transforms():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomCrop(size=20),
        transforms.Resize(size=28),
        transforms.RandomRotation(degrees=20),
        transforms.Normalize(mean=(0.5,), std=(0.5,))
    ])

    return transform
