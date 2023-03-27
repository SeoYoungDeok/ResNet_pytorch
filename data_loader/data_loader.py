import torch
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
import albumentations as A
from albumentations import pytorch
import os


class OxfordPetDataset(Dataset):
    def __init__(self, path, train=False):
        super(OxfordPetDataset, self).__init__()
        os.makedirs(path, exist_ok=True)
        self.train = train

        if self.train:
            self.dataset = datasets.OxfordIIITPet(
                "data", split="trainval", download=True
            )
        else:
            self.dataset = datasets.OxfordIIITPet("data", split="test", download=True)

        self.train_transform = A.Compose(
            [
                A.OneOf([A.GaussNoise(p=0.5), A.GaussianBlur(p=0.5)], p=0.75),
                A.Resize(250, 250),
                A.RandomCrop(224, 224),
                A.OneOf(
                    [
                        A.RandomRotate90(p=0.5),
                        A.HorizontalFlip(p=0.5),
                        A.VerticalFlip(p=0.5),
                    ],
                    p=0.75,
                ),
                A.ColorJitter(),
                A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
                pytorch.transforms.ToTensorV2(),
            ]
        )
        self.test_transform = A.Compose(
            [
                A.Resize(224, 224),
                A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
                A.pytorch.transforms.ToTensorV2(),
            ]
        )

        self.classes = self.dataset.classes
        self.class_idx = self.dataset.class_to_idx

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, label = self.dataset[idx]

        if self.train:
            img = self.train_transform(image=np.array(img))["image"]
            label = torch.tensor(label)
        else:
            img = self.test_transform(image=np.array(img))["image"]
            label = torch.tensor(label)

        return img, label


class Cifar10Dataset(Dataset):
    def __init__(self, path, train=False):
        super(Cifar10Dataset, self).__init__()
        os.makedirs(path, exist_ok=True)
        self.train = train

        if self.train:
            self.dataset = datasets.CIFAR10("data", train=True, download=True)
        else:
            self.dataset = datasets.CIFAR10("data", train=False, download=True)

        self.train_transform = A.Compose(
            [
                A.GaussNoise(p=0.5),
                A.Resize(132, 132),
                A.RandomCrop(128, 128),
                A.OneOf(
                    [
                        A.RandomRotate90(p=0.5),
                        A.HorizontalFlip(p=0.5),
                        A.VerticalFlip(p=0.5),
                    ],
                    p=0.75,
                ),
                A.ColorJitter(),
                A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
                pytorch.transforms.ToTensorV2(),
            ]
        )
        self.test_transform = A.Compose(
            [
                A.Resize(128, 128),
                A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
                A.pytorch.transforms.ToTensorV2(),
            ]
        )

        self.classes = self.dataset.classes
        self.class_idx = self.dataset.class_to_idx

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, label = self.dataset[idx]

        if self.train:
            img = self.train_transform(image=np.array(img))["image"]
            label = torch.tensor(label)
        else:
            img = self.test_transform(image=np.array(img))["image"]
            label = torch.tensor(label)

        return img, label


def get_dataloader(path, batch_size):
    train_dataset = Cifar10Dataset(path, train=True)
    test_dataset = Cifar10Dataset(path, train=False)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size)

    return train_dataloader, test_dataloader


# import matplotlib.pyplot as plt
# import numpy as np

# train_loader, test_loader = get_dataloader("data", 32)

# imgs, labels = next(iter(train_loader))

# fig, axes = plt.subplots(figsize=(10, 10), nrows=8, ncols=4)

# for i, ax in enumerate(axes.flatten()):
#     ax.imshow(imgs.permute(0, 2, 3, 1)[i])
