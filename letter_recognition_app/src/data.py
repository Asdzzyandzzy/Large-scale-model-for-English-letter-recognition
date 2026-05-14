from dataclasses import dataclass

import torch
import torchvision
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from .config import BATCH_SIZE, DATA_DIR, NUM_WORKERS


@dataclass
class DatasetInfo:
    split: str
    num_classes: int
    labels: list[str]


class LabelOffsetDataset(Dataset):
    """把 EMNIST letters 的 1-26 标签改成 0-25。"""

    def __init__(self, dataset: Dataset, offset: int):
        self.dataset = dataset
        self.offset = offset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        image, label = self.dataset[index]
        return image, int(label) - self.offset


def get_dataset_info(split: str) -> DatasetInfo:
    if split == "letters":
        return DatasetInfo(split="letters", num_classes=26, labels=list("ABCDEFGHIJKLMNOPQRSTUVWXYZ"))
    if split == "byclass":
        labels = list("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz")
        return DatasetInfo(split="byclass", num_classes=62, labels=labels)
    raise ValueError("split must be 'byclass' or 'letters'")


def build_train_transform():
    return transforms.Compose([
        # 轻微旋转和平移可以让模型更适应手写输入，但幅度不能太大
        transforms.RandomAffine(degrees=10, translate=(0.08, 0.08), scale=(0.90, 1.10)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])


def build_eval_transform():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])


def build_dataset(split: str, train: bool, download: bool):
    transform = build_train_transform() if train else build_eval_transform()
    dataset = torchvision.datasets.EMNIST(
        root=DATA_DIR,
        split=split,
        train=train,
        download=download,
        transform=transform,
    )

    # letters 数据集的标签从 1 开始，训练前必须减 1
    if split == "letters":
        dataset = LabelOffsetDataset(dataset, offset=1)
    return dataset


def build_dataloaders(split: str, batch_size: int = BATCH_SIZE, download: bool = False):
    train_dataset = build_dataset(split=split, train=True, download=download)
    test_dataset = build_dataset(split=split, train=False, download=download)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )
    return train_loader, test_loader
