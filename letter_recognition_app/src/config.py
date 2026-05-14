from pathlib import Path

import torch


# 这里集中放项目路径和训练参数，后面改起来比较方便
ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
MODEL_PATH = ROOT_DIR / "emnist_cnn.pt"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BATCH_SIZE = 128
LEARNING_RATE = 1e-3
EPOCHS = 15
NUM_WORKERS = 0

IMAGE_SIZE = 28
DRAWING_SIZE = 20

BYCLASS_LABELS = list("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz")
LETTERS_LABELS = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
