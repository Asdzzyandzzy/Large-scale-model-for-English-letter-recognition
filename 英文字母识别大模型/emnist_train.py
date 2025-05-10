# -*- coding: utf-8 -*-
"""
cross_validate.py  ——   EMNIST‑byclass 5‑fold 交叉验证完整脚本
------------------------------------------------------------
• 使用 StrongerCNN (4×Conv + BN + Dropout) 作为基模型
• StratifiedKFold 保证 62 类在每折中分布一致
• 每折训练 10 epoch，保存模型到 model_fold{i}.pt
• 最终输出各折准确率与平均 ± 标准差

运行示例：
    python cross_validate.py            # CPU 或自动用 GPU
    python cross_validate.py --epochs 8 # 自定义每折 epoch 数

脚本假设：
    1. 你已将 EMNIST 数据放到 ./data/EMNIST/raw/ 下
    2. 已安装依赖：torch torchvision sklearn numpy tqdm pillow
"""

# ---------------------------------------------------------------
# 1. 标准库 & 配置
# ---------------------------------------------------------------
import sys, time, argparse
from pathlib import Path
from statistics import mean, stdev

# ---------------------------------------------------------------
# 2. 第三方库
# ---------------------------------------------------------------
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm

# ---------------------------------------------------------------
# 3. 全局常量
# ---------------------------------------------------------------
DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"
DATA_DIR      = Path("data")
BATCH_SIZE    = 128
LEARNING_RATE = 1e-3
N_SPLITS      = 5                      # 折数

# EMNIST 62 类字符映射表（供调试打印，可删除）
IDX2CHAR = list("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz")

# ---------------------------------------------------------------
# 4. 数据集 & 预处理
# ---------------------------------------------------------------
transform = transforms.Compose([
    transforms.RandomAffine(degrees=15, translate=(0.1, 0.1)),  # 轻度数据增强
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

def load_full_dataset():
    """一次性加载整个 EMNIST‑byclass 训练集 Dataset"""
    return torchvision.datasets.EMNIST(root=DATA_DIR,
                                       split="byclass",
                                       train=True,
                                       download=False,
                                       transform=transform)

# ---------------------------------------------------------------
# 5. 模型定义：比 SmallCNN 更强的卷积网络
# ---------------------------------------------------------------
class StrongerCNN(nn.Module):
    def __init__(self, num_classes: int = 62):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2),  # 28→14

            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(2),  # 14→7
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 7 * 7, 512), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)

# ---------------------------------------------------------------
# 6. 单折训练 + 验证
# ---------------------------------------------------------------

def train_one_fold(model, train_loader, val_loader, epochs: int):
    model.to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=5, gamma=0.5)

    for epoch in range(1, epochs + 1):
        model.train(); running_loss = 0
        for X, y in tqdm(train_loader, leave=False, desc=f"Epoch {epoch}/{epochs}"):
            X, y = X.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            loss = F.cross_entropy(model(X), y)
            loss.backward(); opt.step(); running_loss += loss.item()
        scheduler.step()

    # ------ 验证 ------
    model.eval(); correct = total = 0
    with torch.no_grad():
        for X, y in val_loader:
            X, y = X.to(DEVICE), y.to(DEVICE)
            preds = model(X).argmax(1)
            correct += (preds == y).sum().item(); total += y.size(0)
    return correct / total  # 返回准确率

# ---------------------------------------------------------------
# 7. 交叉验证主逻辑
# ---------------------------------------------------------------

def cross_validate(epochs_per_fold: int):
    full_ds = load_full_dataset()
    y = np.array(full_ds.targets)  # 标签数组供 StratifiedKFold 使用

    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
    fold_acc = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(y)), y), 1):
        print(f"\n===== Fold {fold}/{N_SPLITS} =====")
        train_loader = DataLoader(Subset(full_ds, train_idx), batch_size=BATCH_SIZE,
                                  shuffle=True,  num_workers=4)
        val_loader   = DataLoader(Subset(full_ds, val_idx),   batch_size=BATCH_SIZE,
                                  shuffle=False, num_workers=4)

        model = StrongerCNN()
        acc = train_one_fold(model, train_loader, val_loader, epochs_per_fold)
        fold_acc.append(acc)
        # 保存模型权重
        fname = Path(f"model_fold{fold}.pt")
        torch.save(model.state_dict(), fname)
        print(f"Fold {fold} ACC = {acc:.4%} | 模型已保存为 {fname}")

    # 汇总结果
    print("\n=== 交叉验证结果 ===")
    for i, acc in enumerate(fold_acc, 1):
        print(f"Fold {i}: {acc:.4%}")
    print(f"平均准确率: {mean(fold_acc):.4%} ± {stdev(fold_acc):.4%}")

# ---------------------------------------------------------------
# 8. 主入口 & CLI
# ---------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="EMNIST 5‑fold 交叉验证")
    parser.add_argument("--epochs", type=int, default=10, help="每折训练 epoch 数 (默认 10)")
    args = parser.parse_args()

    start = time.time()
    cross_validate(args.epochs)
    print(f"\n总耗时 {(time.time()-start)/60:.1f} 分钟")

if __name__ == "__main__":
    main()
