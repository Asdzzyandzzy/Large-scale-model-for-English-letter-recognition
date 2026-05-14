from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import tqdm

from .config import DEVICE, LEARNING_RATE


def train_one_epoch(model, loader, optimizer):
    model.train()
    total_loss = 0.0

    for images, labels in tqdm(loader, desc="training", leave=False):
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()
        loss = F.cross_entropy(model(images), labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / max(len(loader), 1)


def evaluate(model, loader):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)
            prediction = model(images).argmax(dim=1)
            correct += (prediction == labels).sum().item()
            total += labels.size(0)

    return correct / total if total else 0.0


def train_model(model, train_loader, test_loader, epochs: int, model_path: Path):
    model.to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_acc = 0.0
    for epoch in range(1, epochs + 1):
        loss = train_one_epoch(model, train_loader, optimizer)
        acc = evaluate(model, test_loader)
        scheduler.step()

        print(f"Epoch {epoch:02d}: loss={loss:.4f}, test_acc={acc:.2%}")

        # 只保存当前最好的模型，避免最后一轮波动导致效果变差
        if acc >= best_acc:
            best_acc = acc
            torch.save(model.state_dict(), model_path)
            print(f"Saved best model to {model_path} ({best_acc:.2%})")

    return best_acc
