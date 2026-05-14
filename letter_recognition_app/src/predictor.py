import torch

from .config import DEVICE, MODEL_PATH
from .image_processing import grid_to_tensor
from .model import BetterCNN, LegacyCNN


def load_state_dict(model_path):
    try:
        # 新版本 PyTorch 支持 weights_only，更适合只加载权重文件
        return torch.load(model_path, map_location=DEVICE, weights_only=True)
    except TypeError:
        return torch.load(model_path, map_location=DEVICE)


def build_model_for_state_dict(state_dict, num_classes: int):
    if any(key.startswith("net.") for key in state_dict):
        return LegacyCNN(num_classes=num_classes)
    return BetterCNN(num_classes=num_classes)


def load_model(num_classes: int, model_path=MODEL_PATH):
    state_dict = load_state_dict(model_path)
    model = build_model_for_state_dict(state_dict, num_classes)
    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()
    return model


def predict_grid(model, grid, labels: list[str]):
    tensor = grid_to_tensor(grid).to(DEVICE)

    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1)[0]
        top_probs, top_ids = torch.topk(probs, k=min(3, len(labels)))

    # 返回前三个结果，GUI 可以显示置信度
    return [(labels[i], float(prob)) for prob, i in zip(top_probs.cpu(), top_ids.cpu())]
