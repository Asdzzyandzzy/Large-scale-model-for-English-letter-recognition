import numpy as np
from PIL import Image
from torchvision import transforms

from .config import DRAWING_SIZE, IMAGE_SIZE


INFERENCE_TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),
])


def grid_to_emnist_image(grid: np.ndarray) -> Image.Image:
    """把 28x28 手写网格整理成接近 EMNIST 的输入图片。"""
    grid = np.asarray(grid, dtype=np.uint8)
    active = np.argwhere(grid > 0)

    if active.size == 0:
        return Image.new("L", (IMAGE_SIZE, IMAGE_SIZE), 0)

    # 先裁掉空白区域，避免用户写得太偏时模型看不清
    y_min, x_min = active.min(axis=0)
    y_max, x_max = active.max(axis=0) + 1
    crop = grid[y_min:y_max, x_min:x_max] * 255

    height, width = crop.shape
    side = max(height, width)
    square = np.zeros((side, side), dtype=np.uint8)
    y_offset = (side - height) // 2
    x_offset = (side - width) // 2
    square[y_offset:y_offset + height, x_offset:x_offset + width] = crop

    # EMNIST/MNIST 常见做法：字符主体约 20x20，再放回 28x28 中心
    image = Image.fromarray(square, mode="L").resize((DRAWING_SIZE, DRAWING_SIZE), Image.Resampling.LANCZOS)
    canvas = Image.new("L", (IMAGE_SIZE, IMAGE_SIZE), 0)
    paste_xy = ((IMAGE_SIZE - DRAWING_SIZE) // 2, (IMAGE_SIZE - DRAWING_SIZE) // 2)
    canvas.paste(image, paste_xy)
    return canvas


def grid_to_tensor(grid: np.ndarray):
    image = grid_to_emnist_image(grid)
    return INFERENCE_TRANSFORM(image).unsqueeze(0)
