import sys

import numpy as np
import pygame

from .config import IMAGE_SIZE
from .predictor import predict_grid


CELL_SIZE = 16
SIDE_PANEL = 220
BRUSH_RADIUS = 1


def draw_brush(grid: np.ndarray, row: int, col: int):
    # 用小笔刷填周围格子，鼠标移动快的时候线条不会断得太明显
    for r in range(row - BRUSH_RADIUS, row + BRUSH_RADIUS + 1):
        for c in range(col - BRUSH_RADIUS, col + BRUSH_RADIUS + 1):
            if 0 <= r < IMAGE_SIZE and 0 <= c < IMAGE_SIZE:
                grid[r, c] = 1


def run_gui(model, labels: list[str]):
    pygame.init()
    canvas_size = IMAGE_SIZE * CELL_SIZE
    screen = pygame.display.set_mode((canvas_size + SIDE_PANEL, canvas_size))
    pygame.display.set_caption("EMNIST Letter Recognition")

    grid = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8)
    font_big = pygame.font.SysFont("arial", 60)
    font_small = pygame.font.SysFont("arial", 22)
    result = []
    drawing = False

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                drawing = True
            if event.type == pygame.MOUSEBUTTONUP and event.button == 1:
                drawing = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_c:
                    grid.fill(0)
                    result = []
                if event.key == pygame.K_RETURN:
                    result = predict_grid(model, grid, labels)

        if drawing and pygame.mouse.get_pressed()[0]:
            x, y = pygame.mouse.get_pos()
            if x < canvas_size and y < canvas_size:
                row = y // CELL_SIZE
                col = x // CELL_SIZE
                draw_brush(grid, row, col)

        screen.fill((245, 246, 248))

        # 绘制 28x28 输入区域：黑底白字，和 EMNIST 训练图片更接近
        for row in range(IMAGE_SIZE):
            for col in range(IMAGE_SIZE):
                value = 255 if grid[row, col] else 20
                rect = pygame.Rect(col * CELL_SIZE, row * CELL_SIZE, CELL_SIZE, CELL_SIZE)
                pygame.draw.rect(screen, (value, value, value), rect)

        for i in range(IMAGE_SIZE + 1):
            pos = i * CELL_SIZE
            pygame.draw.line(screen, (70, 70, 70), (pos, 0), (pos, canvas_size), 1)
            pygame.draw.line(screen, (70, 70, 70), (0, pos), (canvas_size, pos), 1)

        panel_x = canvas_size + 24
        title = font_small.render("Press Enter to predict", True, (30, 30, 30))
        clear = font_small.render("Press C to clear", True, (30, 30, 30))
        screen.blit(title, (panel_x, 28))
        screen.blit(clear, (panel_x, 58))

        if result:
            best_text = font_big.render(result[0][0], True, (20, 20, 20))
            screen.blit(best_text, (panel_x, 120))

            y = 205
            for label, prob in result:
                line = font_small.render(f"{label}: {prob:.1%}", True, (45, 45, 45))
                screen.blit(line, (panel_x, y))
                y += 30

        pygame.display.flip()
