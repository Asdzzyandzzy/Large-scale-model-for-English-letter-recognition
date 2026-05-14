import argparse

from src.config import BATCH_SIZE, EPOCHS, MODEL_PATH
from src.data import build_dataloaders, get_dataset_info
from src.model import BetterCNN
from src.trainer import train_model


def main():
    parser = argparse.ArgumentParser(description="Train an EMNIST recognition model.")
    parser.add_argument("--split", choices=["byclass", "letters"], default="byclass")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--download", action="store_true")
    args = parser.parse_args()

    info = get_dataset_info(args.split)
    train_loader, test_loader = build_dataloaders(
        split=args.split,
        batch_size=args.batch_size,
        download=args.download,
    )

    model = BetterCNN(num_classes=info.num_classes)
    best_acc = train_model(model, train_loader, test_loader, args.epochs, MODEL_PATH)
    print(f"Best test accuracy: {best_acc:.2%}")


if __name__ == "__main__":
    main()
