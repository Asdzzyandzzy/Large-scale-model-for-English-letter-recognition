import argparse

from src.config import BATCH_SIZE, MODEL_PATH
from src.data import build_dataloaders, get_dataset_info
from src.predictor import load_model
from src.trainer import evaluate


def main():
    parser = argparse.ArgumentParser(description="Evaluate the saved EMNIST model.")
    parser.add_argument("--split", choices=["byclass", "letters"], default="byclass")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--model", default=str(MODEL_PATH))
    args = parser.parse_args()

    info = get_dataset_info(args.split)
    _, test_loader = build_dataloaders(split=args.split, batch_size=args.batch_size, download=False)
    model = load_model(num_classes=info.num_classes, model_path=args.model)
    acc = evaluate(model, test_loader)
    print(f"Test accuracy: {acc:.2%}")


if __name__ == "__main__":
    main()
