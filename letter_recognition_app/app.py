import argparse

from src.config import MODEL_PATH
from src.data import get_dataset_info
from src.gui import run_gui
from src.predictor import load_model


def main():
    parser = argparse.ArgumentParser(description="Run the drawing recognition app.")
    parser.add_argument("--split", choices=["byclass", "letters"], default="byclass")
    parser.add_argument("--model", default=str(MODEL_PATH))
    args = parser.parse_args()

    info = get_dataset_info(args.split)
    model = load_model(num_classes=info.num_classes, model_path=args.model)
    run_gui(model, info.labels)


if __name__ == "__main__":
    main()
