# English Letter Recognition

This is a small student project for recognizing handwritten English characters with EMNIST and PyTorch.

The old version put training, prediction, and the Pygame window in one file. I reorganized it into several smaller files so it is easier to read and change.

## Project Structure

```text
letter_recognition_app/
  app.py                 # start the drawing window
  train.py               # train the model
  evaluate.py            # test the saved model
  emnist_cnn.pt          # saved model weights
  data/                  # EMNIST data folder
  src/
    config.py            # paths and basic settings
    data.py              # dataset and dataloader code
    gui.py               # Pygame drawing window
    image_processing.py  # convert drawing grid to model input
    model.py             # CNN model
    predictor.py         # load model and predict
    trainer.py           # training and evaluation loops
```

## What Was Improved

The drawing input is now processed more like EMNIST images. The app crops the drawing, centers it, resizes it, and uses a black background with white strokes. This should help a lot because the previous version used the opposite color style, so the model often saw a very different image from the training data.

The code is also split into simple files. I kept the comments in Chinese because that makes the logic easier for me to explain while reading the code.

## Install

```bash
cd letter_recognition_app
pip install -r requirements.txt
```

## Run the App

```bash
python app.py
```

Draw a character in the grid, press `Enter` to predict, and press `C` to clear.

## Train Again

The default setting uses the EMNIST `byclass` split, which has digits, uppercase letters, and lowercase letters.

```bash
python train.py --split byclass --epochs 15
```

If the dataset is not already downloaded:

```bash
python train.py --split byclass --epochs 15 --download
```

For only uppercase letters:

```bash
python train.py --split letters --epochs 15 --download
```

## Evaluate

```bash
python evaluate.py --split byclass
```

## Notes

This is not a huge model. It is a normal CNN, but it is enough for this course-style project and runs faster on a laptop. The biggest practical improvement is the preprocessing before prediction, because a clean input image matters a lot for handwritten character recognition.
