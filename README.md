# Face Mask Detection

This project builds a binary image classifier (Mask vs No Mask) using transfer learning with VGG16. The model is trained on images in two classes and then used with OpenCV to perform real-time webcam prediction.

## Project Structure

- code.ipynb: End-to-end notebook for training and live detection.
- dataset/with_mask: Images of people wearing masks.
- dataset/without_mask: Images of people not wearing masks.

## Features

- Transfer learning with VGG16 for mask classification.
- Image preprocessing to 224x224 model input size.
- Train/test split and model evaluation in notebook workflow.
- Real-time webcam inference with on-frame status labels.

## Requirements

- Python 3.9+ recommended.
- Webcam access enabled for your terminal/Jupyter environment.
- Python packages:
  - tensorflow (or keras with compatible tensorflow backend)
  - opencv-python
  - numpy
  - scikit-learn
  - matplotlib
  - jupyter

Install dependencies:

```bash
pip install -r requirements.txt
```

## Dataset Setup

Make sure your dataset is organized like this:

```text
dataset/
  with_mask/
  without_mask/
```

Each subfolder should contain only images for that class.

## How to Run

1. Start Jupyter:

```bash
jupyter notebook
```

2. Open code.ipynb.
3. Run cells in order from top to bottom.
4. During live detection, press x to close the webcam window.

## Model Pipeline

1. Load images from both class folders.
2. Preprocess and resize images to 224x224.
3. Encode labels and split data into training and testing sets.
4. Load VGG16 and replace the final layer for binary classification.
5. Train model, validate, and use it for real-time webcam predictions.

## Notes

- This repository is currently notebook-first. There is no standalone detect_mask.py script yet.
- On macOS, if webcam does not open, verify camera permission for the app/process running Jupyter.

## License

This project is licensed under the MIT License.
