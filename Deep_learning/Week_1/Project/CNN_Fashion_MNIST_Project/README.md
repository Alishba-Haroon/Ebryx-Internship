# Fashion-MNIST Image Classification using CNN

## Project overview
This project trains a Convolutional Neural Network (CNN) to classify 28x28 grayscale clothing images into 10 Fashion-MNIST classes.

## Main features
- Convolution layers and learned filters
- ReLU activation and feature maps
- Same padding and stride 1 (Keras default for Conv2D)
- Max pooling with 2x2 windows
- Batch Normalization
- Dropout regularization
- Data augmentation (rotation, zoom, translation)
- Early stopping and learning-rate reduction
- Test evaluation, classification report and confusion matrix

## Setup (Windows / Linux / Google Colab)
```bash
python -m venv .venv
```
Activate the environment, then:
```bash
pip install -r requirements.txt
python src/train_fashion_mnist.py
```
The Fashion-MNIST dataset downloads automatically on the first run. Training can take several minutes depending on hardware.

## Outputs
The script creates:
- `models/best_fashion_mnist_cnn.keras`
- `models/final_fashion_mnist_cnn.keras`
- `outputs/training_curves.png`
- `outputs/confusion_matrix.png`
- `outputs/metrics.json`
- `outputs/classification_report.json`

## Model architecture
Input (28x28x1) -> Conv2D(32) -> BatchNorm -> ReLU -> MaxPool ->
Conv2D(64) -> BatchNorm -> ReLU -> MaxPool ->
Conv2D(128) -> BatchNorm -> ReLU -> MaxPool ->
Flatten -> Dense(128) -> Dropout(0.40) -> Dense(10, Softmax).

## Bonus: custom-dataset transfer learning
Put your images in class-specific folders under `data/custom_dataset/`, with at least two class folders. Then run:
```bash
python src/train_custom_transfer_learning.py
```
The bonus script uses ImageNet-pretrained MobileNetV2. It is separate from the Fashion-MNIST experiment because Fashion-MNIST images are grayscale and only 28x28, while MobileNetV2 expects RGB images of a larger size.

## Important reporting note
Do not invent accuracy values. Run the experiment and copy the actual test accuracy and generated plots into the report.
