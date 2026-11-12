"""
Fashion-MNIST Image Classification using CNN
Run: python src/train_fashion_mnist.py
The script downloads Fashion-MNIST automatically through Keras.
"""
import os
import json
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

SEED = 42
tf.keras.utils.set_random_seed(SEED)
os.makedirs("outputs", exist_ok=True)
os.makedirs("models", exist_ok=True)

CLASS_NAMES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]

def load_data():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
    # Scale pixel values from [0, 255] to [0, 1] and add channel dimension.
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0
    x_train = np.expand_dims(x_train, axis=-1)
    x_test = np.expand_dims(x_test, axis=-1)
    return (x_train, y_train), (x_test, y_test)

def build_model(use_augmentation=True):
    inputs = tf.keras.Input(shape=(28, 28, 1))
    x = inputs

    if use_augmentation:
        x = tf.keras.Sequential([
            tf.keras.layers.RandomRotation(0.05),
            tf.keras.layers.RandomZoom(0.10),
            tf.keras.layers.RandomTranslation(0.05, 0.05),
        ], name="data_augmentation")(x)

    # Block 1: learn simple edges and textures.
    x = tf.keras.layers.Conv2D(32, (3, 3), padding="same", use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)

    # Block 2: learn more complex shapes.
    x = tf.keras.layers.Conv2D(64, (3, 3), padding="same", use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)

    # Block 3.
    x = tf.keras.layers.Conv2D(128, (3, 3), padding="same", use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)

    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.40)(x)
    outputs = tf.keras.layers.Dense(10, activation="softmax")(x)

    model = tf.keras.Model(inputs, outputs, name="fashion_mnist_cnn")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model

def save_training_plots(history):
    hist = history.history
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(hist["accuracy"], label="Training accuracy")
    axes[0].plot(hist["val_accuracy"], label="Validation accuracy")
    axes[0].set_title("Model Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()
    axes[1].plot(hist["loss"], label="Training loss")
    axes[1].plot(hist["val_loss"], label="Validation loss")
    axes[1].set_title("Model Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()
    plt.tight_layout()
    plt.savefig("outputs/training_curves.png", dpi=150)
    plt.close()

def main():
    (x_train, y_train), (x_test, y_test) = load_data()

    model = build_model(use_augmentation=True)
    model.summary()

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=4, restore_best_weights=True
        ),
        tf.keras.callbacks.ModelCheckpoint(
            "models/best_fashion_mnist_cnn.keras",
            monitor="val_accuracy", save_best_only=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6
        )
    ]

    history = model.fit(
        x_train, y_train,
        validation_split=0.10,
        epochs=20,
        batch_size=64,
        callbacks=callbacks,
        verbose=1
    )
    save_training_plots(history)

    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
    probabilities = model.predict(x_test, verbose=0)
    predictions = np.argmax(probabilities, axis=1)

    report = classification_report(
        y_test, predictions, target_names=CLASS_NAMES, output_dict=True
    )
    with open("outputs/classification_report.json", "w") as f:
        json.dump(report, f, indent=2)

    cm = confusion_matrix(y_test, predictions)
    display = ConfusionMatrixDisplay(cm, display_labels=CLASS_NAMES)
    fig, ax = plt.subplots(figsize=(11, 11))
    display.plot(ax=ax, xticks_rotation=45, cmap="Blues", colorbar=False)
    plt.title("Fashion-MNIST Confusion Matrix")
    plt.tight_layout()
    plt.savefig("outputs/confusion_matrix.png", dpi=150)
    plt.close()

    metrics = {
        "test_loss": float(test_loss),
        "test_accuracy": float(test_accuracy),
        "epochs_completed": len(history.history["loss"])
    }
    with open("outputs/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    model.save("models/final_fashion_mnist_cnn.keras")
    print("\nTEST RESULTS")
    print(f"Test loss: {test_loss:.4f}")
    print(f"Test accuracy: {test_accuracy:.4f}")
    print("Saved model, metrics, classification report and plots in outputs/ and models/.")

if __name__ == "__main__":
    main()
