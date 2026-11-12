"""
Optional bonus: transfer learning on a custom image dataset.
Expected folder structure:
data/custom_dataset/
    class_one/
        image1.jpg
    class_two/
        image2.jpg

Run: python src/train_custom_transfer_learning.py
Images are resized to 160x160 and loaded with ImageDataGenerator.
"""
import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

DATA_DIR = "data/custom_dataset"
IMG_SIZE = (160, 160)
BATCH_SIZE = 32

if not os.path.isdir(DATA_DIR) or not any(
    os.path.isdir(os.path.join(DATA_DIR, name)) for name in os.listdir(DATA_DIR)
):
    raise SystemExit(
        "Add at least two class folders with images inside data/custom_dataset/ before running."
    )

datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=15,
    zoom_range=0.10,
    width_shift_range=0.10,
    height_shift_range=0.10,
    horizontal_flip=True
)

train_data = datagen.flow_from_directory(
    DATA_DIR, target_size=IMG_SIZE, batch_size=BATCH_SIZE,
    class_mode="categorical", subset="training", seed=42
)
val_data = datagen.flow_from_directory(
    DATA_DIR, target_size=IMG_SIZE, batch_size=BATCH_SIZE,
    class_mode="categorical", subset="validation", seed=42
)

base_model = tf.keras.applications.MobileNetV2(
    input_shape=IMG_SIZE + (3,), include_top=False,
    weights="imagenet"
)
base_model.trainable = False

inputs = tf.keras.Input(shape=IMG_SIZE + (3,))
x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs * 255.0)
x = base_model(x, training=False)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dropout(0.3)(x)
outputs = tf.keras.layers.Dense(train_data.num_classes, activation="softmax")(x)
model = tf.keras.Model(inputs, outputs)

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-3),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)
model.fit(
    train_data, validation_data=val_data, epochs=10,
    callbacks=[tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=3, restore_best_weights=True
    )]
)
os.makedirs("models", exist_ok=True)
model.save("models/custom_mobilenetv2_transfer_learning.keras")
print("Class mapping:", train_data.class_indices)
print("Saved transfer-learning model.")
