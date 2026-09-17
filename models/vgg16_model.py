import os
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models
from tensorflow.keras.applications import VGG16

# ======================================================
# DATASET PATH (Project Folder)
# ======================================================
TRAIN_DIR = "dataset/train"
VALID_DIR = "dataset/valid"

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
NUM_CLASSES = 38
EPOCHS = 10

# ======================================================
# LOAD DATASET
# ======================================================
train_ds = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=True
)

valid_ds = tf.keras.utils.image_dataset_from_directory(
    VALID_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)

AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.map(
    lambda x, y: (tf.keras.applications.vgg16.preprocess_input(x), y)
).prefetch(AUTOTUNE)

valid_ds = valid_ds.map(
    lambda x, y: (tf.keras.applications.vgg16.preprocess_input(x), y)
).prefetch(AUTOTUNE)

# ======================================================
# VGG16 BASE MODEL
# ======================================================
base_model = VGG16(
    weights="imagenet",
    include_top=False,
    input_shape=(224, 224, 3)
)

base_model.trainable = False

# ======================================================
# BUILD MODEL
# ======================================================
model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(256, activation="relu"),
    layers.Dropout(0.5),
    layers.Dense(NUM_CLASSES, activation="softmax")
])

# ======================================================
# COMPILE
# ======================================================
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# ======================================================
# EARLY STOPPING
# ======================================================
early_stop = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=3,
    restore_best_weights=True
)

# ======================================================
# TRAIN MODEL
# ======================================================
history = model.fit(
    train_ds,
    validation_data=valid_ds,
    epochs=EPOCHS,
    callbacks=[early_stop]
)

# ======================================================
# SAVE MODEL
# ======================================================
os.makedirs("models", exist_ok=True)
model.save("models/vgg16_model.keras")

print("VGG16 Model Saved Successfully!")

# ======================================================
# SAVE ACCURACY GRAPH
# ======================================================
os.makedirs("results", exist_ok=True)

plt.figure(figsize=(8,5))
plt.plot(history.history["accuracy"], marker="o",
         linewidth=2, label="Training Accuracy")
plt.plot(history.history["val_accuracy"], marker="s",
         linewidth=2, label="Validation Accuracy")

plt.title("Training vs Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)
plt.savefig("results/accuracy.png", dpi=300)
plt.close()

# ======================================================
# SAVE LOSS GRAPH
# ======================================================
plt.figure(figsize=(8,5))
plt.plot(history.history["loss"], marker="o",
         linewidth=2, label="Training Loss")
plt.plot(history.history["val_loss"], marker="s",
         linewidth=2, label="Validation Loss")

plt.title("Training vs Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.savefig("results/loss.png", dpi=300)
plt.close()

# ======================================================
# PRINT FINAL RESULT
# ======================================================
print("\n========== TRAINING SUMMARY ==========")
print(f"Training Accuracy      : {history.history['accuracy'][-1]*100:.2f}%")
print(f"Validation Accuracy    : {history.history['val_accuracy'][-1]*100:.2f}%")
print(f"Training Loss          : {history.history['loss'][-1]:.4f}")
print(f"Validation Loss        : {history.history['val_loss'][-1]:.4f}")
print("Graphs saved in results folder!")