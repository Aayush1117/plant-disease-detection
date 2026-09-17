import os
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models
from tensorflow.keras.applications import VGG19
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# ======================================================
# DATASET PATH
# ======================================================
TRAIN_DIR = "dataset/train"
VALID_DIR = "dataset/valid"

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
NUM_CLASSES = 38
EPOCHS = 25

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

train_ds = train_ds.prefetch(AUTOTUNE)
valid_ds = valid_ds.prefetch(AUTOTUNE)

# ======================================================
# DATA AUGMENTATION
# ======================================================
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.15),
    layers.RandomZoom(0.15),
    layers.RandomContrast(0.10)
])

# ======================================================
# VGG19 BASE MODEL
# ======================================================
base_model = VGG19(
    weights="imagenet",
    include_top=False,
    input_shape=(224, 224, 3)
)

base_model.trainable = False

# ======================================================
# BUILD MODEL
# ======================================================
model = models.Sequential([
    data_augmentation,
    layers.Rescaling(1./255),

    base_model,

    layers.GlobalAveragePooling2D(),

    layers.Dense(256, activation="relu"),
    layers.Dropout(0.3),

    layers.Dense(128, activation="relu"),
    layers.Dropout(0.2),

    layers.Dense(NUM_CLASSES, activation="softmax")
])

# ======================================================
# COMPILE MODEL
# ======================================================
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# ======================================================
# CALLBACKS
# ======================================================
early_stop = EarlyStopping(
    monitor="val_loss",
    patience=3,
    restore_best_weights=True,
    verbose=1
)

checkpoint = ModelCheckpoint(
    "models/vgg19_model.keras",
    monitor="val_accuracy",
    save_best_only=True,
    verbose=1
)

# ======================================================
# TRAIN MODEL
# ======================================================
history = model.fit(
    train_ds,
    validation_data=valid_ds,
    epochs=EPOCHS,
    callbacks=[early_stop, checkpoint]
)

# ======================================================
# CREATE RESULT FOLDER
# ======================================================
os.makedirs("results", exist_ok=True)

# ======================================================
# ACCURACY GRAPH
# ======================================================
plt.figure(figsize=(8,5))
plt.plot(history.history["accuracy"], label="Training Accuracy", linewidth=2)
plt.plot(history.history["val_accuracy"], label="Validation Accuracy", linewidth=2)
plt.title("Training vs Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.grid(True)
plt.legend()
plt.savefig("results/accuracy.png", dpi=300)
plt.close()

# ======================================================
# LOSS GRAPH
# ======================================================
plt.figure(figsize=(8,5))
plt.plot(history.history["loss"], label="Training Loss", linewidth=2)
plt.plot(history.history["val_loss"], label="Validation Loss", linewidth=2)
plt.title("Training vs Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True)
plt.legend()
plt.savefig("results/loss.png", dpi=300)
plt.close()

# ======================================================
# FINAL RESULT
# ======================================================
print("\n========== TRAINING COMPLETED ==========")
print(f"Training Accuracy     : {history.history['accuracy'][-1]*100:.2f}%")
print(f"Validation Accuracy   : {history.history['val_accuracy'][-1]*100:.2f}%")
print(f"Training Loss         : {history.history['loss'][-1]:.4f}")
print(f"Validation Loss       : {history.history['val_loss'][-1]:.4f}")

print("\nGraphs saved in results/")
print("Best model saved as models/vgg19_model.keras")