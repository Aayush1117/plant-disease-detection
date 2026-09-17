import tensorflow as tf

TEST_DIR = "/mnt/c/PlantData/New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)/valid"

test_ds = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    image_size=(224,224),
    batch_size=8,
    shuffle=False
)

model = tf.keras.models.load_model("models/cnn_baseline.keras", compile=False)

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

loss, acc = model.evaluate(test_ds, verbose=1)

print(f"Accuracy: {acc*100:.2f}%")
print(f"Loss: {loss:.4f}")