import tensorflow as tf

TEST_DIR = "/mnt/c/PlantData/New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)/valid"

test_ds = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    image_size=(224,224),
    batch_size=8,
    shuffle=False
)

test_ds = test_ds.map(
    lambda x, y: (tf.keras.applications.vgg16.preprocess_input(x), y)
)

model = tf.keras.models.load_model(
    "models/vgg16_model.keras",
    compile=False
)

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

loss, acc = model.evaluate(test_ds, verbose=1)

print("\n===== FINAL RESULT =====")
print(f"Accuracy: {acc*100:.2f}%")
print(f"Loss: {loss:.4f}")