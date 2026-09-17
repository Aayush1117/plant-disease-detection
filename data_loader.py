import tensorflow as tf

TRAIN_DIR = r"C:\PlantData\New Plant Diseases Dataset(Augmented)\New Plant Diseases Dataset(Augmented)\train"
VALID_DIR = r"C:\PlantData\New Plant Diseases Dataset(Augmented)\New Plant Diseases Dataset(Augmented)\valid"

IMG_SIZE = (224, 224)
BATCH_SIZE = 32

print("Loading training dataset...")

train_ds = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=True
)

print("\nLoading validation dataset...")

valid_ds = tf.keras.utils.image_dataset_from_directory(
    VALID_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)

print("\nNumber of classes:", len(train_ds.class_names))

print("\nClasses:")
for i, class_name in enumerate(train_ds.class_names):
    print(i, ":", class_name)

print("\nDataset loaded successfully!")