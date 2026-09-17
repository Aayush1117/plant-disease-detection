import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image

# Load model
model = tf.keras.models.load_model(
    "models/vgg16_model.keras",
    compile=False
)

# Dataset path
DATASET_DIR = "/mnt/c/PlantData/New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)/train"

# Class names
class_names = sorted(tf.io.gfile.listdir(DATASET_DIR))

# Test image path
IMG_PATH = "/mnt/c/Users/Asus/PythonProjects/Plant_Disease_Detection/test.jpg"

# Load image
img = image.load_img(IMG_PATH, target_size=(224, 224))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)

# Preprocess
img_array = tf.keras.applications.vgg16.preprocess_input(img_array)

# Predict
pred = model.predict(img_array, verbose=0)
pred_class = np.argmax(pred)
confidence = np.max(pred) * 100

print("\n===== PREDICTION RESULT =====")
print("Disease:", class_names[pred_class])
print(f"Confidence: {confidence:.2f}%")