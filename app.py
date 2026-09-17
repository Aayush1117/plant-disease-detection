from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import cv2
import os
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt

app = Flask(__name__)

# =========================
# MODEL
# =========================
model = tf.keras.models.load_model(
    "models/vgg16_model.keras",
    compile=False
)

# =========================
# DATASET PATH (WINDOWS)
# Change only if your dataset is somewhere else
# =========================
DATASET_DIR = "dataset/train"

class_names = sorted(tf.io.gfile.listdir(DATASET_DIR))

# =========================
# FOLDERS
# =========================
UPLOAD_FOLDER = "static/uploads"
CHART_FOLDER = "static/charts"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(CHART_FOLDER, exist_ok=True)

HISTORY_FILE = "history.csv"

# Create CSV if not exists
if not os.path.exists(HISTORY_FILE):
    pd.DataFrame(columns=[
        "Date", "Plant", "Disease",
        "Confidence", "Severity"
    ]).to_csv(HISTORY_FILE, index=False)


# =========================
# DISEASE INFO
# =========================
disease_info = {
    "Apple___Black_rot": {
        "info": "Black Rot is a fungal disease that creates dark brown or black circular lesions on leaves and fruits.",
        "treatment": "Spray Mancozeb or Captan fungicide every 10–14 days and remove infected leaves.",
        "prevention": "Prune infected branches, destroy fallen leaves and maintain proper air circulation."
    },

    "Strawberry___Leaf_scorch": {
        "info": "Leaf Scorch is a fungal disease that causes purple or brown spots and drying of leaf edges.",
        "treatment": "Apply appropriate fungicide and remove severely infected leaves.",
        "prevention": "Avoid overhead irrigation and maintain good field sanitation."
    }
}


# =========================
# HOME
# =========================
@app.route("/", methods=["GET", "POST"])
def home():

    prediction = None
    confidence = None
    severity = None
    img_path = None
    infected_path = None
    details = None

    if request.method == "POST":

        file = request.files["image"]

        if file:

            filepath = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(filepath)

            # ---------- Prediction ----------
            img = image.load_img(filepath, target_size=(224, 224))
            img = image.img_to_array(img)
            img = np.expand_dims(img, axis=0)
            img = tf.keras.applications.vgg16.preprocess_input(img)

            pred = model.predict(img, verbose=0)

            idx = np.argmax(pred)
            prediction = class_names[idx]
            confidence = round(float(np.max(pred)) * 100, 2)

            # ---------- Severity Detection ----------
            original = cv2.imread(filepath)

            hsv = cv2.cvtColor(original, cv2.COLOR_BGR2HSV)

            lower = np.array([5, 60, 20])
            upper = np.array([35, 255, 255])

            mask = cv2.inRange(hsv, lower, upper)

            kernel = np.ones((3, 3), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

            contours, _ = cv2.findContours(
                mask,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE
            )

            infected = original.copy()
            cv2.drawContours(infected, contours, -1, (0, 0, 255), 2)

            infected_name = "result_" + file.filename
            infected_path = os.path.join(UPLOAD_FOLDER, infected_name)

            cv2.imwrite(infected_path, infected)

            total = original.shape[0] * original.shape[1]
            infected_pixels = cv2.countNonZero(mask)

            severity = round((infected_pixels / total) * 100, 2)

            img_path = filepath.replace("\\", "/")
            infected_path = infected_path.replace("\\", "/")

            # ---------- Save History ----------
            plant = prediction.split("___")[0]
            disease = prediction.split("___")[1]

            row = pd.DataFrame([{
                "Date": datetime.now().strftime("%d-%m-%Y %H:%M"),
                "Plant": plant,
                "Disease": disease,
                "Confidence": confidence,
                "Severity": severity
            }])

            row.to_csv(
                HISTORY_FILE,
                mode="a",
                header=False,
                index=False
            )

            # ---------- Disease Info ----------
            details = disease_info.get(
                prediction,
                {
                    "info": "Information not available.",
                    "treatment": "Consult agriculture expert.",
                    "prevention": "Maintain healthy crop practices."
                }
            )

    return render_template(
        "index.html",
        prediction=prediction,
        confidence=confidence,
        severity=severity,
        img_path=img_path,
        infected_path=infected_path,
        details=details
    )


# =========================
# HISTORY
# =========================
@app.route("/history")
def history():

    rows = pd.read_csv(HISTORY_FILE)

    return render_template(
        "history.html",
        rows=rows.values.tolist()
    )


# =========================
# DASHBOARD
# =========================
@app.route("/dashboard")
def dashboard():

    df = pd.read_csv(HISTORY_FILE)

    labels = df["Disease"].value_counts().index
    values = df["Disease"].value_counts().values

    plt.figure(figsize=(6, 6))
    plt.pie(
        values,
        labels=labels,
        autopct="%1.1f%%"
    )

    plt.title("Disease Distribution")

    chart_path = "static/charts/chart.png"

    plt.savefig(chart_path)
    plt.close()

    return render_template(
        "dashboard.html",
        total=len(df),
        avg_conf=round(df["Confidence"].mean(), 2),
        avg_severity=round(df["Severity"].mean(), 2),
        chart=chart_path
    )


# =========================
# RUN
# =========================
if __name__ == "__main__":
    app.run(debug=True)