from flask import Flask, render_template, request
import numpy as np
import cv2
import os
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model

# -------------------------------
# Base Path
# -------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# -------------------------------
# Flask Setup
# -------------------------------
app = Flask(
    __name__,
    template_folder=os.path.join(BASE_DIR, "..", "Frontend", "templates"),
    static_folder=os.path.join(BASE_DIR, "..", "Frontend", "static")
)

# -------------------------------
# Load Models
# -------------------------------
IMG_SIZE = 64

# Load leaf model (Keras .h5)
leaf_model = load_model(os.path.join(BASE_DIR, "leaf_model.h5"))

# Load weed model — supports both Keras (.h5) and sklearn (.pkl)
weed_model = None
weed_model_type = None

weed_h5_path = os.path.join(BASE_DIR, "weed_model.h5")
weed_pkl_path = os.path.join(BASE_DIR, "weed_model.pkl")

if os.path.exists(weed_h5_path):
    weed_model = load_model(weed_h5_path)
    weed_model_type = "keras"
    print("[INFO] Loaded weed_model.h5 (Keras)")
elif os.path.exists(weed_pkl_path):
    import joblib
    weed_model = joblib.load(weed_pkl_path)
    weed_model_type = "sklearn"
    print("[INFO] Loaded weed_model.pkl (sklearn)")
else:
    print("[WARNING] No weed model found. Weed detection will be unavailable.")

# -------------------------------
# Preprocess Image
# -------------------------------
def preprocess_for_keras(image_path):
    """Preprocess image for a Keras CNN model — returns (1, 64, 64, 3)."""
    img = cv2.imread(image_path)
    if img is None:
        return None
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img.astype("float32") / 255.0
    img = img.reshape(1, IMG_SIZE, IMG_SIZE, 3)
    return img

def preprocess_for_sklearn(image_path):
    """Preprocess image for a sklearn model — returns (1, 64*64*3)."""
    img = cv2.imread(image_path)
    if img is None:
        return None
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img.flatten().reshape(1, -1)
    return img

# -------------------------------
# Route
# -------------------------------
@app.route("/", methods=["GET", "POST"])
def index():

    if request.method == "POST":

        file = request.files["file"]

        if file.filename == "":
            return "No file selected"

        # Upload folder
        upload_folder = os.path.join(BASE_DIR, "..", "Frontend", "static")
        os.makedirs(upload_folder, exist_ok=True)

        filename = secure_filename(file.filename)
        filepath = os.path.join(upload_folder, filename)
        file.save(filepath)

        selected_model = request.form.get("model")

        # ---------------------------
        # Leaf Health (Keras)
        # ---------------------------
        if selected_model == "leaf":
            img = preprocess_for_keras(filepath)
            if img is None:
                return render_template("index.html", result="Error reading image", image_path=filename)
            pred = np.argmax(leaf_model.predict(img), axis=1)[0]
            result = "Healthy" if pred == 0 else "Unhealthy"

        # ---------------------------
        # Weed Detection
        # ---------------------------
        elif selected_model == "weed":
            if weed_model is None:
                result = "Weed model not found. Please add weed_model.h5 or weed_model.pkl to the Backend folder."
            elif weed_model_type == "keras":
                img = preprocess_for_keras(filepath)
                if img is None:
                    return render_template("index.html", result="Error reading image", image_path=filename)
                pred = np.argmax(weed_model.predict(img), axis=1)[0]
                result = "Weed Detected" if pred == 1 else "No Weed"
            else:
                # sklearn model
                img = preprocess_for_sklearn(filepath)
                if img is None:
                    return render_template("index.html", result="Error reading image", image_path=filename)
                pred = weed_model.predict(img)[0]
                result = "Weed Detected" if pred == 1 else "No Weed"

        else:
            result = "Invalid selection"

        return render_template(
            "index.html",
            result=result,
            image_path=filename
        )

    return render_template("index.html")


# -------------------------------
# Run
# -------------------------------
if __name__ == "__main__":
    app.run(debug=True)
