from flask import Flask, request, jsonify, render_template
from tensorflow import keras
import numpy as np

app = Flask(__name__)

# Load your trained model
model = keras.models.load_model('mnist_cnn_model.h5')

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    img = np.array(data["image"]).astype("float32").reshape(1, 28, 28, 1)
    prediction = model.predict(img)
    digit = int(np.argmax(prediction))
    return jsonify({"digit": digit})

if __name__ == "__main__":
    app.run(debug=True)
