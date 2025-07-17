import streamlit as st
import numpy as np
from tensorflow import keras
from PIL import Image
from streamlit_drawable_canvas import st_canvas

@st.cache_resource
def load_model():
    return keras.models.load_model('mnist_cnn_model.h5')

model = load_model()
st.title("Handwritten Digit Recognizer")
st.write("Draw a digit below or upload an image (28x28 grayscale or any size, will auto-resize):")

tab1, tab2 = st.tabs(["Draw Digit", "Upload Image"])

with tab1:
    canvas_image = st_canvas(
        fill_color="black",
        stroke_width=15,
        stroke_color="white",
        background_color="black",
        width=280,
        height=280,
        drawing_mode="freedraw",
        key="canvas",
    )

    if canvas_image.image_data is not None:
        img = Image.fromarray((255 - canvas_image.image_data[:, :, 0]).astype(np.uint8))
        img = img.resize((28, 28)).convert("L")
        arr = np.array(img).astype("float32") / 255.0
        arr = arr.reshape(1, 28, 28, 1)
        if st.button("Predict Drawn Digit"):
            pred = model.predict(arr)
            st.success(f"Predicted Digit: {np.argmax(pred)}")
        st.image(img, caption="28x28 Preprocessed Input")

with tab2:
    uploaded_file = st.file_uploader("Choose a digit image...")
    if uploaded_file:
        img = Image.open(uploaded_file).convert("L").resize((28, 28))
        st.image(img, caption="Uploaded Image (Resized)", width=140)
        arr = np.array(img).astype("float32") / 255.0
        arr = arr.reshape(1, 28, 28, 1)
        if st.button("Predict Uploaded Image"):
            pred = model.predict(arr)
            st.success(f"Predicted Digit: {np.argmax(pred)}")
