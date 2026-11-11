import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps

st.title("Handwritten Character Recognition")
st.write("Upload a handwritten digit image (0-9).")

model = tf.keras.models.load_model(
    "handwritten_digit_model.keras"
)

uploaded_file = st.file_uploader(
    "Upload your digit image",
    type=["png", "jpg", "jpeg"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("L")
    image = ImageOps.invert(image)
    image = ImageOps.fit(image, (28, 28))

    image_array = np.array(image) / 255.0
    image_array = image_array.reshape(1, 28, 28, 1)

    prediction = model.predict(image_array)
    digit = np.argmax(prediction)

    st.image(image, caption="Uploaded digit")
    st.success(f"Predicted Digit: {digit}")
confidence = np.max(prediction) * 100
st.write(f"Confidence: {confidence:.2f}%")