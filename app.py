import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np

# ------------------------------
# Load both YOLO models
# ------------------------------
helmet_model_path = "hemletYoloV8.pt"
plate_model_path = "license_plate_detector.pt"

helmet_model = YOLO(helmet_model_path)
plate_model = YOLO(plate_model_path)

st.title("Helmet + Number Plate Detection App")

# ------------------------------
# Upload Image
# ------------------------------
uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Convert PIL → NumPy array
    img = np.array(image)

    # ------------------------------
    # Helmet Detection
    # ------------------------------
    st.subheader("🔵 Helmet Detection")
    helmet_results = helmet_model(img)
    helmet_out = helmet_results[0].plot()[:, :, ::-1]  # Convert BGR→RGB
    st.image(helmet_out, caption="Helmet Detection Result", use_container_width=True)

    # ------------------------------
    # Number Plate Detection
    # ------------------------------
    st.subheader("🟡 Number Plate Detection")
    plate_results = plate_model(img)
    plate_out = plate_results[0].plot()[:, :, ::-1]  # Convert BGR→RGB
    st.image(plate_out, caption="Plate Detection Result", use_container_width=True)
