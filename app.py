import streamlit as st
from PIL import Image
import numpy as np
import onnxruntime as ort
import pytesseract
import cv2
import re

helmet_model_path = "hemletYoloV8.onnx"
plate_model_path = "license_plate_detector.onnx"

helmet_session = ort.InferenceSession(helmet_model_path)
plate_session = ort.InferenceSession(plate_model_path)

def preprocess(img):
    img = img.resize((640, 640))
    arr = np.array(img)
    arr = arr.transpose(2, 0, 1)
    arr = arr.astype(np.float32) / 255.0
    arr = np.expand_dims(arr, axis=0)
    return arr

def format_plate(plate):
    plate = re.sub(r'[^A-Za-z0-9]', '', plate).upper()
    pattern = r'^[A-Z]{2}\d{1,2}[A-Z]{1,2}\d{1,4}$'
    return plate if re.match(pattern, plate) else None

def run_onnx(session, img):
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    return session.run([output_name], {input_name: img})[0]

st.title("Helmet + Number Plate Detection (Streamlit Cloud Compatible)")

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    input_data = preprocess(image)

    helmet_output = run_onnx(helmet_session, input_data)
    plate_output = run_onnx(plate_session, input_data)

    st.subheader("Raw Helmet Model Output")
    st.write(helmet_output)

    st.subheader("Raw Plate Model Output")
    st.write(plate_output)

    # OCR using pytesseract
    img_np = np.array(image)
    gray = cv2.cvtColor(img_np, cv2.COLOR_BGR2GRAY)

    ocr_text = pytesseract.image_to_string(gray)
    plate = format_plate(ocr_text)

    st.subheader("Recognized Plate Number")
    st.write(plate if plate else "Unable to read plate")
