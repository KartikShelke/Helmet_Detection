import streamlit as st
from PIL import Image
import numpy as np
import onnxruntime as ort
import easyocr
import re

# ------------------ Load ONNX Models ------------------
helmet_model_path = "hemletYoloV8.onnx"
plate_model_path = "license_plate_detector.onnx"

helmet_session = ort.InferenceSession(helmet_model_path)
plate_session = ort.InferenceSession(plate_model_path)

reader = easyocr.Reader(['en'])


# ------------------ Utility Functions ------------------
def preprocess(img):
    img = img.resize((640, 640))
    arr = np.array(img)
    arr = arr.transpose(2, 0, 1)  # HWC → CHW
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


# ------------------ Streamlit UI ------------------
st.title("Helmet + Number Plate Detection")
uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess
    input_data = preprocess(image)

    # Run Helmet Model
    helmet_output = run_onnx(helmet_session, input_data)
    st.write("Helmet Model Output:", helmet_output)

    # Run Plate Model
    plate_output = run_onnx(plate_session, input_data)
    st.write("Plate Model Output:", plate_output)

    # OCR
    ocr_result = reader.readtext(np.array(image))
    text = "".join([res[1] for res in ocr_result])
    plate = format_plate(text)

    st.subheader("Detected Plate Number")
    st.write(plate if plate else "Unable to read plate")
