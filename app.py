import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import easyocr
import pandas as pd
import re
import os
from PIL import Image

# -------------------- CONFIGURATION --------------------
helmet_model_path = "hemletYoloV8.pt"
plate_model_path = "license_plate_detector.pt"
excel_file = "violations.xlsx"

# -------------------- LOAD MODELS --------------------
helmet_model = YOLO(helmet_model_path)
plate_model = YOLO(plate_model_path)
reader = easyocr.Reader(['en'])

# -------------------- EXCEL SETUP --------------------
if os.path.exists(excel_file):
    df = pd.read_excel(excel_file)
else:
    df = pd.DataFrame(columns=["Plate_Number"])
    df.to_excel(excel_file, index=False)

# -------------------- HELPER FUNCTIONS --------------------
def format_plate(plate):
    plate = re.sub(r'[^A-Za-z0-9]', '', plate).upper()
    pattern = r'^[A-Z]{2}\d{1,2}[A-Z]{1,2}\d{1,4}$'
    return plate if re.match(pattern, plate) else None

def save_plate_to_excel(plate_number):
    global df
    new_row = {"Plate_Number": plate_number}
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    df.to_excel(excel_file, index=False)

def detect_objects(image):
    frame = np.array(image)
    helmet_results = helmet_model(frame)
    plate_results = plate_model(frame)
    return helmet_results, plate_results, frame

def draw_boxes(frame, results, color, label_name):
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label_name, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

# -------------------- STREAMLIT UI --------------------
st.title("🛵 Helmet & Number Plate Detection System")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    st.write("🔍 Processing...")

    # Detection
    helmet_results, plate_results, frame = detect_objects(image)

    no_helmet_detected = False
    detected_plate_number = None

    # Check helmet detection
    for r in helmet_results:
        for box in r.boxes:
            label = helmet_model.names[int(box.cls[0])].lower()
            if "helmet" not in label:
                no_helmet_detected = True
            draw_boxes(frame, [r], (0, 255, 0), label)

    # If no helmet → detect number plate
    if no_helmet_detected:
        for r in plate_results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                roi = frame[y1:y2, x1:x2]

                # OCR extraction
                ocr_result = reader.readtext(roi)
                plate_text = "".join([res[1] for res in ocr_result]).strip()
                formatted_plate = format_plate(plate_text)

                if formatted_plate:
                    detected_plate_number = formatted_plate

                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)
                cv2.putText(frame, formatted_plate or "Unreadable",
                            (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (255, 255, 0), 2)

    # Display output image
    st.image(frame, channels="BGR", caption="Processed Result")

    # Save plate button
    if detected_plate_number:
        st.success(f"Detected Plate: **{detected_plate_number}**")
        if st.button("Save to Excel"):
            save_plate_to_excel(detected_plate_number)
            st.success("Saved to violations.xlsx")
