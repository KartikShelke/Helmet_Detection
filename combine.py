import cv2
import os
import pandas as pd
import easyocr
from ultralytics import YOLO
import re

# -------------------- CONFIGURATION --------------------
helmet_model_path = r"C:\Users\DELL\Desktop\ANNA\hemletYoloV8.pt"  # your trained helmet model
plate_model_path = r"C:\Users\DELL\Desktop\ANNA\license_plate_detector.pt"  # license plate model
excel_file = "violations.xlsx"

# -------------------- MODEL LOADING --------------------
helmet_model = YOLO(helmet_model_path)
plate_model = YOLO(plate_model_path)
reader = easyocr.Reader(['en'])

# -------------------- EXCEL SETUP --------------------
if os.path.exists(excel_file):
    df = pd.read_excel(excel_file)
else:
    df = pd.DataFrame(columns=["Plate_Number"])

# -------------------- HELPER FUNCTIONS --------------------
def format_plate(plate):
    """Clean and format number plate to standard pattern like MH15AB0101"""
    plate = re.sub(r'[^A-Za-z0-9]', '', plate).upper()
    pattern = r'^[A-Z]{2}\d{1,2}[A-Z]{1,2}\d{1,4}$'
    if re.match(pattern, plate):
        return plate
    return None

def save_plate_to_excel(plate_number):
    """Save plate number to Excel file"""
    global df
    new_row = {"Plate_Number": plate_number}
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    df.to_excel(excel_file, index=False)
    print(f"✅ Saved to Excel: {plate_number}")

# -------------------- CAMERA INITIALIZATION --------------------
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Could not open camera.")
    exit()
print("✅ Camera opened. Press 'q' to quit.")

# -------------------- MAIN LOOP --------------------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # ---- STEP 1: Helmet Detection ----
    helmet_results = helmet_model(frame, stream=True)

    helmet_detected = False
    no_helmet_detected = False

    for r in helmet_results:
        for box in r.boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            label = helmet_model.names[cls].lower()

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            color = (0, 255, 0) if "helmet" in label else (0, 0, 255)

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            if "helmet" in label:
                helmet_detected = True
            else:
                no_helmet_detected = True

    # ---- STEP 2: If no helmet, detect number plate ----
    if no_helmet_detected:
        plate_results = plate_model(frame, stream=True)
        for r in plate_results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                roi = frame[y1:y2, x1:x2]

                # OCR
                ocr_result = reader.readtext(roi)
                plate_number = "".join([res[1] for res in ocr_result]).strip()
                formatted_plate = format_plate(plate_number)

                # Draw plate box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)
                cv2.putText(frame, f"Plate: {formatted_plate or 'Unreadable'}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

                # Ask for confirmation before saving
                if formatted_plate:
                    print(f"⚠️ Detected possible plate: {formatted_plate}")
                    confirm = input("Save this plate to Excel? (y/n): ").strip().lower()
                    if confirm == 'y':
                        save_plate_to_excel(formatted_plate)

    # ---- STEP 3: Display ----
    cv2.imshow("Helmet + Number Plate Detection", frame)

    # Quit on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
