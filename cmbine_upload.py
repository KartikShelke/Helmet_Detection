import cv2
import easyocr
from ultralytics import YOLO
import pandas as pd
import os
import re
from tkinter import Tk
from tkinter.filedialog import askopenfilename

# === Hide Tkinter window and open file dialog ===
Tk().withdraw()
image_path = askopenfilename(title="Select an image for detection", filetypes=[("Image Files", "*.jpg *.jpeg *.png")])

if not image_path:
    print("❌ No image selected.")
    exit()

print(f"📸 Selected Image: {image_path}")

# === Model paths ===
helmet_model_path = r"C:\Users\DELL\Desktop\ANNA_Project\hemletYoloV8.pt"      # Helmet detection model
plate_model_path = r"C:\Users\DELL\Desktop\ANNA_Project\license_plate_detector.pt"  # License plate detection model

# === Load YOLO models ===
helmet_model = YOLO(helmet_model_path)
plate_model = YOLO(plate_model_path)

# === Initialize OCR ===
reader = easyocr.Reader(['en'])

# === Excel file setup ===
excel_file = "detected_plates.xlsx"
if os.path.exists(excel_file):
    df = pd.read_excel(excel_file)
else:
    df = pd.DataFrame(columns=["Plate_Number"])

# === Load image ===
frame = cv2.imread(image_path)
if frame is None:
    print("❌ Failed to read image.")
    exit()

# --- Helmet Detection ---
results_helmet = helmet_model(frame, verbose=False)
no_helmet_detected = False

for r in results_helmet:
    for box in r.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cls = int(box.cls[0])
        label = helmet_model.names[cls].lower()

        color = (0, 255, 0) if "helmet" in label else (0, 0, 255)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        if "helmet" not in label:
            no_helmet_detected = True

# --- License Plate Detection (only if no helmet) ---
if no_helmet_detected:
    print("⚠️ No helmet detected! Searching for license plate...")

    results_plate = plate_model(frame, verbose=False)
    for r in results_plate:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # Draw detected plate region
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)

    # Run OCR on entire image (not cropped region)
    ocr_result = reader.readtext(frame)
    plate_number = " ".join([res[1] for res in ocr_result]).strip()
    plate_number = re.sub(r'[^A-Z0-9]', '', plate_number.upper())

    if len(plate_number) >= 6:
        print(f"🔍 Detected Plate: {plate_number}")
        confirm = input(f"💾 Save {plate_number} to Excel? (y/n): ")
        if confirm.lower() == "y":
            new_row = {"Plate_Number": plate_number}
            df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
            df.to_excel(excel_file, index=False)
            print("✅ Plate number saved successfully!")
    else:
        print("⚠️ Could not detect a valid plate number.")
else:
    print("✅ Helmet detected. No violation found.")

# === Show final result ===
cv2.imshow("Helmet & Plate Detection Result", frame)
cv2.waitKey(0)
cv2.destroyAllWindows()
