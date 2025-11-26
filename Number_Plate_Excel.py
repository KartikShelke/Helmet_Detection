import cv2
import easyocr
from ultralytics import YOLO
import pandas as pd
import os

# Initialize YOLO model (plate detection)
model = YOLO("license_plate_detector.pt")  # your trained license plate model

# Initialize EasyOCR for text recognition
reader = easyocr.Reader(['en'])

# Excel file setup
excel_file = "detected_numbers.xlsx"
if os.path.exists(excel_file):
    df = pd.read_excel(excel_file)
else:
    df = pd.DataFrame(columns=["Plate_Number"])

# Open camera
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Could not open camera")
    exit()
print("✅ Camera opened. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Detect plates with YOLO
    results = model(frame, verbose=False)
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            roi = frame[y1:y2, x1:x2]

            # OCR on the detected plate region
            ocr_result = reader.readtext(roi)
            plate_number = ""
            for res in ocr_result:
                plate_number += res[1] + " "

            plate_number = plate_number.strip()

            # Draw bounding box + text
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, plate_number, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            # Save to Excel if not empty
            if plate_number != "":
                new_row = {"Plate_Number": plate_number}
                df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
                df.to_excel(excel_file, index=False)

    # Display frame (⚠️ may not work in VS Code, run in terminal instead)
    try:
        cv2.imshow("Real-Time Plate Detection + Number", frame)
    except:
        pass

    # Quit on 'q'
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
