import cv2

from ultralytics import YOLO

# Correct path and spelling
model = YOLO(r"C:\Users\DELL\Desktop\ANNA_Project\hemletYoloV8.pt") #Pathhhhhhhhhhhhhh


# Open webcam (0 = default camera)
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Camera not opened")
    exit()

print("✅ Camera opened. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Failed to grab frame")
        break

    # Run YOLO detection
    results = model(frame, stream=True)

    # Loop through results
    for r in results:
        for box in r.boxes:
            # Get bounding box coordinates
            x1, y1, x2, y2 = box.xyxy[0].int().tolist()
            conf = float(box.conf[0])  # confidence score
            cls = int(box.cls[0])      # class id
            label = model.names[cls]   # class name

            # Draw bounding box (green = helmet, red = no helmet)
            color = (0, 255, 0) if "helmet" in label.lower() else (0, 0, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # Put label text
            text = f"{label} {conf:.2f}"
            cv2.putText(frame, text, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # Show live feed
    cv2.imshow("Helmet Detection", frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
