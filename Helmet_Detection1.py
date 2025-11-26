import cv2
from ultralytics import YOLO
import numpy as np

# Load YOLO model
model = YOLO(r"C:\Users\DELL\Desktop\ANNA\hemletYoloV8.pt")  # Path to your weights

# Path to your input image
image_path = r"C:\Users\DELL\Desktop\ANNA\Sample_Images\riders_7.png"  

# Read the image
frame = cv2.imread(image_path)
if frame is None:
    print("❌ Could not read image")
    exit()

# Make a copy of the original image
original_frame = frame.copy()

# Run YOLO detection
results = model(frame)

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

# Combine both images side by side
comparison = np.hstack((original_frame, frame))

# Resize to fit window (e.g., 1000px width)
scale_width = 1000
scale_height = int(comparison.shape[0] * (scale_width / comparison.shape[1]))
comparison_resized = cv2.resize(comparison, (scale_width, scale_height))

# Show comparison
cv2.imshow("Original (Left) vs Predicted (Right)", comparison_resized)
cv2.waitKey(0)
cv2.destroyAllWindows()
