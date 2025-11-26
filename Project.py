from ultralytics import YOLO

# Helmet model
YOLO("hemletYoloV8.pt").export(format="onnx")

# Plate model
YOLO("license_plate_detector.pt").export(format="onnx")
