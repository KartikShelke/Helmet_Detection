import os
import pandas as pd
from PIL import Image

# Path to your dataset folder (change for train/valid/test)
dataset_dir = "Helmet_Dataset/valid"

# Find the CSV file in the folder
csv_files = [f for f in os.listdir(dataset_dir) if f.endswith(".csv")]
if not csv_files:
    print(" No CSV file found in:", dataset_dir)
    exit()

csv_file = os.path.join(dataset_dir, csv_files[0])
print(" Using CSV:", csv_file)

# Load annotations
df = pd.read_csv(csv_file)
print(" Columns in CSV:", df.columns.tolist())

for i, row in df.iterrows():
    image_name = str(row['filename']).strip()
    img_path = os.path.join(dataset_dir, image_name)

    if not os.path.exists(img_path):
        print(f"⚠️ Image not found: {img_path}")
        continue

    # Open image to normalize bbox
    img = Image.open(img_path)
    img_w, img_h = img.size

    try:
        xmin, ymin, xmax, ymax = row['xmin'], row['ymin'], row['xmax'], row['ymax']
    except KeyError:
        print(" Missing columns! Found columns:", df.columns.tolist())
        break

    # Convert to YOLO format
    x_center = ((xmin + xmax) / 2) / img_w
    y_center = ((ymin + ymax) / 2) / img_h
    width = (xmax - xmin) / img_w
    height = (ymax - ymin) / img_h

    # Class ID (0 = helmet for now, change if multiple classes)
    class_id = 0

    # Save annotation
    txt_path = os.path.join(dataset_dir, image_name.rsplit(".", 1)[0] + ".txt")
    with open(txt_path, "w") as f:
        f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

    print(f" Saved {txt_path}")
