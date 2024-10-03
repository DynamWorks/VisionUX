import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO

# Load YOLOv8 model
model = YOLO("yolov8n.pt")
# model = YOLO('./runs/detect/train/weights/best.pt')

# Load image
original_image = cv2.imread("GettyImages-AB27006.jpg")
original_height, original_width = original_image.shape[:2]

# YOLOv8 detection
results = model(original_image)

# Process YOLOv8 output
boxes = []
confidences = []
class_ids = []

for r in results:
    for box in r.boxes:
        x1, y1, x2, y2 = box.xyxy[0]
        x, y, w, h = int(x1), int(y1), int(x2 - x1), int(y2 - y1)
        conf = float(box.conf)
        cls = int(box.cls)
        boxes.append([x, y, w, h])
        confidences.append(conf)
        class_ids.append(cls)

# Get class names
classes = model.names

# Define vehicle classes
vehicle_classes = ['car', 'truck', 'bus', 'motorbike']

# Create DataFrame for analytics
data = []
# Visualize results
output_image = original_image.copy()
for box, confidence, class_id in zip(boxes, confidences, class_ids):
    class_name = classes[class_id]
    if class_name in vehicle_classes:
        x, y, w, h = box
        label = f"{class_name}: {confidence:.2f}"
        color = (0, 255, 0)  # Green color for bounding box
        
        cv2.rectangle(output_image, (x, y), (x + w, y + h), color, 2)
        cv2.putText(output_image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Add to data list
        data.append({
            'name': class_name,
            'confidence': confidence,
            'xmin': x,
            'ymin': y,
            'xmax': x + w,
            'ymax': y + h
        })

# Create DataFrame from collected data
df = pd.DataFrame(data)

# Save output image
cv2.imwrite("vehicle_detection_opencv.png", output_image)

# Analytics
vehicle_classes = ['car', 'truck', 'bus', 'motorbike']
vehicle_df = df[df['name'].isin(vehicle_classes)]
total_vehicles = len(vehicle_df)

print(f"Total vehicles detected: {total_vehicles}")

if total_vehicles > 0:
    vehicle_counts = vehicle_df['name'].value_counts()
    print("\nVehicle counts by type:")
    print(vehicle_counts)

    print("\nAverage confidence by vehicle type:")
    print(vehicle_df.groupby('name')['confidence'].mean())

    print("\nSpatial distribution:")
    print(f"X-range: {vehicle_df['xmin'].min():.2f} to {vehicle_df['xmax'].max():.2f}")
    print(f"Y-range: {vehicle_df['ymin'].min():.2f} to {vehicle_df['ymax'].max():.2f}")

    # Calculate and print lane estimates
    x_positions = (vehicle_df['xmin'] + vehicle_df['xmax']) / 2
    lane_count = max(1, round((x_positions.max() - x_positions.min()) / 100))  # Assuming average lane width of 100 pixels
    print(f"\nEstimated number of lanes: {lane_count}")

    # Traffic density using bounding boxes
    image_area = original_height * original_width
    vehicle_df['area'] = (vehicle_df['xmax'] - vehicle_df['xmin']) * (vehicle_df['ymax'] - vehicle_df['ymin'])
    total_vehicle_area = vehicle_df['area'].sum()
    density = total_vehicle_area / image_area
    print(f"\nTraffic density (fraction of image covered by vehicles): {density:.2%}")

    # Average vehicle size by type
    print("\nAverage vehicle size by type (in pixels):")
    print(vehicle_df.groupby('name')['area'].mean())
else:
    print("No vehicles detected in the image.")

print("\nAll detected objects:")
print(df['name'].value_counts())
