import cv2
import numpy as np
import pandas as pd

# Load YOLOv3-tiny
net = cv2.dnn.readNetFromDarknet(cv2.samples.findFile("yolov3-tiny.cfg"),
                                 cv2.samples.findFile("yolov3-tiny.weights"))
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

# Load class names
classes = open(cv2.samples.findFile("coco.names")).read().strip().split('\n')

# Load image
image = cv2.imread("GettyImages-AB27006.jpg")
height, width = image.shape[:2]

# YOLO detection
blob = cv2.dnn.blobFromImage(image, 1/255.0, (416, 416), swapRB=True, crop=False)
net.setInput(blob)
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
outputs = net.forward(output_layers)

# Process YOLO output
boxes = []
confidences = []
class_ids = []

for output in outputs:
    for detection in output:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        if confidence > 0.5:
            center_x = int(detection[0] * width)
            center_y = int(detection[1] * height)
            w = int(detection[2] * width)
            h = int(detection[3] * height)
            x = int(center_x - w / 2)
            y = int(center_y - h / 2)
            boxes.append([x, y, w, h])
            confidences.append(float(confidence))
            class_ids.append(class_id)

# Apply non-maximum suppression
indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

# Create DataFrame for analytics
df = pd.DataFrame(columns=['name', 'confidence', 'xmin', 'ymin', 'xmax', 'ymax'])

# Visualize results
output_image = image.copy()
for i in indices:
    i = i[0] if isinstance(i, (tuple, list)) else i  # Adjust for OpenCV version differences
    box = boxes[i]
    x, y, w, h = box
    label = f"{classes[class_ids[i]]}: {confidences[i]:.2f}"
    color = (0, 255, 0)  # Green color for bounding box
    
    cv2.rectangle(output_image, (x, y), (x + w, y + h), color, 2)
    cv2.putText(output_image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    # Add to DataFrame
    df = df.append({
        'name': classes[class_ids[i]],
        'confidence': confidences[i],
        'xmin': x,
        'ymin': y,
        'xmax': x + w,
        'ymax': y + h
    }, ignore_index=True)

# Save output image
cv2.imwrite("vehicle_detection_opencv.png", output_image)

# Analytics
vehicle_classes = ['car', 'truck', 'bus', 'motorbike']
vehicle_df = df[df['name'].isin(vehicle_classes)]
vehicle_counts = vehicle_df['name'].value_counts()
total_vehicles = vehicle_counts.sum()

print(f"Total vehicles detected: {total_vehicles}")
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
image_area = height * width
vehicle_df['area'] = (vehicle_df['xmax'] - vehicle_df['xmin']) * (vehicle_df['ymax'] - vehicle_df['ymin'])
total_vehicle_area = vehicle_df['area'].sum()
density = total_vehicle_area / image_area
print(f"\nTraffic density (fraction of image covered by vehicles): {density:.2%}")

# Average vehicle size by type
print("\nAverage vehicle size by type (in pixels):")
print(vehicle_df.groupby('name')['area'].mean())
