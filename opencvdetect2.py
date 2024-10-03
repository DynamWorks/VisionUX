import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from scipy import stats

# Load YOLOv8 model
yolo_model = YOLO("yolov8n.pt")

# Load CLIP model
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

def validate_detection(image, bbox, initial_class):
    x1, y1, x2, y2 = bbox
    cropped_image = Image.fromarray(cv2.cvtColor(image[y1:y2, x1:x2], cv2.COLOR_BGR2RGB))
    
    texts = ["a car", "a truck", "a bus", "a motorcycle"]
    
    inputs = clip_processor(text=texts, images=cropped_image, return_tensors="pt", padding=True)
    
    with torch.no_grad():
        outputs = clip_model(**inputs)
    logits_per_image = outputs.logits_per_image
    probs = logits_per_image.softmax(dim=1)
    
    best_class = texts[probs.argmax().item()].split()[-1]
    confidence = probs.max().item()
    
    return best_class, confidence

def density_based_estimate(image_shape, traffic_density, avg_vehicle_size):
    image_area = image_shape[0] * image_shape[1]
    vehicle_area = image_area * traffic_density
    return vehicle_area / avg_vehicle_size

def lane_based_estimate(image_shape, num_lanes, avg_vehicle_length_pixels):
    lane_length = image_shape[0]
    return num_lanes * (lane_length / avg_vehicle_length_pixels)

def combine_estimates(estimates, weights):
    weighted_mean = np.average(estimates, weights=weights)
    error_margin = stats.sem(estimates) * 1.96  # 95% confidence interval
    return round(weighted_mean), round(error_margin, 2)

# Load and process image
original_image = cv2.imread("GettyImages-AB27006.jpg")
original_height, original_width = original_image.shape[:2]
resized_image = cv2.resize(original_image, (640, 640))

# YOLOv8 detection
results = yolo_model(resized_image)

# Process YOLOv8 output
boxes = []
confidences = []
class_ids = []

for r in results:
    for box in r.boxes:
        x1, y1, x2, y2 = box.xyxy[0]
        x1 = int(x1 * original_width / 640)
        y1 = int(y1 * original_height / 640)
        x2 = int(x2 * original_width / 640)
        y2 = int(y2 * original_height / 640)
        x, y, w, h = x1, y1, x2 - x1, y2 - y1
        conf = float(box.conf)
        cls = int(box.cls)
        boxes.append([x, y, w, h])
        confidences.append(conf)
        class_ids.append(cls)

classes = yolo_model.names
vehicle_classes = ['car', 'truck', 'bus', 'motorcycle']

# Create DataFrame for analytics
data = []
output_image = original_image.copy()

for box, confidence, class_id in zip(boxes, confidences, class_ids):
    class_name = classes[class_id]
    if class_name in vehicle_classes:
        x, y, w, h = box
        
        validated_class, clip_confidence = validate_detection(original_image, (x, y, x+w, y+h), class_name)
        
        # Combine YOLO and CLIP confidences
        combined_confidence = (confidence + clip_confidence) / 2
        
        label = f"{validated_class}: {combined_confidence:.2f}"
        color = (0, 255, 0)
        
        cv2.rectangle(output_image, (x, y), (x + w, y + h), color, 2)
        cv2.putText(output_image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        data.append({
            'name': validated_class,
            'original_name': class_name,
            'confidence': combined_confidence,
            'xmin': x,
            'ymin': y,
            'xmax': x + w,
            'ymax': y + h
        })

df = pd.DataFrame(data)

cv2.imwrite("vehicle_detection_opencv_improved.png", output_image)

# Analytics
vehicle_df = df[df['name'].isin(vehicle_classes)]
yolo_vehicle_count = len(vehicle_df)

# Calculate traffic density
image_area = original_height * original_width
vehicle_df['area'] = (vehicle_df['xmax'] - vehicle_df['xmin']) * (vehicle_df['ymax'] - vehicle_df['ymin'])
total_vehicle_area = vehicle_df['area'].sum()
traffic_density = total_vehicle_area / image_area

# Estimate number of lanes
x_positions = (vehicle_df['xmin'] + vehicle_df['xmax']) / 2
lane_count = max(1, round((x_positions.max() - x_positions.min()) / 100))

# Additional estimates
avg_vehicle_size = vehicle_df['area'].mean()
avg_vehicle_length = vehicle_df['xmax'] - vehicle_df['xmin'].mean()

density_estimate = density_based_estimate((original_height, original_width), traffic_density, avg_vehicle_size)
lane_estimate = lane_based_estimate((original_height, original_width), lane_count, avg_vehicle_length)

# Combine estimates
estimates = [yolo_vehicle_count, density_estimate, lane_estimate]
weights = [0.5, 0.25, 0.25]  # Adjust weights based on confidence in each method
final_count, error_margin = combine_estimates(estimates, weights)

print(f"Estimated total vehicles: {final_count} ± {error_margin}")
print(f"\nYOLO detected vehicles: {yolo_vehicle_count}")
print(f"Density-based estimate: {round(density_estimate)}")
print(f"Lane-based estimate: {round(lane_estimate)}")

print("\nVehicle counts by type:")
print(vehicle_df['name'].value_counts())

print("\nAverage confidence by vehicle type:")
print(vehicle_df.groupby('name')['confidence'].mean())

print("\nSpatial distribution:")
print(f"X-range: {vehicle_df['xmin'].min():.2f} to {vehicle_df['xmax'].max():.2f}")
print(f"Y-range: {vehicle_df['ymin'].min():.2f} to {vehicle_df['ymax'].max():.2f}")

print(f"\nEstimated number of lanes: {lane_count}")
print(f"\nTraffic density: {traffic_density:.2%}")

print("\nAverage vehicle size by type (in pixels):")
print(vehicle_df.groupby('name')['area'].mean())

print("\nAll detected objects:")
print(df['name'].value_counts())

# Error analysis
print("\nError Analysis:")
print(f"YOLO confidence range: {df['confidence'].min():.2f} to {df['confidence'].max():.2f}")
print(f"Average YOLO confidence: {df['confidence'].mean():.2f}")
print(f"Standard deviation of YOLO confidence: {df['confidence'].std():.2f}")

# Misclassification analysis
misclassified = df[df['name'] != df['original_name']]
print(f"\nMisclassified vehicles: {len(misclassified)}")
if len(misclassified) > 0:
    print("Misclassification details:")
    print(misclassified[['original_name', 'name', 'confidence']])
