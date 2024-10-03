import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from scipy import stats
import time
from ultralytics import YOLO, SAM

# Load models
yolo_model = YOLO("yolov8n.pt")
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
sam_model = SAM('sam_b.pt')

def validate_detection(image, bbox, initial_class):
    x1, y1, x2, y2 = bbox
    cropped_image = Image.fromarray(cv2.cvtColor(image[y1:y2, x1:x2], cv2.COLOR_BGR2RGB))
    
    texts = ["a car", "a truck", "a bus", "a motorcycle"]
    
    inputs = clip_processor(text=texts, images=[cropped_image], return_tensors="pt", padding=True)
    
    with torch.no_grad():
        outputs = clip_model(**inputs)
    logits_per_image = outputs.logits_per_image
    probs = logits_per_image.softmax(dim=1)
    
    best_class = texts[probs.argmax().item()].split()[-1]
    confidence = probs.max().item()
    
    return best_class, confidence

def segment_vehicles(image, boxes):
    masks = []
    for box in boxes:
        x, y, w, h = box
        results = sam_model(image, bboxes=[box])
        mask = results[0].masks.data[0].cpu().numpy()
        masks.append(mask)
    return masks

def refine_vehicle_count(masks):
    # Use connected component analysis to separate potentially merged vehicles
    refined_count = 0
    for mask in masks:
        num_labels, labels = cv2.connectedComponents(mask.astype(np.uint8))
        refined_count += num_labels - 1  # Subtract 1 to exclude background
    return refined_count

def estimate_vehicle_count(image_shape, masks):
    total_area = image_shape[0] * image_shape[1]
    vehicle_area = sum([mask.sum() for mask in masks])
    avg_vehicle_size = vehicle_area / len(masks)
    return vehicle_area / avg_vehicle_size

def combine_estimates(estimates, weights):
    weighted_estimate = sum(e * w for e, w in zip(estimates, weights))
    error_margin = np.std(estimates)
    return round(weighted_estimate), round(error_margin, 2)

# Main processing
start_time = time.time()
original_image = cv2.imread("GettyImages-AB27006.jpg")
original_height, original_width = original_image.shape[:2]

# YOLOv8 detection
results = yolo_model(original_image)

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

classes = yolo_model.names
vehicle_classes = ['car', 'truck', 'bus', 'motorcycle']

# Perform SAM segmentation
masks = segment_vehicles(original_image, [box[:4] for box in boxes])  # Use only x, y, w, h

# Create DataFrame for analytics
data = []
output_image = original_image.copy()

for box, confidence, class_id, mask in zip(boxes, confidences, class_ids, masks):
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
        
        # Overlay segmentation mask
        mask_overlay = output_image.copy()
        mask_overlay[mask] = mask_overlay[mask] * 0.5 + np.array([0, 255, 0]) * 0.5
        cv2.addWeighted(mask_overlay, 0.5, output_image, 0.5, 0, output_image)
        
        data.append({
            'name': validated_class,
            'original_name': class_name,
            'confidence': combined_confidence,
            'xmin': x,
            'ymin': y,
            'xmax': x + w,
            'ymax': y + h,
            'mask_area': mask.sum()
        })

df = pd.DataFrame(data)

cv2.imwrite("vehicle_detection_sam2.png", output_image)

end_time = time.time()
processing_time = end_time - start_time

# Analytics
yolo_vehicle_count = len(df)
refined_count = refine_vehicle_count(masks)
area_based_estimate = estimate_vehicle_count((original_height, original_width), masks)

# Combine estimates
estimates = [float(yolo_vehicle_count), float(refined_count), float(area_based_estimate)]
weights = [0.3, 0.4, 0.3]  # Adjust weights based on confidence in each method
final_count, error_margin = combine_estimates(estimates, weights)

print(f"Estimated total vehicles: {final_count} ± {error_margin}")
print(f"\nYOLO detected vehicles: {yolo_vehicle_count}")
print(f"Refined count (based on segmentation): {refined_count}")
print(f"Area-based estimate: {round(area_based_estimate)}")

print("\nVehicle counts by type:")
print(df['name'].value_counts())

print("\nAverage confidence by vehicle type:")
print(df.groupby('name')['confidence'].mean())

print("\nSpatial distribution:")
print(f"X-range: {df['xmin'].min():.2f} to {df['xmax'].max():.2f}")
print(f"Y-range: {df['ymin'].min():.2f} to {df['ymax'].max():.2f}")

print("\nAverage vehicle size by type (in pixels):")
print(df.groupby('name')['mask_area'].mean())

# Error analysis
print("\nError Analysis:")
print(f"YOLO+CLIP confidence range: {df['confidence'].min():.2f} to {df['confidence'].max():.2f}")
print(f"Average confidence: {df['confidence'].mean():.2f}")
print(f"Standard deviation of confidence: {df['confidence'].std():.2f}")

# Misclassification analysis
misclassified = df[df['name'] != df['original_name']]
print(f"\nMisclassified vehicles: {len(misclassified)}")
if len(misclassified) > 0:
    print("Misclassification details:")
    print(misclassified[['original_name', 'name', 'confidence']])

print(f"\nTotal processing time: {processing_time:.2f} seconds")
