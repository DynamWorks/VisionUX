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
import matplotlib.pyplot as plt
import multiprocessing

# Load models
yolo_model = YOLO("yolov8n.pt")
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
sam_model = SAM('sam2_b.pt')

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

def segment_image(image):
    results = sam_model(image, device='cpu', verbose=False, retina_masks=True, 
                        imgsz=1024, conf=0.4, iou=0.9, 
                        agnostic_nms=False, max_det=300, 
                        half=False, batch=1)
    masks = results[0].masks.data.cpu().numpy()
    return masks

def classify_mask(mask, yolo_results):
    mask_indices = np.where(mask)
    for r in yolo_results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            if np.any((mask_indices[0] >= y1) & (mask_indices[0] <= y2) & 
                      (mask_indices[1] >= x1) & (mask_indices[1] <= x2)):
                return int(box.cls), float(box.conf)
    return None, 0.0

def save_segmentation_result(image, masks, output_path):
    plt.figure(figsize=(12, 8))
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    for i, mask in enumerate(masks):
        color = np.random.rand(3)
        plt.imshow(mask, alpha=0.5, cmap='jet')
    plt.axis('off')
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
    plt.close()

def detect_trucks(image):
    # Load the pre-trained YOLO model for truck detection
    truck_model = YOLO('yolov8n-cls.pt')
    
    # Run the truck detection on the input image
    results = truck_model(image, device='cpu', verbose=False, imgsz=1024, conf=0.4, iou=0.5,
                         agnostic_nms=False, max_det=100, half=False, batch=1)
    
    # Extract the truck detections
    trucks = []
    if results and len(results) > 0:
        for result in results:
            if result.boxes is not None and hasattr(result.boxes, 'data'):
                boxes = result.boxes.data.cpu().numpy()
                for box in boxes:
                    x1, y1, x2, y2, conf, cls = box
                    if int(cls) == 7:  # Class 7 corresponds to trucks
                        trucks.append({
                            'box': [x1, y1, x2-x1, y2-y1],
                            'confidence': conf
                        })
    
    return trucks

def detect_buses(image):
    # Load the pre-trained YOLO model for bus detection
    bus_model = YOLO('yolov8n-cls.pt')
    
    # Run the bus detection on the input image
    results = bus_model(image, device='cpu', verbose=False, imgsz=1024, conf=0.4, iou=0.5,
                        agnostic_nms=False, max_det=100, half=False, batch=1)
    
    # Extract the bus detections
    buses = []
    for result in results:
        boxes = result.boxes.data.cpu().numpy()
        for box in boxes:
            x1, y1, x2, y2, conf, cls = box
            if int(cls) == 5:  # Class 5 corresponds to buses
                buses.append({
                    'box': [x1, y1, x2-x1, y2-y1],
                    'confidence': conf
                })
    
    return buses

def save_yolo_result(image, yolo_results, output_path):
    result_image = image.copy()
    yolo_mask = np.zeros(image.shape[:2], dtype=np.uint8)
    for r in yolo_results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(yolo_mask, (x1, y1), (x2, y2), 255, -1)
            label = f"{r.names[int(box.cls)]}: {box.conf.item():.2f}"
            cv2.putText(result_image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    return result_image, yolo_mask

def refine_vehicle_count(masks):
    refined_count = len(masks)
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
yolo_results = yolo_model(original_image)

# Get YOLO result and mask
yolo_result_image, yolo_mask = save_yolo_result(original_image, yolo_results, "yolo_detection_result.png")

# SAM segmentation
masks = segment_image(original_image)

# Create SAM mask
sam_mask = np.zeros(original_image.shape[:2], dtype=np.uint8)
for mask in masks:
    sam_mask = np.logical_or(sam_mask, mask).astype(np.uint8) * 255

# Create overlap mask
overlap_mask = cv2.bitwise_and(yolo_mask, sam_mask)

# Create highlighted image
highlighted_image = original_image.copy()
highlighted_image[yolo_mask > 0] = cv2.addWeighted(highlighted_image[yolo_mask > 0], 0.7, np.full_like(highlighted_image[yolo_mask > 0], [255, 0, 0]), 0.3, 0)
highlighted_image[sam_mask > 0] = cv2.addWeighted(highlighted_image[sam_mask > 0], 0.7, np.full_like(highlighted_image[sam_mask > 0], [0, 255, 0]), 0.3, 0)
highlighted_image[overlap_mask > 0] = cv2.addWeighted(highlighted_image[overlap_mask > 0], 0.7, np.full_like(highlighted_image[overlap_mask > 0], [0, 0, 255]), 0.3, 0)

# Save highlighted result
cv2.imwrite("highlighted_detection_result.png", highlighted_image)

# Save segmentation result
save_segmentation_result(original_image, masks, "sam_segmentation_result.png")

# Detect trucks
trucks = detect_trucks(original_image)

# Process the truck detections
if trucks:
    for truck in trucks:
        box = truck['box']
        confidence = truck['confidence']
        print(f"Truck detected: {box}, confidence: {confidence}")
else:
    print("No trucks detected in the image.")

classes = yolo_model.names
vehicle_classes = ['car', 'truck', 'bus', 'motorcycle']

# Create DataFrame for analytics
data = []
output_image = original_image.copy()

for i, mask in enumerate(masks):
    class_id, yolo_confidence = classify_mask(mask, yolo_results)
    
    if class_id is not None and classes[class_id] in vehicle_classes:
        class_name = classes[class_id]
        
        # Get bounding box of the mask
        y, x = np.where(mask)
        x1, x2, y1, y2 = x.min(), x.max(), y.min(), y.max()
        
        validated_class, clip_confidence = validate_detection(original_image, (x1, y1, x2, y2), class_name)
        
        # Combine YOLO and CLIP confidences
        combined_confidence = (yolo_confidence + clip_confidence) / 2
        
        label = f"{validated_class}: {combined_confidence:.2f}"
        color = (0, 255, 0)
        
        cv2.rectangle(output_image, (x1, y1), (x2, y2), color, 2)
        cv2.putText(output_image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Overlay segmentation mask
        mask_overlay = output_image.copy()
        mask_overlay[mask] = mask_overlay[mask] * 0.5 + np.array([0, 255, 0]) * 0.5
        cv2.addWeighted(mask_overlay, 0.5, output_image, 0.5, 0, output_image)
        
        data.append({
            'name': validated_class,
            'original_name': class_name,
            'confidence': combined_confidence,
            'xmin': x1,
            'ymin': y1,
            'xmax': x2,
            'ymax': y2,
            'mask_area': mask.sum()
        })

df = pd.DataFrame(data)

# Save final result
cv2.imwrite("final_detection_result.png", output_image)

end_time = time.time()
processing_time = end_time - start_time

# Analytics
sam_vehicle_count = len(df)
refined_count = refine_vehicle_count(masks)
area_based_estimate = estimate_vehicle_count((original_height, original_width), masks)

# Combine estimates
estimates = [float(sam_vehicle_count), float(refined_count), float(area_based_estimate)]
weights = [0.3, 0.4, 0.3]  # Adjust weights based on confidence in each method
final_count, error_margin = combine_estimates(estimates, weights)

print(f"Estimated total vehicles: {final_count} ± {error_margin}")
print(f"\nSAM detected vehicles: {sam_vehicle_count}")
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
