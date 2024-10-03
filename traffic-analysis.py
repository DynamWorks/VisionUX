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
import json

# Load models
yolo_model = YOLO("yolov8n.pt")
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
sam_model = SAM('sam2_b.pt')

def validate_detection(image, bbox, initial_class):
    x1, y1, x2, y2 = map(int, bbox)
    cropped_image = Image.fromarray(cv2.cvtColor(image[y1:y2, x1:x2], cv2.COLOR_BGR2RGB))
    
    texts = ["a car", "a pickup truck", "a large truck", "a bus", "a motorcycle"]
    
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

def combine_detections(yolo_results, masks, original_image):
    combined_detections = []
    classes = yolo_model.names
    vehicle_classes = ['car', 'truck', 'bus', 'motorcycle']
    image_height, image_width = original_image.shape[:2]

    for r in yolo_results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            class_id = int(box.cls)
            yolo_confidence = float(box.conf)
            
            if classes[class_id] in vehicle_classes:
                overlapping_masks = []
                for mask in masks:
                    if np.any(mask[y1:y2, x1:x2]):
                        overlapping_masks.append(mask)
                
                if len(overlapping_masks) > 0:
                    for i, mask in enumerate(overlapping_masks):
                        mask_y, mask_x = np.where(mask)
                        mask_x1, mask_x2 = max(0, mask_x.min()), min(image_width - 1, mask_x.max())
                        mask_y1, mask_y2 = max(0, mask_y.min()), min(image_height - 1, mask_y.max())
                        
                        validated_class, clip_confidence = validate_detection(original_image, (mask_x1, mask_y1, mask_x2, mask_y2), classes[class_id])
                        
                        combined_confidence = (yolo_confidence + clip_confidence) / 2
                        
                        combined_detections.append({
                            'name': validated_class,
                            'original_name': classes[class_id],
                            'confidence': combined_confidence,
                            'xmin': int(mask_x1),
                            'ymin': int(mask_y1),
                            'xmax': int(mask_x2),
                            'ymax': int(mask_y2),
                            'mask_area': mask.sum()
                        })
                else:
                    validated_class, clip_confidence = validate_detection(original_image, (x1, y1, x2, y2), classes[class_id])
                    combined_confidence = (yolo_confidence + clip_confidence) / 2
                    
                    combined_detections.append({
                        'name': validated_class,
                        'original_name': classes[class_id],
                        'confidence': combined_confidence,
                        'xmin': max(0, x1),
                        'ymin': max(0, y1),
                        'xmax': min(image_width - 1, x2),
                        'ymax': min(image_height - 1, y2),
                        'mask_area': (min(image_width - 1, x2) - max(0, x1)) * (min(image_height - 1, y2) - max(0, y1))
                    })

    return combined_detections

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

# Combine YOLO and SAM detections
combined_detections = combine_detections(yolo_results, masks, original_image)

# Create DataFrame for analytics
df = pd.DataFrame(combined_detections)

# Save final result
output_image = original_image.copy()
for detection in combined_detections:
    x1, y1, x2, y2 = detection['xmin'], detection['ymin'], detection['xmax'], detection['ymax']
    label = f"{detection['name']}: {detection['confidence']:.2f}"
    color = (0, 255, 0)
    cv2.rectangle(output_image, (x1, y1), (x2, y2), color, 2)
    cv2.putText(output_image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

cv2.imwrite("final_detection_result.png", output_image)

end_time = time.time()
processing_time = end_time - start_time

# Analytics
vehicle_count = len(df)
area_based_estimate = df['mask_area'].sum() / df['mask_area'].mean()

def combine_estimates(estimates, weights):
    weighted_estimate = sum(e * w for e, w in zip(estimates, weights))
    error_margin = np.std(estimates)
    return round(weighted_estimate), round(error_margin, 2)

# Combine estimates
estimates = [float(vehicle_count), float(area_based_estimate)]
weights = [0.6, 0.4]  # Adjust weights based on confidence in each method
final_count, error_margin = combine_estimates(estimates, weights)

def json_serialize(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj

# Prepare analytics data
analytics_data = {
    "estimated_total_vehicles": final_count,
    "error_margin": error_margin,
    "detected_vehicles": vehicle_count,
    "area_based_estimate": round(area_based_estimate),
    "vehicle_counts_by_type": json_serialize(df['name'].value_counts().to_dict()),
    "average_confidence_by_type": json_serialize(df.groupby('name')['confidence'].mean().to_dict()),
    "spatial_distribution": {
        "x_range": {"min": float(df['xmin'].min()), "max": float(df['xmax'].max())},
        "y_range": {"min": float(df['ymin'].min()), "max": float(df['ymax'].max())}
    },
    "average_vehicle_size_by_type": json_serialize(df.groupby('name')['mask_area'].mean().to_dict()),
    "confidence_stats": {
        "min": float(df['confidence'].min()),
        "max": float(df['confidence'].max()),
        "mean": float(df['confidence'].mean()),
        "std": float(df['confidence'].std())
    },
    "misclassified_vehicles": len(df[df['name'] != df['original_name']]),
    "processing_time": processing_time
}

# Combine detections and analytics data
output_data = {
    "detections": json_serialize(combined_detections),
    "analytics": analytics_data
}

# Save to JSON file
with open("traffic_analysis_results.json", "w") as f:
    json.dump(output_data, f, indent=2, default=json_serialize)

print("Analysis complete. Results saved to traffic_analysis_results.json")
