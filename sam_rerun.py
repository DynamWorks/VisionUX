from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
from typing import Final
from urllib.parse import urlparse

import cv2
import numpy as np
import requests
import rerun as rr  # pip install rerun-sdk
import rerun.blueprint as rrb
import torch
import torchvision
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry, SamPredictor
from segment_anything.modeling import Sam
from tqdm import tqdm
from ultralytics import YOLO
import pdb


model = YOLO('/home/thebird/Dynamworks/LLM_Module/Hackathon/runs/detect/train/weights/best.pt')
model.fuse()

DESCRIPTION = """
Example of using Rerun to log and visualize the output of [Segment Anything](https://segment-anything.com/).

The full source code for this example is available [on GitHub](https://github.com/rerun-io/rerun/blob/latest/examples/python/segment_anything_model).
""".strip()

MODEL_DIR: Final = os.path.join(os.getcwd(),"models/")
MODEL_URLS: Final = {
    "vit_h": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
    "vit_l": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",
    "vit_b": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
}

def create_sam(model: str, device: str, model_name: str) -> Sam:
    """Load the segment-anything model, fetching the model-file as necessary."""
    model_path = os.path.join(MODEL_DIR, model_name)

    logging.info(f"PyTorch version: {torch.__version__}")
    logging.info(f"Torchvision version: {torchvision.__version__}")
    logging.info(f"CUDA is available: {torch.cuda.is_available()}")

    logging.info(f"Building sam from: {model_path}")
    sam = sam_model_registry[model](checkpoint=model_path)
    return sam.to(device=device)

def run_segmentation(mask_generator: SamAutomaticMaskGenerator, image: cv2.typing.MatLike) -> None:
    """Run segmentation on a single image."""
    rr.log("image", rr.Image(image))

    logging.info("Finding masks")
    masks = mask_generator.generate(image)

    logging.info(f"Found {len(masks)} masks")

    # Log all the masks stacked together as a tensor
    # TODO(jleibs): Tensors with class-ids and annotation-coloring would make this much slicker
    mask_tensor = (
        np.dstack([np.zeros((image.shape[0], image.shape[1]))] + [m["segmentation"] for m in masks]).astype("uint8")
        * 128
    )
    rr.log("mask_tensor", rr.Tensor(mask_tensor))

    # Note: for stacking, it is important to sort these masks by area from largest to smallest
    # this is because the masks are overlapping and we want smaller masks to
    # be drawn on top of larger masks.
    # TODO(jleibs): we could instead draw each mask as a separate image layer, but the current layer-stacking
    # does not produce great results.
    masks_with_ids = list(enumerate(masks, start=1))
    masks_with_ids.sort(key=(lambda x: x[1]["area"]), reverse=True)  # type: ignore[no-any-return]

    # Layer all of the masks together, using the id as class-id in the segmentation
    segmentation_img = np.zeros((image.shape[0], image.shape[1]))
    for id, m in masks_with_ids:
        segmentation_img[m["segmentation"]] = id

    rr.log("image/masks", rr.SegmentationImage(segmentation_img.astype(np.uint8)))

    mask_bbox = np.array([m["bbox"] for _, m in masks_with_ids])
    rr.log(
        "image/boxes",
        rr.Boxes2D(array=mask_bbox,
                   array_format=rr.Box2DFormat.XYWH,
                   class_ids=[id for id, _ in masks_with_ids]),
    )

#function to get masks from detections
def get_predicted_masks(model, mask_predictor, frame, width, height):
    # Run frame through YOLOv8 to get detections
    detections = model.predict(frame, conf=0.7)

    # Check if there are fish detections
    if len(detections[0].boxes) == 0:
        return None, None, None, None  # Skip processing for frames without fish detections

    # Run frame and detections through SAM to get masks
    transformed_boxes = mask_predictor.transform.apply_boxes_torch(
        detections[0].boxes.xyxy, [width, height]
    )
    mask_predictor.set_image(frame)
    masks, scores, logits = mask_predictor.predict_torch(
        boxes=transformed_boxes,
        multimask_output=False,
        point_coords=None,
        point_labels=None
    )
    return masks, scores, logits, detections

# function to run segmentation with mask prediction SAM1 and YOLO
def run_segmentation_yolo(mask_predictor: SamPredictor, image: cv2.typing.MatLike, width, height) -> None:
    """Run segmentation on a single image."""
    rr.log("image", rr.Image(image))

    logging.info("Finding masks")
    
    # masks = mask_generator.generate(image)
    
    masks, scores, logits, detections = get_predicted_masks(
        model=model,
        mask_predictor=mask_predictor,
        frame=image,
        width=width,
        height=height
    )

    if masks == None:
        logging.info("No masks found.")
        rr.log("image/boxes", rr.Clear(recursive=False))
        return

    logging.info(f"Found {len(masks)} masks")

    # Log all the masks stacked together as a tensor
    # TODO(jleibs): Tensors with class-ids and annotation-coloring would make this much slicker
    mask_tensor = (
        # np.dstack([np.zeros((image.shape[0], image.shape[1]))] + [m for m in masks]).astype("uint8")
        # * 128
        np.dstack([np.zeros((image.shape[0], image.shape[1]))] + 
          [m.cpu().reshape(image.shape[0], image.shape[1]) for m in masks]
         ).astype("uint8") * 128
    )
    rr.log("mask_tensor", rr.Tensor(mask_tensor))

    # Note: for stacking, it is important to sort these masks by area from largest to smallest
    # this is because the masks are overlapping and we want smaller masks to
    # be drawn on top of larger masks.
    # TODO(jleibs): we could instead draw each mask as a separate image layer, but the current layer-stacking
    # does not produce great results.
    masks_with_ids = list(enumerate(masks, start=1))
    masks_with_ids.sort(key=(lambda x: x[1].sum().item()), reverse=True)  # type: ignore[no-any-return]

    # Layer all of the masks together, using the id as class-id in the segmentation
    segmentation_img = np.zeros((image.shape[0], image.shape[1]))
    # pdb.set_trace()
    for id, m in masks_with_ids:
        segmentation_img[m.cpu()[0]] = id

    rr.log("image/masks", rr.SegmentationImage(segmentation_img.astype(np.uint8)))

    # mask_bbox = np.array([m.cpu() for _, m in masks_with_ids])
    rr.log(
        "image/boxes",
        rr.Boxes2D(array=detections[0].cpu().boxes.xywh,
                   array_format=rr.Box2DFormat.XYWH,
                   class_ids=[id for id, _ in masks_with_ids]),
    )

def load_image(image) -> cv2.typing.MatLike:
    """Conditionally download an image from URL or load it from disk."""
    logging.info(f"Loading: image")

    # Rerun can handle BGR as well, but SAM requires RGB.
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

# def main():
if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Run the Facebook Research Segment Anything example.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    rr.script_add_args(parser)
    args = parser.parse_args()

    blueprint = rrb.Vertical(
            rrb.Spatial2DView(name="Image and segmentation mask", origin="/image"),
            rrb.Horizontal(
                rrb.TextLogView(name="Log", origin="/logs"),
                rrb.TextDocumentView(name="Description", origin="/description"),
                column_shares=[2, 1],
            ),
            row_shares=[3, 1],
        )

    rr.script_setup(args, "rerun_example_segment_anything_model", default_blueprint=blueprint)
    logging.getLogger().addHandler(rr.LoggingHandler("logs"))
    logging.getLogger().setLevel(logging.INFO)

    rr.log("description", rr.TextDocument(DESCRIPTION, media_type=rr.MediaType.MARKDOWN), timeless=True)

    sam = create_sam("vit_h", "cuda", "sam_vit_h_4b8939.pth")

    mask_config = {"points_per_batch": 32}
    mask_generator = SamAutomaticMaskGenerator(sam, **mask_config)
    mask_predictor = SamPredictor(sam)

    cap = cv2.VideoCapture("/home/thebird/Dynamworks/LLM_Module/Hackathon/working/sample_video.mp4")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    images_in_video = []
    frame_num = 1

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_num += 1
        if frame_num > 8:
            images_in_video.append(frame)
            # break

    cap.release()
    print(f"Len:{len(images_in_video)}")

    for n, image_uri in enumerate(images_in_video):
        rr.set_time_sequence("image", n)
        image = load_image(image_uri)
        # run_segmentation(mask_generator, image)
        run_segmentation_yolo(image=image,
                              mask_predictor=mask_predictor,
                              width=width,
                              height=height)
        print(n)

    # main()
