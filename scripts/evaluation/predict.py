"""
predict.py
==========
Inference script for the Dual-Branch Spatio-Frequency Deepfake Detector.

Key Capabilities
----------------
1. Accepts ANY full-resolution image (raw photo, selfie, portrait, group photo).
2. Automatically detects faces using OpenCV's built-in face detector.
3. Crops and aligns each detected face to 256x256 (with graceful fallback if already cropped).
4. Runs inference through the trained DualBranchNet model.
5. Prints classification verdicts (REAL vs FAKE) with confidence percentages.
6. Optionally saves an annotated image with colored bounding boxes:
   - GREEN box : REAL (Authentic Human)
   - RED box   : FAKE (AI-Generated / Deepfake)

Usage
-----
::

    # Basic prediction on any full photo:
    py -3.12 predict.py --image "my_photo.jpg"

    # Specify custom checkpoint and custom output image path:
    py -3.12 predict.py --image "my_photo.jpg" --checkpoint "./checkpoints/best_model.pth" --output "result.jpg"
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import List, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms

from config import Config, cfg
from models.dual_branch_net import DualBranchNet
from utils import load_checkpoint


# --------------------------------------------------------------------------- #
#  Face Detection Helpers
# --------------------------------------------------------------------------- #

def detect_faces(
    image_bgr: np.ndarray,
    scale_factor: float = 1.1,
    min_neighbors: int = 5,
    min_size: Tuple[int, int] = (40, 40),
) -> List[Tuple[int, int, int, int]]:
    """Detect faces in a BGR image using OpenCV's Haar Cascade.

    Returns
    -------
    list of (x, y, w, h) bounding boxes.
    """
    try:
        if hasattr(cv2, "CascadeClassifier") and hasattr(cv2, "data") and hasattr(cv2.data, "haarcascades"):
            cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            if os.path.exists(cascade_path):
                gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
                face_cascade = cv2.CascadeClassifier(cascade_path)
                faces = face_cascade.detectMultiScale(
                    gray,
                    scaleFactor=scale_factor,
                    minNeighbors=min_neighbors,
                    minSize=min_size,
                    flags=getattr(cv2, "CASCADE_SCALE_IMAGE", 0),
                )
                if len(faces) > 0:
                    return [(int(x), int(y), int(w), int(h)) for (x, y, w, h) in faces]
    except Exception:
        pass
    return []


def crop_face_with_margin(
    image_bgr: np.ndarray,
    bbox: Tuple[int, int, int, int],
    margin: float = 0.20,
    target_size: int = 256,
) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
    """Crop a face box with an added margin to capture hair/chin context (like FFHQ).

    Returns
    -------
    (cropped_256x256_rgb, (x1, y1, x2, y2))
    """
    img_h, img_w = image_bgr.shape[:2]
    x, y, w, h = bbox

    # Expand box by margin
    margin_x = int(w * margin)
    margin_y = int(h * margin)

    x1 = max(0, x - margin_x)
    y1 = max(0, y - margin_y)
    x2 = min(img_w, x + w + margin_x)
    y2 = min(img_h, y + h + margin_y)

    crop = image_bgr[y1:y2, x1:x2]
    crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    crop_resized = cv2.resize(crop_rgb, (target_size, target_size), interpolation=cv2.INTER_LINEAR)

    return crop_resized, (x1, y1, x2, y2)


# --------------------------------------------------------------------------- #
#  Inference Pipeline
# --------------------------------------------------------------------------- #

def preprocess_crop(crop_rgb: np.ndarray, device: torch.device) -> torch.Tensor:
    """Transform a 256x256 RGB image into a normalized PyTorch tensor."""
    transform = transforms.Compose([
        transforms.ToTensor(),  # [0, 255] -> [0.0, 1.0], shape (3, H, W)
        transforms.Normalize(mean=cfg.img_mean, std=cfg.img_std),
    ])
    tensor = transform(crop_rgb).unsqueeze(0)  # (1, 3, 256, 256)
    return tensor.to(device)


@torch.no_grad()
def predict_single_crop(model: nn.Module, tensor: torch.Tensor) -> float:
    """Run model forward pass and return deepfake probability (0.0 to 1.0)."""
    model.eval()
    logit = model(tensor)  # (1, 1)
    prob = torch.sigmoid(logit).item()
    return float(prob)


# --------------------------------------------------------------------------- #
#  Main Prediction Routine
# --------------------------------------------------------------------------- #

def run_prediction(
    image_path: str,
    checkpoint_path: str = "./checkpoints/best_model.pth",
    output_path: str | None = None,
    threshold: float = 0.50,
    margin: float = 0.20,
) -> None:
    """Predict real vs fake on any arbitrary full image."""
    if not os.path.isfile(image_path):
        print(f"[ERROR] Image file not found: {image_path}")
        sys.exit(1)

    if not os.path.isfile(checkpoint_path):
        print(f"[ERROR] Model checkpoint not found: {checkpoint_path}")
        print("        Ensure you have a trained model in ./checkpoints/best_model.pth")
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")
    print(f"[INFO] Loading checkpoint: {checkpoint_path}")

    # ── Load Model ────────────────────────────────────────────────────
    model = DualBranchNet(
        spatial_backbone=cfg.spatial_backbone,
        spatial_pretrained=False,
        wavelet=cfg.wavelet_name,
        wavelet_level=cfg.wavelet_decomp_level,
        embed_dim=cfg.embed_dim,
    ).to(device)

    load_checkpoint(checkpoint_path, model, device=device)
    model.eval()

    # ── Read Image ────────────────────────────────────────────────────
    image_bgr = cv2.imread(image_path)
    if image_bgr is None:
        print(f"[ERROR] Could not decode image at: {image_path}")
        sys.exit(1)

    orig_h, orig_w = image_bgr.shape[:2]
    print(f"[INFO] Loaded image: {image_path} ({orig_w}x{orig_h})")

    # ── Detect Faces ──────────────────────────────────────────────────
    face_boxes = detect_faces(image_bgr)
    annotated = image_bgr.copy()

    print(f"\n{'='*60}")
    print(f"  PREDICTION REPORT: {os.path.basename(image_path)}")
    print(f"{'='*60}")

    if not face_boxes:
        print("[INFO] No distinct face box detected by face detector.")
        print("       Evaluating full image directly as a cropped face ...")

        # Smart crop: Preserve aspect ratio and focus on upper-center region (portrait face location)
        if orig_h > orig_w:
            # Portrait (e.g. 1024x1536): take upper square containing face/head
            crop_bgr = image_bgr[:orig_w, :]
        elif orig_w > orig_h:
            # Landscape: center square
            start_x = (orig_w - orig_h) // 2
            crop_bgr = image_bgr[:, start_x:start_x + orig_h]
        else:
            crop_bgr = image_bgr

        crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        crop_resized = cv2.resize(crop_rgb, (cfg.image_size, cfg.image_size))
        tensor = preprocess_crop(crop_resized, device)
        prob_fake = predict_single_crop(model, tensor)

        is_fake = prob_fake >= threshold
        label = "FAKE" if is_fake else "REAL"
        confidence = prob_fake if is_fake else (1.0 - prob_fake)

        print(f"\n  Final Verdict : {label}")
        print(f"  Probability   : {prob_fake * 100:.2f}% Fake  ({(1.0 - prob_fake) * 100:.2f}% Real)")
        print(f"  Confidence    : {confidence * 100:.2f}%")

        # Draw banner on top of full image
        color = (0, 0, 255) if is_fake else (0, 255, 0)  # Red for Fake, Green for Real
        cv2.putText(
            annotated,
            f"{label}: {confidence*100:.1f}%",
            (20, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            color,
            3,
            cv2.LINE_AA,
        )

    else:
        print(f"[INFO] Detected {len(face_boxes)} face(s) in image.\n")

        for idx, bbox in enumerate(face_boxes, 1):
            crop_resized, (x1, y1, x2, y2) = crop_face_with_margin(
                image_bgr, bbox, margin=margin, target_size=cfg.image_size
            )
            tensor = preprocess_crop(crop_resized, device)
            prob_fake = predict_single_crop(model, tensor)

            is_fake = prob_fake >= threshold
            label = "FAKE" if is_fake else "REAL"
            confidence = prob_fake if is_fake else (1.0 - prob_fake)

            print(f"  [Face #{idx}] Box ({x1}, {y1}) -> ({x2}, {y2})")
            print(f"    Verdict    : {label}")
            print(f"    Probability: {prob_fake * 100:.2f}% Fake  ({(1.0 - prob_fake) * 100:.2f}% Real)")
            print(f"    Confidence : {confidence * 100:.2f}%")

            # Draw visual bounding box and label tag
            color = (0, 0, 255) if is_fake else (0, 255, 0)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 3)

            # Label banner above bounding box
            tag = f"{label} {confidence*100:.1f}%"
            font_scale = max(0.5, (x2 - x1) / 300.0)
            text_size = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2)[0]
            tag_y1 = max(0, y1 - text_size[1] - 10)
            cv2.rectangle(annotated, (x1, tag_y1), (x1 + text_size[0] + 10, y1), color, -1)
            cv2.putText(
                annotated,
                tag,
                (x1 + 5, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

    print(f"{'='*60}\n")

    # ── Save Annotated Image ──────────────────────────────────────────
    if output_path is None:
        base, ext = os.path.splitext(image_path)
        output_path = f"{base}_prediction{ext}"

    cv2.imwrite(output_path, annotated)
    print(f"[INFO] Annotated result saved -> {output_path}")


# --------------------------------------------------------------------------- #
#  CLI
# --------------------------------------------------------------------------- #

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run deepfake detection on any full or cropped photo.",
    )
    parser.add_argument(
        "--image", type=str, required=True,
        help="Path to the input image file (full photo or face crop).",
    )
    parser.add_argument(
        "--checkpoint", type=str, default="./checkpoints/best_model.pth",
        help="Path to the trained model checkpoint (.pth).",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Path where the annotated image with bounding boxes will be saved.",
    )
    parser.add_argument(
        "--threshold", type=float, default=0.50,
        help="Classification threshold (default 0.50).",
    )
    parser.add_argument(
        "--margin", type=float, default=0.20,
        help="Margin percentage to expand around detected face box (default 0.20).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_prediction(
        image_path=args.image,
        checkpoint_path=args.checkpoint,
        output_path=args.output,
        threshold=args.threshold,
        margin=args.margin,
    )
