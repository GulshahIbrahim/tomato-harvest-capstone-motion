from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from ultralytics import YOLO

BBox = Tuple[int, int, int, int]


@dataclass
class Detection:
    cls: str
    conf: float
    bbox: BBox


# Output from Stage 2. Stage 3 only processes paired clusters.
@dataclass
class Pair:
    cluster: Detection
    peduncle: Detection
    score: float
    match_type: str
    center_dist_px: float


@dataclass
class ClusterResult:
    pair: Pair
    ripeness: str
    ripeness_conf: float
    target_xy: Tuple[float, float]  # pixel coordinate (cx, cy)
    cluster_bbox: Optional[BBox] = None  # crop bbox in full image


def _load_classifier(ckpt_path: str, device: str):
    ckpt = torch.load(str(ckpt_path), map_location=device)
    class_names = ckpt["class_names"]
    img_size = ckpt.get("img_size", 224)

    model = models.mobilenet_v2(weights=None)
    model.classifier[1] = nn.Linear(model.last_channel, len(class_names))
    model.load_state_dict(ckpt["model_state"])
    model.eval().to(device)

    tfm = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )
    return model, tfm, class_names


def _crop_from_mask(img_bgr: np.ndarray, mask255: np.ndarray, pad_px: int, mask_background: bool):
    """mask255 is HxW uint8 (0 or 255). Returns crop_bgr and bbox in full image coords."""
    height, width = mask255.shape[:2]
    ys, xs = np.where(mask255 > 0)
    if len(xs) == 0:
        return None, None

    x1, x2 = int(xs.min()), int(xs.max())
    y1, y2 = int(ys.min()), int(ys.max())

    x1 = max(0, x1 - pad_px)
    y1 = max(0, y1 - pad_px)
    x2 = min(width - 1, x2 + pad_px)
    y2 = min(height - 1, y2 + pad_px)

    crop = img_bgr[y1 : y2 + 1, x1 : x2 + 1].copy()
    crop_mask = mask255[y1 : y2 + 1, x1 : x2 + 1]

    if mask_background:
        crop[crop_mask == 0] = 0

    return crop, (x1, y1, x2, y2)


def _bbox_center(box: BBox) -> Tuple[float, float]:
    x1, y1, x2, y2 = box
    return 0.5 * (x1 + x2), 0.5 * (y1 + y2)


class RipenessStage:
    """
    Stage 3:
      - run segmentation (cluster masks)
      - mask/crop the cluster
      - classify ripeness (ripe/unripe/mix)
      - return results ONLY for the paired clusters (pairs input)
    """

    def __init__(
        self,
        seg_weights_path: str,
        cls_ckpt_path: str,
        conf_seg: float = 0.25,
        imgsz: int = 640,
        pad_px: int = 10,
        mask_background: bool = True,
        device: Optional[str] = None,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.seg_model = YOLO(str(seg_weights_path))
        self.cls_model, self.tfm, self.class_names = _load_classifier(cls_ckpt_path, self.device)

        self.conf_seg = conf_seg
        self.imgsz = imgsz
        self.pad_px = pad_px
        self.mask_background = mask_background

    def _classify_crop(self, crop_bgr: np.ndarray) -> Tuple[str, float]:
        crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        tensor = self.tfm(crop_rgb).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.cls_model(tensor)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
            pred = int(np.argmax(probs))
            conf = float(probs[pred])

        return self.class_names[pred], conf

    def run(self, frame_bgr: np.ndarray, pairs: List[Pair]) -> List[ClusterResult]:
        if not pairs:
            return []

        height, width = frame_bgr.shape[:2]
        results_out: List[ClusterResult] = []

        # Extra padding around cluster bbox for ROI seg
        roi_pad = 40

        for pair in pairs:
            x1, y1, x2, y2 = pair.cluster.bbox

            # clamp + pad ROI
            rx1 = max(0, x1 - roi_pad)
            ry1 = max(0, y1 - roi_pad)
            rx2 = min(width - 1, x2 + roi_pad)
            ry2 = min(height - 1, y2 + roi_pad)

            roi = frame_bgr[ry1:ry2, rx1:rx2].copy()
            if roi.size == 0:
                continue

            # Segmentation on ROI only
            seg_results = self.seg_model.predict(
                source=roi,
                conf=self.conf_seg,
                imgsz=self.imgsz,
                device=0 if torch.cuda.is_available() else None,
                verbose=False,
            )
            result = seg_results[0]

            crop = None
            crop_bbox_full = None

            if result.masks is not None:
                roi_h, roi_w = roi.shape[:2]
                masks = result.masks.data.cpu().numpy()

                # Pick the biggest mask (can later switch to center-nearest)
                best_area = 0
                best_mask_full = None

                for mask in masks:
                    mask = (mask > 0.5).astype("uint8") * 255
                    mask = cv2.resize(mask, (roi_w, roi_h), interpolation=cv2.INTER_NEAREST)
                    area = int(np.count_nonzero(mask))
                    if area > best_area:
                        best_area = area
                        best_mask_full = mask

                if best_mask_full is not None and best_area > 0:
                    crop_roi, bbox_roi = _crop_from_mask(
                        roi,
                        best_mask_full,
                        pad_px=self.pad_px,
                        mask_background=self.mask_background,
                    )
                    if crop_roi is not None and bbox_roi is not None:
                        # Convert bbox back to full image coords
                        bx1, by1, bx2, by2 = bbox_roi
                        crop_bbox_full = (rx1 + bx1, ry1 + by1, rx1 + bx2, ry1 + by2)
                        crop = crop_roi

            # Fallback: use paired cluster bbox crop
            if crop is None:
                crop = frame_bgr[y1:y2, x1:x2].copy()
                crop_bbox_full = (x1, y1, x2, y2)

            if crop is None or crop.size == 0:
                continue

            label, conf = self._classify_crop(crop)
            target_xy = _bbox_center(crop_bbox_full)

            results_out.append(
                ClusterResult(
                    pair=pair,
                    ripeness=label,
                    ripeness_conf=conf,
                    target_xy=target_xy,
                    cluster_bbox=crop_bbox_full,
                )
            )

        return results_out
