"""
Stage 5 (Robot Mode)
Close perception assuming robot moved and captured a new close-range image.

- No ROI/cropping from far stage
- Runs YOLO-Pose on the full close-range image
- Selects main cluster and closest pedicel
- Returns cutpoint pixel coordinates (u,v) + keypoints
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from ultralytics import YOLO


@dataclass
class CloseRobotOutput:
    # Cluster detection
    cluster_bbox: Optional[Tuple[int, int, int, int]]
    cluster_conf: float

    # Pedicel detection
    pedicel_bbox: Optional[Tuple[int, int, int, int]]
    pedicel_conf: float

    # Keypoints (3 points along pedicel)
    keypoints: List[Tuple[float, float]]
    keypoint_confs: List[float]

    # Cut point (first keypoint)
    cut_point: Optional[Tuple[float, float]]

    detection_success: bool
    debug_info: Dict[str, Any]


def bbox_center_xyxy(box: Tuple[float, float, float, float]) -> Tuple[float, float]:
    x1, y1, x2, y2 = box
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def bbox_area_xyxy(box: Tuple[float, float, float, float]) -> float:
    x1, y1, x2, y2 = box
    return max(0.0, (x2 - x1)) * max(0.0, (y2 - y1))


class ClosePerceptionRobotStage:
    """Close-range perception for robot mode."""

    def __init__(
        self,
        model_path: str,
        imgsz: int = 768,
        conf_thres: float = 0.25,
        iou_thres: float = 0.45,
        kpt_conf_thres: float = 0.3,
        cluster_cls: int = 0,
        pedicel_cls: int = 1,
    ):
        self.model = YOLO(model_path)
        self.imgsz = imgsz
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.kpt_conf_thres = kpt_conf_thres
        self.cluster_cls = cluster_cls
        self.pedicel_cls = pedicel_cls

    def run(self, frame: np.ndarray) -> CloseRobotOutput:
        device = 0 if torch.cuda.is_available() else "cpu"

        preds = self.model.predict(
            source=frame,
            conf=self.conf_thres,
            iou=self.iou_thres,
            imgsz=self.imgsz,
            verbose=False,
            device=device,
        )
        result = preds[0]

        if result.boxes is None or len(result.boxes) == 0:
            return CloseRobotOutput(
                cluster_bbox=None,
                cluster_conf=0.0,
                pedicel_bbox=None,
                pedicel_conf=0.0,
                keypoints=[],
                keypoint_confs=[],
                cut_point=None,
                detection_success=False,
                debug_info={"failure_reason": "no_detections"},
            )

        boxes = result.boxes
        cls_ids = boxes.cls.cpu().numpy().astype(int)
        confs = boxes.conf.cpu().numpy()
        xyxy = boxes.xyxy.cpu().numpy()

        # 1) Choose the main cluster as the largest cluster.
        cluster_mask = cls_ids == self.cluster_cls
        if not cluster_mask.any():
            return CloseRobotOutput(
                cluster_bbox=None,
                cluster_conf=0.0,
                pedicel_bbox=None,
                pedicel_conf=0.0,
                keypoints=[],
                keypoint_confs=[],
                cut_point=None,
                detection_success=False,
                debug_info={"failure_reason": "no_cluster"},
            )

        cluster_idxs = np.where(cluster_mask)[0]
        chosen_cluster_idx = None
        best_area = -1.0

        for idx in cluster_idxs:
            area = bbox_area_xyxy(tuple(xyxy[idx]))
            if area > best_area:
                best_area = area
                chosen_cluster_idx = idx

        cx1, cy1, cx2, cy2 = xyxy[chosen_cluster_idx]
        cluster_bbox = (int(cx1), int(cy1), int(cx2), int(cy2))
        cluster_conf = float(confs[chosen_cluster_idx])
        cluster_center = bbox_center_xyxy((cx1, cy1, cx2, cy2))

        # 2) Choose pedicel closest to that cluster center.
        ped_mask = cls_ids == self.pedicel_cls
        if not ped_mask.any():
            return CloseRobotOutput(
                cluster_bbox=cluster_bbox,
                cluster_conf=cluster_conf,
                pedicel_bbox=None,
                pedicel_conf=0.0,
                keypoints=[],
                keypoint_confs=[],
                cut_point=None,
                detection_success=False,
                debug_info={"failure_reason": "no_pedicel"},
            )

        ped_idxs = np.where(ped_mask)[0]
        chosen_ped_idx = None
        best_dist2 = 1e18

        for idx in ped_idxs:
            px1, py1, px2, py2 = xyxy[idx]
            ped_center = bbox_center_xyxy((px1, py1, px2, py2))
            dist2 = (ped_center[0] - cluster_center[0]) ** 2 + (ped_center[1] - cluster_center[1]) ** 2
            if dist2 < best_dist2:
                best_dist2 = dist2
                chosen_ped_idx = idx

        px1, py1, px2, py2 = xyxy[chosen_ped_idx]
        pedicel_bbox = (int(px1), int(py1), int(px2), int(py2))
        pedicel_conf = float(confs[chosen_ped_idx])

        # 3) Extract keypoints for chosen pedicel.
        keypoints: List[Tuple[float, float]] = []
        keypoint_confs: List[float] = []
        cut_point: Optional[Tuple[float, float]] = None

        if result.keypoints is not None and len(result.keypoints.data) > 0:
            # shape: [num_dets, num_kpts, 3]
            kpts = result.keypoints.data[chosen_ped_idx].cpu().numpy()
            for i in range(min(3, kpts.shape[0])):
                kx, ky, kconf = kpts[i]
                if float(kconf) > self.kpt_conf_thres:
                    keypoints.append((float(kx), float(ky)))
                    keypoint_confs.append(float(kconf))

            if keypoints:
                cut_point = keypoints[0]

        detection_success = cut_point is not None

        return CloseRobotOutput(
            cluster_bbox=cluster_bbox,
            cluster_conf=cluster_conf,
            pedicel_bbox=pedicel_bbox,
            pedicel_conf=pedicel_conf,
            keypoints=keypoints,
            keypoint_confs=keypoint_confs,
            cut_point=cut_point,
            detection_success=detection_success,
            debug_info={
                "num_detections": int(len(boxes)),
                "num_clusters": int(cluster_mask.sum()),
                "num_pedicels": int(ped_mask.sum()),
                "chosen_cluster_idx": int(chosen_cluster_idx),
                "chosen_pedicel_idx": int(chosen_ped_idx),
                "chosen_pedicel_dist2": float(best_dist2),
            },
        )
