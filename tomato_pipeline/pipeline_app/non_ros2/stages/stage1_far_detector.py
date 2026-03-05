from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
from ultralytics import YOLO

BBox = Tuple[int, int, int, int]  # x1, y1, x2, y2


@dataclass
class Detection:
    cls: str  # e.g. "cluster" or "peduncle"
    conf: float
    bbox: BBox


class FarDetector:
    """
    Stage 1: Far detector for clusters + peduncles using Ultralytics YOLO.

    Usage:
        detector = FarDetector(weights_path="models/yolo_far.pt")
        detections = detector.run(frame)  # frame is a numpy BGR image (OpenCV)
    """

    def __init__(self, weights_path: str, conf_thres: float = 0.25, iou_thres: float = 0.7):
        self.model = YOLO(weights_path)
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres

        # Class name mapping from model (Ultralytics provides this)
        # Example: {0: 'cluster', 1: 'peduncle'}
        self.names = self.model.names

    def run(self, frame: np.ndarray) -> List[Detection]:
        """Run inference on a single frame and return Detection objects."""
        results = self.model.predict(
            source=frame,
            conf=self.conf_thres,
            iou=self.iou_thres,
            verbose=False,
        )

        detections: List[Detection] = []

        result = results[0]
        if result.boxes is None or len(result.boxes) == 0:
            return detections

        # boxes: xyxy + conf + cls
        xyxy = result.boxes.xyxy.cpu().numpy()
        confs = result.boxes.conf.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy().astype(int)

        for (x1, y1, x2, y2), conf, cls_id in zip(xyxy, confs, classes):
            cls_name = self.names.get(cls_id, str(cls_id))
            detections.append(
                Detection(
                    cls=cls_name,
                    conf=float(conf),
                    bbox=(int(x1), int(y1), int(x2), int(y2)),
                )
            )

        return detections
