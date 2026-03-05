from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

BBox = Tuple[int, int, int, int]  # x1,y1,x2,y2


@dataclass
class Detection:
    cls: str  # "cluster" or "peduncle"/"pedicel"
    conf: float
    bbox: BBox


# Stage 2 output: one peduncle matched to one cluster, plus debug metadata.
@dataclass
class Pair:
    cluster: Detection
    peduncle: Detection
    score: float
    match_type: str
    center_dist_px: float


def _box_center(box: BBox) -> Tuple[float, float]:
    x1, y1, x2, y2 = box
    return 0.5 * (x1 + x2), 0.5 * (y1 + y2)


def _box_area(box: BBox) -> float:
    x1, y1, x2, y2 = box
    return max(0.0, (x2 - x1)) * max(0.0, (y2 - y1))


def _dist(ax, ay, bx, by) -> float:
    return math.sqrt((ax - bx) ** 2 + (ay - by) ** 2)


def _clamp(value, lo, hi):
    return max(lo, min(hi, value))


def _expand_box(box: BBox, pad_frac: float, width: int, height: int) -> Tuple[float, float, float, float]:
    x1, y1, x2, y2 = box
    box_w = x2 - x1
    box_h = y2 - y1
    pad_x = pad_frac * box_w
    pad_y = pad_frac * box_h

    ex1 = _clamp(int(round(x1 - pad_x)), 0, width - 1)
    ey1 = _clamp(int(round(y1 - pad_y)), 0, height - 1)
    ex2 = _clamp(int(round(x2 + pad_x)), 0, width - 1)
    ey2 = _clamp(int(round(y2 + pad_y)), 0, height - 1)
    return float(ex1), float(ey1), float(ex2), float(ey2)


def _point_in_box(px, py, x1, y1, x2, y2) -> bool:
    return (px >= x1) and (px <= x2) and (py >= y1) and (py <= y2)


def _intersection_area(a: BBox, b: BBox) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    return iw * ih


def _overlap_frac_of_pedicel(ped: BBox, cluster: BBox) -> float:
    inter = _intersection_area(ped, cluster)
    ped_area = max(1.0, _box_area(ped))
    return inter / ped_area


class Pairer:
    """
    Stage 2: Pair peduncle detections to cluster detections.

    Rules:
    A0: peduncle box overlaps original cluster box by >= MIN_A0_OVERLAP_FRAC
    A1: peduncle center inside expanded cluster box
    A2: nearest cluster center with distance gate
    """

    def __init__(
        self,
        cluster_cls: str = "cluster",
        peduncle_cls: str = "pedicel",  # model may use "pedicel"
        cluster_pad_frac: float = 0.15,
        max_nearest_dist_frac: float = 0.60,
        enforce_one_peduncle_per_cluster: bool = True,
        min_a0_overlap_frac: float = 0.01,
        enforce_peduncle_above_cluster: bool = True,
        y_margin_px: int = 0,
    ):
        self.cluster_cls = cluster_cls
        self.peduncle_cls = peduncle_cls
        self.cluster_pad_frac = cluster_pad_frac
        self.max_nearest_dist_frac = max_nearest_dist_frac
        self.enforce_one = enforce_one_peduncle_per_cluster
        self.min_a0_overlap_frac = min_a0_overlap_frac
        self.enforce_above = enforce_peduncle_above_cluster
        self.y_margin_px = y_margin_px

    def run(self, detections: List[Detection], frame_shape: Tuple[int, int]) -> List[Pair]:
        """frame_shape: (h, w) from frame.shape[:2]."""
        height, width = frame_shape

        clusters: List[Dict] = []
        peduncles: List[Dict] = []

        for detection in detections:
            if detection.cls == self.cluster_cls:
                cx, cy = _box_center(detection.bbox)
                area = _box_area(detection.bbox)
                ex1, ey1, ex2, ey2 = _expand_box(detection.bbox, self.cluster_pad_frac, width, height)
                clusters.append(
                    {
                        "det": detection,
                        "cx": cx,
                        "cy": cy,
                        "area": area,
                        "ex": (ex1, ey1, ex2, ey2),
                    }
                )
            elif detection.cls == self.peduncle_cls:
                cx, cy = _box_center(detection.bbox)
                peduncles.append({"det": detection, "cx": cx, "cy": cy})

        if not clusters or not peduncles:
            return []

        # Stable physical order: top -> bottom
        peduncles.sort(key=lambda item: item["cy"])

        used_clusters = set()
        pairs: List[Pair] = []

        for peduncle in peduncles:
            ped_det: Detection = peduncle["det"]
            px, py = peduncle["cx"], peduncle["cy"]

            best_i: Optional[int] = None
            best_d: Optional[float] = None
            match_type = "unmatched"

            # A0: overlap on original cluster box
            cand_a0 = []
            for i, cluster in enumerate(clusters):
                if self.enforce_one and i in used_clusters:
                    continue

                frac = _overlap_frac_of_pedicel(ped_det.bbox, cluster["det"].bbox)
                if frac < self.min_a0_overlap_frac:
                    continue

                if self.enforce_above and not (py + self.y_margin_px < cluster["cy"]):
                    continue

                distance = _dist(px, py, cluster["cx"], cluster["cy"])
                cand_a0.append((distance, i))

            if cand_a0:
                cand_a0.sort(key=lambda pair: pair[0])
                best_d, best_i = cand_a0[0]
                match_type = "contained_orig_overlap"

            # A1: pedicel center inside expanded cluster box
            if best_i is None:
                cand_a1 = []
                for i, cluster in enumerate(clusters):
                    if self.enforce_one and i in used_clusters:
                        continue

                    if self.enforce_above and not (py + self.y_margin_px < cluster["cy"]):
                        continue

                    ex1, ey1, ex2, ey2 = cluster["ex"]
                    if not _point_in_box(px, py, ex1, ey1, ex2, ey2):
                        continue

                    distance = _dist(px, py, cluster["cx"], cluster["cy"])
                    cand_a1.append((distance, i))

                if cand_a1:
                    cand_a1.sort(key=lambda pair: pair[0])
                    best_d, best_i = cand_a1[0]
                    match_type = "contained_expanded"

            # A2: nearest + distance gate
            if best_i is None:
                nearest = []
                for i, cluster in enumerate(clusters):
                    if self.enforce_one and i in used_clusters:
                        continue

                    if self.enforce_above and not (py + self.y_margin_px < cluster["cy"]):
                        continue

                    distance = _dist(px, py, cluster["cx"], cluster["cy"])
                    nearest.append((distance, i))

                if nearest:
                    nearest.sort(key=lambda pair: pair[0])
                    best_d, best_i = nearest[0]
                    match_type = "nearest"

                    cluster_best = clusters[best_i]
                    gate = self.max_nearest_dist_frac * math.sqrt(max(cluster_best["area"], 1.0))
                    if best_d > gate:
                        best_i = None
                        best_d = None
                        match_type = "unmatched"

            if best_i is not None:
                if self.enforce_one:
                    used_clusters.add(best_i)

                cluster_det = clusters[best_i]["det"]
                score = float(ped_det.conf * cluster_det.conf)

                pairs.append(
                    Pair(
                        cluster=cluster_det,
                        peduncle=ped_det,
                        score=score,
                        match_type=match_type,
                        center_dist_px=float(best_d if best_d is not None else 0.0),
                    )
                )

        return pairs
