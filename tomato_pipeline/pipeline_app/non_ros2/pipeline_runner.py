"""
Robot Harvesting Pipeline (MID-MARCH SCOPE)
Far capture -> Far perception -> coarse XYZ -> robot approach move
-> Close capture -> Close perception -> cutpoint XYZ.

This module supports two execution styles:
1) `HarvestPipelineRunner`: long-lived, efficient multi-run object.
2) `run_pipeline(...)`: backward-compatible one-shot wrapper.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from pipeline_app.non_ros2.stages.stage1_far_detector import FarDetector, Detection
from pipeline_app.non_ros2.stages.stage2_pairing import Pairer
from pipeline_app.non_ros2.stages.stage3_ripeness import RipenessStage
from pipeline_app.non_ros2.stages.stage5_close_robot import ClosePerceptionRobotStage
from pipeline_app.non_ros2.vision.camera import Camera
from pipeline_app.ros2.robot_client import RobotClientROS2


class PipelineRunStatus:
    SUCCESS = "SUCCESS"
    NO_RIPE_TARGET = "NO_RIPE_TARGET"
    INVALID_FAR_DEPTH = "INVALID_FAR_DEPTH"
    MOTION_FAILED = "MOTION_FAILED"
    CLOSE_DETECTION_FAILED = "CLOSE_DETECTION_FAILED"
    INVALID_CUT_DEPTH = "INVALID_CUT_DEPTH"


@dataclass(frozen=True)
class PipelineRunResult:
    status: str
    message: str
    timestamp: float
    run_id: int | None = None
    cluster_bbox: tuple[int, int, int, int] | None = None
    cut_point: tuple[float, float] | None = None
    coarse_xyz_cam: tuple[float, float, float] | None = None
    cut_xyz_cam: tuple[float, float, float] | None = None
    run_json_path: str | None = None
    output_dir: str | None = None


# -------------------------
# Depth + Projection Helpers
# -------------------------

def depth_at(depth_img, u, v, win=7, invalid=0, scale=1.0):
    """
    Robust depth sampling: take a small window around (u,v), filter invalid/zero,
    return median depth * scale.
    """
    height, width = depth_img.shape[:2]
    u, v = int(round(u)), int(round(v))
    radius = win // 2
    x1, x2 = max(0, u - radius), min(width, u + radius + 1)
    y1, y2 = max(0, v - radius), min(height, v + radius + 1)

    patch = depth_img[y1:y2, x1:x2].astype(np.float32).reshape(-1)
    patch = patch[(patch != invalid) & (patch > 0)]
    if patch.size == 0:
        return None

    return float(np.median(patch)) * scale


def uvz_to_xyz(u, v, z, fx, fy, cx, cy):
    """
    Pinhole back-projection: (u,v,z) -> (x,y,z) in CAMERA frame.
    Units: z should be in meters (after applying depth_scale).
    """
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    return x, y, z


def _to_serializable(value):
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(k): _to_serializable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_serializable(v) for v in value]
    return str(value)


class HarvestPipelineRunner:
    """Long-lived runner that keeps heavy resources initialized across runs."""

    def __init__(self, log: Callable[[str], None] = print, interactive: bool = None):
        self.log = log
        if interactive is None:
            self.interactive = os.environ.get("PIPELINE_INTERACTIVE", "false").lower() in ("true", "1", "yes")
        else:
            self.interactive = interactive
        self.gui_available = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
        if self.interactive and not self.gui_available:
            self.log("[init] PIPELINE_INTERACTIVE is enabled but no DISPLAY/WAYLAND_DISPLAY is available. Falling back to non-interactive mode.")
            self.interactive = False
        self._closed = False
        self.robot = None
        self.camera = None
        self.detector = None
        self.pairer = None
        self.ripeness = None
        self.close_stage = None
        self.run_counter = 0

        self.project_root = Path(__file__).resolve().parents[2]

        model_dir = self.project_root / "models"
        far_model = str(model_dir / "yolo_far.pt")
        seg_model = str(model_dir / "seg_best.pt")
        cls_model = str(model_dir / "mobilenetv2_best.pt")
        close_model = str(model_dir / "yolo_close.pt")

        # Motion/configuration defaults. Can be overridden via env.
        self.standoff_m = float(os.environ.get("PIPELINE_STANDOFF_M", "0.20"))
        self.move_vel = int(os.environ.get("PIPELINE_MOVE_VEL", "40"))
        self.move_acc = int(os.environ.get("PIPELINE_MOVE_ACC", "10"))
        self.move_timeout = float(os.environ.get("PIPELINE_MOVE_TIMEOUT_S", "30.0"))
        self.warmup_frames = int(os.environ.get("PIPELINE_CAMERA_WARMUP_FRAMES", "20"))

        # Output organization defaults.
        self.data_root = Path(os.environ.get("PIPELINE_DATA_ROOT", "/data"))
        self.trial_num = str(os.environ.get("PIPELINE_TRIAL_NUM", "1")).strip() or "1"

        # SG2 URDF D405 frame (default aligns with gigas-superproject semantics).
        self.camera_frame = os.environ.get("D405_TF_FRAME", "tool_d405_camera_origin").strip()
        if not self.camera_frame:
            raise ValueError("D405_TF_FRAME must be non-empty if provided.")

        try:
            init_t0 = time.time()
            self.log("[init] Initializing long-lived pipeline resources...")

            # Initialize heavy resources once.
            self.robot = RobotClientROS2()
            self.camera = Camera()
            self.detector = FarDetector(far_model)
            self.cluster_cls_name = "c"
            self.peduncle_cls_name = "p"
            self.pairer = Pairer(
                cluster_cls=self.cluster_cls_name,
                peduncle_cls=self.peduncle_cls_name,
            )
            self.ripeness = RipenessStage(
                seg_weights_path=seg_model,
                cls_ckpt_path=cls_model,
                imgsz=320,
                conf_seg=0.30,
            )
            self.close_stage = ClosePerceptionRobotStage(
                model_path=close_model,
                imgsz=768,
            )

            self.log(f"[init] Using RealSense D405 serial: {self.camera.serial}")
            self.log(f"[init] Using D405 TF frame: {self.camera_frame}")

            # Warm up camera once for exposure/depth stabilization.
            self.log(f"[init] Warming camera for {self.warmup_frames} frames...")
            for _ in range(self.warmup_frames):
                self.camera.get_frames()

            intrinsics = self.camera.get_intrinsics()
            self.fx, self.fy, self.cx, self.cy = (
                intrinsics.fx,
                intrinsics.fy,
                intrinsics.cx,
                intrinsics.cy,
            )
            self.log(
                "[init] Camera intrinsics (from active stream): "
                f"fx={self.fx:.3f}, fy={self.fy:.3f}, cx={self.cx:.3f}, cy={self.cy:.3f}, "
                f"width={intrinsics.width}, height={intrinsics.height}"
            )
            self.log(f"[init] Resource initialization completed in {time.time() - init_t0:.2f}s")
        except Exception:
            self.close()
            raise

    def _save_image(self, path: Path, image: np.ndarray):
        ok = cv2.imwrite(str(path), image)
        if ok:
            self.log(f"[artifact] Saved image: {path}")
        else:
            self.log(f"[warn] Failed to save image: {path}")
        return ok

    def _normalize_detection_cls(self, cls_name: str) -> str:
        cls_norm = str(cls_name).strip().lower()
        if cls_norm in ("c", "cluster"):
            return self.cluster_cls_name
        if cls_norm in ("p", "peduncle", "pedicel"):
            return self.peduncle_cls_name
        return cls_norm

    def _annotate_stage1(self, frame: np.ndarray, detections):
        out = frame.copy()
        for idx, detection in enumerate(detections):
            x1, y1, x2, y2 = detection.bbox
            cls_norm = self._normalize_detection_cls(detection.cls)
            if cls_norm == self.cluster_cls_name:
                color = (0, 255, 0)
                cls_text = "cluster"
            elif cls_norm == self.peduncle_cls_name:
                color = (0, 165, 255)
                cls_text = "peduncle"
            else:
                color = (200, 200, 200)
                cls_text = str(detection.cls)

            cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
            label = f"{cls_text}#{idx} {float(detection.conf):.2f}"
            cv2.putText(
                out,
                label,
                (x1, max(20, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2,
                cv2.LINE_AA,
            )
        return out

    def _confirm_detections(self, frame: np.ndarray, detections, stage_name: str) -> tuple[bool, list]:
        """Display detections and wait for user confirmation. Returns (proceed, updated_detections)."""
        for detection in detections:
            detection.cls = self._normalize_detection_cls(detection.cls)

        if not self.interactive:
            return True, detections

        if not detections:
            self.log("[interactive] No Stage 1 detections found. Opening edit mode so you can draw manual boxes.")
            detections = self._edit_detections(frame, detections)
            for detection in detections:
                detection.cls = self._normalize_detection_cls(detection.cls)

        while True:
            annotated = self._annotate_stage1(frame, detections)
            cv2.imshow(f"Confirm {stage_name} Detections", annotated)
            self.log(f"[interactive] Confirming {stage_name} detections. Press 'y' to proceed, 'n' to skip run, 'e' to edit.")
            key = cv2.waitKey(0) & 0xFF
            if key == ord('y'):
                has_cluster = any(detection.cls == self.cluster_cls_name for detection in detections)
                has_peduncle = any(detection.cls == self.peduncle_cls_name for detection in detections)
                if not (has_cluster and has_peduncle):
                    self.log(
                        "[interactive] Need at least one cluster and one peduncle box to continue. "
                        "Press 'e' to add missing boxes."
                    )
                    continue
                cv2.destroyWindow(f"Confirm {stage_name} Detections")
                return True, detections
            elif key == ord('n'):
                cv2.destroyWindow(f"Confirm {stage_name} Detections")
                return False, detections
            elif key == ord('e'):
                cv2.destroyWindow(f"Confirm {stage_name} Detections")
                detections = self._edit_detections(frame, detections)
                # Loop back to confirm again
            else:
                continue
        return True, detections

    def _edit_detections(self, frame: np.ndarray, detections) -> list:
        """Allow user to add manual detections by drawing boxes."""
        self.log("[interactive] Edit mode: Click and drag to draw a box, then press 'c' for cluster or 'p' for peduncle to add. Press 'd' when done.")
        
        drawing = False
        start_point = None
        end_point = None
        temp_frame = self._annotate_stage1(frame, detections).copy()
        
        def mouse_callback(event, x, y, flags, param):
            nonlocal drawing, start_point, end_point, temp_frame
            if event == cv2.EVENT_LBUTTONDOWN:
                drawing = True
                start_point = (x, y)
            elif event == cv2.EVENT_MOUSEMOVE:
                if drawing:
                    temp_frame = self._annotate_stage1(frame, detections).copy()
                    cv2.rectangle(temp_frame, start_point, (x, y), (255, 0, 0), 2)
            elif event == cv2.EVENT_LBUTTONUP:
                drawing = False
                end_point = (x, y)
                temp_frame = self._annotate_stage1(frame, detections).copy()
                cv2.rectangle(temp_frame, start_point, end_point, (255, 0, 0), 2)
        
        cv2.namedWindow("Edit Detections")
        cv2.setMouseCallback("Edit Detections", mouse_callback)
        
        while True:
            cv2.imshow("Edit Detections", temp_frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('c') and start_point and end_point:
                x1, y1 = start_point
                x2, y2 = end_point
                bbox = (min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2))
                detections.append(Detection(cls=self.cluster_cls_name, conf=1.0, bbox=bbox))
                self.log(f"[interactive] Added cluster: {bbox}")
                start_point = None
                end_point = None
                temp_frame = self._annotate_stage1(frame, detections).copy()
            elif key == ord('p') and start_point and end_point:
                x1, y1 = start_point
                x2, y2 = end_point
                bbox = (min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2))
                detections.append(Detection(cls=self.peduncle_cls_name, conf=1.0, bbox=bbox))
                self.log(f"[interactive] Added peduncle: {bbox}")
                start_point = None
                end_point = None
                temp_frame = self._annotate_stage1(frame, detections).copy()
            elif key == ord('d'):
                break
        
        cv2.destroyWindow("Edit Detections")
        return detections

    def _confirm_pairs(self, frame: np.ndarray, pairs, stage_name: str) -> bool:
        if not self.interactive:
            return True

        window_name = f"Confirm {stage_name} Pairs"
        while True:
            annotated = self._annotate_stage2(frame, pairs)
            cv2.imshow(window_name, annotated)
            self.log(f"[interactive] Confirming {stage_name} pairs. Press 'y' to proceed or 'n' to skip run.")
            key = cv2.waitKey(0) & 0xFF
            if key == ord('y'):
                cv2.destroyWindow(window_name)
                return True
            if key == ord('n'):
                cv2.destroyWindow(window_name)
                return False

    def _annotate_stage2(self, frame: np.ndarray, pairs):
        out = frame.copy()
        for idx, pair in enumerate(pairs):
            cx1, cy1, cx2, cy2 = pair.cluster.bbox
            px1, py1, px2, py2 = pair.peduncle.bbox
            cv2.rectangle(out, (cx1, cy1), (cx2, cy2), (0, 255, 0), 2)
            cv2.rectangle(out, (px1, py1), (px2, py2), (0, 165, 255), 2)
            ccx, ccy = int(0.5 * (cx1 + cx2)), int(0.5 * (cy1 + cy2))
            pcx, pcy = int(0.5 * (px1 + px2)), int(0.5 * (py1 + py2))
            cv2.line(out, (ccx, ccy), (pcx, pcy), (255, 255, 0), 2)
            text = f"pair{idx} {pair.match_type} d={pair.center_dist_px:.1f}"
            cv2.putText(out, text, (px1, max(20, py1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2, cv2.LINE_AA)
        return out

    def _annotate_stage3(self, frame: np.ndarray, results, chosen_idx: int | None):
        out = frame.copy()
        for idx, res in enumerate(results):
            if res.cluster_bbox is None:
                continue
            x1, y1, x2, y2 = res.cluster_bbox
            is_chosen = chosen_idx is not None and idx == chosen_idx
            color = (0, 255, 0) if is_chosen else (255, 200, 0)
            cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
            label = f"{res.ripeness}:{res.ripeness_conf:.2f}"
            if is_chosen:
                label += " [chosen]"
            cv2.putText(out, label, (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)
        return out

    def _annotate_stage5(self, frame: np.ndarray, close_out):
        out = frame.copy()
        if close_out.cluster_bbox is not None:
            x1, y1, x2, y2 = close_out.cluster_bbox
            cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
        if close_out.pedicel_bbox is not None:
            x1, y1, x2, y2 = close_out.pedicel_bbox
            cv2.rectangle(out, (x1, y1), (x2, y2), (0, 165, 255), 2)
        for i, point in enumerate(close_out.keypoints):
            px, py = int(point[0]), int(point[1])
            cv2.circle(out, (px, py), 4, (255, 255, 255), -1)
            cv2.putText(out, f"k{i}", (px + 6, py - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        if close_out.cut_point is not None:
            cx, cy = int(close_out.cut_point[0]), int(close_out.cut_point[1])
            cv2.circle(out, (cx, cy), 7, (0, 0, 255), -1)
            cv2.putText(out, "cut", (cx + 8, cy - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)
        return out

    def _prepare_output_paths(self, run_id: int, run_timestamp: float):
        cur_date = datetime.utcnow().strftime("%Y-%m-%d")
        parent_dir = self.data_root / cur_date / self.trial_num
        images_dir = parent_dir / "images"
        labelled_dir = parent_dir / "labelled_images"
        parent_dir.mkdir(parents=True, exist_ok=True)
        images_dir.mkdir(parents=True, exist_ok=True)
        labelled_dir.mkdir(parents=True, exist_ok=True)

        base = f"harvest{run_id}_d405_{run_timestamp:.4f}"
        return {
            "parent_dir": parent_dir,
            "images_dir": images_dir,
            "labelled_dir": labelled_dir,
            "far_rgb": images_dir / f"{base}_far_rgb.jpg",
            "far_depth": images_dir / f"{base}_far_depth.png",
            "close_rgb": images_dir / f"{base}_close_rgb.jpg",
            "close_depth": images_dir / f"{base}_close_depth.png",
            "ann_stage1": labelled_dir / f"{base}_stage1_detections.jpg",
            "ann_stage2": labelled_dir / f"{base}_stage2_pairs.jpg",
            "ann_stage3": labelled_dir / f"{base}_stage3_ripeness.jpg",
            "ann_stage5": labelled_dir / f"{base}_stage5_close_perception.jpg",
        }

    def _write_run_json(self, parent_dir: Path, run_id: int, run_payload: dict):
        ts_str = datetime.utcnow().strftime("%Y-%m-%d_%H-%M-%S") + " UTC"
        json_path = parent_dir / f"trial-{self.trial_num}-pipeline-run{run_id}-{ts_str}.json"
        with open(json_path, "w", encoding="utf-8") as outfile:
            json.dump(_to_serializable(run_payload), outfile, indent=4)
        return json_path

    def run_once(self) -> PipelineRunResult:
        if self._closed:
            raise RuntimeError("HarvestPipelineRunner is closed.")

        self.run_counter += 1
        run_id = self.run_counter
        run_start = time.time()
        run_start_utc = datetime.utcnow().strftime("%Y-%m-%d_%H-%M-%S UTC")

        output_paths = self._prepare_output_paths(run_id=run_id, run_timestamp=run_start)
        self.log(f"[run {run_id}] Artifacts directory: {output_paths['parent_dir']}")
        run_payload = {
            "run_id": run_id,
            "start_time_utc": run_start_utc,
            "config": {
                "camera_frame": self.camera_frame,
                "standoff_m": self.standoff_m,
                "move_vel": self.move_vel,
                "move_acc": self.move_acc,
                "move_timeout_s": self.move_timeout,
                "trial_num": self.trial_num,
                "data_root": str(self.data_root),
            },
            "stages": {},
            "artifacts": {
                "far_rgb": str(output_paths["far_rgb"]),
                "far_depth": str(output_paths["far_depth"]),
                "close_rgb": str(output_paths["close_rgb"]),
                "close_depth": str(output_paths["close_depth"]),
                "ann_stage1": str(output_paths["ann_stage1"]),
                "ann_stage2": str(output_paths["ann_stage2"]),
                "ann_stage3": str(output_paths["ann_stage3"]),
                "ann_stage5": str(output_paths["ann_stage5"]),
            },
        }

        self.log(f"[run {run_id}] Starting pipeline run")

        def _finalize(status: str, message: str, **extra):
            run_payload["status"] = status
            run_payload["message"] = message
            run_payload["end_time_utc"] = datetime.utcnow().strftime("%Y-%m-%d_%H-%M-%S UTC")
            run_payload["duration_s"] = round(time.time() - run_start, 4)
            run_payload.update(extra)
            json_path = self._write_run_json(output_paths["parent_dir"], run_id, run_payload)
            self.log(f"[run {run_id}] Completed with status={status}. JSON: {json_path}")
            return PipelineRunResult(
                status=status,
                message=message,
                timestamp=run_start,
                run_id=run_id,
                cluster_bbox=extra.get("cluster_bbox"),
                cut_point=extra.get("cut_point"),
                coarse_xyz_cam=extra.get("coarse_xyz_cam"),
                cut_xyz_cam=extra.get("cut_xyz_cam"),
                run_json_path=str(json_path),
                output_dir=str(output_paths["parent_dir"]),
            )

        # FAR CAPTURE
        stage_t0 = time.time()
        self.log(f"[run {run_id}] Stage FAR capture")
        rgb_far, depth_far = self.camera.get_frames()
        self._save_image(output_paths["far_rgb"], rgb_far)
        self._save_image(output_paths["far_depth"], depth_far)
        run_payload["stages"]["far_capture"] = {
            "duration_s": round(time.time() - stage_t0, 4),
            "rgb_path": str(output_paths["far_rgb"]),
            "depth_path": str(output_paths["far_depth"]),
        }

        # FAR DETECTION + PAIRING + RIPENESS
        stage_t0 = time.time()
        detections = self.detector.run(rgb_far)
        proceed, detections = self._confirm_detections(rgb_far, detections, "Stage 1")
        if not proceed:
            return _finalize(
                PipelineRunStatus.NO_RIPE_TARGET,
                "User rejected detections.",
            )
        self._save_image(output_paths["ann_stage1"], self._annotate_stage1(rgb_far, detections))
        self.log(f"[run {run_id}] Stage 1 detections: {len(detections)}")

        pairs = self.pairer.run(detections, rgb_far.shape[:2])
        while self.interactive and not pairs:
            self.log("[interactive] Stage 2 could not form any pairs. Returning to Stage 1 confirm/edit.")
            proceed, detections = self._confirm_detections(rgb_far, detections, "Stage 1")
            if not proceed:
                return _finalize(
                    PipelineRunStatus.NO_RIPE_TARGET,
                    "User rejected detections.",
                )
            pairs = self.pairer.run(detections, rgb_far.shape[:2])

        if not self._confirm_pairs(rgb_far, pairs, "Stage 2"):
            return _finalize(
                PipelineRunStatus.NO_RIPE_TARGET,
                "User rejected pairs.",
            )
        self._save_image(output_paths["ann_stage2"], self._annotate_stage2(rgb_far, pairs))
        self.log(f"[run {run_id}] Stage 2 pairs: {len(pairs)}")

        results = self.ripeness.run(rgb_far, pairs)
        ripe_indices = [i for i, result in enumerate(results) if result.ripeness == "ripe"]
        chosen_idx = ripe_indices[0] if ripe_indices else None
        self._save_image(output_paths["ann_stage3"], self._annotate_stage3(rgb_far, results, chosen_idx))
        self.log(
            f"[run {run_id}] Stage 3 ripeness results: total={len(results)}, ripe={len(ripe_indices)}"
        )

        run_payload["stages"]["far_perception"] = {
            "duration_s": round(time.time() - stage_t0, 4),
            "num_detections": len(detections),
            "num_pairs": len(pairs),
            "num_ripeness_results": len(results),
            "num_ripe": len(ripe_indices),
        }

        if not ripe_indices:
            return _finalize(
                PipelineRunStatus.NO_RIPE_TARGET,
                "No ripe clusters detected.",
            )

        chosen_target = results[chosen_idx]
        if chosen_target.cluster_bbox is None:
            return _finalize(
                PipelineRunStatus.NO_RIPE_TARGET,
                "Chosen ripe target is missing cluster bbox.",
            )

        # FAR COARSE XYZ
        x1, y1, x2, y2 = chosen_target.cluster_bbox
        u_far = 0.5 * (x1 + x2)
        v_far = 0.5 * (y1 + y2)
        z_far = depth_at(depth_far, u_far, v_far, scale=self.camera.depth_scale)
        if z_far is None:
            return _finalize(
                PipelineRunStatus.INVALID_FAR_DEPTH,
                "Invalid depth at FAR bbox center.",
                cluster_bbox=chosen_target.cluster_bbox,
            )

        coarse_xyz_cam = uvz_to_xyz(u_far, v_far, z_far, self.fx, self.fy, self.cx, self.cy)
        run_payload["stages"]["far_target"] = {
            "bbox_center_uv": [u_far, v_far],
            "depth_m": z_far,
            "coarse_xyz_cam_m": coarse_xyz_cam,
        }
        self.log(f"[run {run_id}] FAR coarse XYZ_cam: {coarse_xyz_cam}")

        # ROBOT MOVE
        stage_t0 = time.time()
        approach_xyz_cam = (
            coarse_xyz_cam[0],
            coarse_xyz_cam[1],
            coarse_xyz_cam[2] - self.standoff_m,
        )
        self.log(f"[run {run_id}] Robot approach target XYZ_cam: {approach_xyz_cam}")
        ok = self.robot.move_xyz_cam_to_tool(
            xyz_cam=approach_xyz_cam,
            camera_frame=self.camera_frame,
            tool_frame=self.camera_frame,
            velocity=self.move_vel,
            acceleration=self.move_acc,
            timeout_sec=self.move_timeout,
        )
        run_payload["stages"]["robot_move"] = {
            "duration_s": round(time.time() - stage_t0, 4),
            "approach_xyz_cam_m": approach_xyz_cam,
            "success": bool(ok),
        }
        if not ok:
            return _finalize(
                PipelineRunStatus.MOTION_FAILED,
                "Robot motion failed.",
                cluster_bbox=chosen_target.cluster_bbox,
                coarse_xyz_cam=coarse_xyz_cam,
            )

        # CLOSE CAPTURE
        stage_t0 = time.time()
        self.log(f"[run {run_id}] Stage CLOSE capture")
        rgb_close, depth_close = self.camera.get_frames()
        self._save_image(output_paths["close_rgb"], rgb_close)
        self._save_image(output_paths["close_depth"], depth_close)
        run_payload["stages"]["close_capture"] = {
            "duration_s": round(time.time() - stage_t0, 4),
            "rgb_path": str(output_paths["close_rgb"]),
            "depth_path": str(output_paths["close_depth"]),
        }

        # CLOSE PERCEPTION
        stage_t0 = time.time()
        close_out = self.close_stage.run(rgb_close)
        self._save_image(output_paths["ann_stage5"], self._annotate_stage5(rgb_close, close_out))
        run_payload["stages"]["close_perception"] = {
            "duration_s": round(time.time() - stage_t0, 4),
            "detection_success": bool(close_out.detection_success),
            "cluster_bbox": close_out.cluster_bbox,
            "pedicel_bbox": close_out.pedicel_bbox,
            "cut_point": close_out.cut_point,
            "keypoints": close_out.keypoints,
            "debug_info": close_out.debug_info,
        }

        if not close_out.detection_success or close_out.cut_point is None:
            return _finalize(
                PipelineRunStatus.CLOSE_DETECTION_FAILED,
                "Close perception failed to return a valid cutpoint.",
                cluster_bbox=chosen_target.cluster_bbox,
                coarse_xyz_cam=coarse_xyz_cam,
            )

        # CUTPOINT XYZ
        u_cut, v_cut = close_out.cut_point
        z_cut = depth_at(depth_close, u_cut, v_cut, scale=self.camera.depth_scale)
        if z_cut is None:
            return _finalize(
                PipelineRunStatus.INVALID_CUT_DEPTH,
                "Invalid depth at cutpoint.",
                cluster_bbox=chosen_target.cluster_bbox,
                cut_point=close_out.cut_point,
                coarse_xyz_cam=coarse_xyz_cam,
            )

        cut_xyz_cam = uvz_to_xyz(u_cut, v_cut, z_cut, self.fx, self.fy, self.cx, self.cy)
        run_payload["stages"]["cutpoint"] = {
            "cut_uv": [u_cut, v_cut],
            "cut_depth_m": z_cut,
            "cut_xyz_cam_m": cut_xyz_cam,
        }
        self.log(f"[run {run_id}] Cutpoint XYZ_cam: {cut_xyz_cam}")

        return _finalize(
            PipelineRunStatus.SUCCESS,
            "Pipeline run completed successfully.",
            cluster_bbox=chosen_target.cluster_bbox,
            cut_point=close_out.cut_point,
            coarse_xyz_cam=coarse_xyz_cam,
            cut_xyz_cam=cut_xyz_cam,
        )

    def close(self):
        if self._closed:
            return

        self._closed = True
        if self.camera is not None:
            try:
                self.camera.stop()
            except Exception:
                pass

        if self.robot is not None:
            try:
                self.robot.destroy_node()
            except Exception:
                pass


def run_pipeline(log=print, interactive=False):
    """
    Backward-compatible one-shot wrapper used by older call sites/tools.
    """
    runner = HarvestPipelineRunner(log=log, interactive=interactive)
    try:
        return runner.run_once()
    finally:
        runner.close()
