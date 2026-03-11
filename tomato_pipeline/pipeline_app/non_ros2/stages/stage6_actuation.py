from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


Vec3 = Tuple[float, float, float]


@dataclass(frozen=True)
class Stage6ActuationResult:
    success: bool
    status: str
    message: str
    keypoints_xyz_cam: List[Vec3]
    peduncle_axis_cam: Optional[Vec3]
    grasp_xyz_cam: Optional[Vec3]
    cut_xyz_cam: Optional[Vec3]
    orientation_xyzw_base: Optional[Tuple[float, float, float, float]]
    step_results: Dict[str, bool]
    debug_info: Dict[str, Any]


class Stage6Actuation:
    """Actuation stage for peduncle grasp-and-cut after close perception."""

    def __init__(self, robot, tool, log=print):
        self.robot = robot
        self.tool = tool
        self.log = log

        self.tool_frame = os.environ.get("PIPELINE_TOOL_FRAME", "tcp").strip() or "tcp"
        self.approach_offset_m = float(os.environ.get("PIPELINE_STAGE6_APPROACH_OFFSET_M", "0.05"))
        self.grasp_offset_m = float(os.environ.get("PIPELINE_STAGE6_GRASP_OFFSET_M", "0.01"))
        self.retreat_offset_m = float(os.environ.get("PIPELINE_STAGE6_RETREAT_OFFSET_M", "0.08"))
        self.cut_offset_m = float(os.environ.get("PIPELINE_STAGE6_CUT_OFFSET_M", "0.003"))
        self.move_vel = int(os.environ.get("PIPELINE_STAGE6_MOVE_VEL", "20"))
        self.move_acc = int(os.environ.get("PIPELINE_STAGE6_MOVE_ACC", "10"))
        self.move_timeout_s = float(os.environ.get("PIPELINE_STAGE6_MOVE_TIMEOUT_S", "30.0"))
        self.deposit_xyz_base = (
            float(os.environ.get("PIPELINE_DEPOSIT_X_M", "0.30")),
            float(os.environ.get("PIPELINE_DEPOSIT_Y_M", "-0.30")),
            float(os.environ.get("PIPELINE_DEPOSIT_Z_M", "0.35")),
        )
        self.deposit_q_base = (
            float(os.environ.get("PIPELINE_DEPOSIT_QX", "0.0")),
            float(os.environ.get("PIPELINE_DEPOSIT_QY", "0.0")),
            float(os.environ.get("PIPELINE_DEPOSIT_QZ", "0.0")),
            float(os.environ.get("PIPELINE_DEPOSIT_QW", "1.0")),
        )

    def run(
        self,
        depth_img,
        close_out,
        *,
        fx: float,
        fy: float,
        cx: float,
        cy: float,
        depth_scale: float,
        camera_frame: str,
    ) -> Stage6ActuationResult:
        step_results = {
            "open_gripper": False,
            "move_pregrasp": False,
            "move_grasp": False,
            "close_gripper": False,
            "move_cut": False,
            "trigger_cut": False,
            "move_retreat": False,
            "move_deposit": False,
            "release_deposit": False,
        }

        if close_out.cut_point is None or not close_out.keypoints:
            return Stage6ActuationResult(
                success=False,
                status="ACTUATION_INVALID_DEPTH",
                message="Stage 5 did not provide the required cut point/keypoints.",
                keypoints_xyz_cam=[],
                peduncle_axis_cam=None,
                grasp_xyz_cam=None,
                cut_xyz_cam=None,
                orientation_xyzw_base=None,
                step_results=step_results,
                debug_info={"failure_reason": "missing_keypoints"},
            )

        projected = self._project_keypoints(
            depth_img=depth_img,
            keypoints=close_out.keypoints,
            fx=fx,
            fy=fy,
            cx=cx,
            cy=cy,
            depth_scale=depth_scale,
        )
        cut_xyz_cam = projected[0]
        keypoints_xyz_cam = [point for point in projected if point is not None]
        if cut_xyz_cam is None:
            return Stage6ActuationResult(
                success=False,
                status="ACTUATION_INVALID_DEPTH",
                message="Keypoint1 depth is invalid; cannot compute cut target.",
                keypoints_xyz_cam=keypoints_xyz_cam,
                peduncle_axis_cam=None,
                grasp_xyz_cam=None,
                cut_xyz_cam=None,
                orientation_xyzw_base=None,
                step_results=step_results,
                debug_info={"failure_reason": "invalid_keypoint1_depth"},
            )
        if len(keypoints_xyz_cam) < 2:
            return Stage6ActuationResult(
                success=False,
                status="ACTUATION_INVALID_DEPTH",
                message="Not enough valid 3D keypoints to run Stage 6.",
                keypoints_xyz_cam=keypoints_xyz_cam,
                peduncle_axis_cam=None,
                grasp_xyz_cam=None,
                cut_xyz_cam=cut_xyz_cam,
                orientation_xyzw_base=None,
                step_results=step_results,
                debug_info={"failure_reason": "insufficient_3d_keypoints"},
            )

        peduncle_axis_cam = self._fit_axis(keypoints_xyz_cam)
        if peduncle_axis_cam is None:
            return Stage6ActuationResult(
                success=False,
                status="ACTUATION_INVALID_ORIENTATION",
                message="Failed to estimate peduncle orientation from keypoints.",
                keypoints_xyz_cam=keypoints_xyz_cam,
                peduncle_axis_cam=None,
                grasp_xyz_cam=None,
                cut_xyz_cam=cut_xyz_cam,
                orientation_xyzw_base=None,
                step_results=step_results,
                debug_info={"failure_reason": "axis_fit_failed"},
            )

        grasp_xyz_cam = self._choose_grasp_point(keypoints_xyz_cam)
        orientation_xyzw_base = self._compute_tool_orientation_base(
            peduncle_axis_cam=peduncle_axis_cam,
            camera_frame=camera_frame,
        )
        if orientation_xyzw_base is None:
            return Stage6ActuationResult(
                success=False,
                status="ACTUATION_INVALID_ORIENTATION",
                message="Could not compute a valid gripper orientation.",
                keypoints_xyz_cam=keypoints_xyz_cam,
                peduncle_axis_cam=peduncle_axis_cam,
                grasp_xyz_cam=grasp_xyz_cam,
                cut_xyz_cam=cut_xyz_cam,
                orientation_xyzw_base=None,
                step_results=step_results,
                debug_info={"failure_reason": "orientation_failed"},
            )

        pregrasp_xyz_cam = (
            grasp_xyz_cam[0],
            grasp_xyz_cam[1],
            grasp_xyz_cam[2] - self.approach_offset_m,
        )
        grasp_pose_xyz_cam = (
            grasp_xyz_cam[0],
            grasp_xyz_cam[1],
            grasp_xyz_cam[2] - self.grasp_offset_m,
        )
        cut_pose_xyz_cam = (
            cut_xyz_cam[0],
            cut_xyz_cam[1],
            cut_xyz_cam[2] - self.cut_offset_m,
        )
        retreat_xyz_cam = (
            cut_xyz_cam[0],
            cut_xyz_cam[1],
            cut_xyz_cam[2] - self.retreat_offset_m,
        )

        if not self.tool.open_gripper():
            return self._result(
                False,
                "GRASP_FAILED",
                "Failed to open gripper before grasp.",
                keypoints_xyz_cam,
                peduncle_axis_cam,
                grasp_xyz_cam,
                cut_xyz_cam,
                orientation_xyzw_base,
                step_results,
                {
                    "pregrasp_xyz_cam": pregrasp_xyz_cam,
                    "grasp_pose_xyz_cam": grasp_pose_xyz_cam,
                    "cut_pose_xyz_cam": cut_pose_xyz_cam,
                    "retreat_xyz_cam": retreat_xyz_cam,
                },
            )
        step_results["open_gripper"] = True

        if not self._move_cam_pose(pregrasp_xyz_cam, orientation_xyzw_base, camera_frame):
            return self._result(
                False,
                "GRASP_FAILED",
                "Failed to move to pre-grasp pose.",
                keypoints_xyz_cam,
                peduncle_axis_cam,
                grasp_xyz_cam,
                cut_xyz_cam,
                orientation_xyzw_base,
                step_results,
                {"pregrasp_xyz_cam": pregrasp_xyz_cam},
            )
        step_results["move_pregrasp"] = True

        if not self._move_cam_pose(grasp_pose_xyz_cam, orientation_xyzw_base, camera_frame):
            return self._result(
                False,
                "GRASP_FAILED",
                "Failed to move to grasp pose.",
                keypoints_xyz_cam,
                peduncle_axis_cam,
                grasp_xyz_cam,
                cut_xyz_cam,
                orientation_xyzw_base,
                step_results,
                {"grasp_pose_xyz_cam": grasp_pose_xyz_cam},
            )
        step_results["move_grasp"] = True

        if not self.tool.close_gripper():
            return self._result(
                False,
                "GRASP_FAILED",
                "Failed to close gripper on peduncle.",
                keypoints_xyz_cam,
                peduncle_axis_cam,
                grasp_xyz_cam,
                cut_xyz_cam,
                orientation_xyzw_base,
                step_results,
                {},
            )
        step_results["close_gripper"] = True

        if not self._move_cam_pose(cut_pose_xyz_cam, orientation_xyzw_base, camera_frame):
            return self._result(
                False,
                "CUT_FAILED",
                "Failed to move to cut pose.",
                keypoints_xyz_cam,
                peduncle_axis_cam,
                grasp_xyz_cam,
                cut_xyz_cam,
                orientation_xyzw_base,
                step_results,
                {"cut_pose_xyz_cam": cut_pose_xyz_cam},
            )
        step_results["move_cut"] = True

        if not self.tool.trigger_cut():
            return self._result(
                False,
                "CUT_FAILED",
                "Failed to trigger cutter.",
                keypoints_xyz_cam,
                peduncle_axis_cam,
                grasp_xyz_cam,
                cut_xyz_cam,
                orientation_xyzw_base,
                step_results,
                {},
            )
        step_results["trigger_cut"] = True

        if not self._move_cam_pose(retreat_xyz_cam, orientation_xyzw_base, camera_frame):
            return self._result(
                False,
                "DEPOSIT_FAILED",
                "Failed to retreat after cut.",
                keypoints_xyz_cam,
                peduncle_axis_cam,
                grasp_xyz_cam,
                cut_xyz_cam,
                orientation_xyzw_base,
                step_results,
                {"retreat_xyz_cam": retreat_xyz_cam},
            )
        step_results["move_retreat"] = True

        if not self.robot.move_pose(
            position=self.deposit_xyz_base,
            position_frame="base_link",
            orientation_xyzw=self.deposit_q_base,
            orientation_frame="base_link",
            target_frame=self.tool_frame,
            velocity=self.move_vel,
            acceleration=self.move_acc,
            timeout_sec=self.move_timeout_s,
        ):
            return self._result(
                False,
                "DEPOSIT_FAILED",
                "Failed to move to deposit pose.",
                keypoints_xyz_cam,
                peduncle_axis_cam,
                grasp_xyz_cam,
                cut_xyz_cam,
                orientation_xyzw_base,
                step_results,
                {"deposit_xyz_base": self.deposit_xyz_base},
            )
        step_results["move_deposit"] = True

        if not self.tool.release_for_deposit():
            return self._result(
                False,
                "DEPOSIT_FAILED",
                "Failed to release gripper at deposit pose.",
                keypoints_xyz_cam,
                peduncle_axis_cam,
                grasp_xyz_cam,
                cut_xyz_cam,
                orientation_xyzw_base,
                step_results,
                {},
            )
        step_results["release_deposit"] = True

        return self._result(
            True,
            "SUCCESS",
            "Stage 6 actuation completed successfully.",
            keypoints_xyz_cam,
            peduncle_axis_cam,
            grasp_xyz_cam,
            cut_xyz_cam,
            orientation_xyzw_base,
            step_results,
            {
                "pregrasp_xyz_cam": pregrasp_xyz_cam,
                "grasp_pose_xyz_cam": grasp_pose_xyz_cam,
                "cut_pose_xyz_cam": cut_pose_xyz_cam,
                "retreat_xyz_cam": retreat_xyz_cam,
                "deposit_xyz_base": self.deposit_xyz_base,
            },
        )

    def _move_cam_pose(self, xyz_cam, orientation_xyzw_base, camera_frame: str) -> bool:
        return self.robot.move_pose(
            position=xyz_cam,
            position_frame=camera_frame,
            orientation_xyzw=orientation_xyzw_base,
            orientation_frame="base_link",
            target_frame=self.tool_frame,
            velocity=self.move_vel,
            acceleration=self.move_acc,
            timeout_sec=self.move_timeout_s,
        )

    def _project_keypoints(self, depth_img, keypoints, fx, fy, cx, cy, depth_scale) -> List[Optional[Vec3]]:
        points = []
        for keypoint in keypoints:
            depth_m = self._depth_at(depth_img, keypoint[0], keypoint[1], scale=depth_scale)
            if depth_m is None:
                points.append(None)
                continue
            points.append(self._uvz_to_xyz(keypoint[0], keypoint[1], depth_m, fx, fy, cx, cy))
        return points

    @staticmethod
    def _depth_at(depth_img, u, v, win=7, invalid=0, scale=1.0):
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

    @staticmethod
    def _uvz_to_xyz(u, v, z, fx, fy, cx, cy) -> Vec3:
        x = (u - cx) * z / fx
        y = (v - cy) * z / fy
        return float(x), float(y), float(z)

    @staticmethod
    def _fit_axis(points: Sequence[Vec3]) -> Optional[Vec3]:
        if len(points) < 2:
            return None
        pts = np.asarray(points, dtype=np.float64)
        if pts.shape[0] == 2:
            axis = pts[1] - pts[0]
        else:
            centroid = pts.mean(axis=0)
            demeaned = pts - centroid
            _, _, vh = np.linalg.svd(demeaned, full_matrices=False)
            axis = vh[0]
            if np.dot(axis, pts[-1] - pts[0]) < 0:
                axis = -axis
        norm = np.linalg.norm(axis)
        if norm < 1e-8:
            return None
        axis = axis / norm
        return float(axis[0]), float(axis[1]), float(axis[2])

    @staticmethod
    def _choose_grasp_point(points: Sequence[Vec3]) -> Vec3:
        if len(points) >= 3:
            grasp = 0.5 * (np.asarray(points[1]) + np.asarray(points[2]))
        elif len(points) == 2:
            grasp = np.asarray(points[1])
        else:
            grasp = np.asarray(points[0])
        return float(grasp[0]), float(grasp[1]), float(grasp[2])

    def _compute_tool_orientation_base(
        self,
        *,
        peduncle_axis_cam: Vec3,
        camera_frame: str,
    ) -> Optional[Tuple[float, float, float, float]]:
        cam_quat = self.robot.get_current_frame_orientation(camera_frame, target_frame="base_link")
        if cam_quat is None:
            return None

        r_base_cam = self._quat_to_matrix(cam_quat)
        ped_axis_base = r_base_cam @ np.asarray(peduncle_axis_cam, dtype=np.float64)
        ped_axis_base = ped_axis_base / max(np.linalg.norm(ped_axis_base), 1e-8)

        current_tool_quat = self.robot.get_current_frame_orientation(self.tool_frame, target_frame="base_link")
        if current_tool_quat is None:
            return None
        r_base_tool_current = self._quat_to_matrix(current_tool_quat)
        approach_axis = r_base_tool_current[:, 2]
        jaw_axis = np.cross(approach_axis, ped_axis_base)
        if np.linalg.norm(jaw_axis) < 1e-6:
            fallback = np.array([1.0, 0.0, 0.0])
            if abs(np.dot(fallback, ped_axis_base)) > 0.9:
                fallback = np.array([0.0, 1.0, 0.0])
            jaw_axis = np.cross(fallback, ped_axis_base)
        jaw_axis = jaw_axis / max(np.linalg.norm(jaw_axis), 1e-8)
        approach_axis = np.cross(ped_axis_base, jaw_axis)
        approach_axis = approach_axis / max(np.linalg.norm(approach_axis), 1e-8)

        # Tool frame convention: x = jaw closing direction, y = peduncle axis, z = approach.
        r_base_tool_target = np.column_stack((jaw_axis, ped_axis_base, approach_axis))
        return self._matrix_to_quat(r_base_tool_target)

    @staticmethod
    def _quat_to_matrix(quat_xyzw) -> np.ndarray:
        x, y, z, w = [float(v) for v in quat_xyzw]
        xx, yy, zz = x * x, y * y, z * z
        xy, xz, yz = x * y, x * z, y * z
        wx, wy, wz = w * x, w * y, w * z
        return np.array(
            [
                [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
                [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
                [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
            ],
            dtype=np.float64,
        )

    @staticmethod
    def _matrix_to_quat(matrix: np.ndarray) -> Tuple[float, float, float, float]:
        m = matrix
        trace = float(m[0, 0] + m[1, 1] + m[2, 2])
        if trace > 0.0:
            s = math.sqrt(trace + 1.0) * 2.0
            w = 0.25 * s
            x = (m[2, 1] - m[1, 2]) / s
            y = (m[0, 2] - m[2, 0]) / s
            z = (m[1, 0] - m[0, 1]) / s
        elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
            s = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
            w = (m[2, 1] - m[1, 2]) / s
            x = 0.25 * s
            y = (m[0, 1] + m[1, 0]) / s
            z = (m[0, 2] + m[2, 0]) / s
        elif m[1, 1] > m[2, 2]:
            s = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
            w = (m[0, 2] - m[2, 0]) / s
            x = (m[0, 1] + m[1, 0]) / s
            y = 0.25 * s
            z = (m[1, 2] + m[2, 1]) / s
        else:
            s = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
            w = (m[1, 0] - m[0, 1]) / s
            x = (m[0, 2] + m[2, 0]) / s
            y = (m[1, 2] + m[2, 1]) / s
            z = 0.25 * s
        quat = np.array([x, y, z, w], dtype=np.float64)
        quat /= max(np.linalg.norm(quat), 1e-8)
        return float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3])

    @staticmethod
    def _result(
        success,
        status,
        message,
        keypoints_xyz_cam,
        peduncle_axis_cam,
        grasp_xyz_cam,
        cut_xyz_cam,
        orientation_xyzw_base,
        step_results,
        debug_info,
    ) -> Stage6ActuationResult:
        return Stage6ActuationResult(
            success=bool(success),
            status=status,
            message=message,
            keypoints_xyz_cam=list(keypoints_xyz_cam),
            peduncle_axis_cam=peduncle_axis_cam,
            grasp_xyz_cam=grasp_xyz_cam,
            cut_xyz_cam=cut_xyz_cam,
            orientation_xyzw_base=orientation_xyzw_base,
            step_results=dict(step_results),
            debug_info=dict(debug_info),
        )
