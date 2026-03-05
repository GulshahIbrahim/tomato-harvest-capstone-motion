import os
from dataclasses import dataclass

import numpy as np
import pyrealsense2 as rs


def get_realsense_serial() -> str:
    """Return the RealSense serial to use."""
    serial = os.environ.get("REALSENSE_SERIAL_ID_D405", "218622279985")
    # Compose/.env files sometimes include quotes; strip them defensively.
    serial = serial.strip().strip('"').strip("'")
    if not serial:
        raise ValueError("REALSENSE_SERIAL_ID_D405 is empty.")
    return serial


def _get_realsense_serial() -> str:
    """Backward-compatible alias for older imports."""
    return get_realsense_serial()


@dataclass(frozen=True)
class CameraIntrinsics:
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int


class Camera:
    """
    Runtime camera API used by the pipeline runner.

    Provides:
      - get_frames() -> (color_bgr_np, depth_np)
      - get_intrinsics() -> CameraIntrinsics
      - depth_scale (meters per raw depth unit)
      - stop()
    """

    def __init__(self, width=1280, height=720, fps=30, serial: str | None = None):
        self.serial = serial.strip() if serial is not None else get_realsense_serial()
        if not self.serial:
            raise ValueError("RealSense serial must be non-empty.")

        self.pipeline = rs.pipeline()
        cfg = rs.config()

        # Force the intended device when multiple RealSense cameras are connected.
        cfg.enable_device(self.serial)

        cfg.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
        cfg.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)

        self.profile = self.pipeline.start(cfg)
        self.align = rs.align(rs.stream.color)
        self.depth_scale = self.profile.get_device().first_depth_sensor().get_depth_scale()
        self._intrinsics: CameraIntrinsics | None = None

    @staticmethod
    def _to_intrinsics(color_frame) -> CameraIntrinsics:
        intr = color_frame.profile.as_video_stream_profile().intrinsics
        return CameraIntrinsics(
            fx=float(intr.fx),
            fy=float(intr.fy),
            cx=float(intr.ppx),
            cy=float(intr.ppy),
            width=int(intr.width),
            height=int(intr.height),
        )

    def get_frames(self):
        while True:
            frames = self.pipeline.wait_for_frames()
            aligned = self.align.process(frames)
            depth = aligned.get_depth_frame()
            color = aligned.get_color_frame()
            if not depth or not color:
                continue

            self._intrinsics = self._to_intrinsics(color)
            color_img = np.asanyarray(color.get_data())
            depth_img = np.asanyarray(depth.get_data())
            return color_img, depth_img

    def get_intrinsics(self) -> CameraIntrinsics:
        if self._intrinsics is None:
            self.get_frames()
        return self._intrinsics

    def capture_color_depth(self):
        """Capture one aligned color/depth pair and include stream intrinsics."""
        color_img, depth_img = self.get_frames()
        return color_img, depth_img, self.get_intrinsics()

    def stop(self):
        try:
            self.pipeline.stop()
        except Exception:
            pass

    def release(self):
        self.stop()
