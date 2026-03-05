from pipeline_app.non_ros2.vision.camera import Camera


def main():
    camera = Camera(width=1280, height=720, fps=30)
    try:
        # Pull at least one aligned frame so intrinsics are read from the active stream.
        camera.get_frames()
        intr = camera.get_intrinsics()
        print(f"RealSense serial: {camera.serial}")
        print("fx, fy, cx, cy =", intr.fx, intr.fy, intr.cx, intr.cy)
    finally:
        camera.stop()


if __name__ == "__main__":
    main()
