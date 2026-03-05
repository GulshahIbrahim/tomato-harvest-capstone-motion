import threading
import traceback

import rclpy
from rclpy.node import Node
from std_msgs.msg import Empty

from pipeline_app.non_ros2.pipeline_runner import (
    HarvestPipelineRunner,
    PipelineRunStatus,
)


class TomatoPipelineNode(Node):
    def __init__(self):
        super().__init__("tomato_pipeline_node")
        self.runner = HarvestPipelineRunner(log=self.get_logger().info)
        self._busy = False
        self._lock = threading.Lock()
        self._run_count = 0
        self._worker = None

        self.trigger_sub = self.create_subscription(
            Empty,
            "/tomato_pipeline/trigger",
            self._on_trigger,
            10,
        )

        self.get_logger().info(
            "Tomato pipeline node initialized. Waiting for trigger on /tomato_pipeline/trigger"
        )

    def _on_trigger(self, _msg: Empty):
        with self._lock:
            if self._busy:
                self.get_logger().warning("Trigger ignored: pipeline is already running.")
                return
            self._busy = True
            self._run_count += 1
            run_id = self._run_count

        self.get_logger().info(f"Accepted trigger #{run_id}. Starting pipeline run.")
        self._worker = threading.Thread(
            target=self._execute_run,
            args=(run_id,),
            daemon=True,
        )
        self._worker.start()

    def _execute_run(self, run_id: int):
        try:
            result = self.runner.run_once()
            if result.status == PipelineRunStatus.SUCCESS:
                self.get_logger().info(
                    f"Run #{run_id} finished with status={result.status}: {result.message}"
                )
            else:
                self.get_logger().warning(
                    f"Run #{run_id} finished with status={result.status}: {result.message}"
                )
        except Exception:
            self.get_logger().error(
                f"Run #{run_id} crashed with an unexpected exception:\n{traceback.format_exc()}"
            )
            if rclpy.ok():
                rclpy.shutdown()
        finally:
            with self._lock:
                self._busy = False

    def destroy_node(self):
        try:
            self.runner.close()
        finally:
            super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = None

    try:
        node = TomatoPipelineNode()
        rclpy.spin(node)
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
