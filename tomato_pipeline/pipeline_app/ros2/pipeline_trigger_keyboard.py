import rclpy
from rclpy.node import Node
from std_msgs.msg import Empty


class PipelineTriggerKeyboard(Node):
    def __init__(self):
        super().__init__("pipeline_trigger_keyboard")
        self.publisher = self.create_publisher(Empty, "/tomato_pipeline/trigger", 10)
        self.trigger_count = 0

    def publish_trigger(self):
        self.publisher.publish(Empty())
        self.trigger_count += 1
        self.get_logger().info(f"Published trigger #{self.trigger_count}")


def main(args=None):
    rclpy.init(args=args)
    node = PipelineTriggerKeyboard()
    print("Keyboard trigger ready. Press Enter to run pipeline; type 'q' then Enter to quit.")

    try:
        while rclpy.ok():
            user_input = input()
            if user_input.strip().lower() == "q":
                break
            node.publish_trigger()
            # Process publish-related callbacks immediately in this interactive loop.
            rclpy.spin_once(node, timeout_sec=0.0)
    except (EOFError, KeyboardInterrupt):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
