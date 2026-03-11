import time

import rclpy
from geometry_msgs.msg import PointStamped, QuaternionStamped
from rclpy.action import ActionClient
from rclpy.node import Node
from robot_interfaces.action import RobotJoints
from robot_interfaces.srv import ComplexIK, GetTransform, SetRdo


def create_position_message(position, frame_id):
    msg = PointStamped()
    msg.header.frame_id = frame_id
    x, y, z = position
    msg.point.x = float(x)
    msg.point.y = float(y)
    msg.point.z = float(z)
    return msg


def create_quaternion_message(orientation, frame_id):
    msg = QuaternionStamped()
    msg.header.frame_id = frame_id
    x, y, z, w = orientation
    msg.quaternion.x = float(x)
    msg.quaternion.y = float(y)
    msg.quaternion.z = float(z)
    msg.quaternion.w = float(w)
    return msg


class RobotClientROS2(Node):
    def __init__(self):
        super().__init__("pipeline_robot_client")

        # IK service
        self.complex_ik_client = self.create_client(ComplexIK, "complex_ik")
        while not self.complex_ik_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info("Waiting for complex_ik service...")

        # Transform service
        self.tf_client = self.create_client(GetTransform, "get_transform")
        while not self.tf_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info("Waiting for get_transform service...")

        self.set_rdo_client = self.create_client(SetRdo, "/set_rdo")
        while not self.set_rdo_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info("Waiting for set_rdo service...")

        # Motion action
        self.joint_client = ActionClient(self, RobotJoints, "joint_controller")
        self.get_logger().info("Waiting for joint_controller action server...")
        self.joint_client.wait_for_server()
        self.get_logger().info("Connected to joint_controller.")

    def get_current_frame_transform(self, current_frame: str, target_frame: str = "base_link"):
        req = GetTransform.Request()
        req.current_frame = current_frame
        req.target_frame = target_frame

        future = self.tf_client.call_async(req)
        while rclpy.ok() and not future.done():
            rclpy.spin_once(self, timeout_sec=0.05)

        if future.exception() is not None:
            self.get_logger().error(
                f"Transform request {current_frame} -> {target_frame} raised: {future.exception()}"
            )
            return None

        resp = future.result()
        if resp is None or not resp.success:
            self.get_logger().error(f"Failed to get transform {current_frame} -> {target_frame}")
            return None

        return resp.transform

    def get_current_frame_orientation(self, current_frame: str, target_frame: str = "base_link"):
        """Query frame orientation in target_frame coordinates."""
        transform = self.get_current_frame_transform(current_frame, target_frame)
        if transform is None:
            return None

        quat = transform.rotation
        return [quat.x, quat.y, quat.z, quat.w]

    def move_pose(
        self,
        position,
        position_frame,
        orientation_xyzw,
        orientation_frame="base_link",
        target_frame="tcp",
        velocity=40,
        acceleration=10,
        cnt_val=100,
        timeout_sec=30.0,
    ):
        req = ComplexIK.Request()
        req.target_frame = target_frame
        req.position = create_position_message(list(position), position_frame)
        req.orientation = create_quaternion_message(list(orientation_xyzw), orientation_frame)

        ik_future = self.complex_ik_client.call_async(req)

        start = time.time()
        while rclpy.ok() and not ik_future.done():
            if time.time() - start > timeout_sec:
                self.get_logger().error("IK timed out")
                return False
            rclpy.spin_once(self, timeout_sec=0.05)

        if ik_future.exception() is not None:
            self.get_logger().error(f"Complex IK request raised: {ik_future.exception()}")
            return False

        ik_resp = ik_future.result()
        if ik_resp is None or not ik_resp.success:
            self.get_logger().error("Complex IK failed")
            return False

        goal = RobotJoints.Goal()
        goal.joint_state = ik_resp.joint_state
        goal.velocity = velocity
        goal.acceleration = acceleration
        goal.cnt_val = cnt_val

        send_future = self.joint_client.send_goal_async(goal)
        while rclpy.ok() and not send_future.done():
            rclpy.spin_once(self, timeout_sec=0.05)

        if send_future.exception() is not None:
            self.get_logger().error(f"Sending motion goal raised: {send_future.exception()}")
            return False

        goal_handle = send_future.result()
        if goal_handle is None or not goal_handle.accepted:
            self.get_logger().error("Motion goal rejected")
            return False

        result_future = goal_handle.get_result_async()

        start = time.time()
        while rclpy.ok() and not result_future.done():
            if time.time() - start > timeout_sec:
                self.get_logger().error("Motion timed out")
                return False
            rclpy.spin_once(self, timeout_sec=0.05)

        if result_future.exception() is not None:
            self.get_logger().error(f"Motion result raised: {result_future.exception()}")
            return False

        result_msg = result_future.result()
        if result_msg is None:
            self.get_logger().error("Motion returned no result")
            return False

        result = result_msg.result
        self.get_logger().info("Motion finished")

        return bool(result.success)

    def move_xyz_cam_to_tool(
        self,
        xyz_cam,
        camera_frame,
        tool_frame="tcp",
        velocity=40,
        acceleration=10,
        cnt_val=100,
        timeout_sec=30.0,
    ):
        # Keep the target frame orientation fixed in base_link while moving in camera-relative XYZ.
        quat_xyzw = self.get_current_frame_orientation(tool_frame, target_frame="base_link")
        if quat_xyzw is None:
            return False
        self.get_logger().info(f"Using current {tool_frame} orientation in base_link: {quat_xyzw}")
        return self.move_pose(
            position=xyz_cam,
            position_frame=camera_frame,
            orientation_xyzw=quat_xyzw,
            orientation_frame="base_link",
            target_frame=tool_frame,
            velocity=velocity,
            acceleration=acceleration,
            cnt_val=cnt_val,
            timeout_sec=timeout_sec,
        )

    def set_rdo(self, num: int, data: bool, timeout_sec: float = 5.0) -> bool:
        req = SetRdo.Request()
        req.num = int(num)
        req.data = bool(data)
        future = self.set_rdo_client.call_async(req)

        start = time.time()
        while rclpy.ok() and not future.done():
            if time.time() - start > timeout_sec:
                self.get_logger().error(f"set_rdo timed out for RDO[{num}]")
                return False
            rclpy.spin_once(self, timeout_sec=0.05)

        if future.exception() is not None:
            self.get_logger().error(f"set_rdo raised: {future.exception()}")
            return False

        resp = future.result()
        return bool(resp is not None and resp.success)
