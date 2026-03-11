import os
import time


class ToolClientROS2:
    """Minimal tool-control wrapper backed by robot RDO outputs."""

    def __init__(self, robot_client, log=print):
        self.robot = robot_client
        self.log = log
        self.gripper_rdo = int(os.environ.get("PIPELINE_GRIPPER_RDO", "3"))
        self.cut_rdo = int(os.environ.get("PIPELINE_CUT_RDO", "4"))
        self.cut_pulse_s = float(os.environ.get("PIPELINE_CUT_PULSE_S", "0.5"))
        self.tool_delay_s = float(os.environ.get("PIPELINE_TOOL_DELAY_S", "0.25"))
        self.gripper_closed_state = os.environ.get("PIPELINE_GRIPPER_CLOSED_STATE", "true").lower() in ("1", "true", "yes")
        self.cut_active_state = os.environ.get("PIPELINE_CUT_ACTIVE_STATE", "true").lower() in ("1", "true", "yes")

    def _set_output(self, rdo_num: int, state: bool, label: str) -> bool:
        ok = self.robot.set_rdo(rdo_num, state)
        if ok:
            self.log(f"[tool] {label}: RDO[{rdo_num}]={int(state)}")
            time.sleep(self.tool_delay_s)
        else:
            self.log(f"[tool] Failed {label}: RDO[{rdo_num}]={int(state)}")
        return ok

    def open_gripper(self) -> bool:
        return self._set_output(self.gripper_rdo, not self.gripper_closed_state, "open_gripper")

    def close_gripper(self) -> bool:
        return self._set_output(self.gripper_rdo, self.gripper_closed_state, "close_gripper")

    def trigger_cut(self) -> bool:
        if not self._set_output(self.cut_rdo, self.cut_active_state, "cut_on"):
            return False
        time.sleep(self.cut_pulse_s)
        return self._set_output(self.cut_rdo, not self.cut_active_state, "cut_off")

    def release_for_deposit(self) -> bool:
        return self.open_gripper()
