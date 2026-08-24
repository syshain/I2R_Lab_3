"""Thin wrapper around the xArm SDK for the Lite 6.

Keeps every hardware call in one place so the phase logic in path_tracker.py
never touches the SDK directly. All angles are returned as (6,) numpy arrays
in RADIANS; all cartesian positions as (6,) vectors [x, y, z, roll, pitch, yaw]
with position in MM and rpy in DEGREES (the SDK's native units).
"""

import time

from cv2.cuda import Event_BLOCKING_SYNC
import numpy as np

from ik_solver import LITE6_JOINT_LIMITS_DEG


ROBOT_IP = '192.168.1.153'
SERVO_SPEED = 30          # % speed for each servo move
POSITION_MODE = 0         # position control mode (programmed motion)


def connect_arm(ip=ROBOT_IP):
    """Connect to the robot and bring it to a clean, movable state.

    Returns an XArmAPI instance. Raises RuntimeError if the connection or
    setup fails (caller should print and exit cleanly).
    """
    from xarm.wrapper import XArmAPI

    arm = XArmAPI(ip)
    try:
        arm.reset()
        arm.clean_error()
        arm.motion_enable(enable=True)
        arm.set_mode(POSITION_MODE)
        arm.set_state(0)
        time.sleep(2)
    except Exception as e:
        arm.disconnect()
        raise RuntimeError("Cannot connect to robot. Check IP and/or enable "
                           "robot in web app: %s" % e)
    return arm


def read_joints_rad(arm):
    """Current joint angles as a (6,) array in radians."""
    code, res = arm.get_servo_angle(is_radian=True)
    if code != 0:
        raise RuntimeError("Failed to read joint angles (code %d)" % code)
    return np.asarray(res[:6], dtype=float)


def read_pose_mm_deg(arm):
    """Current flange pose as a (6,) vector [x,y,z mm; roll,pitch,yaw deg]."""
    code, res = arm.get_position(is_radian=False)
    if code != 0:
        raise RuntimeError("Failed to read pose (code %d)" % code)
    return np.asarray(res[:6], dtype=float)


def within_joint_limits(q_deg):
    """True if a (6,) degree vector is inside the Lite 6 joint limits."""
    q_deg = np.asarray(q_deg, dtype=float)
    return bool(np.all((q_deg >= LITE6_JOINT_LIMITS_DEG[:, 0]) &
                       (q_deg <= LITE6_JOINT_LIMITS_DEG[:, 1])))


def move_to_joints(arm, q_target_rad, speed=SERVO_SPEED, wait=True):
    """Command the servos to q_target_rad (radians).

    wait=True blocks until the controller reports arrival (single moves);
    wait=False queues the move and returns immediately (smooth playback of a
    dense trajectory — the controller streams the queued points continuously).

    Returns (ok, message). Refuses to command out-of-limit solutions.
    """
    q_target_rad = np.asarray(q_target_rad, dtype=float)
    if not within_joint_limits(np.rad2deg(q_target_rad)):
        return False, 'IK solution outside joint limits'
    try:
        code = arm.set_servo_angle(angle=list(np.rad2deg(q_target_rad)),
                                   speed=speed, is_radian=False, wait=wait)
        if code == 0:
            return True, 'moved'
        return False, 'servo command rejected (code %d)' % code
    except Exception as e:  # network/timeout errors surface here
        return False, 'servo command error: %s' % e


def wait_for_idle(arm, timeout_s=120.0, poll_s=0.1):
    """Block until the controller reports no active movement."""
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            moving = arm.is_moving()
        except Exception:
            return False
        if not moving:
            return True
        time.sleep(poll_s)
    return False


HOME_JOINTS_DEG = [0, 0, 20, 0, 0, 0]

def go_home(arm):
    """Move the arm back to its home position (blocking). Returns (ok, msg)."""
    return move_to_joints(arm, np.deg2rad(HOME_JOINTS_DEG))


def open_gripper(arm):
    """
    Open the gripper if not opened
    """
    opened = 0
    try:
        code = arm.open_lite6_gripper()
        opened += 1
    except Exception as e:
        print("ERROR: Could not open gripper")
    return opened

def close_gripper(arm):
    """
    Close the gripper if not opened
    """
    closed = 0
    try:
        code = arm.close_lite6_gripper()
        closed += 1
    except Exception as e:
        print("ERROR: Could not close gripper")
    return closed
