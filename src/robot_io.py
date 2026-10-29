"""Thin wrapper around the xArm SDK for the Lite 6."""

import os
import time

import numpy as np
from xarm.wrapper import XArmAPI

from ik_solver import LITE6_JOINT_LIMITS_DEG
from lab_config import ROBOT_IP

SERVO_SPEED = 30
POSITION_MODE = 0


def connect_arm(ip=ROBOT_IP):
    """Connect + enable motion; returns an XArmAPI. No reset() so we don't wipe."""

    arm = XArmAPI(ip, do_not_open=True)
    arm.connect()
    if not arm.connected:
        raise RuntimeError('could not connect to %s' % ip)
    time.sleep(0.5)
    if arm.get_is_moving() or arm.get_state() == 1:
        print('  warning: arm is in motion at connect - holding')
        arm.emergency_stop(30)
        time.sleep(1.0)
    return arm


def read_joints_rad(arm):
    """Current joint angles as a (6,) array in radians."""
    code, angles_deg = arm.get_servo_angle()
    if code != 0:
        raise RuntimeError('joint read failed (code %d)' % code)
    return np.deg2rad(np.asarray(angles_deg[:6], dtype=float))


def within_joint_limits(q_deg):
    """True if a (6,) degree vector is inside the Lite 6 joint limits."""
    q_deg = np.asarray(q_deg, dtype=float)
    return bool(np.all(q_deg >= LITE6_JOINT_LIMITS_DEG[:, 0])
                and np.all(q_deg <= LITE6_JOINT_LIMITS_DEG[:, 1]))


def move_to_joints(arm, q_target_rad, speed=SERVO_SPEED, wait=True):
    """Drive servos to q_target_rad (radians). wait=True blocks on arrival;."""

    q_deg = np.rad2deg(np.asarray(q_target_rad, dtype=float))
    if not within_joint_limits(q_deg):
        return False, 'target outside joint limits: %s' % np.round(q_deg, 1)
    code = arm.set_servo_angle(angle=q_deg.tolist(), speed=speed,
                               is_radian=False, wait=wait)
    if code != 0:
        return False, 'servo move failed (code %d)' % code
    return True, 'ok'


HOME_JOINTS_DEG = [0, 0, 20, 0, 0, 0]


def go_home(arm):
    """Move the arm back to its home position (blocking). Returns (ok, msg)."""
    return move_to_joints(arm, np.deg2rad(HOME_JOINTS_DEG))


def open_gripper(arm):
    """Engage suction. Returns 1 on success, 0 otherwise."""
    try:
        return 1 if arm.open_lite6_gripper() == 0 else 0
    except Exception as e:
        print('  ! open_gripper: %s' % e)
        return 0


def close_gripper(arm):
    """Release suction. Returns 1 on success, 0 otherwise."""
    try:
        return 1 if arm.close_lite6_gripper() == 0 else 0
    except Exception as e:
        print('  ! close_gripper: %s' % e)
        return 0


def stop_gripper(arm):
    """Best-effort stop of the vacuum pump (disable it). Returns True on success."""
    try:
        return arm.stop_lite6_gripper()
    except Exception as e:
        print('  ! stop_gripper: %s' % e)
        return False
