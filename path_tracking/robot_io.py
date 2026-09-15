"""Thin wrapper around the xArm SDK for the Lite 6.

Keeps every hardware call in one place so the phase logic in pick_and_place.py
never touches the SDK directly. All angles are returned as (6,) numpy arrays
in RADIANS; all cartesian positions as (6,) vectors [x, y, z, roll, pitch, yaw]
with position in MM and rpy in DEGREES (the SDK's native units).
"""

import time

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
        # NOTE: no arm.reset() here — it drives the arm home on every connect,
        # which wipes out whatever pose you left it in between runs. We only
        # clear faults and enable motion so the arm stays exactly where it is.
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


HOME_JOINTS_DEG = [0, 0, 20, 0, 0, 0]

def go_home(arm):
    """Move the arm back to its home position (blocking). Returns (ok, msg)."""
    return move_to_joints(arm, np.deg2rad(HOME_JOINTS_DEG))


def open_bio_gripper(arm):
    """Open (release) the Lite6 vacuum gripper. Returns 1 on success, 0 otherwise.

    Uses arm.open_lite6_gripper(); the SDK returns a code where 0 == success, so we
    surface that rather than assuming success just because no exception was raised.
    """
    try:
        code = arm.open_lite6_gripper()
        if code != 0:
            print("WARN: open_lite6_gripper returned code %s" % code)
            return 0
        return 1
    except Exception as e:
        print("ERROR: Could not open bio gripper:", e)
        return 0


def close_bio_gripper(arm):
    """Close (engage suction) the Lite6 vacuum gripper. Returns 1 on success, 0 otherwise.

    Uses arm.close_lite6_gripper(); the SDK returns a code where 0 == success.
    """
    try:
        code = arm.close_lite6_gripper()
        print(code)
        if code != 0:
            print("WARN: close_lite6_gripper returned code %s" % code)
            return 0
        return 1
    except Exception as e:
        print("ERROR: Could not close bio gripper:", e)
        return 0


def stop_bio_gripper(arm):
    """Stop the bio/vacuum gripper pump/motor. Best-effort; returns True unless it raises."""
    try:
        arm.stop_lite6_gripper()
        return True
    except Exception as e:
        print("WARNING: could not stop bio gripper (%s)." % e)
        return False
