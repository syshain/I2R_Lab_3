#!/usr/bin/env python3
"""Lab 3 — Recorded-trajectory replay with IK-based via planning.

Three phases:

  PHASE 1 — RECORD (manual): the operator drags the arm (teach mode) from a
      safe start pose to the final grasp pose over the block, checking that
      opening/closing the gripper would grab it WITHOUT closing yet. The wrist
      camera feed is shown the whole time and every joint configuration +
      flange pose reached during the drag is logged. Pressing [g] at the
      camera window confirms the final position and saves the recorded
      trajectory.

  PHASE 2 — PLAN: the recorded path is divided into VIA_COUNT equally spaced
      via points. A trajectory generator (trajectory_interp.py — linear /
      cubic / 4th-order / sinusoidal / quintic) fills in a dense stream of
      cartesian poses between the vias, and every pose is back-generated to
      joint angles with the Python IK solver (ik_solver.py — the Lab 2
      method), seeded from the previous waypoint so the arm stays on one
      elbow/wrist branch. Endpoints are the recorded configs verbatim, so the
      trajectory starts and ends exactly where the student taught.

  PHASE 3 — EXECUTE: the arm is returned to home, then driven through the
      generated waypoints with blocking servo moves. No vision decisions are
      made here — the camera simply keeps showing the live feed so the
      operator can watch the replay.

Keys (camera window):  [g] confirm/save   [q] quit

Usage:
    cd Lab_3/python_cube_nav/path_tracking
    python path_tracker.py
"""

import time
from pathlib import Path

import numpy as np
import cv2

_SCRIPT_DIR = Path(__file__).resolve().parent          # .../path_tracking
_NAV_DIR = _SCRIPT_DIR.parent                          # .../python_cube_nav
_CALIB_DIR = _NAV_DIR.parent / 'Calibration'           # .../Lab_3/Calibration
_DATA_DIR = _CALIB_DIR.parent / 'Data'                 # .../Lab_3/Data

CAMERA_INDEX = 4    # wrist USB camera device index
VIA_COUNT = 5       # number of via points the recorded path is split into
# NOTE on smoothness: 'linear' gives constant speed with no pause at the vias,
# which is what makes the Phase 3 replay look continuous. The other profiles
# (cubic/fourth/sinusoidal/quintic) ease to a full stop at every via — great for
# comparing shapes in plots, but visibly jerky when played back on the arm.
INTERPOLATOR = 'cubic'    # trajectory generator: linear/cubic/fourth/sinusoidal/quintic
SEG_SAMPLES = 10            # dense samples per segment (incl. shared endpoints)
RECORD_TRAJ_NAME = 'lab3_recorded'   # controller-side name (.traj file)

WINDOW = "Lab 3 : Trajectory Planning"


# ---------------------------------------------------------------------------
# Calibration loading
# ---------------------------------------------------------------------------
def load_camera_intrinsics():
    """Load camera matrix and distortion coefficients."""
    K = np.load(str(_CALIB_DIR / 'camera_matrix.npy'))
    dist = np.load(str(_CALIB_DIR / 'dist_coeffs.npy'))
    return K, dist


def load_calibration_results(use_calibrated_dh=False):
    """Load DH params (zeta) and T_ee_cam from the Lab 2 calibration results."""
    cal_file = str(_CALIB_DIR / 'calibration_results.npy')
    try:
        data = np.load(cal_file, allow_pickle=True).item()
        T_ee_cam = data['T_ee_cam_joint'] if 'T_ee_cam_joint' in data else data['T_ee_cam']
        if use_calibrated_dh and 'zeta_calibrated' in data:
            zeta = data['zeta_calibrated']
            print("Loaded CALIBRATED DH + hand-eye from:", cal_file)
        else:
            zeta = data['zeta_nominal'] if 'zeta_nominal' in data else get_nominal_zeta()
            print("Loaded NOMINAL DH (factory) + hand-eye from:", cal_file)
        return zeta, T_ee_cam
    except FileNotFoundError:
        print("WARNING: No calibration results found. Using nominal DH + identity T_ee_cam.")
        return get_nominal_zeta(), np.eye(4)


def get_nominal_zeta():
    """Return nominal DH parameter vector (metres, radians)."""
    dh_raw = np.array([
        [  0,   243.3,  -90,    0],
        [-90,     0.0,  180,  200],
        [-90,     0.0,   90,   87],
        [  0,   227.6,   90,    0],
        [  0,     0.0,  -90,    0],
        [  0,    61.5,    0,    0],
    ])
    a = dh_raw[:, 3] / 1000.0
    alpha = np.deg2rad(dh_raw[:, 2])
    d = dh_raw[:, 1] / 1000.0
    theta_off = np.deg2rad(dh_raw[:, 0])
    return np.concatenate([a, alpha, d, theta_off])


# ---------------------------------------------------------------------------
# Camera helpers
# ---------------------------------------------------------------------------
def open_camera(cam_index=CAMERA_INDEX):
    """Open the wrist camera; raises RuntimeError if it cannot be opened."""
    cap = cv2.VideoCapture(cam_index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera at index {cam_index}")
    for _ in range(5):  # warm-up frames
        cap.read()
    return cap


def size_window_quarter_screen(window_name):
    """Resize an OpenCV window to half screen width x half screen height."""
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    import tkinter as tk
    root = tk.Tk()
    root.withdraw()
    sw = root.winfo_screenwidth()
    sh = root.winfo_screenheight()
    root.destroy()
    cv2.resizeWindow(window_name, max(320, int(sw / 2)), max(240, int(sh / 2)))


def draw_overlay(frame, status_line, info_lines=None):
    """Draw a status banner (+ optional cyan info lines) onto the frame."""
    info_lines = info_lines or []
    bar_h = 90 + len(info_lines) * 28
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (frame.shape[1], bar_h), (40, 40, 40), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    lines = [status_line,
             "Keys:  [g] confirm & save   [q] quit"]
    for i, ln in enumerate(lines):
        col = (0, 200, 255) if i == 0 else (200, 200, 200)
        cv2.putText(frame, ln, (10, 28 + i * 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)

    for j, ln in enumerate(info_lines):
        cv2.putText(frame, ln, (10, 28 + 2 * 28 + j * 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    return frame


def show_frame(cap, status_line, info_lines=None):
    """Read one frame, draw the overlay, display it, return pressed key (or None)."""
    ret, frame = cap.read()
    if not ret:
        time.sleep(0.05)
        return None
    cv2.imshow(WINDOW, draw_overlay(frame, status_line, info_lines))
    return cv2.waitKey(1) & 0xFF


# ---------------------------------------------------------------------------
# Phase 1 — manual recording
# ---------------------------------------------------------------------------
def record_phase(arm, cap):
    """Phase 1: operator drags the arm to the grasp pose while we log everything.

    Returns (config_log, pose_log, traj_saved):
        config_log : (N, 6) array of joint angles (radians) sampled during the drag
        pose_log   : (N, 6) array of flange poses [x,y,z mm; rpy deg]
        traj_saved : True if the controller-side .traj was saved
    """
    import robot_io

    print("\n" + "=" * 62)
    print("PHASE 1 — MANUAL POSITIONING & RECORDING")
    print("=" * 62)
    print("Drag the arm (teach mode) until the end-effector sits at the final")
    print("grasp pose over the block. Check that OPENING/CLOSING the gripper")
    print("would grasp the block, but do NOT close it yet.")
    print("When ready, click the camera window and press [g].")
    print("Press [q] to cancel.\n")

    q_rad = robot_io.read_joints_rad(arm)
    print("Starting joints (deg):", np.round(np.rad2deg(q_rad), 2).tolist())

    arm.start_record_trajectory()

    config_log, pose_log = [], []
    confirmed, traj_saved = False, False
    last_status = "Dragging... move the arm to the grasp pose."

    while not confirmed:
        key = show_frame(cap, last_status,
                         ["Samples kept: %d" % len(config_log)])
        if key is None:
            continue

        # Sample the current configuration + pose (transient read failures keep
        # the last good value rather than aborting the recording).
        try:
            q_now = robot_io.read_joints_rad(arm)
            p_now = robot_io.read_pose_mm_deg(arm)
            config_log.append(q_now)
            pose_log.append(p_now)
            last_status = "Recording... samples: %d" % len(config_log)
        except Exception:
            pass

        if key == ord('g'):
            confirmed = True
            # Final sample at the exact confirmation instant.
            try:
                config_log.append(robot_io.read_joints_rad(arm))
                pose_log.append(robot_io.read_pose_mm_deg(arm))
            except Exception:
                pass
            # Stop + save the controller-side trajectory record.
            arm.stop_record_trajectory()
            try:
                code = arm.save_record_trajectory(RECORD_TRAJ_NAME)
                traj_saved = (code == 0)
                print("Recorded trajectory saved to controller: %s.traj (code=%s)"
                      % (RECORD_TRAJ_NAME, code))
            except Exception as e:
                print("Warning: could not save controller trajectory: %s" % e)

            try:
                p_togo = robot_io.read_pose_mm_deg(arm)
                p_togo[2] += 100  # lift the EE 100mm clear of the block before retracting
                x, y, z, roll, pitch, yaw = p_togo
                code = arm.set_position(x=x, y=y, z=z, roll=roll, pitch=pitch, yaw=yaw,
                                        is_radian=False, wait=True)
                if code != 0:
                    print("WARNING: lift-off move rejected by controller (code %d)" % code)
            except Exception as e:
                print("Error: could not lift the end-effector off the block:", e)

        elif key == ord('q'):
            print("Cancelled before confirmation.")
            break

    if not confirmed:
        arm.stop_record_trajectory()
        return None, None, False

    return np.asarray(config_log), np.asarray(pose_log), traj_saved


# ---------------------------------------------------------------------------
# Phase 2 — planning (via selection + IK)
# ---------------------------------------------------------------------------
def plan_via_indices(n_samples, via_count=VIA_COUNT):
    """Indices of the via points within the recorded sample array.

    Equally spaced along the recording, always including the LAST sample
    (the confirmed grasp pose). Returns a strictly increasing index list of
    length min(via_count, n_samples).
    """
    n = min(int(via_count), int(n_samples))
    idx = np.unique(np.linspace(0, n_samples - 1, n, dtype=int))
    idx[-1] = n_samples - 1
    return idx.tolist()


def plan_trajectory(zeta, config_log, pose_log):
    """Phase 2: turn the recorded path into an IK-solved waypoint list.

    Via points are chosen equally spaced along the recording. Each via's target
    pose is built from its recorded FLANGE POSE (position + orientation), and IK
    is solved for it, seeded from the previous via's solution so the arm stays
    on one solution branch. The final waypoint is the EXACT recorded grasp
    configuration (no re-solve), guaranteeing the replay ends where the student
    confirmed it.

    Parameters
    ----------
    zeta : packed DH vector
    config_log : (N, 6) joint angles in radians sampled during the drag
    pose_log   : (N, 6) flange poses [x,y,z mm; rpy deg] sampled alongside

    Returns (waypoints, via_idx):
        waypoints : (M, 6) array of joint targets in RADIANS
        via_idx   : indices into the logs that were used as vias
    """
    from ik_solver import solve_ik

    assert len(config_log) == len(pose_log), \
        "config_log and pose_log must have the same number of samples"
    n = len(config_log)
    via_idx = plan_via_indices(n)
    print("\n" + "=" * 62)
    print("PHASE 2 — TRAJECTORY PLANNING")
    print("=" * 62)
    print("Recording has %d samples; choosing %d via points (equally spaced)."
          % (n, len(via_idx)))

    from trajectory_interp import plan_full, list_interpolators

    if INTERPOLATOR not in list_interpolators():
        raise ValueError("INTERPOLATOR must be one of %s" % list_interpolators())

    print("Interpolator: %-12s   samples/segment: %d" % (INTERPOLATOR, SEG_SAMPLES))
    configs, residuals, traj_poses_m, via_poses_m = plan_full(
        zeta, config_log, pose_log, via_idx,
        n_per_seg=SEG_SAMPLES, method=INTERPOLATOR)

    max_res = float(residuals.max())
    print("\nGenerated %d waypoints through %d vias." % (len(configs), len(via_idx)))
    print("Max IK residual across waypoints: %.2e" % max_res)
    if max_res > 1e-6:
        print("NOTE: some waypoints have large IK residuals — review before Phase 3.")
    return configs, via_idx


def pose_to_transform(pose_mm_deg):
    """Build a 4x4 transform from an SDK pose row [x,y,z mm; roll,pitch,yaw deg]."""
    from ik_solver import pose_to_matrix
    pos_m = np.asarray(pose_mm_deg[:3], dtype=float) / 1000.0
    rpy_rad = np.deg2rad(np.asarray(pose_mm_deg[3:], dtype=float))
    return pose_to_matrix(pos_m, rpy_rad)


# ---------------------------------------------------------------------------
# Phase 3 — execution
# ---------------------------------------------------------------------------
def execute_phase(arm, cap, waypoints):
    """Phase 3: go home, then drive the arm via-to-via (blocking moves).

    The camera keeps displaying throughout; no vision decisions are made.
    Returns the number of waypoints executed.
    """
    import robot_io

    print("\n" + "=" * 62)
    print("PHASE 3 — MOVEMENT (replay through %d vias)" % len(waypoints))
    print("=" * 62)
    print("Returning to home first, then stepping through the planned vias.")
    print("Watch the camera feed — the arm replays the recorded approach.")
    print("[q] abort before the next via\n")

    robot_io.go_home(arm)
    print("At home.")

    executed = 0
    for k, q_wp in enumerate(waypoints):
        status = "Moving to waypoint %d/%d ..." % (k + 1, len(waypoints))
        key = show_frame(cap, status,
                         ["q(deg) = %s" % np.round(np.rad2deg(q_wp), 1).tolist()])
        if key == ord('q'):
            print("\nAborted by operator before waypoint %d." % (k + 1))
            return executed

        ok, msg = robot_io.move_to_joints(arm, q_wp)
        if not ok:
            print("STOPPED at waypoint %d/%d: %s" % (k + 1, len(waypoints), msg))
            return executed

        # Re-read the ACTUAL joints rather than trusting the commanded value.
        try:
            q_actual = robot_io.read_joints_rad(arm)
        except Exception:
            q_actual = np.asarray(q_wp)
        dq = float(np.max(np.abs(q_actual - np.asarray(q_wp))))
        landed = "landed ok" if np.degrees(dq) < 5.0 else "DESYNC %.1f deg" % np.degrees(dq)
        executed += 1
        print("  waypoint %d/%d done (%s)" % (k + 1, len(waypoints), landed))

        show_frame(cap, "Waypoint %d/%d reached." % (k + 1, len(waypoints)))
    return executed

# ---------------------------------------------------------------------------
# Phase 4 — gripper
# ---------------------------------------------------------------------------
def grip_work(arm, cap):
    """Phase 4: After phase 3, grip the block then move 10cm x y z depending on user input

    The camera keeps displaying throughout; no vision decisions are made.
    Returns the number of moves made.
    """
    import robot_io

    print("\n" + "=" * 62)
    print("PHASE 4 — GRIPPER WORK ")
    print("=" * 62)
    executed = 0
    try:
        gripper_open = robot_io.open_gripper(arm)
        if gripper_open != 1:
            raise Exception("Gripper could not be started")
            return executed

        time.sleep(1)

        gripper_closed = robot_io.close_gripper(arm)
        if gripper_closed != 1:
            raise Exception("The gripper could not be closed. Stopping program")
            return executed
    except Exception as e:
        print(f"Issue occurred. See exception {e}")
        return executed

    print("Block grasped. Move robot in x y z using the following keys")

    print("[w] and [s] to move in z, [a] and [d] to move in x, [z] and [c] to move in y\n")
    print("Orientation is kept as is. Loop is set to run for a maximum of 200 moves")
    print("Unless [q] is pressed to abort\n")
    print("When ready, press [o] to open gripper and release block")
    moves = 0

    status = "Moving grasped block..."
    while moves < 200:
        p_current = robot_io.read_pose_mm_deg(arm)

        key = show_frame(cap, status,
                         ["q(deg) = %s" % np.round(np.rad2deg(p_current), 1).tolist()])
        if key == ord('q'):
            print("Done. Resetting! (lift -> release -> home)")
            # Lift the EE clear of the block first so releasing won't snag it.
            try:
                p_togo = robot_io.read_pose_mm_deg(arm)
                p_togo[2] += 100  # lift the EE 100mm clear before retracting
                x, y, z, roll, pitch, yaw = p_togo
                code = arm.set_position(x=x, y=y, z=z, roll=roll, pitch=pitch, yaw=yaw,
                                        is_radian=False, wait=True)
                if code != 0:
                    print("WARNING: lift-off move rejected by controller (code %d)" % code)
            except Exception as e:
                print("Warning: lift-off failed (%s); trying to release anyway." % e)

            # Release the block.
            opened = robot_io.open_gripper(arm)
            if opened != 1:
                print("WARNING: could not open gripper to release the block.")
            else:
                print("Gripper released.")
            try:
                arm.stop_lite6_gripper()
            except Exception as e:
                print("Warning: could not stop gripper motor (%s)." % e)

            # Always return home, regardless of what happened above.
            ok, msg = robot_io.move_to_joints(arm, np.deg2rad([0, 0, 10, 0, 0, 0]))
            if ok:
                print("Returned to home.")
            else:
                print("WARNING: could not return home: %s" % msg)

            return executed
        if key == ord('w'):
            p_current[2] += 100
            x, y, z, roll, pitch, yaw = p_current
            try:
                code = arm.set_position(x,y,z,roll,pitch,yaw,is_radian=False,wait=True)
            except Exception:
                print("Servos rejected move command")
                return 0
            moves += 1

        if key == ord('s'):
            p_current[2] -= 100
            x, y, z, roll, pitch, yaw = p_current
            try:
                code = arm.set_position(x,y,z,roll,pitch,yaw,is_radian=False,wait=True)
            except Exception:
                print("Servos rejected move command")
                return 0
            moves += 1

        if key == ord('a'):
            p_current[0] -= 100
            x, y, z, roll, pitch, yaw = p_current
            try:
                code = arm.set_position(x,y,z,roll,pitch,yaw,is_radian=False,wait=True)
            except Exception:
                print("Servos rejected move command")
                return 0
            moves += 1

        if key == ord('d'):
            p_current[0] += 100
            x, y, z, roll, pitch, yaw = p_current
            try:
                code = arm.set_position(x,y,z,roll,pitch,yaw,is_radian=False,wait=True)
            except Exception:
                print("Servos rejected move command")
                return 0
            moves += 1

        if key == ord('z'):
            p_current[1] -= 100
            x, y, z, roll, pitch, yaw = p_current
            try:
                code = arm.set_position(x,y,z,roll,pitch,yaw,is_radian=False,wait=True)
            except Exception:
                print("Servos rejected move command")
                return 0
            moves += 1

        if key == ord('c'):
            p_current[1] += 100
            x, y, z, roll, pitch, yaw = p_current
            try:
                code = arm.set_position(x,y,z,roll,pitch,yaw,is_radian=False,wait=True)
            except Exception:
                print("Servos rejected move command")
                return 0
            moves += 1

        if key == ord('o'):
            try:
                opened = robot_io.open_gripper(arm)
                if opened != 1:
                    raise Exception("Could not release block")
            except Exception as e:
                print(f"Could not open gripper. See exception {e}")
                return 0

        if key == ord('b'):
            try:
                closed = robot_io.close_gripper(arm)
                if closed != 1:
                    raise Exception("Could not grip block")
            except Exception as e:
                print(f"Could not close gripper. See exception {e}")
                return 0

    executed += 1
    print("Executed a total of %d moves", moves)

    return executed

# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def live_run(K, dist_coeffs, zeta, T_ee_cam):
    """Run all three phases with the camera feed up the entire time."""
    import robot_io

    # --- Connect to the robot ----------------------------------------------
    print("\nConnecting to robot at %s ..." % robot_io.ROBOT_IP)
    try:
        arm = robot_io.connect_arm()
    except RuntimeError as e:
        print(e)
        return

    # --- Open the camera window (guarded) ------------------------------------
    try:
        cap = open_camera()
    except Exception as e:
        print("Could not open camera: %s" % e)
        print("Aborting — check the camera index and connection to the laptop.")
        arm.disconnect()
        return

    size_window_quarter_screen(WINDOW)

    try:
        # ================= PHASE 1: RECORD ===================================
        config_log, pose_log, traj_saved = record_phase(arm, cap)
        if config_log is None or len(config_log) < 2:
            print("No usable recording captured — nothing to plan.")
            return

        # ================= PHASE 2: PLAN =====================================
        waypoints, via_idx = plan_trajectory(zeta, config_log, pose_log)

        # Save the full recording + the generated trajectory for the report.
        _DATA_DIR.mkdir(parents=True, exist_ok=True)
        rec_path = str(_DATA_DIR / 'recorded_trajectory.npy')
        np.save(rec_path, {'config_log': config_log, 'pose_log': pose_log})
        traj_path = str(_DATA_DIR / ('generated_trajectory_%s.npy' % INTERPOLATOR))
        np.save(traj_path, waypoints)
        print("\nSaved recording    ->", rec_path)
        print("Saved trajectory   ->", traj_path)

        # ================= PHASE 3: MOVE =====================================
        executed = execute_phase(arm, cap, waypoints)

        print("\n" + "=" * 62)
        print("TRAJECTORY REPLAY COMPLETE")
        print("=" * 62)
        print("Waypoints executed: %d / %d" % (executed, len(waypoints)))
        if traj_saved:
            print("Controller trajectory '%s.traj' also stored on the robot." % RECORD_TRAJ_NAME)

    # ================= PHASE 4: GRIPPER ==================================
        moved = grip_work(arm, cap)
        print("\n" + "=" * 62)
        print("GRIP WORK COMPLETE.")

    finally:
        # Always clean up: release camera, go home, disconnect.
        cv2.destroyAllWindows()
        cap.release()
        arm.stop_lite6_gripper()
        try:
            robot_io.go_home(arm)
        except Exception as e:
            print("Warning: could not move to home on exit: %s" % e)
        arm.disconnect()


if __name__ == '__main__':
    import sys
    sys.stdout.reconfigure(line_buffering=True)   # print as we go, no buffered bursts
    K, dist = load_camera_intrinsics()
    zeta, T_ee_cam = load_calibration_results()

    print("Camera intrinsics loaded (fx=%.1f, fy=%.1f)" % (K[0, 0], K[1, 1]))
    print("T_ee_cam translation: [%.3f, %.3f, %.3f] m" % tuple(T_ee_cam[:3, 3]))

    live_run(K, dist, zeta, T_ee_cam)
