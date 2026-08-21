#!/usr/bin/env python3
"""Lab 3 — Monocular Vision-Guided Navigation: Main Entry Point.

Pipeline:
  1. Load camera intrinsics and calibrated DH + hand-eye parameters (from Lab 2).
  2. Open the wrist-camera window (wrapped in try/except so a missing camera
     degrades gracefully instead of crashing).
  3. PHASE 1 — MANUAL POSITIONING: the operator moves the arm by hand (teach /
     jog mode) until the dark block is visible in the camera view, then presses
     [g] to begin. No automatic scanning is performed.
  4. PHASE 2 — INTERACTIVE NAVIGATION: each 'w' press steps the end-effector a
     fixed 10 mm along the camera's optical axis toward the block. 'a' aborts,
     'r' resets to home, 'q' quits.
  5. Log the navigation trace and save it.

Inverse kinematics is solved on the robot controller via the xArm SDK
(get_inverse_kinematics), which returns a tuple (code, angles). Motion completion
is handled by set_servo_angle(wait=True); there is no wait_until_ready() API in
this SDK version.

Usage:
    cd Lab_3/python_cube_nav
    python run_lab3.py
"""

import numpy as np
import cv2
import time
from scipy.spatial.transform import Rotation
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_CALIB_DIR = _SCRIPT_DIR.parent.parent / 'Lab_2_revisited' / 'Calibration'

CAMERA_INDEX = 4   # wrist USB camera device index


# ---------------------------------------------------------------------------
# Loading calibration data
# ---------------------------------------------------------------------------
def load_camera_intrinsics():
    """Load camera matrix and distortion coefficients."""
    K = np.load(str(_CALIB_DIR / 'camera_matrix.npy'))
    dist = np.load(str(_CALIB_DIR / 'dist_coeffs.npy'))
    return K, dist


def load_calibration_results(use_calibrated_dh=False):
    """Load DH params and T_ee_cam from Lab 2 results.

    NOTE ON WHICH ZETA TO USE
    -------------------------
    The kinematic-calibrated DH set (zeta_calibrated) was found to TRACK THE
    CONTROLLER WORSE than the factory nominal set at live poses (~24 mm off vs
    ~3 mm) — a classic overfit to the limited calibration snapshots. So by default
    we use the NOMINAL Lite6 DH parameters, which match the controller's own FK.
    Set use_calibrated_dh=True only if you have re-done a good calibration.

    T_ee_cam (hand-eye) is unaffected by this choice and always comes from the
    joint-based result.
    """
    cal_file = str(_CALIB_DIR / 'calibration_results.npy')
    try:
        data = np.load(cal_file, allow_pickle=True).item()
        T_ee_cam = data['T_ee_cam_joint'] if 'T_ee_cam_joint' in data else data['T_ee_cam']
        if use_calibrated_dh and 'zeta_calibrated' in data:
            zeta = data['zeta_calibrated']
            print("Loaded CALIBRATED DH + hand-eye from:", cal_file)
        else:
            zeta = data['zeta_nominal'] if 'zeta_nominal' in data else _get_nominal_zeta()
            print("Loaded NOMINAL DH (factory) + hand-eye from:", cal_file)
            print("  (skipping zeta_calibrated — it tracks the controller worse)")
        return zeta, T_ee_cam
    except FileNotFoundError:
        print("WARNING: No calibration results found. Using nominal DH + identity T_ee_cam.")
        return _get_nominal_zeta(), np.eye(4)


def _get_nominal_zeta():
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
# Camera
# ---------------------------------------------------------------------------
def open_camera(cam_index=CAMERA_INDEX):
    """Open the wrist USB camera and return a capture object.

    Raises RuntimeError if the camera cannot be opened; callers should wrap this
    in try/except so a missing/disconnected camera does not crash the program.
    """
    cap = cv2.VideoCapture(cam_index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera at index {cam_index}")
    for _ in range(5):  # warm-up frames
        cap.read()
    return cap


def _size_window_quarter_screen(window_name):
    """Resize an OpenCV window to occupy a quarter of the screen.

    "Quarter" is interpreted as half the width and half the height (i.e. one of
    four equal quadrants), which keeps the aspect ratio sensible on typical
    displays. The window is created explicitly so the resize applies even before
    the first imshow call.
    """
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    # Query the primary display resolution via the GUI toolkit.
    import tkinter as tk
    root = tk.Tk()
    root.withdraw()
    sw = root.winfo_screenwidth()
    sh = root.winfo_screenheight()
    root.destroy()
    win_w = max(320, int(sw / 2))
    win_h = max(240, int(sh / 2))
    cv2.resizeWindow(window_name, win_w, win_h)


# ---------------------------------------------------------------------------
# Overlay drawing
# ---------------------------------------------------------------------------
def _fmt_xyz(p):
    """Format a 3-vector of metres as an mm string, e.g. '[+239.1, -0.8, +551.3]'."""
    m = np.asarray(p) * 1000.0
    return "[%+.1f, %+7.1f, %+7.1f]" % (m[0], m[1], m[2])


def _draw_overlay(frame, plan, status_line):
    """Draw the detected block + a status banner onto the display frame."""
    if plan.get('detected') and plan.get('bbox_px') is not None:
        x, y, w, h = plan['bbox_px']
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        tag = "%.0fmm" % plan['depth_mm']
        cv2.putText(frame, tag, (x, max(0, y - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # Base-frame readout: WHERE WE THINK THE BLOCK IS vs. WHERE THE FLANGE IS.
    # If the arm's actual motion disagrees with these numbers, the FK / hand-eye
    # chain feeding IK is off — this panel makes that mismatch visible live.
    p_block = plan.get('p_block_base')
    p_flange = plan.get('p_flange_base')
    info_lines = []
    if p_flange is not None:
        info_lines.append("Flange (base) : " + _fmt_xyz(p_flange))
    if p_block is not None:
        info_lines.append("Block centre : " + _fmt_xyz(p_block))
        # Distance from flange to block centre — the gap each [w] step closes.
        gap = plan.get('dist_to_block_mm')
        if gap is None:
            gap = float(np.linalg.norm(np.asarray(p_block) - np.asarray(p_flange))) * 1000.0
        info_lines.append("Gap to close : %.1f mm" % gap)

    # Banner (taller when we have base-frame data to show).
    n_info = len(info_lines)
    bar_h = 90 + n_info * 28
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (frame.shape[1], bar_h), (40, 40, 40), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    lines = [status_line,
             "Keys:  [w] step toward block   [a] abort   [r] reset to home   [q] quit"]
    for i, ln in enumerate(lines):
        col = (0, 200, 255) if i == 0 else (200, 200, 200)
        cv2.putText(frame, ln, (10, 28 + i * 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)

    # Base-frame lines go below the keys line, in cyan so they stand out.
    for j, ln in enumerate(info_lines):
        cv2.putText(frame, ln, (10, 28 + 2 * 28 + j * 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    return frame


def _draw_debug_view(mask, plan):
    """Render the thresholded detection mask as a labelled BGR image.

    The raw mask is binary (white where the detector thinks "block"). We tint it
    red so it reads clearly against black, and overlay the bounding box plus a
    caption explaining what is being shown. This is purely a debugging aid — it
    shows exactly which pixels passed the darkness threshold this frame.
    """
    if mask is None:
        # No frame processed yet; show a blank panel with a hint.
        blank = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(blank, "waiting for frame...", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
        return blank

    coloured = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    coloured[mask > 0] = (0, 0, 255)  # red where the block was segmented

    if plan.get('detected') and plan.get('bbox_px') is not None:
        x, y, w, h = plan['bbox_px']
        cv2.rectangle(coloured, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.putText(coloured, "Thresholded mask (red = dark-region candidates)",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)
    return coloured


# ---------------------------------------------------------------------------
# Live interactive run
# ---------------------------------------------------------------------------
def live_run(K, dist_coeffs, zeta_cal, T_ee_cam):
    """Live interactive run: manual positioning, then key-driven approach."""
    from xarm.wrapper import XArmAPI
    from navigator import (ROBOT_IP, plan_step, execute_move, _read_joints_rad,
                           _solve_ik_controller)

    # --- Connect to the robot ------------------------------------------------
    print("\nConnecting to robot at %s ..." % ROBOT_IP)
    arm = XArmAPI(ROBOT_IP)
    try:
        arm.reset()
        arm.clean_error()
        arm.motion_enable(enable=True)
        arm.set_mode(0)
        arm.set_state(0)
    except Exception as e:
        print("Cannot connect to robot. Check IP and/or enable robot in web app: %s" % e)
        arm.disconnect()
        return

    q_rad = _read_joints_rad(arm)
    print("Current joints (deg):", np.round(np.rad2deg(q_rad), 2).tolist())

    # --- Open the camera window (guarded) ------------------------------------
    try:
        cap = open_camera()
    except Exception as e:
        print("Could not open camera: %s" % e)
        print("Aborting — check the wrist camera connection.")
        arm.disconnect()
        return

    WINDOW = "Lab 3 : Cube Navigation"
    DEBUG_WINDOW = "Lab 3 : Detection Mask"
    _size_window_quarter_screen(WINDOW)
    _size_window_quarter_screen(DEBUG_WINDOW)
    nav_log = []
    step_count = 0

    try:
        # ================= PHASE 1: MANUAL POSITIONING =======================
        print("\n" + "=" * 62)
        print("PHASE 1 — MANUAL POSITIONING")
        print("=" * 62)
        print("Jog the arm (teach pendant / UFactory web UI) until the dark")
        print("block is clearly visible in the camera window below.")
        print("When ready, click the window and press [g] to start navigation.")
        print("Press [q] to cancel.\n")

        started = False
        _last_jt_refresh = 0.0
        while not started:
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.1)
                continue
            # Refresh joint state ~every 0.5 s so the displayed flange tracks the
            # arm WHILE the operator jogs it by hand. Without this, q_rad stays
            # frozen at the pre-positioning pose and the "Flange" readout is wrong
            # for wherever the arm has actually been moved (issue: flange off on
            # restart / after manual positioning).
            now = time.time()
            if now - _last_jt_refresh > 0.5:
                _last_jt_refresh = now
                try:
                    q_rad = _read_joints_rad(arm)
                except Exception:
                    pass  # keep last good value if a transient read fails
            # Show the live feed and a first-pass detection mask for positioning.
            probe = plan_step(frame, K, dist_coeffs, T_ee_cam, zeta_cal, q_rad)
            cv2.imshow(WINDOW, frame)
            cv2.imshow(DEBUG_WINDOW, _draw_debug_view(probe.get('mask'), probe))
            key = cv2.waitKey(1) & 0xFF
            if key == ord('g'):
                started = True
            elif key == ord('q'):
                print("Cancelled before start.")
                break

        if not started:
            return

        # ================= PHASE 2: INTERACTIVE NAVIGATION ===================
        print("\n" + "=" * 62)
        print("PHASE 2 — INTERACTIVE NAVIGATION TO DARK BLOCK")
        print("=" * 62)
        print("[w]=step forward  [a]=abort  [r]=reset home  [q]=quit\n")

        # Take a fresh joint reading at the start of navigation so the first step
        # is planned from the arm's TRUE current pose (not a stale value).
        try:
            q_rad = _read_joints_rad(arm)
            print("Starting navigation. Joints (deg):",
                  np.round(np.degrees(q_rad), 2).tolist())
        except Exception as e:
            print("Warning: could not refresh joints at nav start:", e)

        last_status = "Ready."
        _last_diag_print = 0.0
        while step_count < 200:
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.1)
                continue

            # Plan a step from the current view (does NOT move the robot).
            plan = plan_step(frame, K, dist_coeffs, T_ee_cam, zeta_cal, q_rad)

            # LIVE DIAGNOSTIC (throttled to ~1 Hz): read the controller's OWN
            # reported EE position and print it next to OUR FK flange. Both are
            # base-frame mm. If these disagree by more than a few mm, our FK
            # (zeta / joint reading) is off; if they agree but the arm still
            # misbehaves, the problem is downstream (target build / IK).
            now = time.time()
            if now - _last_diag_print > 1.0:
                _last_diag_print = now
                try:
                    code, ctrl_pos = arm.get_position(is_radian=False)
                    if code == 0:
                        p_fk = plan.get('p_flange_base')
                        if p_fk is not None:
                            d = float(np.linalg.norm(np.asarray(p_fk) * 1000.0 - np.array(ctrl_pos[:3])))
                            # Block pixel + approach direction: if these DON'T change
                            # as the arm moves, the "line sticks" because the detected
                            # block isn't shifting in the image (static/lagging camera
                            # or detection lock), NOT because the planner is frozen.
                            pix = plan.get('center_px')
                            dir_str = ""
                            if plan.get('p_block_base') is not None and p_fk is not None:
                                dv = np.asarray(plan['p_block_base']) - np.asarray(p_fk)
                                nv = np.linalg.norm(dv)
                                if nv > 1e-6:
                                    dir_str = " DIR=%s" % np.round(dv / nv, 2)
                            print("[diag] FK flange(mm)=%s  CTRL EE(mm)=%s  |Δ|=%.1f%s%s" % (
                                np.round(np.asarray(p_fk) * 1000.0, 1),
                                np.round(np.array(ctrl_pos[:3]), 1), d,
                                ("  blk_px=(%d,%d)" % (pix[0], pix[1])) if pix else "",
                                dir_str))
                except Exception as e:
                    print("[diag] could not read controller pos:", e)

            status = last_status or plan.get('reason', '')
            disp = _draw_overlay(frame, plan, status)
            cv2.imshow(WINDOW, disp)
            cv2.imshow(DEBUG_WINDOW, _draw_debug_view(plan.get('mask'), plan))
            key = cv2.waitKey(1) & 0xFF

            if key == ord('w'):
                # ---- take one 10 mm step toward the block -------------------
                if plan.get('T_target') is not None:
                    T_target = plan['T_target']
                    euler_deg = np.rad2deg(Rotation.from_matrix(
                        T_target[:3, :3]).as_euler('xyz'))
                    # Position in MM (SDK expects mm), RPY in DEGREES.
                    pose_mm_deg = np.concatenate([T_target[:3, 3] * 1000.0, euler_deg])

                    ok, q_target, msg = _solve_ik_controller(
                        arm, list(pose_mm_deg), q_current_rad=q_rad)
                    if ok:
                        mok, mmsg = execute_move(arm, q_target)
                        if mok:
                            # Re-read the ACTUAL joint angles from the robot rather
                            # than trusting the IK prediction. The arm's true joints
                            # can differ from q_target (settling, branch choice),
                            # and using the predicted value corrupts every subsequent
                            # FK / target computation -> the "flange jumps 200mm" bug.
                            try:
                                q_actual = _read_joints_rad(arm)
                            except Exception as e:
                                print("[warn] post-move joint re-read failed (%s); "
                                      "keeping IK prediction." % e)
                                q_actual = list(np.asarray(q_target))

                            # Guard: if the real joints are far from what we
                            # commanded, the move did not land where expected.
                            dq_cmd = np.max(np.abs(np.asarray(q_actual) - np.asarray(q_target)))
                            if np.degrees(dq_cmd) > 15.0:
                                print("[!! DESYNC !!] commanded joints vs actual differ by "
                                      "%.1f deg max. Commanded=%s Actual=%s" % (
                                          np.degrees(dq_cmd),
                                          np.round(np.degrees(q_target), 1),
                                          np.round(np.degrees(q_actual), 1)))
                                print("     Possible causes: move interrupted, manual jog in")
                                print("     web app, or servo limit hit. Trusting ACTUAL joints.")
                            q_rad = np.asarray(q_actual)
                            step_count += 1
                            entry = {'step': step_count,
                                     'depth_mm': plan['depth_mm'],
                                     'joints_deg': np.degrees(q_rad).tolist()}
                            nav_log.append(entry)
                            last_status = ("Step %d taken — block at %.0f mm (%s)" %
                                           (step_count, plan['depth_mm'], msg))
                            print(last_status)
                        else:
                            last_status = "Move failed: %s" % mmsg
                            print(last_status)
                    else:
                        last_status = msg
                        print(msg)
                else:
                    last_status = plan.get('reason', 'cannot step yet')
                    print(last_status)

            elif key == ord('a'):
                last_status = "ABORTED by operator."
                print("\nAborted by operator.")
                break

            elif key == ord('r'):
                print("\nResetting to home ...")
                q_rad = _read_joints_rad(arm)
                arm.move_gohome(wait=True)
                last_status = "Returned to home."
                print("At home. Joints (deg):", np.round(np.degrees(q_rad), 2).tolist())

            elif key == ord('q'):
                last_status = "Quit requested."
                print("\nQuitting.")
                break

    finally:
        # Always clean up: release camera, go home, disconnect.
        cv2.destroyAllWindows()
        cap.release()
        try:
            arm.move_gohome(wait=True)
        except Exception as e:
            print("Warning: could not move to home on exit: %s" % e)
        arm.disconnect()

    # --- Summary + save log --------------------------------------------------
    print("\n" + "=" * 62)
    print("NAVIGATION COMPLETE")
    print("=" * 62)
    print("Steps taken: %d" % step_count)
    if nav_log:
        last = nav_log[-1]
        print("Final depth to block: %.1f mm" % last['depth_mm'])

    log_path = str(_SCRIPT_DIR.parent / 'Data' / 'navigation_log.npy')
    np.save(log_path, nav_log)
    print("Navigation log saved to:", log_path)


if __name__ == '__main__':
    K, dist = load_camera_intrinsics()
    zeta_cal, T_ee_cam = load_calibration_results()

    print("Camera intrinsics loaded (fx=%.1f, fy=%.1f)" % (K[0, 0], K[1, 1]))
    print("T_ee_cam translation: [%.3f, %.3f, %.3f] m" % tuple(T_ee_cam[:3, 3]))

    live_run(K, dist, zeta_cal, T_ee_cam)
