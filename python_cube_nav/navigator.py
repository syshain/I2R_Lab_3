"""Interactive Navigation Planner — step the end-effector toward the dark block.

Each "step" is: detect the dark block -> build a target flange pose that is one
fixed 10 mm closer along the camera's optical (Z) axis -> solve IK on the robot
controller for that pose -> command the servos.

The motion is intentionally simple: a pure 10 mm advance in the camera Z
direction, flange orientation held constant. Lateral centring and adaptive step
sizing are deliberately left out so the control loop stays easy to reason about.

Inverse kinematics uses the controller's own solver via the xArm SDK
(get_inverse_kinematics), which returns a tuple (code, angles).
"""

import numpy as np

from cube_detector import detect_block


ROBOT_IP = '192.168.1.153'
APPROACH_STEP_MM = 10      # fixed forward step size per keypress (toward block centre)
MAX_AIM_DEG = 10.0         # max wrist steering rotation allowed in a single step
SERVO_SPEED = 30           # % speed for each servo move


def _read_joints_rad(arm):
    """Read current joint angles in radians (list of 6)."""
    code, res = arm.get_servo_angle(is_radian=True)
    if code != 0:
        raise RuntimeError("Failed to read joint angles (code %d)" % code)
    return list(res[:6])


def _solve_ik_controller(arm, target_pose_mm_euler_deg, q_current_rad=None):
    """Solve IK on the robot controller.

    The SDK expects [x(mm), y(mm), z(mm), roll(deg), pitch(deg), yaw(deg)] when
    input_is_radian=False, and RETURNS joint angles in DEGREES when
    return_is_radian=False. We convert the returned degrees to radians here so the
    rest of the pipeline stays in radians.

    Prints the full target pose and the solved joint angles (plus per-joint delta
    vs. the current pose) so unit problems are visible in the terminal.
    """
    tp = np.asarray(target_pose_mm_euler_deg, dtype=float)
    print("\n--- IK TARGET (sent to controller) ---")
    print("  pos (mm) : [%.3f, %.3f, %.3f]" % (tp[0], tp[1], tp[2]))
    print("  rpy (deg): [%.3f, %.3f, %.3f]" % (tp[3], tp[4], tp[5]))

    code, angles = arm.get_inverse_kinematics(
        list(tp), input_is_radian=False, return_is_radian=False
    )
    if code == 0 and angles is not None and len(angles) >= 6:
        q_deg = list(angles[:6])
        q_rad = np.radians(q_deg)
        print("--- IK SOLUTION (returned by controller) ---")
        print("  joints (deg): [%s]" % ", ".join("%.3f" % a for a in q_deg))
        if q_current_rad is not None:
            dq_deg = np.degrees(np.asarray(q_rad) - np.asarray(q_current_rad))
            print("  Δjoints   : [%s]  (max |Δ| = %.3f deg)" %
                  (", ".join("%+.3f" % d for d in dq_deg),
                   float(np.max(np.abs(dq_deg)))))
        return True, q_rad, 'ok (controller IK)'
    print("  ** IK FAILED, code = %s **" % code)
    return False, None, 'controller IK failed (code %s)' % code


def _within_limits(q_deg):
    """Check joint angles (degrees) against UFACTORY Lite 6 limits."""
    limits_deg = [(-360, 360), (-150, 150), (-3.5, 300), (-360, 360), (-124, 124), (-360, 360)]
    for i, (lo, hi) in enumerate(limits_deg):
        if q_deg[i] < lo or q_deg[i] > hi:
            return False
    return True


def plan_step(frame, K, dist_coeffs, T_ee_cam, zeta_calibrated, q_rad):
    """Plan ONE approach step toward the dark block (hybrid visual servoing).

    Logic:
      1. SEE     — detect the dark block in the frame.
      2. CONFIRM — reject unless it passes the area/aspect/confidence filters
                   (handled inside detect_block).
      3. ESTIMATE— use the known 50 mm size + pinhole model for the camera-to-block
                   depth, and unproject the block's pixel centre to a 3D point in
                   the BASE frame.
      4. AIM     — rotate the camera (in its own frame) so its optical axis points
                   more toward the block, i.e. steer the block toward image centre.
                   Clamped to MAX_AIM_DEG per step so the wrist never snaps.
      5. ADVANCE — translate the (re-aimed) camera forward along its new optical
                   axis by APPROACH_STEP_MM.
      6. CONVERT — turn the desired CAMERA pose back into a FLANGE pose via the
                   hand-eye matrix; that flange pose is what IK commands.

    Doing aim+advance in the camera frame (then converting to flange) is what lets
    an off-axis target actually converge: pure flange translation alone cannot turn
    the camera, so it would asymptotically miss a block that isn't dead ahead.

    Returns a dict with keys:
        'detected'       : bool — was the block found?
        'depth_mm'       : float or None — estimated camera-to-block distance
        'bbox_px'        : (x, y, w, h) or None — block bounding box for display
        'center_px'      : (u, v) or None — detected block pixel centre (diagnostics)
        'mask'           : (H, W) uint8 binary image used for detection (debug view)
        'T_target'       : (4,4) target FLANGE pose in the base frame, or None
        'p_block_base'   : (3,) estimated block centre in the BASE frame (m), or None
        'p_flange_base'  : (3,) current flange position in the BASE frame (m)
        'dist_to_block_mm' : float or None — |flange - block| distance (m->mm)
        'aim_deg'        : float or None — wrist steering applied this step (deg)
        'reason'         : human-readable status string
    """
    from fk_functions import fk_from_zeta
    from scipy.spatial.transform import Rotation

    # 1. SEE the block.
    det = detect_block(frame, K, dist_coeffs)

    out = {'detected': False, 'depth_mm': None, 'bbox_px': None, 'center_px': None,
           'mask': det.get('mask'), 'T_target': None,
           'p_block_base': None, 'p_flange_base': None,
           'dist_to_block_mm': None, 'aim_deg': None, 'reason': ''}

    # 2. CONFIRM — detect_block only returns bbox/centre when the blob passed all
    #    the sanity filters (area, aspect ratio, confidence, min depth).
    if 'bbox_px' not in det:
        out['reason'] = 'block not confirmed in view — reposition the arm'
        return out

    out['detected'] = True
    out['bbox_px'] = det['bbox_px']
    out['center_px'] = det.get('center_px')   # (u, v) — for live diagnostics
    depth_m = det['depth_m']
    out['depth_mm'] = depth_m * 1000.0

    # Current flange + camera poses in the base frame.
    T_flange = fk_from_zeta(np.asarray(q_rad), zeta_calibrated, n=6)
    p_flange = T_flange[:3, 3]
    out['p_flange_base'] = p_flange.copy()
    T_cam = T_flange @ T_ee_cam

    # 3. ESTIMATE — unproject the block's pixel centre out to the measured depth,
    #    giving the block centre in the BASE frame (for readout + gap metric).
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    u, v = det['center_px']
    ray_cam = np.array([(u - cx) / fx, (v - cy) / fy, 1.0])
    ray_cam /= np.linalg.norm(ray_cam)
    p_block_base = T_cam[:3, :3] @ (depth_m * ray_cam) + T_cam[:3, 3]
    out['p_block_base'] = p_block_base

    to_block = p_block_base - p_flange
    dist_to_block = float(np.linalg.norm(to_block))
    out['dist_to_block_mm'] = dist_to_block * 1000.0

    # 4. AIM — small-angle steering so the optical axis tracks the block toward
    #    image centre. Signs verified empirically against the real hand-eye mount:
    #      block RIGHT of centre (u>cx) -> rotate about Y by +err_yaw
    #      block LOW   below centre(v>cy)-> rotate about X by -err_pitch
    err_yaw = (u - cx) / fx
    err_pitch = (v - cy) / fy
    lim = np.deg2rad(MAX_AIM_DEG)
    err_yaw = float(np.clip(err_yaw, -lim, lim))
    err_pitch = float(np.clip(err_pitch, -lim, lim))
    aim_applied = float(np.hypot(err_yaw, err_pitch)) * 180.0 / np.pi
    out['aim_deg'] = aim_applied

    # Rotate the camera about its own axes to steer the block toward centre.
    Rx = Rotation.from_euler('x', -err_pitch).as_matrix()
    Ry = Rotation.from_euler('y', +err_yaw).as_matrix()
    R_cam_new = T_cam[:3, :3] @ (Rx @ Ry)

    # 5. ADVANCE — step forward along the NEW optical axis, capped so we don't
    #    overshoot the block in one press.
    new_axis = R_cam_new[:, 2]
    step_m = min(APPROACH_STEP_MM / 1000.0, max(dist_to_block, 1e-6))
    p_cam_new = T_cam[:3, 3] + new_axis * step_m

    T_desired_cam = np.eye(4)
    T_desired_cam[:3, :3] = R_cam_new
    T_desired_cam[:3, 3] = p_cam_new

    # 6. CONVERT — desired camera pose -> desired flange pose (inverse hand-eye).
    #    T_cam = T_flange @ T_ee_cam  =>  T_flange = T_cam @ inv(T_ee_cam)
    T_target = T_desired_cam @ np.linalg.inv(T_ee_cam)
    out['T_target'] = T_target

    out['reason'] = ('%.0f mm fwd + %.1f deg aim (%.0f mm to block)' %
                     (step_m * 1000.0, aim_applied, dist_to_block * 1000.0))
    return out


def execute_move(arm, q_target_rad):
    """Command the servos to q_target_rad and block until the move completes.

    Returns (ok, message).
    """
    if not _within_limits(np.degrees(q_target_rad)):
        return False, 'IK solution outside joint limits'
    try:
        arm.set_servo_angle(angle=np.degrees(q_target_rad), speed=SERVO_SPEED,
                            is_radian=False, wait=True)
        return True, 'moved'
    except Exception as e:  # network/timeout errors surface here
        return False, 'servo command error: %s' % e
