"""Motion planning + robot-facing move primitives for Lab 3.

Two layers live here:

Pure geometry / planning (no hardware touched):
    flange_pose_base, make_downward_pose, plan_centre_step, plan_descent_step,
    cup_shift_flange, cup_axis_base, compute_topface_target,
    _project_cuptip_marks, _approach_schedule.

Robot-facing moves (solve IK then drive the servos via robot_io):
    move_to_pose, nudge_z, nudge_xy, orient_downward, shift_cup_over_block,
    descend_along_cup_axis.

All internal maths in METRES + RADIANS; the SDK speaks MM + DEGREES (handled
inside robot_io). The coordinate chain is identical to run_lab3.py /
navigator.py:
    T_flange = fk_from_zeta(q, zeta_nominal)   (base -> flange)
    T_cam    = T_flange @ T_ee_cam            (base -> camera)
The TOOL OFFSET shifts from the flange frame to the SUCTION CUP TIP (the actual
contact point), because the cup hangs TOOL_OFFSET_MM below the flange face.
"""

import numpy as np
import cv2
from scipy.spatial.transform import Rotation

from constants import (
    DOWN_RPY_DEG,
    CENTRE_TOL_PX,
    DESCENT_STEP_MM,
    MAX_AIM_DEG_DESCENT,
    TOOL_OFFSET_MM,
    ADVANCE_STEP_MAX_MM,
    ADVANCE_STEP_MIN_MM,
    CLOSE_ZONE_MM,
    FAR_SPEED_PCT,
    NEAR_SPEED_PCT,
)


# ===========================================================================
# Pose construction helpers
# ===========================================================================
def flange_pose_base(q_rad, zeta):
    """Base->flange transform from current joints (metres/radians)."""
    from fk_functions import fk_from_zeta
    return fk_from_zeta(np.asarray(q_rad), zeta, n=6)


def make_downward_pose(p_target_m, rpy_deg=(0.0, 0.0, 0.0)):
    """Build a 4x4 whose position is p_target_m (metres) with the given RPY (deg)."""
    from ik_solver import pose_to_matrix
    return pose_to_matrix(np.asarray(p_target_m, dtype=float),
                          np.deg2rad(np.asarray(rpy_deg, dtype=float)))


# ===========================================================================
# Safety taper
# ===========================================================================
def _approach_schedule(dist_mm):
    """Safety taper: as the flange approaches the block, BOTH the commanded step
    length and the servo speed shrink linearly. Far -> fast & big steps; inside
    CLOSE_ZONE_MM -> creeping & tiny steps, floored at ADVANCE_STEP_MIN_MM so we
    never micro-jitter below 1 mm. Returns (step_mm, speed_pct)."""
    if dist_mm <= 0.0:
        return ADVANCE_STEP_MIN_MM, NEAR_SPEED_PCT
    t = min(1.0, max(0.0, dist_mm / CLOSE_ZONE_MM))   # 1=far, 0=at zone edge
    step_mm = ADVANCE_STEP_MIN_MM + \
        (ADVANCE_STEP_MAX_MM - ADVANCE_STEP_MIN_MM) * t
    speed_pct = NEAR_SPEED_PCT + \
        (FAR_SPEED_PCT - NEAR_SPEED_PCT) * t
    return float(step_mm), int(round(speed_pct))


# ===========================================================================
# Aim / descent planning (ported from python_cube_nav/navigator.py::plan_step)
# ===========================================================================
def plan_centre_step(det, T_flange, T_ee_cam, K, dist_coeffs, max_aim_deg=15.0):
    """Plan the flange pose that RE-CENTRES the block in the image.

    Ported directly from the proven white-block navigator (python_cube_nav/
    navigator.py::plan_step), with the forward ADVANCE removed — Phase B only
    wants to swing the camera's optical axis onto the block, not close in.

    Why this and NOT a naive flange-XY translation: translating the flange
    sideways moves the whole camera, which shifts where EVERYTHING appears in
    the frame. If the depth estimate is even slightly off, that translation
    pushes an already-centred block back out of centre — the exact "it was
    centred once, then moved out" symptom. Instead we AIM: rotate the camera
    about its own X and Y axes by the small pixel error, then solve for the
    flange pose whose optical axis now passes through the block. That is a
    self-consistent correction that converges to centre and STAYS there.

    Steps (mirrors navigator):
      1. err_yaw   = (u-cx)/fx   (block right of centre -> +)
         err_pitch = (v-cy)/fy   (block low below centre -> +)
      2. R_cam_new = R_cam @ Rx(-err_pitch) @ Ry(+err_yaw)
      3. Keep the camera at its CURRENT position (no advance) but with the new
         orientation -> desired camera pose.
      4. T_target_flange = T_desired_cam @ inv(T_ee_cam)

    Returns (T_target_flange, reason_str) or (None, reason) if already centred.
    """
    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])
    u, v = det['center_px']
    px_off = float(np.hypot(u - cx, v - cy))
    if px_off < CENTRE_TOL_PX:
        return None, "centred (%.1f px from centre)" % px_off

    # Current camera pose in the base frame.
    T_cam = T_flange @ T_ee_cam

    err_yaw = (u - cx) / fx               # rad; block RIGHT of centre -> +
    err_pitch = (v - cy) / fy             # rad; block LOW below centre  -> +
    lim = np.deg2rad(max_aim_deg)
    err_yaw = float(np.clip(err_yaw, -lim, lim))
    err_pitch = float(np.clip(err_pitch, -lim, lim))
    aim_deg = float(np.hypot(err_yaw, err_pitch)) * 180.0 / np.pi

    Rx = Rotation.from_euler('x', -err_pitch).as_matrix()
    Ry = Rotation.from_euler('y', +err_yaw).as_matrix()
    R_cam_new = T_cam[:3, :3] @ (Rx @ Ry)

    # Desired camera pose: SAME position as now (pure re-aim, no advance),
    # new orientation so the optical axis points at the block.
    T_desired_cam = np.eye(4)
    T_desired_cam[:3, :3] = R_cam_new
    T_desired_cam[:3, 3] = T_cam[:3, 3]

    # Convert desired camera pose -> flange pose (inverse hand-eye).
    T_target = T_desired_cam @ np.linalg.inv(T_ee_cam)
    reason = ('%.0f px off-centre -> re-aim %.1f deg '
              '(yaw %+0.2f, pitch %+0.2f)'
              % (px_off, aim_deg, np.rad2deg(err_yaw), np.rad2deg(err_pitch)))
    return T_target, reason


def plan_descent_step(det, T_flange, T_ee_cam, K, dist_coeffs,
                      step_mm=DESCENT_STEP_MM, max_aim_deg=MAX_AIM_DEG_DESCENT):
    """Plan ONE descent move: re-aim onto the block AND advance along the view axis.

    This is the FULL white-block navigator step (python_cube_nav/navigator.py::
    plan_step), unlike plan_centre_step which only re-aims. It returns a single
    target FLANGE pose that folds both corrections into one motion:

      1. SEE     — use the detected block centroid + measured depth.
      2. AIM     — rotate the camera about its own X/Y by the clamped pixel error
                   so its optical axis points at the block (steers it to centre).
      3. ADVANCE — translate the re-aimed camera forward along its NEW optical
                   axis by step_mm, capped so we never overshoot the block.
      4. CONVERT — desired camera pose -> flange pose via inverse hand-eye.

    Advancing along the CAMERA optical axis (not base Z) is what keeps this stable:
    one fused aim+advance per press means nothing fights itself, so no fishing. This
    lowers the CAMERA over the block; because the wrist camera sits ~57 mm off to
    one side of the flange, a separate PHASE D shift then carries the CUP onto the
    block (see cup_shift_flange / shift_cup_over_block).

    Returns (T_target_flange, reason_str) or (None, reason) if not confirmed.
    """
    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])
    u, v = det['center_px']

    # Current camera pose in the base frame.
    T_cam = T_flange @ T_ee_cam

    # --- AIM: small-angle steering toward image centre ----------------------
    err_yaw = (u - cx) / fx               # rad; block RIGHT of centre -> +
    err_pitch = (v - cy) / fy             # rad; block LOW below centre  -> +
    lim = np.deg2rad(max_aim_deg)
    err_yaw = float(np.clip(err_yaw, -lim, lim))
    err_pitch = float(np.clip(err_pitch, -lim, lim))
    aim_deg = float(np.hypot(err_yaw, err_pitch)) * 180.0 / np.pi

    Rx = Rotation.from_euler('x', -err_pitch).as_matrix()
    Ry = Rotation.from_euler('y', +err_yaw).as_matrix()
    R_cam_new = T_cam[:3, :3] @ (Rx @ Ry)

    # --- ADVANCE: step forward along the NEW optical axis --------------------
    new_axis = R_cam_new[:, 2]
    dist_to_block_m = float(det['dist_flange_mm']) / 1000.0
    step_m = min(step_mm / 1000.0, max(dist_to_block_m, 1e-6))
    p_cam_new = T_cam[:3, 3] + new_axis * step_m

    T_desired_cam = np.eye(4)
    T_desired_cam[:3, :3] = R_cam_new
    T_desired_cam[:3, 3] = p_cam_new

    # --- CONVERT: desired camera pose -> flange pose (inverse hand-eye) ------
    T_target = T_desired_cam @ np.linalg.inv(T_ee_cam)
    reason = ('%.0f mm fwd + %.1f deg aim (%.0f mm to block)'
              % (step_m * 1000.0, aim_deg, dist_to_block_m * 1000.0))
    return T_target, reason


# ===========================================================================
# Cup offset geometry
# ===========================================================================
def cup_shift_flange(T_ee_cam):
    """Fixed FLANGE-frame XY translation that carries the CUP TIP onto the point
    the camera's view axis is aimed at — i.e. directly over a block centred in
    frame.

    WHY IT EXISTS: the wrist camera sits ~57 mm off to one side of the flange, so
    "block centred in the image" puts the CAMERA'S line of sight over the block,
    not the suction pad. Phase C servos the camera onto the block (keeping vision
    locked); this one-shot shift then moves the flange so the PAD lands on it.

    Both points are rigidly fixed in the flange frame, so the offset is constant
    and computed straight from the hand-eye matrix:
      - footprint = where the cam optical axis crosses the flange plane (over block)
      - tip       = cup tip, TOOL_OFFSET_MM below the flange face along flange Z
    The required flange-frame translation is (footprint_xy - tip_xy). Applying it
    as T_flange @ Trans(d) keeps it correct for ANY wrist orientation.

    Returns (dx_m, dy_m) in the flange frame.
    """
    R = T_ee_cam[:3, :3]
    o = T_ee_cam[:3, 3]
    axis = R[:, 2]                          # optical axis in flange frame
    t_face = -o[2] / axis[2]                # where the axis crosses the flange plane
    footprint = o + axis * t_face           # view-axis footprint at flange plane
    tip = np.array([0.0, 0.0, -(TOOL_OFFSET_MM / 1000.0)])   # cup tip, flange frame
    d = footprint[:2] - tip[:2]             # XY only; keep height unchanged
    return float(d[0]), float(d[1])


def cup_axis_base(T_flange, T_ee_cam):
    """Unit vector, in the BASE frame, pointing FROM the flange origin OUT toward
    the suction-cup tip / table — i.e. the direction we descend along to touch down.

    Why not just use flange -Z: with the wrist flipped to face the table (pitch
    ~180 deg), the flange +Z points DOWN at the table, so the pad hangs out the
    +Z side, not -Z. Hardcoding a sign breaks depending on which way the wrist
    ended up. Instead we anchor to the CAMERA: the pad and the lens face the same
    way, so the cup axis is the camera's own optical axis (+Z) carried into the
    base frame via the full flange->cam chain. Robust to any wrist twist.
    """
    R_f = T_flange[:3, :3]
    R_fc = T_ee_cam[:3, :3]            # flange -> cam rotation
    cam_optical_in_flange = R_fc[:, 2] # cam +Z (optical axis) expressed in flange frame
    return R_f @ cam_optical_in_flange


def compute_topface_target(q_rad, zeta, T_ee_cam, det):
    """Compute the FULL flange pose (position + orientation) for sitting on the
    block's TOP FACE, using one snapshot of the current pose + a fresh detection.

    This is called ONCE at the moment the operator presses [Enter] to shift (start
    of PHASE D), when we are closest to a good aim and the depth estimate is most
    trustworthy. The resulting 4x4 is frozen and handed to fine_position(); pressing
    [g] there simply drives the arm to it — no re-estimation on every press.

    Construction (all in metres/radians, base frame):
      1. Start from the CURRENT flange pose — its orientation is already the
         downward "cup at the table" attitude from Phase A, so we keep it.
      2. Apply the fixed cup-shift XY (see cup_shift_flange) so the PAD, not the
         camera, sits over the block. Orientation unchanged.
      3. Descend along the REAL cup axis (flange Z column, tilt-safe) by the
         bias-corrected camera->block depth, bringing the tip down to the top face.

    Returns a (4,4) base->flange transform, or None if det is unusable.
    """
    from constants import DEPTH_BIAS_MM
    if det is None:
        return None
    T_cur = flange_pose_base(q_rad, zeta)
    R = T_cur[:3, :3]

    # Step 2: lateral cup shift (flange-frame XY), composed as T @ Trans(d).
    dx_m, dy_m = cup_shift_flange(T_ee_cam)
    T_shift = np.eye(4)
    T_shift[0, 3] = dx_m
    T_shift[1, 3] = dy_m
    T_over = T_cur @ T_shift                    # pad now over the block, still hovering

    # Step 3: descend along the REAL cup axis (toward the table/tip) by the
    # bias-corrected camera->block depth. Direction derived from the camera so it
    # is correct for whichever way the wrist is flipped — NOT a hardcoded flange Z.
    descend_m = max(det['depth_mm'] - DEPTH_BIAS_MM, 0.0) / 1000.0
    cup_dir_base = cup_axis_base(T_over, T_ee_cam)
    T_target = T_over.copy()
    T_target[:3, 3] = T_over[:3, 3] + cup_dir_base * descend_m
    return T_target


def _project_cuptip_marks(T_flange, T_ee_cam, K, dist_coeffs):
    """Return an extra-marks list for show_frame() that plots where the CUP TIP
    currently projects in the camera image (a cyan diamond labelled 'CUP').

    Why this exists: instead of trusting a computed sign convention, the operator
    sees the pad's actual image position next to the block centroid crosshair and
    can judge the needed direction by eye. The tip is found along the flange Z
    axis at TOOL_OFFSET_MM below the flange face, transformed base->cam via
    T_flange^-1 @ T_ee_cam, then projected with the pinhole model + distortion.
    Returns [] if the tip is behind the camera or out of frame.
    """
    # Cup tip in FLANGE frame: straight down the flange Z axis by the tool length.
    p_tip_flange = np.array([0.0, 0.0, -(TOOL_OFFSET_MM / 1000.0)])
    # Flange point -> base -> camera:
    p_base = (T_flange @ np.concatenate([p_tip_flange, [1.0]]))[:3]
    p_cam = (np.linalg.inv(T_flange) @ T_ee_cam @ np.concatenate([p_base, [1.0]]))[:3]
    if p_cam[2] <= 1e-3:                       # behind/at the camera plane
        return []
    x, y, z = p_cam
    u = K[0, 0] * x / z + K[0, 2]
    v = K[1, 1] * y / z + K[1, 2]
    # Apply radial/tangential distortion for accuracy near the frame edge.
    try:
        uu, vv = cv2.projectPoints(np.array([[x, y, z]], dtype=np.float64),
                                   np.zeros(3), np.zeros(3), K, dist_coeffs)[0][0]
        u, v = float(uu), float(vv)
    except Exception:
        pass
    H, W = 480, 640                              # rough bounds; off-screen => skip
    if not (-20 < u < W + 20 and -20 < v < H + 20):
        return [(u, v, "CUP(off)", (255, 255, 0))]
    return [(u, v, "CUP", (255, 255, 0))]


# ===========================================================================
# Robot-facing move primitives (IK + servo drive)
# ===========================================================================
def move_to_pose(arm, T_target, zeta, q_seed_rad, speed=None):
    """Solve IK (Python solver, seeded) and drive the servos to T_target.

    speed : optional servo-speed percent override (slower when closing in).
            Defaults to robot_io.SERVO_SPEED when None.

    Returns (ok, msg, q_after_rad_or_None).
    """
    import robot_io
    from ik_solver import solve_ik

    q0 = q_seed_rad if q_seed_rad is not None else np.zeros(6)
    q_sol, ok, err = solve_ik(T_target, q0, zeta)
    if not ok:
        return False, "IK did not converge (resid %.2e)" % err, None
    kwargs = {'wait': True}
    if speed is not None:
        kwargs['speed'] = int(speed)
    mok, mmsg = robot_io.move_to_joints(arm, q_sol, **kwargs)
    if not mok:
        return False, mmsg, None
    # Re-read ACTUAL joints (never trust the prediction — see run_lab3.py).
    try:
        q_actual = robot_io.read_joints_rad(arm)
    except Exception:
        q_actual = np.asarray(q_sol)
    return True, "moved", q_actual


def nudge_z(arm, zeta, q_rad, dz_mm):
    """Move straight up (+dz) or down (-dz) in the base frame, orientation held.

    Used for the manual final descent when the block has left the camera view:
    the operator taps [down]/[up] to creep the cup onto the block by eye.
    """
    T_flange = flange_pose_base(q_rad, zeta)
    T_new = T_flange.copy()
    T_new[2, 3] += dz_mm / 1000.0
    return move_to_pose(arm, T_new, zeta, q_rad)


def nudge_xy(arm, zeta, q_rad, dx_mm, dy_mm):
    """Creep the flange by (dx_mm, dy_mm) in the BASE frame, Z and orientation held.

    Manual XY fine-tuning during fine positioning: the operator taps [z]/[c] for
    X and [a]/[d] for Y to walk the cup onto the block's centre by eye. Base-frame
    axes are used (not flange-relative) so the mapping stays intuitive — +X is
    always "forward" and +Y always "left" regardless of how the wrist is turned.
    """
    T_flange = flange_pose_base(q_rad, zeta)
    T_new = T_flange.copy()
    T_new[0, 3] += dx_mm / 1000.0
    T_new[1, 3] += dy_mm / 1000.0
    return move_to_pose(arm, T_new, zeta, q_rad)


def orient_downward(arm, zeta, q_rad):
    """Flip the wrist so the suction cup points straight DOWN at the table.

    Keeps the current XYZ, only changes orientation to DOWN_RPY_DEG (pitch 180).
    This is the "face the table" step that must happen BEFORE any servoing, so
    the camera optical axis (≈ cup axis) is vertical and lateral alignment is
    unambiguous.
    """
    T_flange = flange_pose_base(q_rad, zeta)
    p = T_flange[:3, 3].copy()
    T_target = make_downward_pose(p, rpy_deg=DOWN_RPY_DEG)
    return move_to_pose(arm, T_target, zeta, q_rad)


def shift_cup_over_block(arm, zeta, q_rad, T_ee_cam, speed=None):
    """Execute the single one-shot flange translation that puts the CUP over the
    block (see cup_shift_flange). Composed in the flange frame so it is valid for
    whatever wrist pose we're in. Returns (ok, msg, q_after_rad_or_None)."""
    dx_m, dy_m = cup_shift_flange(T_ee_cam)
    mag_mm = float(np.hypot(dx_m, dy_m) * 1000.0)
    print("  cup-shift: translating flange by (%+.1f, %+.1f) mm [flange frame]"
          % (dx_m * 1000.0, dy_m * 1000.0))
    T_flange = flange_pose_base(q_rad, zeta)
    T_shift = np.eye(4)
    T_shift[0, 3] = dx_m
    T_shift[1, 3] = dy_m
    T_target = T_flange @ T_shift           # pose-relative, orientation preserved
    return move_to_pose(arm, T_target, zeta, q_rad, speed=speed)


def descend_along_cup_axis(arm, zeta, q_rad, descend_mm, T_ee_cam, speed=None):
    """Lower the CUP TIP by ``descend_mm`` along the REAL cup axis (toward the table).

    Why not just decrement base Z: with the wrist upright the cup hangs straight
    down the base -Z axis, so moving the flange in base Z works. But if Phase A
    leaves the wrist roll/yaw-twisted, the cup axis points somewhere else and a
    pure base-Z move would miss the block laterally. The cup tip lives at
    ``p_flange + R_flange @ [0, 0, -TOOL_OFFSET]``, so to bring it closer to the
    block we translate the flange by ``-R_flange @ [0, 0, descend]`` — i.e. along
    the flange's own Z column, whatever direction that currently points. This
    preserves the current orientation exactly (only position changes), so it is
    valid for any wrist pose. Reduces to a plain base-Z descent when upright.

    Returns (ok, msg, q_after_rad_or_None).
    """
    T_flange = flange_pose_base(q_rad, zeta)
    d_m = descend_mm / 1000.0
    # Unit vector from flange toward its own cup tip / table, in the BASE frame.
    # Derived from the camera optical axis so it is correct for any wrist flip
    # (see cup_axis_base). Moving the flange along this brings the tip closer to
    # the block; orientation is preserved (position-only change).
    cup_dir_base = cup_axis_base(T_flange, T_ee_cam)
    T_new = T_flange.copy()
    T_new[:3, 3] = T_flange[:3, 3] + cup_dir_base * d_m   # move tip down by d_m
    return move_to_pose(arm, T_new, zeta, q_rad, speed=speed)
