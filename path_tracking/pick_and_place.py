#!/usr/bin/env python3
"""Lab 3 — Vision-guided pick & place of two coloured blocks.

Task:
    1. Find the RED 25 mm cube (RGB colour segmentation on the wrist camera),
       iteratively approach it, engage the vacuum gripper and lift it, then lift
       back to a both-blocks-visible overview pose.
    2. From that overview pose, find the GREEN 25 mm cube, approach it with the
       same five-phase navigation, and set it down in place.

Every stage is KEYGATED: the live camera feed is shown the whole time and the
arm only advances to the next stage when you press [Enter]/[n]. This keeps the
first live runs safe — you confirm each detection and each move by eye before
committing. Press [q] at any point to abort (the script lifts, releases and
homes).

Geometry / coordinate chain (identical to run_lab3.py / navigator.py):
    * All internal maths in METRES + RADIANS; the SDK speaks MM + DEGREES.
    * T_flange = fk_from_zeta(q, zeta_nominal)          (base -> flange)
    * T_cam    = T_flange @ T_ee_cam                    (base -> camera)
    * A detected pixel centre is unprojected out to a measured depth to give a
      3-D point in the BASE frame. That base-frame point is what we plan moves
      against, so every command is expressed in the robot's own coordinates.
    * The TOOL OFFSET below shifts from the flange frame to the SUCTION CUP TIP
      (the actual contact point), because the cup hangs TOOL_OFFSET_MM below the
      flange face. UFACTORY does not publish this distance — it is a per-rig
      constant you tune once (see the note under the value).

Keys (camera window):
    [Enter]/[n]   advance to the next stage
    [q]           quit (safe reset: lift -> release -> home)

Module layout (flat siblings, one concern each):
    constants.py     tunable knobs + HSV defaults
    vision.py        colour detection + base-frame resolution
    motion.py        geometry planners + robot-facing move primitives
    ui_display.py    camera / OpenCV UI / calibration loading / key input
    fk_functions.py  forward kinematics
    ik_solver.py     numerical inverse kinematics
    robot_io.py      thin xArm SDK wrapper
This file is the slim ORCHESTRATOR: the interactive phase loops plus the top-
level run() sequence. It owns no geometry or hardware detail of its own.

Usage:
    cd Lab_3/path_tracking
    python pick_and_place.py
"""

import sys
import time

import numpy as np
import cv2

from constants import (
    LATERAL_STEPS,
    CENTRE_TOL_PX,
    DESCENT_STEP_MM,
    MAX_AIM_DEG_DESCENT,
    FINE_Z_STEP_MM,
    FINE_XY_STEP_MM,
    DEPTH_BIAS_MM,
    AUTO_LOWER_MIN_GAP_MM,
    LIFT_BACK_UP_MM,
    NEAR_SPEED_PCT,
    TOOL_OFFSET_MM,
)
from motion import (
    flange_pose_base,
    orient_downward,
    shift_cup_over_block,
    descend_along_cup_axis,
    nudge_z,
    nudge_xy,
    move_to_pose,
    plan_centre_step,
    plan_descent_step,
    cup_shift_flange,
    cup_axis_base,
    compute_topface_target,
    _project_cuptip_marks,
    _approach_schedule,
)
from vision import compute_detection
from ui_display import (
    load_camera_intrinsics,
    load_calibration_results,
    open_camera,
    size_windows,
    create_threshold_trackbars,
    show_frame,
    wait_next,
    tune_masks,
    WINDOW,
    MASK_WINDOWS,
)


# ===========================================================================
# Interactive phase helpers (owned here — they drive the operator loop)
# ===========================================================================
def _fresh_det(cap, q_rad, zeta, K, dist, T_ee_cam, color):
    """Read one frame + current pose and resolve the block in the base frame."""
    T_flange = flange_pose_base(q_rad, zeta)
    ret, frame = cap.read()
    if not ret:
        return None, T_flange
    det = compute_detection(frame, color, K, dist, T_flange, T_ee_cam)
    return det, T_flange


def capture_both_blocks_pose(cap, q_rad, zeta, K, dist, T_ee_cam):
    """If BOTH the red and green blocks are currently in view, return the flange
    pose at this instant; otherwise None.

    This is the "overview" pose — the arm position from which the whole scene
    (both blocks) is visible. We remember the LAST such pose seen before we start
    chasing a single block, so that after picking we can lift straight back to a
    known viewpoint where the next block will still be findable by vision.
    """
    T_flange = flange_pose_base(q_rad, zeta)
    ret, frame = cap.read()
    if not ret:
        return None
    det_red = compute_detection(frame, 'red', K, dist, T_flange, T_ee_cam)
    det_green = compute_detection(frame, 'green', K, dist, T_flange, T_ee_cam)
    if det_red is not None and det_green is not None:
        return T_flange.copy()
    return None


def maybe_update_overview(T_both_visible, cap, q_rad, zeta, K, dist, T_ee_cam,
                          label=""):
    """Try to grab the current both-blocks-visible pose and, if found, replace
    ``T_both_visible`` with it. Returns the (possibly updated) pose.

    Called repeatedly while the arm is parked with the whole scene in view —
    during the free-move prelude AND at Stage 0, i.e. any time BEFORE the wrist
    has been flipped cup-down for a chase. Keeping the freshest such pose means
    the lift-back always returns to a viewpoint where the remaining block is
    still findable by vision.
    """
    T_ov = capture_both_blocks_pose(cap, q_rad, zeta, K, dist, T_ee_cam)
    if T_ov is not None:
        T_both_visible = T_ov
        print("  %soverview pose refreshed (both blocks in view): "
              "flange Z ≈ %.1f mm" % (label, T_ov[2, 3] * 1000.0))
    return T_both_visible


def vertical_nudges(arm, cap, zeta, K, dist, T_ee_cam, color, q_rad, compensated):
    """Keystroke-driven descent onto the block via SINGLE-MOTION optical-axis steps.

    Direct port of the proven white-block navigator's approach (python_cube_nav/
    navigator.py::plan_step). Each press issues ONE IK-solved move that both re-aims
    the camera onto the block AND advances it DESCENT_STEP_MM along its own optical
    axis. Because aim+advance are fused into a single target pose, there is only one
    motion per press — nothing fights itself, so the old "lateral then vertical"
    fishing/hunting disappears.

        [w] / [Up]     advance toward the block (aim + creep forward)
        [s] / [Down]   same as [w] (kept for muscle-memory / symmetry)
        [Enter]/[n]    descent complete -> continue to grip
        [a]            abort this pick

    The block must stay in view for the servoing to work; near contact it may start
    to leave the frame, at which point you're close enough to finish by eye.

    Signature keeps the trailing ``compensated`` arg so the caller is unchanged;
    the separate [c] cam->cup shift is gone because advancing along the view axis
    puts the CUP (which lies on that axis) directly onto the block automatically.

    Returns (done: bool, aborted: bool, q_after_rad, compensated_flag).
    """
    while True:
        T_flange = flange_pose_base(q_rad, zeta)
        det, _ = _fresh_det(cap, q_rad, zeta, K, dist, T_ee_cam, color)
        info = ["flange Z ≈ %.1f mm" % (T_flange[2, 3] * 1000.0)]
        if det is not None:
            info.append("cam->block depth : %.0f mm" % det['depth_mm'])
            info.append("  XY gap         : %.1f mm" % det['xy_gap_mm'])
        else:
            info.append("%s block NOT in view" % color.upper())
        # Project the LIVE CUP TIP into the image so you can watch the pad converge
        # onto the block crosshair as you advance.
        marks = _project_cuptip_marks(T_flange, T_ee_cam, K, dist)
        key = show_frame(
            cap,
            "DESCEND — each press aims + creeps forward along the view axis:",
            info, det, color,
            hint="[w]/up=ADVANCE  [s]/down=ADVANCE  [Enter]=ready  [a]=ABORT"
                  "   (watch the cyan CUP close onto the crosshair)",
            extra_marks=marks, K=K, dist_coeffs=dist)
        if key == ord('q'):
            key = ord('a')
        # Arrow keys arrive masked to their low byte: Down=18, Up=17. Both drive a
        # forward advance (the direction is set by the pose, not the key).
        if key in (ord('w'), ord('s'), 17, 18):
            if det is None:
                print("  ! no block in view — cannot plan an advance; hold still.")
                continue
            _, speed_pct = _approach_schedule(det['dist_flange_mm'])
            T_target, reason = plan_descent_step(
                det, T_flange, T_ee_cam, K, dist,
                step_mm=DESCENT_STEP_MM, max_aim_deg=MAX_AIM_DEG_DESCENT)
            if T_target is None:
                print("  %s — holding." % reason)
                continue
            ok, msg, q_after = move_to_pose(arm, T_target, zeta, q_rad, speed=speed_pct)
            if not ok:
                print("  ! advance failed: %s" % msg)
                return False, False, q_rad, compensated
            q_rad = q_after
            z_now = flange_pose_base(q_rad, zeta)[2, 3] * 1000.0
            print("  advance: %s  (flange Z≈%.1f mm) @ %d%%"
                  % (reason, z_now, speed_pct))
        elif key in (ord('\r'), ord('\n'), ord('n')):
            return True, False, q_rad, compensated
        elif key == ord('a'):
            return False, True, q_rad, compensated
        # any other key: just refresh the feed and wait again


def fine_position(arm, cap, zeta, K, dist, T_ee_cam, color, q_rad, T_target=None):
    """PHASE E — hover-and-settle loop once the cup is over the block.

    After PHASE D's lateral shift we are hovering above the block with the pad
    centred but not touching. This interactive loop lets the operator bring the
    cup down onto the top face and manage suction:

        [s] / down-arrow : creep DOWN  FINE_Z_STEP_MM (base Z)
        [w] / up-arrow   : creep UP    FINE_Z_STEP_MM
        [g]              : DRIVE TO T_TARGET — the full top-face pose (position AND
                           orientation) that was computed ONCE at the Enter-to-shift
                           moment and passed in as ``T_target``. No re-estimation.
        [o]              : ENGAGE suction (CLOSE the vacuum cup onto the block)
        [r]              : RELEASE suction (OPEN the cup)
        [y]              : CONFIRM — only accepted once suction is engaged; ends the
                           phase so the caller can lift off. This is the phase exit.
        [a]              : abort (returns aborted=True; caller releases grip)
        [q]              : quit (mapped to abort)

    The phase does NOT end on its own after engaging suction: the operator must
    press [y] to confirm the grip looks good. That keeps a failed or weak suction
    from being lifted straight away.

    ``T_target`` (4x4 base->flange) is the frozen "sit on the top face" pose from
    compute_topface_target(). If it is None (e.g. no detection at the shift gate),
    [g] falls back to a live descent along the real cup axis by the current
    bias-corrected depth, so the key still does something sensible.

    The live frame shows the block box, the projected CUP tip, and a running readout
    of flange Z, estimated cam->block depth (bias-corrected), and XY gap so you can
    judge touchdown by eye even when the block partly leaves the frame.

    Returns (done: bool, aborted: bool, q_after_rad, gripped: bool).
    ``gripped`` reports whether suction is currently ON so the caller knows whether
    to release on teardown.
    """
    import robot_io

    gripped = False
    while True:
        T_flange = flange_pose_base(q_rad, zeta)
        det, _ = _fresh_det(cap, q_rad, zeta, K, dist, T_ee_cam, color)
        info = ["flange Z ≈ %.1f mm" % (T_flange[2, 3] * 1000.0)]
        if det is not None:
            corrected_depth = max(det['depth_mm'] - DEPTH_BIAS_MM, 0.0)
            info.append("cam->block depth : %.0f mm (raw %.0f, bias -%.0f)"
                        % (corrected_depth, det['depth_mm'], DEPTH_BIAS_MM))
            info.append("  XY gap         : %.1f mm" % det['xy_gap_mm'])
        else:
            info.append("%s block NOT in view — use [s]/[w] by eye" % color.upper())
        marks = _project_cuptip_marks(T_flange, T_ee_cam, K, dist)
        key = show_frame(
            cap,
            "FINE POSITIONING — settle the cup on the %s block" % color.upper(),
            info, det, color,
            hint="[s]/[w]=Z down/up   [z]/[c]=X -/+   [a]/[d]=Y -/+"
                  "   [g]=go-to-seat  [o]=ENGAGE  [r]=release  [y]=CONFIRM  [x]=ABORT",
            extra_marks=marks, K=K, dist_coeffs=dist)
        if key == ord('q'):
            key = ord('a')

        # --- vertical creep --------------------------------------------------
        elif key in (ord('s'), 18):                      # 's' or Down arrow
            ok, msg, q_after = nudge_z(arm, zeta, q_rad, -FINE_Z_STEP_MM)
            if not ok:
                print("  ! down creep failed: %s" % msg)
                return False, False, q_rad, gripped
            q_rad = q_after
            print("  down  %.1f mm  (Z≈%.1f mm)"
                  % (FINE_Z_STEP_MM, flange_pose_base(q_rad, zeta)[2, 3] * 1000.0))
        elif key in (ord('w'), 17):                      # 'w' or Up arrow
            ok, msg, q_after = nudge_z(arm, zeta, q_rad, +FINE_Z_STEP_MM)
            if not ok:
                print("  ! up creep failed: %s" % msg)
                return False, False, q_rad, gripped
            q_rad = q_after
            print("  up    %.1f mm  (Z≈%.1f mm)"
                  % (FINE_Z_STEP_MM, flange_pose_base(q_rad, zeta)[2, 3] * 1000.0))

        # --- horizontal creep (base frame) -----------------------------------
        elif key == ord('z'):                            # -X
            ok, msg, q_after = nudge_xy(arm, zeta, q_rad, -FINE_XY_STEP_MM, 0.0)
            if not ok:
                print("  ! X creep failed: %s" % msg)
                return False, False, q_rad, gripped
            q_rad = q_after
            print("  X-%.1f mm  (X≈%.1f mm)"
                  % (FINE_XY_STEP_MM, flange_pose_base(q_rad, zeta)[0, 3] * 1000.0))
        elif key == ord('c'):                            # +X
            ok, msg, q_after = nudge_xy(arm, zeta, q_rad, +FINE_XY_STEP_MM, 0.0)
            if not ok:
                print("  ! X creep failed: %s" % msg)
                return False, False, q_rad, gripped
            q_rad = q_after
            print("  X+%.1f mm  (X≈%.1f mm)"
                  % (FINE_XY_STEP_MM, flange_pose_base(q_rad, zeta)[0, 3] * 1000.0))
        elif key == ord('a'):                            # -Y
            ok, msg, q_after = nudge_xy(arm, zeta, q_rad, 0.0, -FINE_XY_STEP_MM)
            if not ok:
                print("  ! Y creep failed: %s" % msg)
                return False, False, q_rad, gripped
            q_rad = q_after
            print("  Y-%.1f mm  (Y≈%.1f mm)"
                  % (FINE_XY_STEP_MM, flange_pose_base(q_rad, zeta)[1, 3] * 1000.0))
        elif key == ord('d'):                            # +Y
            ok, msg, q_after = nudge_xy(arm, zeta, q_rad, 0.0, +FINE_XY_STEP_MM)
            if not ok:
                print("  ! Y creep failed: %s" % msg)
                return False, False, q_rad, gripped
            q_rad = q_after
            print("  Y+%.1f mm  (Y≈%.1f mm)"
                  % (FINE_XY_STEP_MM, flange_pose_base(q_rad, zeta)[1, 3] * 1000.0))

        # --- go to the precomputed top-face pose -----------------------------
        elif key == ord('g'):
            if T_target is not None:
                # Primary path: drive straight to the frozen full pose (position +
                # orientation) captured at the Enter-to-shift moment. This is what
                # the operator asked for — one deterministic move to the seat.
                tz = T_target[2, 3] * 1000.0
                cz = flange_pose_base(q_rad, zeta)[2, 3] * 1000.0
                print("  [g] driving to top-face target (Z %.1f -> %.1f mm)..."
                      % (cz, tz))
                ok, msg, q_after = move_to_pose(arm, T_target, zeta, q_rad,
                                                speed=NEAR_SPEED_PCT)
            else:
                # Fallback: no frozen target (no detection at the shift gate). Close
                # the remaining gap along the real cup axis by the live depth.
                if det is None:
                    print("  ! no target pose and no block in view — hold still.")
                    continue
                descend_mm = max(det['depth_mm'] - DEPTH_BIAS_MM, 0.0)
                if descend_mm < AUTO_LOWER_MIN_GAP_MM:
                    print("  already at/near estimated contact (%.0f mm) — skipping."
                          % descend_mm)
                    continue
                print("  [g] (fallback) descending %.0f mm along cup axis"
                      % descend_mm)
                ok, msg, q_after = descend_along_cup_axis(
                    arm, zeta, q_rad, descend_mm, T_ee_cam, speed=NEAR_SPEED_PCT)
            if not ok:
                print("  ! auto-lower failed: %s" % msg)
                return False, False, q_rad, gripped
            q_rad = q_after
            print("  reached target (Z≈%.1f mm)"
                  % (flange_pose_base(q_rad, zeta)[2, 3] * 1000.0))

        # --- suction control -------------------------------------------------
        # NOTE polarity: for this vacuum cup, CLOSE engages suction (grasps the
        # block) and OPEN releases it. Earlier code had these swapped.
        elif key == ord('o'):
            if robot_io.open_bio_gripper(arm) == 1:
                gripped = True
                time.sleep(0.6)
                print("  suction ENGAGED on %s block — press [y] when the grip "
                      "looks good." % color.upper())
            else:
                print("  ! could not engage suction.")
        elif key == ord('r'):
            if robot_io.close_bio_gripper(arm) == 1:
                gripped = False
                print("  suction RELEASED.")
            else:
                print("  ! could not release suction.")

        # --- confirm grip -> end phase ---------------------------------------
        elif key == ord('y'):
            if not gripped:
                print("  ! suction is not engaged — press [o] first, then [y].")
                continue
            print("  grip confirmed by operator — lifting off.")
            return True, False, q_rad, gripped
        elif key == ord('x'):                            # [a] now drives -Y; abort moved here
            return False, True, q_rad, gripped
        # any other key: just refresh the feed and wait again


# ===========================================================================
# High-level stages
# ===========================================================================
def approach_and_pick(arm, cap, zeta, K, dist, T_ee_cam, color, terminal='lift',
                      liftback_pose=None):
    """Close on a coloured block, in five explicit sub-phases, ending on [o]+[y].

    ``liftback_pose`` : optional (4,4) base->flange transform — the "overview" pose
                        from which both blocks were last visible. After the grip is
                        confirmed with [y], the arm returns HERE instead of a blind
                        +Z bump, so the next navigation stage starts from a known
                        viewpoint where the remaining block is still findable by
                        vision. If None, falls back to raising LIFT_BACK_UP_MM along
                        the cup axis from the pre-shift hover pose.

    ``terminal`` controls what happens AFTER the operator confirms the grip with
    [y] inside PHASE E:
      - 'lift'    : raise the held block back to the captured pre-descent hover pose
                    and return it HELD (used for the red pick).
      - 'release' : set the block down in place, release suction, and retract clear
                    (used for the green drop — same navigation/shift/confirm, but we
                    leave the block where it landed rather than carrying it away).

    In BOTH cases the phase ends only when the operator presses [o] (engage) then
    [y] (confirm); nothing lifts or releases before that confirmation.

      PHASE A — ORIENT DOWN: flip the wrist so the cup points straight at the
                  table (pitch 180). Done ONCE, gated by a single keypress.
      PHASE B — CENTRE IN FRAME: keystroke-driven lateral moves that drive the
                  block's pixel centroid to the image centre, i.e. point the
                  camera optical axis at the block. No Z change.
      PHASE C — LOWER THE CAMERA ONTO THE BLOCK: each press issues ONE fused move
                  (re-aim + advance along the view axis), lowering the camera
                  until it sits right over the block. Vision stays locked the whole
                  time because we servo the CAMERA, not the cup. No split motion,
                  hence no fishing.
      PHASE D — SHIFT CUP OVER BLOCK: one-shot flange-frame translation (computed
                  from the hand-eye matrix) that carries the CUP ~57 mm laterally
                  so the pad is centred over the block. Leaves us HOVERING.
      PHASE E — FINE POSITIONING: interactive loop to settle the cup onto the top
                  face ([s]/[w] creep, [g] auto-lower to estimated contact height)
                  and manage suction ([o] engage / [r] release). Lift only after
                  suction is confirmed on.

    WHY C THEN D: the wrist camera sits ~57 mm off to one side of the flange, so
    its view axis does NOT pass through the suction pad. Servoing the cup directly
    would lose sight of the block; instead we keep the camera on the target, then
    apply the fixed rigid offset once (constant in the flange frame, valid for any
    wrist pose). WHY E: the shift centres but does not touch down, and the depth
    estimate over-reads distance, so the final seat is done interactively with a
    bias-corrected auto-lower option.

    Every gate offers an explicit [a] ABORT. Returns (success, final_q_rad).
    """
    import robot_io

    q_rad = robot_io.read_joints_rad(arm)

    # =====================================================================
    # PHASE A — ORIENT DOWN (face the table)
    # =====================================================================
    print("\n  [PHASE A] Orienting cup downward...")
    advanced, aborted = wait_next(
        "PHASE A — [Enter] to orient cup DOWN (pitch 180)",
        cap, arm, zeta, K, dist, T_ee_cam, color,
        ["cup axis -> base -Z"])
    if aborted:
        return False, q_rad
    ok, msg, q_after = orient_downward(arm, zeta, q_rad)
    if not ok:
        print("  ! orientation move failed: %s" % msg)
        return False, q_rad
    q_rad = q_after
    print("  cup now facing the table.")

    # =====================================================================
    # PHASE B — CENTRE THE BLOCK IN FRAME (camera-view servoing, XY only)
    # =====================================================================
    # The wrist cam sits ~57 mm off to one side of the flange, so we steer on
    # where the block sits IN THE IMAGE, not on base-frame XY. Goal: pixel
    # centroid == principal point, i.e. the camera optical axis points at the
    # block. This is what makes the subsequent vertical drop unambiguous.
    print("\n  [PHASE B] Centre the %s block in the frame..." % color.upper())
    centred = False
    for step in range(LATERAL_STEPS):
        advanced, aborted = wait_next(
            "PHASE B — centre block  (press %d/%d)  [Enter]=shift to centre" %
            (step + 1, LATERAL_STEPS),
            cap, arm, zeta, K, dist, T_ee_cam, color,
            ["drive the crosshair to the image centre; Z held constant"])
        if aborted:
            return False, q_rad

        det, T_flange = _fresh_det(cap, q_rad, zeta, K, dist, T_ee_cam, color)
        if det is None:
            print("  ! %s block not visible — reposition and retry." % color.upper())
            continue

        cx, cy = float(K[0, 2]), float(K[1, 2])
        u, v = det['center_px']
        px_off = float(np.hypot(u - cx, v - cy))
        if px_off < CENTRE_TOL_PX:
            print("  block centred (%.1f px from centre)." % px_off)
            centred = True
            break

        # Re-aim the camera onto the block (rotation only; no advance here).
        # Speed still tapers with distance so the final corrections are gentle.
        _, speed_pct = _approach_schedule(det['dist_flange_mm'])
        T_target, reason = plan_centre_step(det, T_flange, T_ee_cam, K, dist)
        if T_target is None:
            print("  press %d: %s — hold." % (step + 1, reason))
            continue
        ok, msg, q_after = move_to_pose(arm, T_target, zeta, q_rad, speed=speed_pct)
        if not ok:
            print("  ! re-aim move failed: %s" % msg)
            break
        q_rad = q_after
        print("  press %d: %s  @ %d%%" % (step + 1, reason, speed_pct))

    if not centred:
        # Operator ran out of presses without converging. Let them decide.
        advanced, aborted = wait_next(
            "Block not fully centred after %d presses. [Enter]=proceed, [%s]=abort" %
            (LATERAL_STEPS, 'a'),
            cap, arm, zeta, K, dist, T_ee_cam, color,
            ["final pixel offset shown live below"], abort_key='a')
        if aborted:
            return False, q_rad

    # =====================================================================
    # PHASE C — CREEP THE CAMERA DOWN ONTO THE BLOCK (optical-axis advances)
    # =====================================================================
    # The camera and the cup are ~57 mm apart laterally, so we CANNOT servo the
    # cup directly without losing sight of the block. Instead Phase C keeps the
    # CAMERA locked on the block (vision stays alive) and lowers until the block
    # fills the frame / the view axis is right above it. Each [w]/[s] press is
    # ONE fused move (re-aim + advance along the view axis) — no split motion,
    # hence no fishing.
    print("\n  [PHASE C] Lower camera onto %s block (%.0f mm per press)..."
          % (color.upper(), DESCENT_STEP_MM))
    advanced, aborted = wait_next(
        "PHASE C — lower camera onto %s block (each press aims + descends)" % color.upper(),
        cap, arm, zeta, K, dist, T_ee_cam, color)
    if aborted:
        return False, q_rad

    compensated = False
    done, aborted, q_after, compensated = vertical_nudges(
        arm, cap, zeta, K, dist, T_ee_cam, color, q_rad, compensated)
    if aborted:
        return False, q_after
    if not done:
        print("  descent did not complete.")
        return False, q_after
    q_rad = q_after
    print("  camera lowered over %s block." % color.upper())

    # ---------------------------------------------------------------------
    # PHASE D — SHIFT CUP ONTO BLOCK (one-shot, calibration-derived)
    # ---------------------------------------------------------------------
    # Now the camera's view axis is over the block, but the CUP is still ~57 mm
    # off to one side. Apply the fixed flange-frame translation once so the pad
    # lands on the block. Dedicated keypress so you can eyeball it before it moves.
    dx_m, dy_m = cup_shift_flange(T_ee_cam)
    shift_mm = float(np.hypot(dx_m, dy_m) * 1000.0)
    print("\n  [PHASE D] Cup is ~%.0f mm off the block — shifting it over." % shift_mm)
    advanced, aborted = wait_next(
        "PHASE D — [Enter] to SHIFT CUP %.0f mm onto %s block" % (shift_mm, color.upper()),
        cap, arm, zeta, K, dist, T_ee_cam, color,
        ["cup offset (flange XY): (%+.1f, %+.1f) mm" % (dx_m * 1000.0, dy_m * 1000.0)])
    if aborted:
        return False, q_rad

    # SNAPSHOT TWO POSES NOW, before we shift/lower. At this instant we're closest
    # to a clean aim with the freshest depth estimate, so freeze both here:
    #   - T_target : the full "sit on the top face" seat pose ([g] drives to it)
    #   - T_liftback: where we RETURN to after grasping = current hover pose raised
    #     LIFT_BACK_UP_MM along the cup axis. Lifting straight back here keeps the
    #     block centred under the pad and clears the table without hunting.
    det_snap, _ = _fresh_det(cap, q_rad, zeta, K, dist, T_ee_cam, color)
    T_target = compute_topface_target(q_rad, zeta, T_ee_cam, det_snap)
    if T_target is not None:
        ttz = T_target[2, 3] * 1000.0
        print("  top-face target captured: flange Z ≈ %.1f mm" % ttz)
    else:
        print("  ! no detection at shift gate — [g] will fall back to live descent.")

    if liftback_pose is not None:
        # Preferred: return to the remembered "both blocks visible" overview pose,
        # so after picking we go back to a viewpoint where the next block is still
        # in view and findable by vision.
        T_liftback = np.asarray(liftback_pose).copy()
        print("  lift-back target = overview pose (both blocks visible): "
              "flange Z ≈ %.1f mm" % (T_liftback[2, 3] * 1000.0))
    else:
        # Fallback: no overview pose was captured — just raise LIFT_BACK_UP_MM along
        # the cup axis from the current pre-shift hover pose.
        T_cur_now = flange_pose_base(q_rad, zeta)
        T_liftback = T_cur_now.copy()
        T_liftback[:3, 3] = T_cur_now[:3, 3] + \
            cup_axis_base(T_cur_now, T_ee_cam) * (-LIFT_BACK_UP_MM / 1000.0)
        print("  lift-back pose (fallback +Z): flange Z ≈ %.1f mm"
              % (T_liftback[2, 3] * 1000.0))

    ok, msg, q_after = shift_cup_over_block(arm, zeta, q_rad, T_ee_cam)
    if not ok:
        print("  ! cup shift failed: %s" % msg)
        return False, q_rad
    q_rad = q_after
    time.sleep(0.4)
    print("  cup shifted over %s block — now hovering." % color.upper())

    # ---------------------------------------------------------------------
    # PHASE E — FINE POSITIONING: settle the cup on the top face + manage grip
    # ---------------------------------------------------------------------
    # The shift left us HOVERING, not touching. This interactive loop lets you
    # creep down ([s]/[w]), auto-lower to the estimated contact height ([g]), and
    # engage/release suction ([o]/[r]) until the block is firmly held. Suction must
    # be ON before we lift, so we require it before proceeding.
    print("\n  [PHASE E] Fine positioning — settle cup on %s block..." % color.upper())
    done, aborted, q_after, gripped = fine_position(
        arm, cap, zeta, K, dist, T_ee_cam, color, q_rad, T_target=T_target)
    if aborted:
        if gripped:
            robot_io.stop_bio_gripper(arm)   # don't drag the block on an abort
        return False, q_after
    q_rad = q_after
    if not gripped:
        print("  ! no suction engaged — cannot proceed. Aborting safely.")
        return False, q_rad

    if terminal == 'release':
        # --- SET DOWN IN PLACE, then release + retract -----------------------
        # Same navigation/shift/confirm as a pick, but instead of carrying the
        # block away we let it drop onto the surface right beneath the pad, open
        # the cup, and lift clear. The block stays where the vision put it.
        print("\n  [RELEASE] setting %s block down in place..." % color.upper())
        advanced, aborted = wait_next(
            "Grip confirmed — [Enter] to RELEASE %s block here, [a] abort"
            % color.upper(),
            cap, arm, zeta, K, dist, T_ee_cam, color)
        if aborted:
            robot_io.stop_bio_gripper(arm)
            return False, q_rad
        if robot_io.open_bio_gripper(arm) != 1:
            print("  ! could not release %s block." % color.upper())
            return False, q_rad
        print("  %s block released." % color.upper())
        time.sleep(0.5)
        ok, msg, q_after = move_to_pose(arm, T_liftback, zeta, q_rad)
        if not ok:
            print("  ! post-release lift-back failed: %s" % msg)
            return False, q_rad
        q_rad = q_after
        print("  %s block set down — lifted clear to hover pose." % color.upper())
        return True, q_rad

    # --- LIFT BACK to the captured pre-descent hover pose ('lift' terminal) --
    # The operator just confirmed the grip with [y]. Raise the held block straight
    # back up to T_liftback — the exact pose we were hovering at when the descent
    # started (captured in PHASE D). Returning here, rather than a blind +Z bump,
    # restores the SAME lateral position and orientation, so the camera re-sees the
    # scene from where it left off and the next navigation stage (find the green
    # block) starts from a known-good, block-clearing height.
    lbz_before = flange_pose_base(q_rad, zeta)[2, 3] * 1000.0
    lbz_target = T_liftback[2, 3] * 1000.0
    print("\n  [LIFT-BACK] returning to pre-descent hover (Z %.1f -> %.1f mm)..."
          % (lbz_before, lbz_target))
    ok, msg, q_after = move_to_pose(arm, T_liftback, zeta, q_rad)
    if not ok:
        print("  ! lift-back failed: %s" % msg)
        return False, q_rad
    q_rad = q_after
    print("  %s block lifted clear — back at hover pose." % color.upper())
    return True, q_rad


def safe_hold(arm, cap):
    """Teardown WITHOUT homing: release the gripper and hold position.

    Used when a stage fails or the operator aborts. The arm stays exactly where
    it is so you can inspect / recover by hand; we only make sure the suction is
    off so a held block isn't dragged around.
    """
    import robot_io
    print("\nStopping in place (NOT homing). Releasing gripper...")
    if robot_io.open_bio_gripper(arm) == 1:
        print("  gripper released.")
    robot_io.stop_bio_gripper(arm)
    print("  Arm held at current pose. Move it manually to recover.")


# ===========================================================================
# Orchestration
# ===========================================================================
def run(K, dist, zeta, T_ee_cam):
    import robot_io

    print("\nConnecting to robot at %s ..." % robot_io.ROBOT_IP)
    try:
        arm = robot_io.connect_arm()
    except RuntimeError as e:
        print(e)
        return

    try:
        cap = open_camera()
    except Exception as e:
        print("Could not open camera: %s" % e)
        arm.disconnect()
        return

    size_windows([WINDOW] + MASK_WINDOWS)
    create_threshold_trackbars()

    succeeded = False   # flipped True only when the whole task completes
    try:
        # ---------------- STEP 0: tune the colour masks ----------------------
        # Before any motion, let the operator dial in the RGB thresholds by eye.
        # This is where a green (or red) mask that "wasn't working" gets fixed —
        # drag each window's sliders until its block is cleanly isolated.
        if not tune_masks(cap, K, dist):
            return

        # ---------------- PRELUDE: manual positioning (feed only) -------------
        # The camera feed is live but the code drives NOTHING. Move the arm by
        # hand into a safe starting pose where BOTH blocks are visible, then press
        # [Enter]/[s] to begin the automated detection chain. [q] quits straight
        # away. While we wait, we continuously remember the LAST flange pose from
        # which both blocks were simultaneously detected — that becomes the
        # lift-back target after each pick, so we always return to a viewpoint
        # where the remaining block is still findable by vision.
        T_both_visible = None
        while True:
            q_probe = robot_io.read_joints_rad(arm)
            key = show_frame(cap,
                "FREE MOVE — position the arm manually. "
                "[Enter]/[s] start   [q] quit",
                K=K, dist_coeffs=dist)
            # Opportunistically grab the current overview pose whenever both
            # blocks are in view; keep the most recent one.
            T_both_visible = maybe_update_overview(
                T_both_visible, cap, q_probe, zeta, K, dist, T_ee_cam, label="")
            if key in (ord('\r'), ord('\n'), ord('s')):
                break
            if key == ord('q'):
                print("Aborted during free move.")
                return
        if T_both_visible is None:
            print("! NOTE: no both-blocks overview yet — Stage 0 gets another "
                  "chance before the cup flips down.")
        print("Free move ended — beginning automated sequence.")

        # ---------------- STAGE 0: start at home -----------------------------
        # Refresh the overview pose one more time here, while the wrist is STILL
        # in its starting attitude (not yet flipped cup-down). This is the last
        # chance to record a both-blocks-visible viewpoint before Phase A turns the
        # cup vertical and narrows the field of view onto a single block.
        q_probe = robot_io.read_joints_rad(arm)
        T_both_visible = maybe_update_overview(
            T_both_visible, cap, q_probe, zeta, K, dist, T_ee_cam,
            label="Stage 0 ")
        advanced, aborted = wait_next(
            "STAGE 0 — arm should be clear. [Enter] to begin",
            cap, arm, zeta, K, dist, T_ee_cam, None)
        if aborted:
            return
        q_rad = robot_io.read_joints_rad(arm)

        # ---------------- STAGE 1: find & pick the RED block -----------------
        print("\n" + "=" * 62)
        print("STAGE 1 — PICK RED BLOCK")
        print("=" * 62)
        ok, q_rad = approach_and_pick(arm, cap, zeta, K, dist, T_ee_cam, 'red',
                                      liftback_pose=T_both_visible)
        if not ok:
            print("Red pick incomplete — aborting safely.")
            return
        # Red is picked and lifted back to the overview pose. No intermediate
        # X-shift or set-down — we carry straight on to the green block next.

        # ---------------- STAGE 2: find & place the GREEN block --------------
        print("\n" + "=" * 62)
        print("STAGE 2 — TRACK & PLACE GREEN BLOCK")
        print("=" * 62)
        # From the lift-back overview pose, run the same five-phase camera
        # navigation (orient -> centre -> lower -> keypress cup-shift ->
        # fine-position), ending on [o] engage then [y] confirm. With
        # terminal='release' it sets the green block DOWN in place and retracts.
        ok, q_rad = approach_and_pick(
            arm, cap, zeta, K, dist, T_ee_cam, 'green', terminal='release',
            liftback_pose=T_both_visible)
        if not ok:
            print("Green placement incomplete — aborting safely.")
            return

        print("\n" + "=" * 62)
        print("PICK & PLACE COMPLETE")
        print("=" * 62)
        succeeded = True

    finally:
        cv2.destroyAllWindows()
        cap.release()
        try:
            if succeeded:
                # Full task done — return to a known-safe home pose.
                ok, msg = robot_io.go_home(arm)
                print("\nTask complete -> homed (%s)." % ("ok" if ok else msg))
            else:
                # Failed or aborted mid-way — HOLD POSITION, don't home.
                safe_hold(arm, cap)
        except Exception as e:
            print("warning: cleanup issue: %s" % e)
        arm.disconnect()


if __name__ == '__main__':
    sys.stdout.reconfigure(line_buffering=True)   # print as we go
    K, dist = load_camera_intrinsics()
    zeta, T_ee_cam = load_calibration_results()
    print("Camera intrinsics loaded (fx=%.1f, fy=%.1f)" % (K[0, 0], K[1, 1]))
    print("T_ee_cam translation: [%.3f, %.3f, %.3f] m" % tuple(T_ee_cam[:3, 3]))
    print("Tool offset (flange->cup): %.0f mm" % TOOL_OFFSET_MM)
    run(K, dist, zeta, T_ee_cam)
