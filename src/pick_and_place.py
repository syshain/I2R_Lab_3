"""Lab 3 stages 2-3: one-shot estimate -> planned transit -> key-gated approach."""


import numpy as np
from scipy.spatial.transform import Rotation

import robot_io

# Diagnostic cross-check of the FK chain against UFactory Studio: at the
# one-shot estimate, print the joints read and the FK tool-tip position so
# they can be compared with the Studio live readout for the same pose.
# Set False once the chain is verified.
ONE_SHOT_DIAG = True

from lab_config import (BLOCK_SIDE_MM, DOWN_RPY_DEG, HOVER_ABOVE_MM,
                        LIFT_CLEAR_MM)
from ik_solver import rotation_log_map, solve_ik
from key_control import LEGEND, flange_for_tip, keygate_approach, tip_position
from motion import flange_pose_base, move_to_pose
from planners import INTERPOLATORS, path_report, plan_via_path
from ui_display import (load_camera_intrinsics, load_calibration_results,
                        load_gripper_calibration, open_camera, tune_masks,
                        wait_next)
from vision import one_shot_estimate


N_PER_SEG = 25              # planner samples per segment
TRANSIT_SPEED = 30          # % speed for planned transits
LIFT_SPEED = 20             # % speed for the post-pick lift


# ===========================================================================
# Student input
# ===========================================================================
def choose_profile():
    """Ask which interpolator to use; default cubic. Returns the key."""
    names = sorted(INTERPOLATORS)
    print("\nInterpolation profiles available:")
    for name in names:
        print("  %-8s" % name)
    while True:
        raw = input("Profile [Enter = cubic]: ").strip().lower()
        if not raw:
            return 'cubic'
        if raw in INTERPOLATORS:
            return raw
        print("  ! pick one of: %s" % ", ".join(names))


def prompt_vias(p_target_m, tool_d_m):
    """Read via points from stdin, as cup-TIP waypoints in mm (base frame)."""

    print("\nVia points for the transit (cup-TIP x y z in mm, base frame).")
    print("Target tip is at (%.0f, %.0f, %.0f) mm."
          % tuple(np.asarray(p_target_m) * 1000.0))
    print("Leave the arm head-room: route AROUND obstacles, then come down")
    print("onto the hover point. Blank line when done.\n")

    R_down = Rotation.from_euler('xyz', np.deg2rad(DOWN_RPY_DEG)).as_matrix()
    vias = []
    while True:
        line = input("  via %d (x y z, blank to finish): "
                     % (len(vias) + 1)).strip()
        if not line:
            break
        parts = line.replace(',', ' ').split()
        if len(parts) != 3:
            print("  ! need exactly three numbers")
            continue
        try:
            xyz = np.asarray([float(p) for p in parts]) / 1000.0
        except ValueError:
            print("  ! could not parse numbers")
            continue
        vias.append(flange_for_tip(xyz, R_down, tool_d_m))
    return vias


# ===========================================================================
# Path execution
# ===========================================================================
def execute_path(arm, zeta, q_rad, poses, label, speed=TRANSIT_SPEED):
    """Solve IK along every planned pose BEFORE moving, then stream."""

    q_chain = np.asarray(q_rad, dtype=float).copy()
    plan = []
    for i, T in enumerate(poses):
        q_sol, ok, err = solve_ik(T, q_chain, zeta)
        if not ok:
            T_end = flange_pose_base(q_sol, zeta)
            print("  ! IK failed at %s pose %d/%d — best it reached was"
                  " %.1f mm, %.2f deg off the pose (resid %.2e) — nothing"
                  " moved"
                  % (label, i + 1, len(poses),
                     np.linalg.norm(T_end[:3, 3] - T[:3, 3]) * 1000.0,
                     np.degrees(np.linalg.norm(
                         rotation_log_map(T_end[:3, :3] @ T[:3, :3].T))),
                     err))
            return q_rad, True
        plan.append(q_sol)
        q_chain = q_sol

    for q in plan[:-1]:
        ok, msg = robot_io.move_to_joints(arm, q, speed=speed, wait=False)
        if not ok:
            print("  ! streaming rejected: %s" % msg)
            return q_rad, True
    ok, msg = robot_io.move_to_joints(arm, plan[-1], speed=speed, wait=True)
    if not ok:
        print("  ! final move rejected: %s" % msg)
        return q_rad, True
    try:
        return robot_io.read_joints_rad(arm), False
    except Exception:
        return plan[-1], False


def transit_to_hover(arm, cap, K, dist, zeta, T_6_C, tool_d_m, q_rad,
                     p_tip_target_m, profile):
    """Plan and fly the transit, stopping HOVER_ABOVE_MM above the target tip."""

    R_down = Rotation.from_euler('xyz', np.deg2rad(DOWN_RPY_DEG)).as_matrix()
    p_hover = np.asarray(p_tip_target_m, dtype=float) \
        + np.array([0.0, 0.0, HOVER_ABOVE_MM / 1000.0])
    T_hover = flange_for_tip(p_hover, R_down, tool_d_m)

    vias = prompt_vias(p_hover, tool_d_m)
    T_start = flange_pose_base(q_rad, zeta)
    poses = plan_via_path(T_start, vias, T_hover, mode=profile,
                          n_per_seg=N_PER_SEG)
    length_mm, max_step_mm = path_report(poses)
    print("  %s: %d via(s), %d poses, path %.0f mm, max step %.2f mm"
          % (profile, len(vias), len(poses), length_mm, max_step_mm))

    return execute_path(arm, zeta, q_rad, poses, label='transit')


# ===========================================================================
# One transfer: estimate -> transit -> keygate -> actuate
# ===========================================================================
def one_shot_gate(arm, cap, K, dist, zeta, T_6_C, tool_d_m, color):
    """Wait for [Enter], take the one-shot estimate. Retry until seen."""

    while True:
        adv, abrt = wait_next(
            "%s block: aim the camera, [Enter] takes the one-shot estimate"
            % color.upper(), cap, arm, zeta, K, dist, T_6_C, color)
        if abrt:
            return None, True
        ret, frame = cap.read()
        q_rad = robot_io.read_joints_rad(arm)
        T_flange = flange_pose_base(q_rad, zeta)
        if ONE_SHOT_DIAG:
            print("  [diag] joints read (deg): %s"
                  % np.array2string(np.degrees(q_rad), precision=2,
                                    suppress_small=True))
            print("  [diag] FK flange   : x %+7.1f  y %+7.1f  z %+7.1f mm"
                  % tuple(T_flange[:3, 3] * 1000.0))
            p_tip = tip_position(T_flange, tool_d_m)
            print("  [diag] FK tool tip : x %+7.1f  y %+7.1f  z %+7.1f mm"
                  % tuple(p_tip * 1000.0))
            print("  [diag] cross-check the tip against UFactory Studio's "
                  "live readout at the same pose")
        est = one_shot_estimate(frame, color, K, dist, T_flange, T_6_C)
        if est is None:
            print("  block not detected — check the mask windows / lighting")
            continue
        p = est['p_block_base']
        d_g = T_flange[:3, :3].T @ (p - tip_position(T_flange, tool_d_m))
        print("\n  ONE-SHOT ESTIMATE (%s) — record for the report:" % color)
        print("  position : x %+7.1f   y %+7.1f   z %+7.1f  mm (top-face centre)"
              % tuple(p * 1000.0))
        print("  gripper  : lateral (%+.1f, %+.1f) mm, %+7.1f mm along the cup axis"
              % (d_g[0] * 1000.0, d_g[1] * 1000.0, d_g[2] * 1000.0))
        print("  yaw      : %+7.1f deg   (mod 90 deg for a square face)"
              % np.degrees(est['yaw_rad']))
        print("  depth    : %7.1f mm   confidence %.2f"
              % (est['depth_mm'], est['confidence']))
        print("  view     : %5.1f deg onto the block%s"
              % (est['tilt_deg'],
                 "   [tilt fit failed, naive depth kept]"
                 if est['naive_used'] else ""))
        print("  image    : centre (%.0f, %.0f) px\n"
              % tuple(est['center_px']))
        return est, False


def do_transfer(arm, cap, K, dist, zeta, T_6_C, tool_d_m, color,
                dest_block, profile):
    """Full pipeline: pick `color`, place it on top of `dest_block`."""

    est, abrt = one_shot_gate(arm, cap, K, dist, zeta, T_6_C, tool_d_m, color)
    if abrt:
        return None, True
    p_target = est['p_block_base']

    # The arm was jogged in Manual Mode to aim the camera after the script
    # read its joints, so the transit MUST be planned from a fresh read:
    # a stale start pose sends the first planned poses back toward the old
    # pose — the arm moves away from the block before arcing to the hover.
    q_rad = robot_io.read_joints_rad(arm)
    T_pre = flange_pose_base(q_rad, zeta)     # pose before the one-shot descent;
                                              # the post-pick lift returns here

    print("  transit: hover %d mm above the estimate tip, profile '%s'"
          % (HOVER_ABOVE_MM, profile))
    q_rad, abrt = transit_to_hover(arm, cap, K, dist, zeta, T_6_C, tool_d_m,
                                   q_rad, p_target, profile)
    if abrt:
        return None, True

    R_flange = flange_pose_base(q_rad, zeta)[:3, :3]
    print("\n  FINAL APPROACH — jog until the cup rests on the block:")
    print("  %s" % LEGEND)
    confirmed, abrt, q_rad = keygate_approach(
        arm, cap, K, dist, zeta, T_6_C, tool_d_m, R_flange,
        p_target, color, mode='pick', q_rad=q_rad)
    if abrt or not confirmed:
        return None, True

    if robot_io.open_gripper(arm) != 1:
        print("  ! vacuum failed to engage — aborting this transfer")
        return None, True
    print("  vacuum engaged.\n")

    # Lift back to the pose the one-shot estimate was taken at — where the
    # descent started from — before any transit with the block held.
    q_rad, abrt = execute_path(arm, zeta, q_rad, [T_pre], label='lift',
                               speed=LIFT_SPEED)
    if abrt:
        robot_io.close_gripper(arm)      # better a drop than a drag
        return None, True

    # ---- place: destination is the top face of the destination block,
    # estimated (one-shot) from wherever the arm is jogged to aim ----
    est_d, abrt = one_shot_gate(arm, cap, K, dist, zeta, T_6_C, tool_d_m,
                                dest_block)
    if abrt:
        robot_io.close_gripper(arm)
        return None, True
    # The cup tip rests on the placed block's top face, one block height
    # above the destination block's top-face centre.
    p_dest_tip_m = (est_d['p_block_base']
                    + np.array([0.0, 0.0, BLOCK_SIDE_MM / 1000.0]))
    print("  destination: on top of the %s block — tip (%.0f, %.0f, %.0f) mm"
          % (dest_block, p_dest_tip_m[0] * 1000.0,
             p_dest_tip_m[1] * 1000.0, p_dest_tip_m[2] * 1000.0))

    # The arm may have been jogged to aim the camera at the destination
    # block: plan the transit from a fresh read, as after the pick estimate.
    q_rad = robot_io.read_joints_rad(arm)

    print("  transit to the destination, again stopping %d mm above:"
          % HOVER_ABOVE_MM)
    q_rad, abrt = transit_to_hover(arm, cap, K, dist, zeta, T_6_C, tool_d_m,
                                   q_rad, p_dest_tip_m, profile)
    if abrt:
        robot_io.close_gripper(arm)
        return None, True

    R_flange = flange_pose_base(q_rad, zeta)[:3, :3]
    print("\n  SEATING — jog until the block rests on what is below it:")
    print("  %s" % LEGEND)
    confirmed, abrt, q_rad = keygate_approach(
        arm, cap, K, dist, zeta, T_6_C, tool_d_m, R_flange,
        p_dest_tip_m, color, mode='place', q_rad=q_rad)
    if abrt or not confirmed:
        robot_io.close_gripper(arm)
        return None, True

    if robot_io.close_gripper(arm) != 1:
        print("  ! vacuum failed to release")
    seat_z_mm = float(tip_position(flange_pose_base(q_rad, zeta),
                                   tool_d_m)[2] * 1000.0)
    print("  released at tip height %.1f mm (record this: the stack builds"
          " on it)" % seat_z_mm)

    # Retreat so the next transit starts clear of the stack.
    T_flange = flange_pose_base(q_rad, zeta)
    T_up = T_flange.copy()
    T_up[2, 3] += LIFT_CLEAR_MM / 1000.0
    q_rad, _ = execute_path(arm, zeta, q_rad, [T_up], label='retreat',
                            speed=LIFT_SPEED)
    return {'seat_z_mm': seat_z_mm}, False


# ===========================================================================
# Orchestrator
# ===========================================================================
def run():
    K, dist = load_camera_intrinsics()
    zeta, T_6_C = load_calibration_results()
    tool_d_m, tool_src = load_gripper_calibration()
    print("Tool offset: %.2f mm  [%s]" % (tool_d_m * 1000.0, tool_src))

    arm = robot_io.connect_arm()
    cap = open_camera()
    try:
        if not tune_masks(cap, K, dist):
            print("Mask tuning aborted — quitting.")
            return
        profile = choose_profile()

        print("\n" + "=" * 62)
        print("TRANSFER — GREEN picked, placed on top of the RED block")
        print("=" * 62)
        state, abrt = do_transfer(arm, cap, K, dist, zeta, T_6_C, tool_d_m,
                                  'green', 'red', profile)
        if abrt:
            print("\nAborted during the transfer — quitting.")
            return

        print("\nTransfer complete. Record both one-shot estimates, the")
        print("seat height and the path reports for the handout.")
    finally:
        robot_io.stop_gripper(arm)
        cap.release()
        arm.disconnect()


if __name__ == '__main__':
    run()
