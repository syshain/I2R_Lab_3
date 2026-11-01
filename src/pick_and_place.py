"""Lab 3 stages 2-3: one-shot estimate -> planned transit -> g-gated motion.

SAFETY GATE: nothing on the arm moves unless [g] is HELD DOWN. Each frame
the key is held advances the arm one small step (KEY_STEP_MM); releasing it
stops the motion immediately. [SPACE] starts (and stops) the vacuum pump
while the arm is stood down during the descent/seating phases, [q] aborts.
Two liftoffs run on their own once the vacuum is toggled: the post-pick
lift starts automatically GRIPPER_SETTLE_S after [SPACE] engages the
vacuum, and the post-seating retreat starts automatically after [SPACE]
releases it, after which the arm is disabled. Both still abort on [q].

The previous behaviour — one-shot streaming of planned paths and the
key-gated final tuning jog — is kept below, commented out, so it can be
restored if needed.
"""


import numpy as np
from scipy.spatial.transform import Rotation

import robot

# Diagnostic cross-check of the FK chain against UFactory Studio: at the
# one-shot estimate, print the joints read and the FK tool-tip position so
# they can be compared with the Studio live readout for the same pose.
# Set False once the chain is verified.
ONE_SHOT_DIAG = True

from lab_config import (BLOCK_SIDE_MM, DOWN_RPY_DEG, HOVER_ABOVE_MM,
                        KEY_SPEED_PCT, KEY_STEP_MM, LIFT_CLEAR_MM, STREAM_DESCENT_MM_S,
                        STREAM_SPEED_MM_S, TABLE_Z_MM)
# rotation_log_map/solve_ik/move_to_pose/LEGEND/keygate_approach serve only
# the commented-out one-shot executor and key-gated approach further down;
# they stay imported so restoring those blocks needs no import edits.
from ik_lite6 import rotation_log_map, solve_ik
from key_control import (LEGEND, flange_for_tip, gated_path, streamed_path,
                         keygate_approach, tip_position)
from path_recorder import PathRecorder, save_path_plots
from motion import flange_pose_base, move_to_pose
from planners import INTERPOLATORS, path_report, plan_via_path
from reporting import banner, field, heading, progress
from ui_display import (load_camera_intrinsics, load_calibration_results,
                        load_gripper_calibration, open_camera, tune_masks,
                        wait_next)
from vision import one_shot_estimate


N_PER_SEG = 25              # planner samples per segment
TRANSIT_SPEED = 30          # % speed for planned transits
LIFT_SPEED = 20             # % speed for the post-pick lift and retreat


# ===========================================================================
# Student input
# ===========================================================================
def choose_profile():
    """Ask which interpolator to use; default linear. Returns the key.

    Linear keeps the velocity continuous through every segment end, so the
    g-gated steps stay even; cubic/quintic ease to a full stop at each via
    and crowd the waypoints there — the motion jolts under the dead-man
    executor.
    """
    names = sorted(INTERPOLATORS)
    print()
    progress("Interpolation profiles available:")
    for name in names:
        progress(name)
    while True:
        raw = input("Profile [Enter = linear]: ").strip().lower()
        if not raw:
            return 'linear'
        if raw in INTERPOLATORS:
            return raw
        progress("! pick one of: %s" % ", ".join(names))


def choose_executor():
    """Ask which path executor to run the planned poses with.

    Both work under the same dead-man gate; students compare the two:
    gated sends one KEY_STEP_MM position-mode step per held-[g] frame (each
    a rest-to-rest trapezoid, so the motion jolts at every step), streamed
    switches to servo motion mode (mode 1) and streams tick poses the
    controller blends into continuous motion.
    """

    print()
    progress("Path executor (the dead-man gate is the same for both):")
    progress("  gated    - position mode, one %g mm step per held-[g] frame"
             % KEY_STEP_MM)
    progress("  streamed - mode 1 servo stream, smooth continuous motion")
    while True:
        raw = input("Executor [Enter = gated]: ").strip().lower()
        if not raw:
            return 'gated'
        if raw in ('gated', 'streamed'):
            return raw
        progress("! pick one of: gated, streamed")


def prompt_vias(p_target_m, tool_d_m, recorder=None):
    """Read via points from stdin, as cup-TIP waypoints in mm (base frame)."""

    print()
    progress("Via points for the transit (cup-TIP x y z in mm, base frame).")
    progress("Target tip is at (%.0f, %.0f, %.0f) mm."
             % tuple(np.asarray(p_target_m) * 1000.0))
    progress("Leave the arm head-room: route AROUND obstacles, then come")
    progress("down onto the hover point. Blank line when done.")

    R_down = Rotation.from_euler('xyz', np.deg2rad(DOWN_RPY_DEG)).as_matrix()
    vias = []
    while True:
        line = input("  via %d (x y z, blank to finish): "
                     % (len(vias) + 1)).strip()
        if not line:
            break
        parts = line.replace(',', ' ').split()
        if len(parts) != 3:
            progress("! need exactly three numbers")
            continue
        try:
            xyz = np.asarray([float(p) for p in parts]) / 1000.0
        except ValueError:
            progress("! could not parse numbers")
            continue
        vias.append(flange_for_tip(xyz, R_down, tool_d_m))
        if recorder is not None:
            recorder.add_via(xyz)
    return vias


# ===========================================================================
# Path execution
# ===========================================================================
def transit_to_hover(arm, cap, K, dist, zeta, T_6_C, tool_d_m, q_rad,
                     p_tip_target_m, profile, recorder=None):
    """Plan the transit, stopping HOVER_ABOVE_MM above the target tip.

    Planning only — execution is gated or streamed in do_transfer.
    """

    R_down = Rotation.from_euler('xyz', np.deg2rad(DOWN_RPY_DEG)).as_matrix()
    p_hover = np.asarray(p_tip_target_m, dtype=float) \
        + np.array([0.0, 0.0, HOVER_ABOVE_MM / 1000.0])
    T_hover = flange_for_tip(p_hover, R_down, tool_d_m)

    vias = prompt_vias(p_hover, tool_d_m, recorder=recorder)
    T_start = flange_pose_base(q_rad, zeta)
    poses = plan_via_path(T_start, vias, T_hover, mode=profile,
                          n_per_seg=N_PER_SEG)
    length_mm, max_step_mm = path_report(poses)
    field("transit", "profile '%s': %d via(s), %d poses, path %.0f mm,"
          " max step %.2f mm"
          % (profile, len(vias), len(poses), length_mm, max_step_mm))

    return poses, vias


# ---------------------------------------------------------------------------
# OLD one-shot executor (kept for restore): streams the whole planned path
# autonomously, with no dead-man switch. To restore the old behaviour,
# uncomment this block and replace the gated_path calls in do_transfer with
# execute_path calls.
# ---------------------------------------------------------------------------
# def execute_path(arm, zeta, q_rad, poses, label, speed=TRANSIT_SPEED):
#     """Solve IK along every planned pose BEFORE moving, then stream."""
#
#     q_chain = np.asarray(q_rad, dtype=float).copy()
#     plan = []
#     for i, T in enumerate(poses):
#         q_sol, ok, err = solve_ik(T, q_chain, zeta)
#         if not ok:
#             T_end = flange_pose_base(q_sol, zeta)
#             progress("! IK failed at %s pose %d/%d — best it reached was"
#                      " %.1f mm, %.2f deg off the pose (resid %.2e) — nothing"
#                      " moved"
#                      % (label, i + 1, len(poses),
#                         np.linalg.norm(T_end[:3, 3] - T[:3, 3]) * 1000.0,
#                         np.degrees(np.linalg.norm(
#                             rotation_log_map(T_end[:3, :3] @ T[:3, :3].T))),
#                         err))
#             return q_rad, True
#         plan.append(q_sol)
#         q_chain = q_sol
#
#     for q in plan[:-1]:
#         ok, msg = robot.move_to_joints(arm, q, speed=speed, wait=False)
#         if not ok:
#             progress("! streaming rejected: %s" % msg)
#             return q_rad, True
#     ok, msg = robot.move_to_joints(arm, plan[-1], speed=speed, wait=True)
#     if not ok:
#         progress("! final move rejected: %s" % msg)
#         return q_rad, True
#     try:
#         return robot.read_joints_rad(arm), False
#     except Exception:
#         return plan[-1], False


# ===========================================================================
# One transfer: estimate -> transit -> g-gated descent -> actuate
# ===========================================================================
def one_shot_gate(arm, cap, K, dist, zeta, T_6_C, tool_d_m, color):
    """Wait for [Enter], take the one-shot estimate. Retry until seen."""

    while True:
        adv, abrt = wait_next(
            "block %s: aim the camera, [Enter] takes the one-shot estimate"
            % color, cap, arm, zeta, K, dist, T_6_C, color)
        if abrt:
            return None, True
        ret, frame = cap.read()
        q_rad = robot.read_joints_rad(arm)
        T_flange = flange_pose_base(q_rad, zeta)
        if ONE_SHOT_DIAG:
            progress("[diag] joints read (deg): %s"
                     % np.array2string(np.degrees(q_rad), precision=2,
                                       suppress_small=True))
            progress("[diag] FK flange   : x %+7.1f  y %+7.1f  z %+7.1f mm"
                     % tuple(T_flange[:3, 3] * 1000.0))
            p_tip = tip_position(T_flange, tool_d_m)
            progress("[diag] FK tool tip : x %+7.1f  y %+7.1f  z %+7.1f mm"
                     % tuple(p_tip * 1000.0))
            progress("[diag] cross-check the tip against UFactory Studio's "
                     "live readout at the same pose")
        est = one_shot_estimate(frame, color, K, dist, T_flange, T_6_C)
        if est is None:
            progress("! block not detected — check the mask windows / lighting")
            continue
        p = est['p_block_base']
        d_g = T_flange[:3, :3].T @ (p - tip_position(T_flange, tool_d_m))
        print()
        heading("ONE-SHOT ESTIMATE (block %s) — record for the report" % color)
        field("position", "x %+7.1f   y %+7.1f   z %+7.1f  mm (top-face centre)"
              % tuple(p * 1000.0))
        field("gripper", "lateral (%+.1f, %+.1f) mm, %+7.1f mm along the cup axis"
              % (d_g[0] * 1000.0, d_g[1] * 1000.0, d_g[2] * 1000.0))
        field("yaw", "%+.1f deg   (mod 90 deg for a square face)"
              % np.degrees(est['yaw_rad']))
        field("depth", "%.1f mm   confidence %.2f"
              % (est['depth_mm'], est['confidence']))
        field("view", "%.1f deg onto the block%s"
              % (est['tilt_deg'],
                 "   [tilt fit failed, naive depth kept]"
                 if est['naive_used'] else ""))
        field("image", "centre (%.0f, %.0f) px"
              % tuple(est['center_px']))
        print()
        return est, False


def do_transfer(arm, cap, K, dist, zeta, T_6_C, tool_d_m, color,
                dest_block, profile, executor, recorder=None):
    """Full pipeline: pick block `color`, place it on top of `dest_block`.

    executor is 'gated' or 'streamed' (choose_executor). With recorder
    (path_recorder.PathRecorder), each executed segment's tip path is
    recorded and the transit vias are attached to their segment for the
    data/trajectories/ plots.
    """

    def execute(q_in, poses, label, speed, allow_gripper=False,
                gripping=False, p_tip_target_m=None, color=None,
                auto=False):
        """One gated phase, run by the chosen executor.

        streamed: descent/seating slow to STREAM_DESCENT_MM_S so the cup
        approach stays gentle; everything else streams at
        STREAM_SPEED_MM_S. gated keeps the caller's % speed. auto=True
        passes through: the phase runs un-gated (post-vacuum liftoff).
        """
        if executor == 'streamed':
            s_speed = (STREAM_DESCENT_MM_S
                       if label in ('descent', 'seating')
                       else STREAM_SPEED_MM_S)
            return streamed_path(arm, cap, K, dist, zeta, T_6_C, tool_d_m,
                                 q_in, poses, label, speed=s_speed,
                                 allow_gripper=allow_gripper,
                                 gripping=gripping,
                                 p_tip_target_m=p_tip_target_m, color=color,
                                 recorder=recorder, auto=auto)
        return gated_path(arm, cap, K, dist, zeta, T_6_C, tool_d_m, q_in,
                          poses, label, speed=speed,
                          allow_gripper=allow_gripper, gripping=gripping,
                          p_tip_target_m=p_tip_target_m, color=color,
                          recorder=recorder, auto=auto)

    est, abrt = one_shot_gate(arm, cap, K, dist, zeta, T_6_C, tool_d_m, color)
    if abrt:
        return None, True
    p_target = est['p_block_base']

    # The arm was jogged in Manual Mode to aim the camera after the script
    # read its joints, so the transit MUST be planned from a fresh read:
    # a stale start pose sends the first planned poses back toward the old
    # pose — the arm moves away from the block before arcing to the hover.
    q_rad = robot.read_joints_rad(arm)
    T_pre = flange_pose_base(q_rad, zeta)     # pose before the one-shot descent;
                                              # the post-pick lift returns here

    field("transit", "hover %d mm above the estimate tip, profile '%s'"
          % (HOVER_ABOVE_MM, profile))
    poses, _ = transit_to_hover(arm, cap, K, dist, zeta, T_6_C, tool_d_m,
                                q_rad, p_target, profile, recorder=recorder)
    q_rad, abrt, _ = execute(q_rad, poses, 'transit', TRANSIT_SPEED)
    if abrt:
        return None, True

    # ------------------------------------------------------------------
    # OLD key-gated final tuning jog (kept for restore): the student jogged
    # in KEY_STEP_MM increments until the cup rested on the block, then
    # confirmed with [SPACE]; the vacuum engaged only after the confirm.
    # Replaced by the gated descent below.
    # ------------------------------------------------------------------
    # R_flange = flange_pose_base(q_rad, zeta)[:3, :3]
    # print()
    # heading("FINAL APPROACH — jog until the cup rests on the block")
    # progress(LEGEND)
    # confirmed, abrt, q_rad = keygate_approach(
    #     arm, cap, K, dist, zeta, T_6_C, tool_d_m, R_flange,
    #     p_target, color, mode='pick', q_rad=q_rad)
    # if abrt or not confirmed:
    #     return None, True
    #
    # if robot.open_gripper(arm) != 1:
    #     progress("! vacuum failed to engage — aborting this transfer")
    #     return None, True
    # progress("vacuum engaged.")
    # print()

    # Descent onto the block, still under the [g] dead-man switch: the cup
    # tip stops at the estimated top-face centre. [SPACE] engages the
    # vacuum once the cup rests on the block.
    R_flange = flange_pose_base(q_rad, zeta)[:3, :3]
    T_touch = flange_for_tip(p_target, R_flange, tool_d_m)
    T_touch[2, 3] = max(T_touch[2, 3], (TABLE_Z_MM + 2.0) / 1000.0)
    print()
    heading("DESCENT — hold [g] until the cup rests on the block, then [SPACE]")
    q_rad, abrt, gripping = execute(q_rad, [T_touch], 'descent',
                                    KEY_SPEED_PCT, allow_gripper=True,
                                    gripping=False, p_tip_target_m=p_target,
                                    color=color)
    if abrt or not gripping:
        robot.close_gripper(arm)      # better a drop than a drag
        if not gripping:
            progress("! vacuum never engaged — aborting this transfer")
        return None, True

    # Lift back to the pose the one-shot estimate was taken at — where the
    # descent started from — before any transit with the block held. This
    # liftoff runs ON ITS OWN (auto=True): the settle wait already ran the
    # moment [SPACE] engaged the vacuum, so the arm climbs GRIPPER_SETTLE_S
    # after the pick whether or not [g] is held; the dead-man gate resumes
    # at the transit below.
    q_rad, abrt, _ = execute(q_rad, [T_pre], 'lift', LIFT_SPEED,
                             allow_gripper=False, gripping=gripping,
                             auto=True)
    if abrt:
        robot.close_gripper(arm)
        return None, True

    # ---- place: destination is the top face of the destination block,
    # estimated (one-shot) from wherever the arm is jogged to aim ----
    est_d, abrt = one_shot_gate(arm, cap, K, dist, zeta, T_6_C, tool_d_m,
                                dest_block)
    if abrt:
        robot.close_gripper(arm)
        return None, True
    # The cup tip rests on the placed block's top face, one block height
    # above the destination block's top-face centre.
    p_dest_tip_m = (est_d['p_block_base']
                    + np.array([0.0, 0.0, BLOCK_SIDE_MM / 1000.0]))
    field("destination", "on top of block %s — tip (%.0f, %.0f, %.0f) mm"
          % (dest_block, p_dest_tip_m[0] * 1000.0,
             p_dest_tip_m[1] * 1000.0, p_dest_tip_m[2] * 1000.0))

    # The arm may have been jogged to aim the camera at the destination
    # block: plan the transit from a fresh read, as after the pick estimate.
    q_rad = robot.read_joints_rad(arm)

    field("transit", "to the destination, again stopping %d mm above"
          % HOVER_ABOVE_MM)
    poses, _ = transit_to_hover(arm, cap, K, dist, zeta, T_6_C, tool_d_m,
                                q_rad, p_dest_tip_m, profile,
                                recorder=recorder)
    q_rad, abrt, _ = execute(q_rad, poses, 'transit 2', TRANSIT_SPEED,
                             allow_gripper=False, gripping=gripping)
    if abrt:
        robot.close_gripper(arm)
        return None, True

    # ------------------------------------------------------------------
    # OLD key-gated seating jog (kept for restore): see the pick note above.
    # ------------------------------------------------------------------
    # R_flange = flange_pose_base(q_rad, zeta)[:3, :3]
    # print()
    # heading("SEATING — jog until the block rests on what is below it")
    # progress(LEGEND)
    # confirmed, abrt, q_rad = keygate_approach(
    #     arm, cap, K, dist, zeta, T_6_C, tool_d_m, R_flange,
    #     p_dest_tip_m, color, mode='place', q_rad=q_rad)
    # if abrt or not confirmed:
    #     robot.close_gripper(arm)
    #     return None, True

    # Seating descent, still gated: [SPACE] releases the vacuum once the
    # block rests on what is below it.
    R_flange = flange_pose_base(q_rad, zeta)[:3, :3]
    T_seat = flange_for_tip(p_dest_tip_m, R_flange, tool_d_m)
    T_seat[2, 3] = max(T_seat[2, 3], (TABLE_Z_MM + 2.0) / 1000.0)
    print()
    heading("SEATING — hold [g] until the block rests on what is below it,"
            " then [SPACE]")
    q_rad, abrt, gripping = execute(q_rad, [T_seat], 'seating',
                                    KEY_SPEED_PCT, allow_gripper=True,
                                    gripping=gripping,
                                    p_tip_target_m=p_dest_tip_m, color=color)
    if abrt:
        robot.close_gripper(arm)
        return None, True
    # Defensive second release: a no-op if [SPACE] already released the
    # vacuum during seating, but it guarantees the block is not dragged
    # along by the retreat.
    if robot.close_gripper(arm) != 1:
        progress("! vacuum failed to release")
    gripping = False

    seat_z_mm = float(tip_position(flange_pose_base(q_rad, zeta),
                                   tool_d_m)[2] * 1000.0)
    field("seat height", "%.1f mm (record this: the stack builds on it)"
          % seat_z_mm)

    # Retreat so the next transit starts clear of the stack — also on its
    # own (auto=True): [SPACE] released the vacuum during seating, so the
    # retreat runs un-gated, and the arm is disabled once clear.
    T_flange = flange_pose_base(q_rad, zeta)
    T_up = T_flange.copy()
    T_up[2, 3] += LIFT_CLEAR_MM / 1000.0
    q_rad, _, _ = execute(q_rad, [T_up], 'retreat', LIFT_SPEED,
                          allow_gripper=False, gripping=gripping, auto=True)
    robot.disable_arm(arm)
    field("robot", "disabled (controller stopped) — re-enable in UFactory "
          "Studio if needed")
    return {'seat_z_mm': seat_z_mm}, False


# ===========================================================================
# Orchestrator
# ===========================================================================
def run():
    K, dist = load_camera_intrinsics()
    zeta, T_6_C = load_calibration_results()
    tool_d_m, tool_src = load_gripper_calibration()
    field("tool offset", "%.2f mm  [%s]" % (tool_d_m * 1000.0, tool_src))

    arm = robot.connect_arm()
    cap = open_camera()
    recorder = PathRecorder()   # executed tip paths -> data/trajectories/
    try:
        if not tune_masks(cap, K, dist):
            progress("! mask tuning aborted — quitting.")
            return
        profile = choose_profile()
        executor = choose_executor()

        banner("TRANSFER — block 1 picked, placed on top of block 2")
        state, abrt = do_transfer(arm, cap, K, dist, zeta, T_6_C, tool_d_m,
                                  '1', '2', profile, executor,
                                  recorder=recorder)
        if abrt:
            print()
            progress("! aborted during the transfer — quitting.")
            return

        print()
        progress("Transfer complete. Record both one-shot estimates, the")
        progress("seat height and the path reports for the handout.")
    finally:
        save_path_plots(recorder)
        robot.stop_gripper(arm)
        robot.disable_arm(arm)
        cap.release()
        arm.disconnect()


if __name__ == '__main__':
    run()
