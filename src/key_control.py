"""Key-gated jogging + the g-gated (dead-man) path executor.

Two control styles live here. `gated_path` is the current safety gate used
by pick_and_place.py: the arm advances one small step per frame ONLY while
[g] is held down, and stops the moment it is released. `keygate_approach`
is the OLD final-approach jog (discrete KEY_STEP_MM steps per key press,
[SPACE] confirms) — no longer called, kept so the old behaviour can be
restored.
"""

import threading
import time

import numpy as np

from ik_lite6 import rotation_log_map, solve_ik
from lab_config import (GRIPPER_SETTLE_S, IK_JUMP_DEG, KEY_SPEED_PCT,
                        KEY_STEP_MM, STREAM_DRAIN_S, STREAM_SETTLE_TIMEOUT_S,
                        STREAM_TICK_S, STREAM_SPEED_MM_S, TABLE_Z_MM)
from motion import flange_pose_base, move_to_pose
from planners import interpolate_pose
from vision import compute_detection, project_base_to_uv
from ui_display import show_frame
from reporting import progress
import robot

KEY_DELTAS = {ord('w'): np.array([1.0, 0.0, 0.0]), ord('s'): np.array([-1.0, 0.0, 0.0]), ord('a'): np.array([0.0, 1.0, 0.0]), ord('d'): np.array([0.0, -1.0, 0.0]), ord('z'): np.array([0.0, 0.0, -1.0]), ord('c'): np.array([0.0, 0.0, 1.0])}
LEGEND = '[w/s]=+/-X [a/d]=+/-Y [z]=down [c]=up [SPACE]=confirm [q]=abort'

HOLD_KEY = ord('g')
HOLD_LEGEND = '[hold g]=move (release to stop) [SPACE]=gripper on/off [q]=abort'
HOLD_LEGEND_NO_GRIPPER = '[hold g]=move (release to stop) [q]=abort'

def flange_for_tip(p_tip_base, R_flange, tool_d_m):
    """Flange pose (4x4) that puts the cup tip at p_tip_base (m)."""

    T = np.eye(4)
    T[:3, :3] = R_flange
    T[:3, 3] = (np.asarray(p_tip_base, dtype=float)
                - R_flange @ np.array([0.0, 0.0, tool_d_m]))
    return T

def tip_position(T_flange, tool_d_m):
    """Current cup-tip position in the base frame (m), from the flange pose."""
    return T_flange[:3, 3] + T_flange[:3, :3] @ np.array([0.0, 0.0, tool_d_m])

def _fine_plan(poses, q_rad, zeta, label, step_mm):
    """Subdivide planned poses into ~step_mm steps and pre-solve their IK.

    Both executors share this: the whole path is tracked before anything
    moves, so a plan that cannot be followed aborts with the arm untouched.
    The interpolation starts from our FK of q_rad, the pose the executor
    was entered with.
    On IK failure the error is printed, with how many fine poses were
    rejected by IK (the failed step and every step after it, which the
    plan drops), and (None, None) is returned; otherwise (fine_poses,
    plan), with plan[i] the joints for fine_poses[i]. A joint jump beyond
    IK_JUMP_DEG between consecutive steps (a wrist flip) rejects the whole
    move the same way, before anything has moved.
    """

    fine = []
    prev_T = flange_pose_base(q_rad, zeta)
    for T in poses:
        d_mm = np.linalg.norm(T[:3, 3] - prev_T[:3, 3]) * 1000.0
        n = max(1, int(np.ceil(d_mm / step_mm)))
        for s in np.linspace(1.0 / n, 1.0, n):
            fine.append(interpolate_pose(prev_T, T, s))
        prev_T = T

    q_chain = np.asarray(q_rad, dtype=float).copy()
    plan = []
    for i, T in enumerate(fine):
        q_sol, ok, err = solve_ik(T, q_chain, zeta)
        if not ok:
            T_end = flange_pose_base(q_sol, zeta)
            progress("! IK failed at %s step %d/%d — best it reached was"
                     " %.1f mm, %.2f deg off the pose (resid %.2e) — nothing"
                     " moved; %d fine pose(s) rejected by IK"
                     % (label, i + 1, len(fine),
                        np.linalg.norm(T_end[:3, 3] - T[:3, 3]) * 1000.0,
                        np.degrees(np.linalg.norm(
                            rotation_log_map(T_end[:3, :3] @ T[:3, :3].T))),
                        err, len(fine) - i))
            return None, None
        # Raw difference, deliberately NOT modulo 360: two commands one
        # full turn apart are different physical motions, and the wrap-
        # equivalent reading would wave a near-full revolution (J6 from
        # +179 to -179) through as a 2-degree step.
        jump_deg = float(np.max(np.abs(np.degrees(q_sol - q_chain))))
        if jump_deg > IK_JUMP_DEG:
            progress("! joint jump %.1f deg at %s step %d/%d (wrist flip)"
                     " — move rejected, nothing moved; %d fine pose(s)"
                     " rejected"
                     % (jump_deg, label, i + 1, len(fine), len(fine) - i))
            return None, None
        plan.append(q_sol)
        q_chain = q_sol
    return fine, plan

def gated_path(arm, cap, K, dist_coeffs, zeta, T_ee_cam, tool_d_m, q_rad,
               poses, label, speed=KEY_SPEED_PCT, allow_gripper=False,
               gripping=False, p_tip_target_m=None, color=None,
               recorder=None, auto=False):
    """G-gated path execution: the arm moves ONLY while [g] is held down.

    The planned poses are subdivided into KEY_STEP_MM steps and their IK is
    solved BEFORE anything moves, so a plan that cannot be tracked aborts
    with the arm untouched. Each frame [g] is held advances one step;
    releasing it stops the motion immediately (dead-man switch). With
    allow_gripper, [SPACE] toggles the vacuum pump; engaging it waits
    GRIPPER_SETTLE_S for the suction to build before any further motion,
    so the post-pick lift never races the pump. A successful [SPACE] toggle
    also ends the phase (engaging the vacuum ends the descent, releasing it
    ends the seating): the automatic liftoff/retreat that follows starts
    right after the toggle, whether or not the planned steps are all done.

    With auto=True the phase runs un-gated (the automatic post-vacuum
    liftoff): every frame advances one step whether or not [g] is held,
    [SPACE] does nothing, and [q] still aborts.

    With recorder (path_recorder.PathRecorder), the executed cup-tip
    position is recorded after every step into a segment named `label`,
    for the 3D path plots in data/trajectories/.

    Returns (q_rad, aborted, gripping).
    """

    q_rad = np.asarray(q_rad, dtype=float).copy()
    if recorder is not None:
        recorder.start_segment(label)
    fine, plan = _fine_plan(poses, q_rad, zeta, label, KEY_STEP_MM)
    if plan is None:
        return q_rad, True, gripping

    aborted = done = False
    i = 0
    # Like the streamed executor, the phase stays open after the last
    # step when [SPACE] is live, so the toggle is never raced by the end
    # of the plan.
    while not aborted and not done and (i < len(plan) or allow_gripper):
        T_flange = flange_pose_base(q_rad, zeta)
        p_tip = tip_position(T_flange, tool_d_m)
        T_cam = T_flange @ T_ee_cam
        lines, marks = [], []
        det = None
        ok_f, frame = cap.read()
        if ok_f:
            if color is not None:
                det = compute_detection(frame, color, K, dist_coeffs,
                                        T_flange, T_ee_cam)
                if det is not None:
                    d_g = T_flange[:3, :3].T @ (det['p_block_base'] - p_tip)
                    lines.append('tip->block: lateral (%+.1f, %+.1f) mm, '
                                 '%+.1f mm along cup axis'
                                 % (d_g[0] * 1000.0, d_g[1] * 1000.0,
                                    d_g[2] * 1000.0))
            if p_tip_target_m is not None:
                for p_mark, lbl, bgr in ((p_tip, 'tip', (255, 0, 255)),
                                         (p_tip_target_m, 'target',
                                          (0, 255, 255))):
                    uv = project_base_to_uv(p_mark, T_cam, K, dist_coeffs)
                    if uv is not None:
                        marks.append((uv[0], uv[1], lbl, bgr))
        if auto:
            title = '%s: auto liftoff — %d/%d — [q] to abort' % (label, i,
                                                                 len(plan))
            hint = '[q]=abort'
        else:
            title = '%s: %d/%d — hold [g] to move' % (label, i, len(plan))
            hint = HOLD_LEGEND if allow_gripper else HOLD_LEGEND_NO_GRIPPER
        key = show_frame(
            cap, title,
            info_lines=lines, det=det, color=color, hint=hint,
            extra_marks=marks, K=K, dist_coeffs=dist_coeffs)
        if key == ord('q'):
            aborted = True
        elif allow_gripper and key == 32:
            # A successful vacuum toggle is the whole purpose of [SPACE] in
            # the phases that allow it (descent: engage, seating: release),
            # so it also ends the phase: the automatic liftoff/retreat that
            # follows must not wait for the remaining planned steps. A
            # failed toggle keeps the phase open so it can be retried.
            if gripping:
                if robot.close_gripper(arm) == 1:
                    gripping = False
                    progress('vacuum released.')
                    done = True
                else:
                    progress('! vacuum failed to release')
            else:
                if robot.open_gripper(arm) == 1:
                    gripping = True
                    progress('vacuum engaged — settling %.0f s before motion.'
                             % GRIPPER_SETTLE_S)
                    time.sleep(GRIPPER_SETTLE_S)
                    done = True
                else:
                    progress('! vacuum failed to engage')
        elif (auto or key == HOLD_KEY) and i < len(plan):      # auto, or held down -> step
            ok_mv, msg = robot.move_to_joints(arm, plan[i], speed=speed,
                                              wait=True)
            if ok_mv:
                try:
                    q_rad = robot.read_joints_rad(arm)
                except Exception:
                    q_rad = plan[i]
                i += 1
                if recorder is not None:
                    recorder.add_tip(tip_position(
                        flange_pose_base(q_rad, zeta), tool_d_m))
            else:
                progress('! move rejected: %s' % msg)
    return q_rad, aborted, gripping

def streamed_path(arm, cap, K, dist_coeffs, zeta, T_ee_cam, tool_d_m, q_rad,
                  poses, label, speed=STREAM_SPEED_MM_S, allow_gripper=False,
                  gripping=False, p_tip_target_m=None, color=None,
                  recorder=None, auto=False):
    """Mode-1 servo-streaming executor: same gate as gated_path, smooth motion.

    The plan is subdivided into ticks of speed*STREAM_TICK_S mm and their IK
    solved BEFORE anything moves (same pre-solve guarantee as gated_path).
    The arm is switched to servo motion mode (mode 1) and a fixed-rate thread
    streams one tick's joint solution per STREAM_TICK_S via set_servo_angle_j; the
    controller blends consecutive targets into continuous motion, so there is
    no rest-to-rest trapezoid per step and hence no jolt. The stream is
    primed before the first planned tick (the controller's own joint readout is
    sent once: servo commands move at the controller's maximum speed with
    no ramp, so tick 1 must start from where the controller really is),
    and after the last planned tick the stream keeps re-sending the final
    pose for STREAM_DRAIN_S, since the controller never queues servo
    commands and would otherwise halt short of the final pose. After the
    drain the executor waits for the controller to report the arm stopped
    (bounded by STREAM_SETTLE_TIMEOUT_S) before the joints are read, so
    the next phase is planned from where the arm actually is.

    The dead-man switch works by pausing the stream: releasing [g] stops new
    ticks, and the arm halts within one tick of the last commanded pose
    (at most speed*STREAM_TICK_S mm of overshoot). [SPACE] toggles the vacuum
    when allow_gripper — handled while the stream is paused, so only one
    thread talks to the arm at a time — and a successful toggle also ends
    the phase (engaging the vacuum ends the descent, releasing it ends the
    seating), so the automatic liftoff/retreat that follows starts right
    after the toggle. Control returns to mode 0 on exit,
    whatever happened. With auto=True the phase streams un-gated (the
    automatic post-vacuum liftoff): no [g] needed, [q] still aborts. With
    recorder, the commanded cup-tip position is recorded for every tick
    sent (a much denser trace than gated_path's).

    Returns (q_rad, aborted, gripping).
    """

    q_rad = np.asarray(q_rad, dtype=float).copy()
    if recorder is not None:
        recorder.start_segment(label)
    fine, plan = _fine_plan(poses, q_rad, zeta, label,
                            speed * STREAM_TICK_S)   # mm/s * s = mm per tick
    if plan is None:
        return q_rad, True, gripping

    if robot.set_control_mode(arm, robot.SERVO_MOTION_MODE) != 0:
        progress('! could not enter servo motion mode (mode 1) — arm untouched')
        return q_rad, True, gripping
    progress('%s: streaming %d ticks at %.0f mm/s (mode 1)%s'
             % (label, len(plan), speed,
                ' — auto liftoff, [q] to abort' if auto
                else ' — hold [g] to stream'))

    arm_lock = threading.Lock()
    state = {'i': 0, 'held': auto, 'run': True, 'error': None, 'done': False,
             'drained': False}

    # Prime the stream: send the JOINTS the controller is actually at
    # before the first planned tick. Servo commands have no speed ramp (the
    # SDK's speed argument is reserved; the controller dashes to each new
    # target at full speed), so without this the first tick is a max-speed
    # dash from wherever the controller sits — the jolt at the start of
    # every streamed phase. Joint targets prime exactly: the readout IS the
    # wire format, so there is no FK-vs-controller disagreement to dash
    # across, which is what removes the jolt the pose stream only dampened.
    try:
        prime = robot.read_joints_rad(arm)
    except Exception as exc:
        progress('! could not read the joints to prime the stream (%s)'
                 ' — leaving servo mode' % exc)
        robot.set_control_mode(arm, robot.POSITION_MODE)
        return q_rad, True, gripping
    if robot.stream_joints(arm, prime) != 0:
        progress('! priming tick rejected — leaving servo mode')
        robot.set_control_mode(arm, robot.POSITION_MODE)
        return q_rad, True, gripping

    def _stream():
        next_t = time.perf_counter()
        drain_deadline = None
        while state['run']:
            next_t += STREAM_TICK_S
            if state['held'] and state['i'] < len(plan):
                with arm_lock:
                    if not state['run']:
                        break
                    code = robot.stream_joints(arm, plan[state['i']])
                if code == 0:
                    if recorder is not None:
                        recorder.add_tip(tip_position(fine[state['i']],
                                                      tool_d_m))
                    state['i'] += 1
                else:
                    state['error'] = ('stream rejected at tick %d/%d '
                                      '(SDK code %d)'
                                      % (state['i'] + 1, len(plan), code))
                    state['run'] = False
                    break
            elif state['i'] >= len(plan):
                # Plan exhausted. The controller does NOT queue servo
                # commands: it chases only the last pose received and
                # halts the moment the stream stops, so going silent here
                # would strand the arm short of the final pose. Keep
                # re-sending the final pose for STREAM_DRAIN_S so the
                # controller actually arrives. The drain runs with [g]
                # released too — the motion was already fully commanded,
                # and the dead-man gate still kills any NEW motion
                # instantly — but never past the deadline.
                if drain_deadline is None:
                    drain_deadline = time.perf_counter() + STREAM_DRAIN_S
                with arm_lock:
                    if not state['run']:
                        break
                    code = robot.stream_joints(arm, plan[-1])
                if code != 0:
                    state['error'] = ('drain tick rejected at %d/%d '
                                      '(SDK code %d)'
                                      % (state['i'], len(plan), code))
                    state['run'] = False
                    break
                if time.perf_counter() >= drain_deadline:
                    state['drained'] = True
                    break
            delay = next_t - time.perf_counter()
            if delay > 0.0:
                time.sleep(delay)
            else:
                next_t = time.perf_counter() + STREAM_TICK_S

    thread = threading.Thread(target=_stream, name='servo-stream', daemon=True)
    thread.start()

    aborted = False
    # The phase stays open after the last tick when [SPACE] is live: the
    # cup reaches the block on the final tick, and ending the loop on
    # plan completion would race the [SPACE] keypress and lose it. It
    # also stays open until the drain has finished re-sending the final
    # pose (state['drained']), so the controller is actually at the
    # final pose before mode 0 is restored.
    while state['run'] and not aborted and not state['done'] \
            and (state['i'] < len(plan) or not state['drained']
                 or allow_gripper):
        # Refresh from the controller every frame: the streamed plan runs
        # ahead of q_rad otherwise, and the tip marker and the tip->block
        # numbers would sit at the phase's starting pose for the whole
        # flight. Under arm_lock so the read never races the stream thread.
        with arm_lock:
            try:
                q_rad = robot.read_joints_rad(arm)
            except Exception:
                pass
        T_flange = flange_pose_base(q_rad, zeta)
        p_tip = tip_position(T_flange, tool_d_m)
        T_cam = T_flange @ T_ee_cam
        lines, marks = [], []
        det = None
        ok_f, frame = cap.read()
        if ok_f:
            if color is not None:
                det = compute_detection(frame, color, K, dist_coeffs,
                                        T_flange, T_ee_cam)
                if det is not None:
                    d_g = T_flange[:3, :3].T @ (det['p_block_base'] - p_tip)
                    lines.append('tip->block: lateral (%+.1f, %+.1f) mm, '
                                 '%+.1f mm along cup axis'
                                 % (d_g[0] * 1000.0, d_g[1] * 1000.0,
                                    d_g[2] * 1000.0))
            if p_tip_target_m is not None:
                for p_mark, lbl, bgr in ((p_tip, 'tip', (255, 0, 255)),
                                         (p_tip_target_m, 'target',
                                          (0, 255, 255))):
                    uv = project_base_to_uv(p_mark, T_cam, K, dist_coeffs)
                    if uv is not None:
                        marks.append((uv[0], uv[1], lbl, bgr))
        if auto:
            title = '%s: auto liftoff — %d/%d — [q] to abort' % (
                label, state['i'], len(plan))
            hint = '[q]=abort'
        else:
            title = '%s: %d/%d — hold [g] to stream' % (label, state['i'],
                                                        len(plan))
            hint = HOLD_LEGEND if allow_gripper else HOLD_LEGEND_NO_GRIPPER
        key = show_frame(
            cap, title,
            info_lines=lines, det=det, color=color, hint=hint,
            extra_marks=marks, K=K, dist_coeffs=dist_coeffs)
        if key == ord('q'):
            aborted = True
        elif allow_gripper and key == 32:
            state['held'] = False               # pause before touching the arm
            with arm_lock:
                if gripping:
                    if robot.close_gripper(arm) == 1:
                        gripping = False
                        progress('vacuum released.')
                        state['done'] = True    # toggle ends the phase
                    else:
                        progress('! vacuum failed to release')
                else:
                    if robot.open_gripper(arm) == 1:
                        gripping = True
                        progress('vacuum engaged — settling %.0f s before'
                                 ' motion.' % GRIPPER_SETTLE_S)
                        time.sleep(GRIPPER_SETTLE_S)
                        state['done'] = True    # toggle ends the phase
                    else:
                        progress('! vacuum failed to engage')
        else:
            state['held'] = auto or (key == HOLD_KEY)   # auto -> always stream

    state['held'] = False
    state['run'] = False
    thread.join()
    # Let the last ticks reach the controller, then wait for the arm to
    # actually STOP before the joints are read: the controller keeps
    # blending after the final tick, so joints read mid-motion would plan
    # the next phase from a pose the arm has already left (the jolt and
    # the offset). Bounded by STREAM_SETTLE_TIMEOUT_S so a controller that
    # never reports idle cannot hang the phase. Same moving test as
    # connect_arm: is_moving, or controller state 1 (in motion).
    time.sleep(0.2)
    settle_deadline = time.perf_counter() + STREAM_SETTLE_TIMEOUT_S
    while time.perf_counter() < settle_deadline:
        try:
            stopped = not arm.get_is_moving() and arm.get_state() != 1
        except Exception:
            stopped = True
        if stopped:
            break
        time.sleep(0.05)
    if state['error']:
        progress('! %s' % state['error'])
        try:
            progress('controller: state %s, mode %s, error %s, warn %s'
                     % (arm.state, arm.mode, arm.error_code,
                        arm.warn_code))
        except Exception:
            progress('controller status unreadable')
        aborted = True
    try:
        q_rad = robot.read_joints_rad(arm)
    except Exception:
        pass
    robot.set_control_mode(arm, robot.POSITION_MODE)
    return q_rad, aborted, gripping

def keygate_approach(arm, cap, K, dist_coeffs, zeta, T_ee_cam, tool_d_m, R_flange, p_tip_target_m, color, mode, q_rad):
    """Interactive base-frame jog from the hover point down to contact.

    OLD behaviour, kept for restore: pick_and_place.py now descends with
    gated_path instead (hold [g] to move, release to stop).
    """

    z_floor_m = (TABLE_Z_MM + 2.0) / 1000.0
    verb = 'CONTACT' if mode == 'pick' else 'SEATED'
    confirmed = aborted = False
    while not confirmed and not aborted:
        T_flange = flange_pose_base(q_rad, zeta)
        p_tip = tip_position(T_flange, tool_d_m)
        T_cam = T_flange @ T_ee_cam

        ok, frame = cap.read()
        det = None
        lines, marks = [], []
        if ok:
            det = compute_detection(frame, color, K, dist_coeffs,
                                    T_flange, T_ee_cam)
            if det is not None:
                d_g = T_flange[:3, :3].T @ (det['p_block_base'] - p_tip)
                lines.append('tip->block: lateral (%+.1f, %+.1f) mm, '
                             '%+.1f mm along cup axis'
                             % (d_g[0] * 1000.0, d_g[1] * 1000.0,
                                d_g[2] * 1000.0))
            for p_mark, label, bgr in ((p_tip, 'tip', (255, 0, 255)),
                                       (p_tip_target_m, 'target', (0, 255, 255))):
                uv = project_base_to_uv(p_mark, T_cam, K, dist_coeffs)
                if uv is not None:
                    marks.append((uv[0], uv[1], label, bgr))

        key = show_frame(
            cap, 'FINAL APPROACH (%s): jog until %s' % (mode.upper(), verb),
            info_lines=lines, det=det, color=color,
            hint='%s   step %d mm' % (LEGEND, KEY_STEP_MM),
            extra_marks=marks, K=K, dist_coeffs=dist_coeffs)

        if key in (32, 13):                       # SPACE (Enter also confirms)
            confirmed = True
        elif key == ord('q'):
            aborted = True
        elif key in KEY_DELTAS:
            p_new = p_tip + KEY_DELTAS[key] * (KEY_STEP_MM / 1000.0)
            p_new[2] = max(p_new[2], z_floor_m)
            T_target = flange_for_tip(p_new, R_flange, tool_d_m)
            ok_mv, msg, q_new = move_to_pose(arm, T_target, zeta, q_rad,
                                             speed=KEY_SPEED_PCT)
            if ok_mv:
                try:
                    q_rad = robot.read_joints_rad(arm)
                except Exception:
                    q_rad = q_new if q_new is not None else q_rad
            else:
                progress('! move rejected: %s' % msg)
    return confirmed, aborted, q_rad

