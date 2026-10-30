"""Key-gated final approach: the last millimetres are driven by hand."""

import numpy as np
from lab_config import KEY_STEP_MM, KEY_SPEED_PCT, TABLE_Z_MM, HOVER_ABOVE_MM
from motion import flange_pose_base, move_to_pose
from vision import compute_detection, project_base_to_uv
from ui_display import show_frame
from reporting import progress
import robot
KEY_DELTAS = {ord('w'): np.array([1.0, 0.0, 0.0]), ord('s'): np.array([-1.0, 0.0, 0.0]), ord('a'): np.array([0.0, 1.0, 0.0]), ord('d'): np.array([0.0, -1.0, 0.0]), ord('z'): np.array([0.0, 0.0, -1.0]), ord('c'): np.array([0.0, 0.0, 1.0])}
LEGEND = '[w/s]=+/-X [a/d]=+/-Y [z]=down [c]=up [SPACE]=confirm [q]=abort'

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

def keygate_approach(arm, cap, K, dist_coeffs, zeta, T_ee_cam, tool_d_m, R_flange, p_tip_target_m, color, mode, q_rad):
    """Interactive base-frame jog from the hover point down to contact."""

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
