"""Via-point path planning between Cartesian poses."""

import numpy as np

from reporting import banner, field, progress
from so3 import rotation_log_map


def _blend_linear(s):
    """Constant-speed blend."""
    return s


def _blend_cubic(s):
    """Hermite blend: zero velocity at both ends."""
    return s * s * (3.0 - 2.0 * s)


def _blend_quintic(s):
    """Quintic blend: zero velocity AND zero acceleration at both ends."""
    return s * s * s * (10.0 - 15.0 * s + 6.0 * s * s)


INTERPOLATORS = {'linear': _blend_linear, 'cubic': _blend_cubic, 'quintic': _blend_quintic}


def _skew(v):
    """3x3 cross-product matrix of v."""
    return np.array([[0.0, -v[2], v[1]],
                     [v[2], 0.0, -v[0]],
                     [-v[1], v[0], 0.0]])


def rotation_exp_map(omega):
    """Rotation matrix from a rotation vector (axis * angle), Rodrigues form."""

    omega = np.asarray(omega, dtype=float)
    theta = float(np.linalg.norm(omega))
    if theta < 1e-8:
        return np.eye(3) + _skew(omega)
    k = omega / theta
    K = _skew(k)
    return np.eye(3) + np.sin(theta) * K + (1.0 - np.cos(theta)) * (K @ K)


def interpolate_pose(T0, T1, blend):
    """Pose at blend factor `blend` in [0, 1] between T0 and T1."""

    T = np.eye(4)
    d_ori = rotation_log_map(T0[:3, :3].T @ T1[:3, :3])
    T[:3, :3] = T0[:3, :3] @ rotation_exp_map(blend * d_ori)
    T[:3, 3] = (1.0 - blend) * T0[:3, 3] + blend * T1[:3, 3]
    return T


def segment_poses(T0, T1, mode='cubic', n=25):
    """Sample one segment into n+1 poses (start included, end included)."""
    return [interpolate_pose(T0, T1, INTERPOLATORS[mode](s))
            for s in np.linspace(0.0, 1.0, n + 1)]


def plan_via_path(T_start, vias, T_end, mode='cubic', n_per_seg=25):
    """Chain segments start -> via1 -> ... -> viaN -> end."""

    keys = [T_start] + list(vias) + [T_end]
    poses = []
    for A, B in zip(keys[:-1], keys[1:]):
        poses += segment_poses(A, B, mode=mode, n=n_per_seg)[1:]
    return poses


def path_report(poses):
    """(total_length_mm, max_step_mm) along a pose list, the numbers that."""

    steps = [np.linalg.norm(poses[i + 1][:3, 3] - poses[i][:3, 3]) * 1000.0
             for i in range(len(poses) - 1)]
    return float(np.sum(steps)), float(np.max(steps))


if __name__ == '__main__':
    from ik_lite6 import pose_to_matrix
    T_start = pose_to_matrix([0.0, 0.0, 0.3], np.deg2rad([0, 180, 0]))
    T_via = pose_to_matrix([0.05, 0.1, 0.35], np.deg2rad([0, 180, 0]))
    T_end = pose_to_matrix([0.15, 0.05, 0.25], np.deg2rad([0, 180, 0]))
    banner('VIA-POINT PLANNER PROFILE COMPARISON')
    progress('2 segments, 25 waypoints each, same start/via/end poses.')
    field('profile', '   total_mm  max_step_mm')
    for mode in ('linear', 'cubic', 'quintic'):
        poses = plan_via_path(T_start, [T_via], T_end, mode=mode, n_per_seg=25)
        total, mx = path_report(poses)
        field(mode, '%9.2f %12.2f' % (total, mx))
    print()
    progress('linear has the shortest path but a velocity discontinuity at')
    progress('the via; cubic/quintic crowd waypoints near segment ends')
    progress('(smooth stops).')
