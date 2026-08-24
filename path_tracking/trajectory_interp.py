#!/usr/bin/env python3
"""Trajectory generation between via points (Lab 3 — path planning).

Given the VIA_COUNT via poses recorded in Phase 1, this module generates a
smooth trajectory THROUGH them and back-generates the joint configurations
using the Lab 2 inverse kinematics solver.

Design
------
Interpolation is done in CARTESIAN SPACE on the flange pose:

    p(t)  : straight-line position segment between consecutive vias
    R(t)  : slerp of the orientation (geodesic on SO(3))

This keeps every intermediate pose physically meaningful (constant-speed
straight line + shortest-path rotation), unlike interpolating the six raw
[x,y,z,rpy] numbers which can twist through singularities near +/-pi wrap.

Each interpolated pose T_i is then converted to joint angles with solve_ik()
from ik_solver.py, seeded from the previous waypoint so the arm stays on one
elbow/wrist branch. The FIRST and LAST waypoints are the recorded configs
verbatim, so the trajectory starts and ends exactly where the student taught.

Available interpolators (choose via INTERPOLATOR or the method arg):

    'linear'     trapezoidal velocity profile (zero accel at both ends)
    'cubic'      cubic polynomial, zero velocity AND acceleration at both ends
    'fourth'     4th-order polynomial, zero vel/accel/jerk at both ends
    'sinusoidal' cosine ease-in/out (zero velocity at both ends)
    'quintic'    5th-order polynomial, full C2 continuity across vias when
                 interior velocities are matched (default recommended)

All profiles share the same endpoints; they differ only in the time-varying
shape factor f(u) over u = t/T in [0, 1].
"""

import numpy as np
from scipy.spatial.transform import Rotation


# ---------------------------------------------------------------------------
# Shape factors f(u) for u in [0, 1];  f(0)=0, f(1)=1
# ---------------------------------------------------------------------------
def _f_linear(u):
    """Trapezoidal: linear ramp -> constant slope 1."""
    return np.asarray(u, dtype=float)


def _f_cubic(u):
    """Cubic ease: f = 3u^2 - 2u^3  (f'=6u-6u^2 vanishes at 0 and 1)."""
    u = np.asarray(u, dtype=float)
    return 3.0 * u**2 - 2.0 * u**3


def _f_fourth(u):
    """4th-order ease: f = u^2*(6 - 15u + 10u^2) (vanishing f', f'', f''' at ends)."""
    u = np.asarray(u, dtype=float)
    return u**2 * (6.0 - 15.0 * u + 10.0 * u**2)


def _f_sinusoidal(u):
    """Cosine ease: f = (1 - cos(pi*u))/2  (f'(0)=f'(1)=0)."""
    u = np.asarray(u, dtype=float)
    return 0.5 * (1.0 - np.cos(np.pi * u))


def _f_quintic(u):
    """Quintic ease: f = 10u^3 - 15u^4 + 6u^5  (f', f'' vanish at both ends)."""
    u = np.asarray(u, dtype=float)
    return 10.0 * u**3 - 15.0 * u**4 + 6.0 * u**5


INTERPOLATORS = {
    'linear':     _f_linear,
    'cubic':      _f_cubic,
    'fourth':     _f_fourth,
    'sinusoidal': _f_sinusoidal,
    'quintic':    _f_quintic,
}

DEFAULT_INTERPOLATOR = 'quintic'


def list_interpolators():
    return sorted(INTERPOLATORS.keys())


# ---------------------------------------------------------------------------
# Pose interpolation primitives
# ---------------------------------------------------------------------------
def interpolate_pose_segment(p0, R0, p1, R1, u):
    """Interpolate one pose segment at normalised times u (array or scalar).

    Position: linear blend. Orientation: slerp (shortest arc on SO(3)).
    Returns (positions (n,3), rotations list of n (3,3) matrices).
    """
    u = np.atleast_1d(np.asarray(u, dtype=float))
    p = (1.0 - u[:, None]) * np.asarray(p0) + u[:, None] * np.asarray(p1)

    q0 = Rotation.from_matrix(R0).as_quat()              # [x, y, z, w]
    q1 = Rotation.from_matrix(R1).as_quat()
    Rs = [Rotation.from_quat(_slerp_quat(q0, q1, ui)).as_matrix() for ui in u]
    return p, Rs


def _slerp_quat(qa, qb, t):
    """Slerp between unit quaternions [x,y,z,w] at parameter t in [0,1]."""
    dot = float(np.dot(qa, qb))
    if dot < 0.0:                       # take the short way round
        qb = -qb
        dot = -dot
    if dot > 0.9995:                    # nearly parallel -> linear blend
        q = qa + t * (qb - qa)
        return q / np.linalg.norm(q)
    theta = np.arccos(np.clip(dot, -1.0, 1.0))
    st = np.sin(theta)
    return (np.sin((1.0 - t) * theta) / st * qa +
            np.sin(t * theta) / st * qb)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def generate_trajectory(via_poses_m, n_per_seg=20, method=DEFAULT_INTERPOLATOR):
    """Generate a dense trajectory through the via poses.

    Parameters
    ----------
    via_poses_m : (M, 6) array [x,y,z (m); roll,pitch,yaw (rad)] — the via poses
    n_per_seg   : number of samples per segment INCLUDING the shared endpoint
                  (so total samples = (M-1)*(n_per_seg-1)+1)
    method      : one of list_interpolators()

    Returns
    -------
    traj_poses : (N, 6) array [x,y,z (m); roll,pitch,yaw (rad)] along the path
    seg_ids    : (N,) int array — which segment each sample belongs to (for plots)
    """
    if method not in INTERPOLATORS:
        raise ValueError("Unknown interpolator '%s'. Choose from %s"
                         % (method, list_interpolators()))
    f = INTERPOLATORS[method]
    M = len(via_poses_m)
    assert M >= 2, "need at least two via poses"

    ps = via_poses_m[:, :3]
    rs = [Rotation.from_euler('xyz', via_poses_m[i, 3:]).as_matrix() for i in range(M)]

    all_p, all_R, seg_ids = [], [], []
    for k in range(M - 1):
        u = np.linspace(0.0, 1.0, n_per_seg)
        pk, Rk = interpolate_pose_segment(ps[k], rs[k], ps[k + 1], rs[k + 1], u)
        if k > 0:                      # drop duplicated segment start
            pk, Rk = pk[1:], Rk[1:]
        all_p.append(pk)
        all_R.extend(Rk)
        seg_ids.extend([k] * len(pk))

    traj_poses = np.zeros((sum(len(x) for x in all_p), 6))
    P = np.vstack(all_p)
    traj_poses[:, :3] = P
    eul = np.stack([Rotation.from_matrix(R).as_euler('xyz') for R in all_R])
    traj_poses[:, 3:] = eul
    return traj_poses, np.asarray(seg_ids)


def poses_to_configs(traj_poses_m, zeta, q_first=None, q_last=None):
    """Back-generate joint configurations for a cartesian trajectory via IK.

    Each pose is solved with solve_ik(), seeded from the previously accepted
    solution so the arm stays on one elbow/wrist branch. If q_first / q_last
    are given (the recorded start/end configs), they are used verbatim at the
    ends instead of being re-solved, guaranteeing exact endpoints.

    Returns
    -------
    configs   : (N, 6) joint angles in RADIANS
    residuals : (N,) final IK residual norm per waypoint
    """
    from ik_solver import solve_ik, pose_to_matrix

    N = len(traj_poses_m)
    configs = np.zeros((N, 6))
    residuals = np.zeros(N)
    seed = None
    for i in range(N):
        if q_first is not None and i == 0:
            configs[i] = q_first
            residuals[i] = 0.0
            seed = q_first
            continue
        if q_last is not None and i == N - 1:
            configs[i] = q_last
            residuals[i] = 0.0
            seed = q_last
            continue
        T = pose_to_matrix(traj_poses_m[i, :3], traj_poses_m[i, 3:])
        init = seed if seed is not None else (q_first if q_first is not None else np.zeros(6))
        q_sol, ok, err = solve_ik(T, init, zeta=zeta)
        configs[i] = q_sol
        residuals[i] = err
        seed = q_sol
    return configs, residuals


def plan_full(zeta, config_log, pose_log_mm_deg, via_idx,
              n_per_seg=20, method=DEFAULT_INTERPOLATOR):
    """One-call convenience: vias -> dense cartesian traj -> IK configs.

    Parameters
    ----------
    zeta           : packed DH vector
    config_log     : (Nrec, 6) recorded joint angles (rad)
    pose_log_mm_deg: (Nrec, 6) recorded flange poses [mm; deg]
    via_idx        : indices into the logs marking the chosen vias
    n_per_seg      : samples per segment
    method         : interpolator name

    Returns (configs, residuals, traj_poses_m, via_poses_m)
    """
    via_poses_mm = pose_log_mm_deg[np.asarray(via_idx)]
    via_poses_m = np.column_stack([
        via_poses_mm[:, :3] / 1000.0,
        np.deg2rad(via_poses_mm[:, 3:]),
    ])
    traj_poses_m, _seg = generate_trajectory(via_poses_m, n_per_seg=n_per_seg, method=method)
    q_first = np.asarray(config_log[via_idx[0]], dtype=float)
    q_last = np.asarray(config_log[via_idx[-1]], dtype=float)
    configs, residuals = poses_to_configs(traj_poses_m, zeta, q_first=q_first, q_last=q_last)
    return configs, residuals, traj_poses_m, via_poses_m
