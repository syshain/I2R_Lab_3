"""Numerical inverse kinematics for the UFACTORY Lite 6."""

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from fk_lite6 import fk_from_zeta

LITE6_JOINT_LIMITS_DEG = np.array([[-360.0, 360.0], [-150.0, 150.0], [-3.5, 300.0], [-360.0, 360.0], [-124.0, 124.0], [-360.0, 360.0]])
LITE6_JOINT_LIMITS_RAD = np.deg2rad(LITE6_JOINT_LIMITS_DEG)

# Nominal (UFACTORY handbook) rows: [theta_offset_deg, d_mm, alpha_deg, a_mm].
DH_NOMINAL_RAW = [
    [0.0, 243.3, 0.0, 0.0],      # joint 1
    [-90.0, 0.0, -90.0, 0.0],    # joint 2
    [-90.0, 0.0, 180.0, 200.0],  # joint 3
    [0.0, 227.6, 90.0, 87.0],    # joint 4
    [0.0, 0.0, 90.0, 0.0],       # joint 5
    [0.0, 61.5, -90.0, 0.0],     # joint 6
]

# A solution counts as converged only when the POSE residual is small, not
# merely the step: a tiny damped step far from an unreachable target must
# stay NOT CONVERGED (the Lab 1 lesson).
RESIDUAL_TOL = 1e-3   # ~1 mm-equivalent in the [m; rad] residual


def get_nominal_zeta():
    """Factory Lite 6 modified (Craig) D-H params as a packed zeta
    [a|alpha|d|theta_off], lengths in mm + angles in rad, matching
    fk_from_zeta and the calibrated zeta_stage2."""

    theta_off = np.deg2rad([r[0] for r in DH_NOMINAL_RAW])
    d = np.array([r[1] for r in DH_NOMINAL_RAW])
    alpha = np.deg2rad([r[2] for r in DH_NOMINAL_RAW])
    a = np.array([r[3] for r in DH_NOMINAL_RAW])
    return np.concatenate([a, alpha, d, theta_off])


def rotation_log_map(R):
    """SO(3) log map: omega in R^3 with exp(hat(omega)) = R."""
    cosang = float(np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0))
    theta = np.arccos(cosang)
    if theta < 1e-8:
        skew = (R - R.T) / 2.0
        return np.array([skew[2, 1], skew[0, 2], skew[1, 0]])
    if cosang < -1.0 + 1e-12:
        # Rotation by ~pi: sin(theta) -> 0 kills the scale, so recover the
        # axis as the +1 eigenvector of the symmetric part instead.
        w, V = np.linalg.eigh((R + R.T) / 2.0)
        axis = V[:, int(np.argmax(w))]
        return np.pi * (-axis if axis[0] < 0.0 else axis)
    scale = theta / (2.0 * np.sin(theta))
    return np.array([R[2, 1] - R[1, 2],
                     R[0, 2] - R[2, 0],
                     R[1, 0] - R[0, 1]]) * scale


def ik_objective(q, T_target, zeta):
    """6-vector pose error [pos_err (m); orient_err (axis-angle, rad)]."""
    T_cur = fk_from_zeta(q, zeta)
    e_pos = T_cur[:3, 3] - T_target[:3, 3]
    e_ori = rotation_log_map(T_cur[:3, :3] @ T_target[:3, :3].T)
    return np.concatenate([e_pos, e_ori])


def solve_ik(T_target, q0, zeta=None, joint_limits=LITE6_JOINT_LIMITS_RAD, verbose=False):
    """Bounded-Levenberg-Marquardt IK."""

    if zeta is None:
        zeta = get_nominal_zeta()
    q0 = np.asarray(q0, dtype=float).copy()
    lb = None if joint_limits is None else joint_limits[:, 0]
    ub = None if joint_limits is None else joint_limits[:, 1]

    res = least_squares(ik_objective, q0, args=(T_target, zeta),
                        bounds=(lb, ub), method='trf',
                        xtol=1e-12, ftol=1e-12, gtol=1e-12, max_nfev=500)
    err = float(np.linalg.norm(res.fun))
    if verbose:
        print("  ik: resid %.2e  pos %.2f mm  ori %.3f deg"
              % (err, np.linalg.norm(res.fun[:3]) * 1000.0,
                 np.degrees(np.linalg.norm(res.fun[3:]))))
    return res.x, (err < RESIDUAL_TOL), err


def pose_to_matrix(pos_xyz, euler_xyz):
    """4x4 transform from pos_xyz (m) + euler_xyz [r,p,y] (rad, XYZ intrinsic)."""
    T = np.eye(4)
    T[:3, :3] = Rotation.from_euler('xyz', np.asarray(euler_xyz, dtype=float)).as_matrix()
    T[:3, 3] = np.asarray(pos_xyz, dtype=float)
    return T


if __name__ == '__main__':
    zeta = get_nominal_zeta()
    lim = LITE6_JOINT_LIMITS_RAD
    tests = [('Home', np.deg2rad([-180, 0.0, 4.0, 0.0, -45.0, 0.0]), np.array([0.1, -1.2, 0.1, 0.1, 0.1, 0.1])), ('ElbowDown', np.deg2rad([-180, -18.8, 4.0, 0.0, -45.0, 0.0]), np.array([0.1, -0.8, 0.3, 0.1, 0.1, 0.1])), ('Complex', np.deg2rad([-302.2, -19.9, 37.2, -207.4, -108.7, -39.9]), np.array([0.3, -0.8, 0.5, 0.2, 0.3, 0.1]))]
    print('=== Python IK self-test (nominal Lite 6 DH) ===\n')
    for name, q_true, q0 in tests:
        T_t = fk_from_zeta(q_true, zeta, n=6)
        q_sol, ok, en = solve_ik(T_t, q0, zeta, lim)
        T_v = fk_from_zeta(q_sol, zeta, n=6)
        pos_err_mm = np.linalg.norm(T_v[:3, 3] - T_t[:3, 3]) * 1000.0
        ang_err_deg = np.degrees(np.linalg.norm(rotation_log_map(T_v[:3, :3] @ T_t[:3, :3].T)))
        print('[%s] converged=%s  resid=%.2e' % (name, ok, en))
        print('   true (deg): %s' % np.round(np.rad2deg(q_true), 2).tolist())
        print('   sol  (deg): %s' % np.round(np.rad2deg(q_sol), 2).tolist())
        print('   check: pos_err=%.4f mm  orient_err=%.4f deg\n' % (pos_err_mm, ang_err_deg))
