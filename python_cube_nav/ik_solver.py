"""Numerical Inverse Kinematics for the UFACTORY Lite 6 (Python).

Direct port of the Lab 2 MATLAB solver:

    Lab 2 (MATLAB)                          Lab 3 (this file)
    ----------------------                  -------------------------------
    ik_objective.m -> err                   ik_objective(q, T_target, zeta)
        position error p_cur - p_tgt              same formula
        orientation error = axis-angle of         rotation_log_map(R_cur @ R_tgt.T)
        R_cur * R_tgt'                              (identical to so3.py from Lab 2's
                                                    calibration code)
    ik_solve.m -> lsqnonlin(fsolve LM)          solve_ik(...) -> scipy.optimize.least_squares
                                                (method='trf' == bounded Levenberg-Marquardt)

Nothing about the method changed between labs — only the language. The residual is
the 6-vector e(q) = [p_FK(q) - p_target ; log(R_FK(q) @ R_target.T)], minimised over
the six joint angles with the UFACTORY Lite 6 joint limits enforced as bounds.

Conventions match the rest of this folder:
    * FK via fk_from_zeta(q, zeta) in fk_functions.py (standard D-H).
    * zeta is the (4*n,) packed DH vector [a | alpha | d | theta_off], metres/radians.
    * All positions in METRES, all angles in RADIANS.

Usage:
    from ik_solver import solve_ik, pose_to_matrix, matrix_to_pose, LITE6_JOINT_LIMITS_RAD

    T_target = pose_to_matrix([0.3, 0.0, 0.35], [0.0, 1.5708, 0.0])   # xyz + rpy (rad)
    q_sol, ok, err_norm = solve_ik(T_target, q0=current_joints_rad, zeta=zeta)
"""

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from fk_functions import fk_from_zeta


# ---------------------------------------------------------------------------
# Joint limits — UFACTORY Lite 6 (identical to ik_numerical.m in Lab 2)
# ---------------------------------------------------------------------------
LITE6_JOINT_LIMITS_DEG = np.array([
    [-360.0,  360.0],   # J1
    [-150.0,  150.0],   # J2
    [  -3.5,  300.0],   # J3 (asymmetric minimum)
    [-360.0,  360.0],   # J4
    [-124.0,  124.0],   # J5
    [-360.0,  360.0],   # J6
])
LITE6_JOINT_LIMITS_RAD = np.deg2rad(LITE6_JOINT_LIMITS_DEG)


def get_nominal_zeta():
    """Nominal (factory) Lite 6 DH parameters as a packed zeta vector.

    Mirrors _get_nominal_zeta() in run_lab3.py and the first `dh` block of
    fk_lite6.m in Lab 2. Returns [a | alpha | d | theta_off] in metres/radians.
    """
    dh_raw = np.array([
        [  0,   243.3,  -90,    0],
        [-90,     0.0,  180,  200],
        [-90,     0.0,   90,   87],
        [  0,   227.6,   90,    0],
        [  0,     0.0,  -90,    0],
        [  0,    61.5,    0,    0],
    ])
    a = dh_raw[:, 3] / 1000.0
    alpha = np.deg2rad(dh_raw[:, 2])
    d = dh_raw[:, 1] / 1000.0
    theta_off = np.deg2rad(dh_raw[:, 0])
    return np.concatenate([a, alpha, d, theta_off])


# ---------------------------------------------------------------------------
# SO(3) log map — same routine as so3.py used in Lab 2's calibration
# ---------------------------------------------------------------------------
def rotation_log_map(R):
    """Vector logarithm of a 3x3 rotation matrix.

    Returns omega in R^3 such that exp(hat(omega)) = R; ||omega|| is the
    geodesic angle on SO(3). Identical to rotation_log_map() in Lab 2's
    python_dh_optimization/so3.py.
    """
    theta = np.arccos(np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0))
    if theta < 1e-8:
        skew = (R - R.T) / 2.0
        return np.array([skew[2, 1], skew[0, 2], skew[1, 0]])
    scale = theta / (2.0 * np.sin(theta))
    return np.array([R[2, 1] - R[1, 2],
                     R[0, 2] - R[2, 0],
                     R[1, 0] - R[0, 1]]) * scale


# ---------------------------------------------------------------------------
# IK objective — port of ik_objective.m
# ---------------------------------------------------------------------------
def ik_objective(q, T_target, zeta):
    """Compute the 6-vector pose error for a candidate joint vector.

    Parameters
    ----------
    q : (6,) joint angles in radians
    T_target : (4, 4) target end-effector transform in the base frame
    zeta : (4*6,) packed DH parameter vector

    Returns
    -------
    err : (6,) = [position_error (m); orientation_error (axis-angle, rad)]
    """
    T_current = fk_from_zeta(q, zeta, n=6)

    # Position error (metres)
    err_pos = T_current[:3, 3] - T_target[:3, 3]

    # Orientation error: axis-angle of R_current * R_target'
    R_current = T_current[:3, :3]
    R_target = T_target[:3, :3]
    R_error = R_current @ R_target.T

    err_rot = rotation_log_map(R_error)

    return np.concatenate([err_pos, err_rot])


# ---------------------------------------------------------------------------
# IK solver — port of ik_solve.m (lsqnonlin / fsolve LM -> bounded least_squares)
# ---------------------------------------------------------------------------
def solve_ik(T_target, q0, zeta=None, joint_limits=LITE6_JOINT_LIMITS_RAD, verbose=False):
    """Solve inverse kinematics numerically with bounded Levenberg-Marquardt.

    Parameters
    ----------
    T_target : (4, 4) target end-effector transform in the base frame
    q0 : (6,) initial guess in radians. Seed with the arm's CURRENT joint angles
        each step to stay on the same solution branch (elbow/wrist config).
    zeta : (4*6,) packed DH vector. Defaults to the nominal Lite 6 set.
    joint_limits : (6, 2) array [min_rad, max_rad] per joint. Pass None or an
        empty array to remove bounds (unconstrained solve, like plain fsolve).
    verbose : print iteration info when True.

    Returns
    -------
    q_solution : (6,) solved joint angles in radians
    success : bool — did the optimizer converge?
    error_norm : float — final norm of the 6-vector residual
    """
    if zeta is None:
        zeta = get_nominal_zeta()
    q0 = np.asarray(q0, dtype=float).reshape(-1)
    assert q0.shape == (6,), "q0 must be a 6-element vector"

    def fun(q):
        return ik_objective(q, T_target, zeta)

    if joint_limits is not None and np.size(joint_limits) > 0:
        lb, ub = joint_limits[:, 0], joint_limits[:, 1]
    else:
        lb, ub = -np.inf, np.inf

    res = least_squares(fun, q0, bounds=(lb, ub), method='trf',
                        max_nfev=5000, xtol=1e-12, ftol=1e-10, gtol=1e-10,
                        verbose=int(verbose))

    q_solution = res.x
    error_norm = float(np.linalg.norm(res.fun))
    success = bool(res.success)

    if not success:
        print("WARNING: IK solver did not converge. Status: %s (%s)"
              % (res.status, res.message.strip()))

    return q_solution, success, error_norm


# ---------------------------------------------------------------------------
# Pose <-> matrix helpers — ports of pose_to_matrix.m / matrix_to_pose.m
# ---------------------------------------------------------------------------
def pose_to_matrix(pos_xyz, euler_xyz):
    """Build a 4x4 transform from position + Euler angles.

    pos_xyz : (3,) in metres
    euler_xyz : (3,) [roll, pitch, yaw] in radians (XYZ fixed/intrinsic convention,
        matching rotm2eul(...,'zyx') in the MATLAB helper).
    """
    R = Rotation.from_euler('xyz', np.asarray(euler_xyz)).as_matrix()
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = np.asarray(pos_xyz, dtype=float)
    return T


def matrix_to_pose(T):
    """Extract [pos_xyz, euler_xyz] from a 4x4 transform.

    Returns (pos_xyz in metres, euler_xyz = [roll, pitch, yaw] in radians).
    """
    pos_xyz = T[:3, 3].copy()
    euler_xyz = Rotation.from_matrix(T[:3, :3]).as_euler('xyz')
    return pos_xyz, euler_xyz


# ---------------------------------------------------------------------------
# Self-test: mirrors the three scenarios in ik_numerical.m (Lab 2)
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    zeta = get_nominal_zeta()
    lim = LITE6_JOINT_LIMITS_RAD

    tests = [
        ("Home",      np.deg2rad([-180,   0.0,   4.0,    0.0,  -45.0,    0.0]),
                      np.array([0.1, -1.2, 0.1, 0.1, 0.1, 0.1])),
        ("ElbowDown", np.deg2rad([-180, -18.8,   4.0,    0.0,  -45.0,    0.0]),
                      np.array([0.1, -0.8, 0.3, 0.1, 0.1, 0.1])),
        ("Complex",   np.deg2rad([-302.2, -19.9, 37.2, -207.4, -108.7,  -39.9]),
                      np.array([0.3, -0.8, 0.5, 0.2, 0.3, 0.1])),
    ]

    print("=== Python IK self-test (nominal Lite 6 DH) ===\n")
    for name, q_true, q0 in tests:
        T_t = fk_from_zeta(q_true, zeta, n=6)
        q_sol, ok, en = solve_ik(T_t, q0, zeta, lim)
        # verify by forward kinematics
        T_v = fk_from_zeta(q_sol, zeta, n=6)
        pos_err_mm = np.linalg.norm(T_v[:3, 3] - T_t[:3, 3]) * 1000.0
        ang_err_deg = np.degrees(np.linalg.norm(rotation_log_map(
            T_v[:3, :3] @ T_t[:3, :3].T)))
        print("[%s] converged=%s  resid=%.2e" % (name, ok, en))
        print("   true (deg): %s" % np.round(np.rad2deg(q_true), 2).tolist())
        print("   sol  (deg): %s" % np.round(np.rad2deg(q_sol), 2).tolist())
        print("   check: pos_err=%.4f mm  orient_err=%.4f deg\n" % (pos_err_mm, ang_err_deg))
