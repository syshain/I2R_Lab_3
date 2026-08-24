import numpy as np


def fk_from_zeta(q, zeta, n=6):
    """Standard D-H forward kinematics.

    Parameters
    ----------
    q : (n,) joint angles in radians
    zeta : (4*n,) DH parameter vector packed as [a | alpha | d | theta_off],
        where a and d are in metres and alpha and theta_off in radians.
    n : number of joints (default 6)

    Returns
    -------
    T : (4, 4) base-to-flange homogeneous transform.
    """
    a = zeta[0:n]
    alpha = zeta[n:2*n]
    d = zeta[2*n:3*n]
    theta_off = zeta[3*n:4*n]
    T = np.eye(4)

    for i in range(n):
        theta = q[i] + theta_off[i]
        ct, st = np.cos(theta), np.sin(theta)
        ca, sa = np.cos(alpha[i]), np.sin(alpha[i])
        T = T @ np.array([
            [ct, -st*ca,  st*sa, a[i]*ct],
            [st,  ct*ca, -ct*sa, a[i]*st],
            [0,    sa,     ca,   d[i]],
            [0,    0,      0,    1 ],
        ])
    return T
