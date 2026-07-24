import numpy as np

from .math import KVExpr, \
                  KVSymbol, \
                  KVArray, \
                  VectorizedEvalHandler as VEval, \
                  Symbol,  \
                  Position, \
                  Velocity, \
                  Acceleration, \
                  Jerk, \
                  Snap, \
                  is_symbolic, \
                  sqrt, \
                  abs,  \
                  sin,  \
                  cos,  \
                  asin,  \
                  acos,  \
                  arcsin,  \
                  arccos,  \
                  asinh,  \
                  acosh,  \
                  arcsinh,  \
                  arccosh,  \
                  exp,  \
                  log,  \
                  tan,  \
                  atan,  \
                  atan2,  \
                  arctan,  \
                  tanh,  \
                  atanh,  \
                  arctanh,  \
                  expr,    \
                  array,  \
                  asarray, \
                  diag,   \
                  diag_view, \
                  eye,   \
                  batched_eye, \
                  ones,  \
                  zeros, \
                  tri, \
                  trace, \
                  vstack, \
                  hstack, \
                  stack, \
                  concatenate, \
                  min, \
                  max, \
                  EvaluationError


def point3(x, y, z):
    return KVArray([x, y, z, 1]).reshape((4, 1))

def vector3(x, y, z):
    return KVArray([x, y, z, 0]).reshape((4, 1))

unitX = vector3(1, 0, 0)
unitY = vector3(0, 1, 0)
unitZ = vector3(0, 0, 1)

def norm(a, axis=None):
    rt = sqrt((a**2).sum(axis=axis))
    return rt

def cross(u, v):
    """Computes the cross product between two vectors."""
    # u = np.squeeze(u)
    # v = np.squeeze(v)

    return KVArray([[u[..., 1] * v[..., 2] - u[..., 2] * v[..., 1],
                     u[..., 2] * v[..., 0] - u[..., 0] * v[..., 2],
                     u[..., 0] * v[..., 1] - u[..., 1] * v[..., 0], 0]]).T

def skew(v : KVArray) -> KVArray:
    """Batched hat operator: maps a (..., 3) vector to a (..., 3, 3) skew-symmetric matrix."""
    dtype = float if not isinstance(v, KVArray) or not v.is_symbolic else object
    out   = zeros((3, 3), dtype=dtype) if v.ndim == 1 else zeros(v.shape[:-1] + (3, 3), dtype=dtype)
    out[..., 0, 1] = -v[..., 2]
    out[..., 0, 2] =  v[..., 1]
    out[..., 1, 0] =  v[..., 2]
    out[..., 1, 2] = -v[..., 0]
    out[..., 2, 0] = -v[..., 1]
    out[..., 2, 1] =  v[..., 0]
    return out


def unskew(m : KVArray) -> KVArray:
    """Batched vee operator: maps a (..., 3, 3) skew-symmetric matrix to a (..., 3) vector."""
    return (m[..., (2, 0, 1), (1, 2, 0)] - m[..., (1, 2, 0), (2, 0, 1)]) / 2


def operational_space_inertia(mass_matrix, jacobian) -> np.ndarray:
    """Operational-space (task-space) inertia `Lambda = (J M^-1 J^T)^-1`.

    **Numeric**, not symbolic: it takes already-evaluated `mass_matrix` (n x n) and
    `jacobian` (m x n, e.g. the 6 x n output of `Graph.body_jacobian` for a target frame)
    and returns an `m x m` ndarray. There is no symbolic matrix inverse in this backend, and
    a symbolic `M^-1` would blow up for realistic DoF counts, so evaluate `M` and `J` at a
    configuration first, then call this. Unlike the mass matrix (which aggregates over all
    bodies) this legitimately depends on a chosen target frame, via its Jacobian.
    """
    M = np.asarray(mass_matrix, dtype=float)
    J = np.asarray(jacobian,    dtype=float)
    return np.linalg.inv(J.dot(np.linalg.inv(M)).dot(J.T))


def diff(expr : KVArray, symbol) -> KVArray:
    """Derivative of a KVArray w.r.t. a single symbol, same shape as `expr`.

    Thin wrapper over `KVArray.jacobian`, which appends the symbol axis and then squeezes
    the length-1 axis away — so a 4x4 transform yields a 4x4 derivative. A fully numeric
    `expr` yields a same-shaped zero array.
    """
    return expr.jacobian([symbol])


def jacobian_column(fk : KVArray, symbol) -> KVArray:
    """One column of a 6xn spatial body Jacobian, in [omega; v] ordering.

    `fk` is the 4x4 homogeneous transform of the body in the reference frame. The angular
    part uses the **spatial** angular velocity vee(dR @ R^T) (expressed in the reference
    frame), required for consistency with `spatial_inertia`. Using R^T @ dR instead would
    give the body angular velocity and is wrong here.
    """
    dT = diff(fk, symbol)               # 4x4
    R  = fk[:3, :3]
    v  = dT[:3,  3]                      # linear part, taken directly
    w  = unskew(dT[:3, :3].dot(R.T))     # angular part via vee(R_dot @ R^T)
    return hstack([w, v])               # [omega; v]


def adjoint(T : KVArray) -> KVArray:
    """6x6 adjoint of a 4x4 homogeneous transform, in [omega; v] (angular-first) ordering.

    Maps a spatial twist expressed in the child frame to the same twist expressed in the
    parent frame:  Ad_T = [[R, 0], [skew(p) R, R]].
    """
    R  = T[:3, :3]
    p  = T[:3,  3]
    Ad = zeros((6, 6), dtype=object if is_symbolic(T) else float)
    Ad[:3, :3] = R
    Ad[3:, :3] = skew(p).dot(R)
    Ad[3:, 3:] = R
    return Ad


def spatial_inertia(inertial, ref_T_body : KVArray) -> KVArray:
    """6x6 spatial inertia about the **body frame origin**, in [omega; v] ordering.

    `inertial` is duck-typed (fields `mass`, `origin`, `moments`) to avoid importing
    `model.Inertial` here — `model` already imports `spatial`. `moments` is the inertia
    tensor about the CoM expressed in the frame defined by `origin` (URDF convention).

    The reference point is the body frame origin, matching `jacobian_column`, whose linear
    part is the velocity of that same frame origin (the hybrid geometric Jacobian). The
    parallel-axis offset `d` is therefore the CoM position **relative to the body frame
    origin**, not the absolute CoM position — using the absolute position would double-count
    the offset and inflate `J^T I J`. Both J and I are referenced at the body's own frame
    origin; the resulting per-body `J^T I J` term is internally consistent, and summing them
    yields the correct configuration-space mass matrix (which is frame-independent).
    """
    m         = inertial.mass
    ref_T_com = ref_T_body.dot(inertial.origin)
    R         = ref_T_com[:3, :3]
    d         = ref_T_com[:3, 3] - ref_T_body[:3, 3]   # CoM offset from body frame origin, ref frame

    I  = R.dot(inertial.moments).dot(R.T)      # inertia about CoM, rotated into ref frame
    dx = skew(d)

    M6 = zeros((6, 6), dtype=object if (is_symbolic(ref_T_com) or is_symbolic(I)) else float)
    M6[:3, :3] = I + m * dx.dot(dx.T)          # parallel-axis shift from CoM to frame origin
    M6[:3, 3:] = m * dx
    M6[3:, :3] = m * dx.T
    M6[3:, 3:] = m * eye(3)
    return M6


def rotation_vector_from_matrix(rotation_matrix):
    rm = rotation_matrix

    angle = (np.diag(rm[:3, :3]) - 1) / 2
    angle = acos(angle)
    x = (rm[2, 1] - rm[1, 2])
    y = (rm[0, 2] - rm[2, 0])
    z = (rm[1, 0] - rm[0, 1])

    return vector3(x, y, z)

def axis_angle_from_matrix(rotation_matrix):
    rm = rotation_matrix

    v_rot = rotation_vector_from_matrix(rm)

    angle = norm(v_rot)
    axis  = v_rot / angle
    return axis, angle

def translation3(x, y, z, w=1):
    """Creates a homogenous translation transformation."""
    return KVArray([[1, 0, 0, x],
                    [0, 1, 0, y],
                    [0, 0, 1, z],
                    [0, 0, 0, w]])

def plane_projection(normal : KVArray) -> KVArray:
    return eye(normal.shape[-2]) - normal @ normal.T


class Transform:
    @staticmethod
    def from_xyz(x, y, z):
        return KVArray([[1, 0, 0, x],
                        [0, 1, 0, y],
                        [0, 0, 1, z],
                        [0, 0, 0, 1]])
    
    @staticmethod
    def from_euler(roll, pitch, yaw):
        rx = KVArray([[1, 0, 0, 0],
                      [0, cos(roll), -sin(roll), 0],
                      [0, sin(roll), cos(roll), 0],
                      [0, 0, 0, 1]])
        ry = KVArray([[cos(pitch), 0, sin(pitch), 0],
                      [0, 1, 0, 0],
                      [-sin(pitch), 0, cos(pitch), 0],
                      [0, 0, 0, 1]])
        rz = KVArray([[cos(yaw), -sin(yaw), 0, 0],
                      [sin(yaw), cos(yaw), 0, 0],
                      [0, 0, 1, 0],
                      [0, 0, 0, 1]])
        return rz.dot(ry).dot(rx)
    
    @staticmethod
    def from_axis_angle(axis, angle):
        """ Conversion of unit axis and angle to 4x4 rotation matrix according to:
            http://www.euclideanspace.com/maths/geometry/rotations/conversions/angleToMatrix/index.htm
        """
        axis = axis.squeeze()
        ct = cos(angle)
        st = sin(angle)
        vt = 1 - ct
        m_vt_0 = vt * axis[0]
        m_vt_1 = vt * axis[1]
        m_vt_2 = vt * axis[2]
        m_st_0 = axis[0] * st
        m_st_1 = axis[1] * st
        m_st_2 = axis[2] * st
        m_vt_0_1 = m_vt_0 * axis[1]
        m_vt_0_2 = m_vt_0 * axis[2]
        m_vt_1_2 = m_vt_1 * axis[2]
        return KVArray([[ct + m_vt_0 * axis[0], -m_st_2 + m_vt_0_1, m_st_1 + m_vt_0_2, 0],
                        [m_st_2 + m_vt_0_1, ct + m_vt_1 * axis[1], -m_st_0 + m_vt_1_2, 0],
                        [-m_st_1 + m_vt_0_2, m_st_0 + m_vt_1_2, ct + m_vt_2 * axis[2], 0],
                        [0, 0, 0, 1]])
    
    @staticmethod
    def from_quat(x, y, z, w):
        """ Unit quaternion to 4x4 rotation matrix according to:
            https://github.com/orocos/orocos_kinematics_dynamics/blob/master/orocos_kdl/src/frames.cpp#L167
        """
        x2 = x * x
        y2 = y * y
        z2 = z * z
        w2 = w * w
        return KVArray([[w2 + x2 - y2 - z2, 2 * x * y - 2 * w * z, 2 * x * z + 2 * w * y, 0],
                        [2 * x * y + 2 * w * z, w2 - x2 + y2 - z2, 2 * y * z - 2 * w * x, 0],
                        [2 * x * z - 2 * w * y, 2 * y * z + 2 * w * x, w2 - x2 - y2 + z2, 0],
                        [0, 0, 0, 1]])

    @staticmethod
    def from_xyz_euler(x, y, z, rr, rp, ry):
        return Transform.from_xyz(x, y, z).dot(Transform.from_euler(rr, rp, ry))

    @staticmethod
    def from_xyz_aa(x, y, z, axis, angle):
        return Transform.from_xyz(x, y, z).dot(Transform.from_axis_angle(axis, angle))

    @staticmethod
    def from_xyz_quat(x, y, z, qx, qy, qz, qw):
        return Transform.from_xyz(x, y, z).dot(Transform.from_quat(qx, qy, qz, qw))

    @staticmethod
    def inverse(tf):
        inv = eye(4)
        inv = inv.astype(object) if isinstance(tf, KVArray) and tf.is_symbolic else inv
        inv[..., :3, :3] = tf[..., :3, :3].T
        inv[..., :3,  3] = -inv[..., :3, :3].dot(tf[..., :3, 3])
        return inv
    
    @staticmethod
    def identity():
        return eye(4)
    
    @staticmethod
    def pos(tf):
        return tf[:, 3].reshape((4, 1))
    
    @staticmethod
    def rot(tf):
        out = tf * 1
        out[:3, 3] = 0
        return out

    @staticmethod
    def trans(tf):
        out = Transform.identity()
        out[:3, 3] = tf[:3, 3]
        return out
    
    @staticmethod
    def x(tf):
        return tf[:, 0].reshape((4, 1))
    
    @staticmethod
    def y(tf):
        return tf[:, 1].reshape((4, 1))
    
    @staticmethod
    def z(tf):
        return tf[:, 2].reshape((4, 1))
    
    @staticmethod
    def w(tf):
        return tf[:, 3].reshape((4, 1))