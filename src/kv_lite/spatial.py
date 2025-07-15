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
        inv = inv.astype(object) if tf.is_symbolic else inv
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