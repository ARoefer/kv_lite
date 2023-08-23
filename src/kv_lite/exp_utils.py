from typing import Tuple

from . import spatial as gm
from .model import ConstrainedEdge
import numpy as np

def twist_to_se3_special(q, v, w, epsilon=1e-6) -> gm.KVArray:
    """Generate a SE(3) transform from a twist in exponential coordinates.
        Check for prismatic special case and treat differently
        q: position
        v: linear part of the twist
        w: angular part of the twist 
    """
    if w.ndim < 2:
        w = w.reshape((len(w), 1))

    if np.linalg.norm(w) < epsilon:
        return gm.Transform.from_xyz(*(v[:3].flatten() * q))

    return twist_to_se3(q, v, w, epsilon=1e-6)

def twist_to_se3(q, v, w, epsilon=1e-6) -> gm.KVArray:
    """Generate a SE(3) transform from a twist in exponential coordinates.
        q: position
        v: linear part of the twist
        w: angular part of the twist 
    """
    if w.ndim < 2:
        w = w.reshape((len(w), 1))
    w_hat = gm.KVArray([[       0, -w[2, 0],  w[1, 0]],
                        [ w[2, 0],        0, -w[0, 0]],
                        [-w[1, 0],  w[0, 0],       0]])
    e_wq  = gm.eye(3) + gm.sin(q) * w_hat + w_hat.dot(w_hat) * (1 - gm.cos(q))
    h     = w[:3].T.dot(v[:3]) / (gm.norm(w[:3]**2) + epsilon)
    lin   = (gm.eye(3) - e_wq).dot(gm.cross(w[:3], v[:3])[:3]) + w[:3].dot(w[:3].T)[:3].dot(v[:3]) * q
    tf         = gm.eye(4).astype(object)
    tf[:3, :3] = e_wq
    tf[:3,  3] = lin.flatten()

    return tf


class TwistJointEdge(ConstrainedEdge):
    def __init__(self, parent, child, linear, angular, position, constraints=None) -> None:
        super().__init__(parent, child, constraints)
        self.linear   = linear
        self.angular  = angular
        self.position = position

    def eval(self, graph, current_tf):
        return twist_to_se3(self.linear, self.angular, self.position).dot(current_tf)

