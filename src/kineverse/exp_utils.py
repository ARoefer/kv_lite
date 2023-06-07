from typing import Tuple

from . import spatial as gm
from .model import ConstrainedEdge


def twist_to_se3(q, v, w) -> gm.KVArray:
    """Generate a SE(3) transform from a twist in exponential coordinates.
        q: position
        v: linear part of the twist
        w: angular part of the twist 
    """
    tf = gm.KVArray([[    0, -w[2],  w[1], v[0]]
                     [ w[2],     0, -w[0], v[1]],
                     [-w[1],  w[0],     0, v[2]],
                     [    0,     0,     0,    0]])
    tf = gm.exp(tf * q)
    tf[3, :3] = 0 # Exponential makes all zeros 1
    return tf


def se3_to_twist(tf) -> Tuple[gm.KVArray, gm.KVArray]:
    """Maps a SE(3) transform back into a twist."""
    lin = gm.log(gm.KVArray(tf.T[3, :3]))
    ang = gm.log(gm.KVArray([-tf[1, 2], tf[0, 2], -tf[0, 1]]))
    return lin, ang


class TwistJointEdge(ConstrainedEdge):
    def __init__(self, parent, child, linear, angular, position, constraints=None) -> None:
        super().__init__(parent, child, constraints)
        self.linear   = linear
        self.angular  = angular
        self.position = position

    def eval(self, graph, current_tf):
        return twist_to_se3(self.linear, self.angular, self.position).dot(current_tf)

