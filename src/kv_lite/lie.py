import numpy as np

from . import spatial as kv


def _skew(v : kv.KVArray) -> kv.KVArray:
    return kv.array([[  0.0, -v[2],  v[1]], 
                     [ v[2],   0.0, -v[0]],
                     [-v[1],  v[0],   0.0]])

def _unskew(m : kv.KVArray) -> kv.KVArray:
    return kv.array([m[2, 1] - m[1, 2],
                     m[0, 2] - m[2, 0],
                     m[1, 0] - m[0, 1]]) / 2

class SO3:
    @staticmethod
    def expmap(w, epsilon=1e-6):
        theta = kv.norm(w)
        w_n   = w / (theta + epsilon)

        skw = _skew(w_n)

        return (kv.eye(skw.shape[0])
                + kv.sin(theta) * skw 
                + (1.0 - kv.cos(theta)) * skw @ skw
               )

    @staticmethod
    def logmap(R, epsilon=1e-6):
        R  = R[:3, :3]
        tr = (kv.trace(R) - 1) / 2
        # min for inaccuracies near identity yielding trace > 3
        theta = kv.acos(tr)
        st = kv.sin(theta)
        
        skw = (R - R.T) / (2 * st + epsilon)
        return _unskew(skw) * theta



class SE3:
    EXP_MAP_G = np.array([[[ 0,  0,  0, 0],
                           [ 0,  0, -1, 0],
                           [ 0,  1,  0, 0],
                           [ 0,  0,  0, 0]],
                          [[ 0,  0,  1, 0],
                           [ 0,  0,  0, 0],
                           [-1,  0,  0, 0],
                           [ 0,  0,  0, 0]],
                          [[ 0, -1,  0, 0],
                           [ 1,  0,  0, 0],
                           [ 0,  0,  0, 0],
                           [ 0,  0,  0, 0]],
                          [[ 0,  0,  0, 1],
                           [ 0,  0,  0, 0],
                           [ 0,  0,  0, 0],
                           [ 0,  0,  0, 0]],
                          [[ 0,  0,  0, 0],
                           [ 0,  0,  0, 1],
                           [ 0,  0,  0, 0],
                           [ 0,  0,  0, 0]],
                          [[ 0,  0,  0, 0],
                           [ 0,  0,  0, 0],
                           [ 0,  0,  0, 1],
                           [ 0,  0,  0, 0]]])
    EXP_MAP_G = np.transpose(EXP_MAP_G, (1, 2, 0))

    @staticmethod
    def expmap(w, v, epsilon=1e-6):
        R   = SO3.expmap(w, epsilon=epsilon)
        theta = kv.norm(w)
        theta_sq = w @ w
        t_parallel = w * (w @ v)
        w_cross    = np.cross(w, v)
        
        # Need to actively make the array symbolic
        out = kv.eye(4)
        if R.is_symbolic:
            out = out.astype(object)
        out[:3, :3] = R
        out[:3,  3] = (w_cross - R @ w_cross + t_parallel) / (theta_sq + epsilon)

        return out

    @staticmethod
    def logmap(mat : np.ndarray, epsilon=1e-6) -> np.ndarray:
        w = SO3.logmap(mat, epsilon=epsilon)
        theta = kv.norm(w) + epsilon
        S = _skew(w) 

        Ginv = (kv.eye(3)
                - S / 2
                + (1 / theta - 1 / kv.tan(theta / 2) / 2) / theta * S @ S
               )
        t = mat[:3, 3]
        v = Ginv @ t
        return kv.hstack((w, v))
