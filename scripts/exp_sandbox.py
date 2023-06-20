import kineverse.exp_utils as ke
from kineverse import gm

if __name__ == '__main__':

    q = gm.KVSymbol('q')

    t_trans = ke.twist_to_se3(q, gm.vector3(1, 0, 0), gm.zeros(3))

    print(t_trans)