import kv_lite.exp_utils as ke
from kv_lite import gm

if __name__ == '__main__':

    q = gm.KVSymbol('q')

    vee = gm.vector3(0, 1, 0)
    omega = gm.zeros(3)

    t_trans = ke.twist_to_se3_special(q, vee, omega)

    print(t_trans)
    print(t_trans.eval({q: 0}))
    print(t_trans.eval({q: 1}))
    print(t_trans.eval({q: 2}))
