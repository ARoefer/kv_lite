import kv_lite as kv
import numpy   as np
import rospy

from math         import prod
from prime_bullet import res_pkg_path
from tqdm         import tqdm


from kv_lite import gm
from kv_lite.ros_utils import ModelTFBroadcaster

if __name__ == '__main__':
    rospy.init_node('kv_lite_sandbox')

    x, y, z = [gm.KVSymbol(c) for c in 'xyz']

    print(f'x >= x: {x >= x}')
    print(f'x > y: {x > y}')
    print(f'x < y: {x < y}')
    print(f'x == x: {x == x}')
    print(f'x == y: {x == y}')
    print(f'x == new x: {x == gm.KVSymbol("x")}')
    print(f'id(x) == id(new x): {id(x) == id(gm.KVSymbol("x"))}')

    e1 = x * 4
    print(f'{e1} ({type(e1)}): {e1.symbols}')
    e2 = y - 4
    print(f'{e2} ({type(e2)}): {e2.symbols}')
    e3 = e1 - e2
    print(f'{e3} ({type(e3)}): {e3.symbols}')

    jac_e2 = e2.jacobian([x, y])
    print(f'{jac_e2} ({type(jac_e2)})')

    e4 = e1 * 0 + 3
    print(f'{e4} ({type(e4)}): {e4.symbols}')

    try:
        float(e2)
        print('Float conversion of symbolic expression did NOT raise exception')
    except RuntimeError as e:
        print('Float conversion of symbolic expression raised exception')
    
    try:
        float(e4)
        print('Float conversion of non-symbolic expression did not raise exception')
    except RuntimeError as e:
        print('Float conversion of not-symbolic expression DID raise exception')

    d1 = gm.KVArray(np.diag([x, y, z]))

    print(f'{d1}')
    print(f'Is symbolic: {d1.is_symbolic}')
    print(f'Symbols: {d1.symbols}')

    d2 = gm.KVArray(np.eye(3)) * x + np.diag([1, 1], 1) * y
    print(f'{d2} ({type(d2)})')
    print(f'Is symbolic: {d2.is_symbolic}')
    print(f'Symbols: {d2.symbols}')

    d3 = gm.eye(3) * x + gm.diag([1, 1], 1) * y
    print(f'{d3} ({type(d3)})')
    print(f'Is symbolic: {d3.is_symbolic}')
    print(f'Symbols: {d3.symbols}')

    print(f'd3.T:\n{d3.T} ({type(d3.T)})')
    print(f'Is symbolic: {d3.T.is_symbolic}')
    print(f'Symbols: {d3.T.symbols}')

    print(f'sin(4): {gm.sin(4)}')
    print(f'sin(e1): {gm.sin(e1)}')
    print(f'sin(d3): {gm.sin(d3)}')

    # print(f'eval({e1}, x: 2) = {e1.eval({x: 2})}')
    # print(f'eval({e3}, x: 2, y: 5) = {e3.eval({x: 2, y: 5})}')

    # f  = gm._speed_up(m, [x, y, z], (3, 3))
    args = np.array([1.0, 2.0, 3.0]) # .reshape((3, 1))
    print(d1)
    print(d1.eval({x: 1, y: 2, z: 3}))
    
    print(d2)
    print(d2.eval({x: 1, y: 2, z: 3}))
    
    print(d3)
    print(d3.eval({x: 1, y: 2, z: 3}))

    km = kv.Model()
    with open(res_pkg_path('/home/russell/git/iiwa_grasping/urdf/yang_grasping.urdf'), 'r') as f:
        windmill = kv.urdf.load_urdf(km, f.read())

    km.add_edge(kv.TransformEdge('world', windmill.root, gm.Transform.from_xyz(0.1, 0, 0)))

    for ln in windmill.links:
        print(f'Link: {ln}')

    for jn in windmill.joints:
        print(f'Joint: {jn}')

    world_T_r_arm_5 = windmill.get_fk('iiwa_link_4')
    print(world_T_r_arm_5)
    joint_symbols = world_T_r_arm_5.transform.symbols
    for j in joint_symbols:
        print(j)

    constraints = km.get_constraints(joint_symbols.union({s.derivative() for s in joint_symbols}))
    print(f'Number of constraints: {len(constraints)}')

    for cn, c in constraints.items():
        print(f'{cn}: {c}')

    d_target = gm.norm(gm.point3(2.0, 1.0, 1.0) - gm.Transform.pos(world_T_r_arm_5.transform))

    tf_pub = ModelTFBroadcaster(km)

    full_q_symbols = tf_pub.symbols

    for x in tqdm(range(100000), desc='FK eval speed'):
        js = dict(zip(full_q_symbols, np.sin(np.ones(len(full_q_symbols)) * (x * 0.1))))
        d_target.eval(js)
        tf_pub.update(js)
        if rospy.is_shutdown():
            exit(0)
        rospy.sleep(0.05)
        # bla = input('press enter for next config')
        # if bla == 'q':
        #     exit(0)


    J_target = d_target.jacobian(joint_symbols)

    for x in tqdm(range(100000), desc='J eval speed'):
        J_target.eval(dict(zip(joint_symbols, np.random.uniform(-1, 1, len(joint_symbols)))))

    t_target = d_target.tangent()
    tangent_symbols = t_target.symbols

    for x in tqdm(range(100000), desc='Tangent eval speed'):
        t_target.eval(dict(zip(tangent_symbols, np.random.uniform(-1, 1, len(tangent_symbols)))))

    # print(f'bTb:\n{windmill.get_fk("base", "base")}')
    # print(f'bTw:\n{windmill.get_fk("base")}')
    # print(f'hTb:\n{windmill.get_fk("wings", "base")}')
    # print(f'bTw:\n{windmill.get_fk("base", "wings")}')