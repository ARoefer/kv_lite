import kineverse as kv
import numpy     as np

from math           import prod
from iai_bullet_sim import res_pkg_path

from kineverse import gm
from tqdm      import tqdm


if __name__ == '__main__':
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
    with open(res_pkg_path('package://iai_bullet_sim/src/iai_bullet_sim/data/urdf/windmill.urdf'), 'r') as f:
        windmill = kv.urdf.load_urdf(km, f.read())

    km.add_edge(kv.TransformEdge('world', windmill.root, gm.Transform.from_xyz(0.1, 0, 0)))

    for ln in windmill.links:
        print(f'Link: {ln}')

    for jn in windmill.joints:
        print(f'Joint: {jn}')

    world_T_r_arm_5 = windmill.get_fk('RARM_JOINT5_Link')
    print(world_T_r_arm_5)
    joint_symbols = world_T_r_arm_5.transform.symbols
    for j in joint_symbols:
        print(j)

    constraints = km.get_constraints(joint_symbols.union({s.derivative() for s in joint_symbols}))
    print(f'Number of constraints: {len(constraints)}')

    for cn, c in constraints.items():
        print(f'{cn}: {c}')

    for x in tqdm(range(100000)):
        world_T_r_arm_5.transform.eval(dict(zip(joint_symbols, 
                                                np.random.uniform(-1, 1, len(joint_symbols)))))

    # print(f'bTb:\n{windmill.get_fk("base", "base")}')
    # print(f'bTw:\n{windmill.get_fk("base")}')
    # print(f'hTb:\n{windmill.get_fk("wings", "base")}')
    # print(f'bTw:\n{windmill.get_fk("base", "wings")}')