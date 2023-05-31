import kineverse as kv
import numpy     as np

from kineverse import gm
from math      import prod

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

    ca = gm.ca

    # m = ca.SX.sym('m', 3, 3)  # ca.SX.eye(3)
    # # m[0, 0] = x._ca_data
    # # m[1, 1] = y._ca_data
    # # m[2, 2] = z._ca_data
    # m[0, 1:] = 0
    # m[1,  0] = 0
    # m[1,  2] = 0
    # m[2, :2] = 0

    # print(m)

    # os = list(d3.symbols)


    # m = gm._Matrix([[x._ca_data, 0, 0],
    #                 [0, y._ca_data, 0],
    #                 [0, 0, z._ca_data]])

    # f  = gm._speed_up(m, [x, y, z], (3, 3))
    args = np.array([1.0, 2.0, 3.0]) # .reshape((3, 1))
    print(d1)
    print(d1.eval({x: 1, y: 2, z: 3}))
    
    print(d2)
    print(d2.eval({x: 1, y: 2, z: 3}))
    
    print(d3)
    print(d3.eval({x: 1, y: 2, z: 3}))

    # f = ca.Function('f', ca.symvar(m), [ca.densify(m)])

    # buf, f_eval = f.buffer()
    
    # print(f'symvar(m): {ca.symvar(m)}')
    
    # # buf.set_arg(0, memoryview(args))
    # for i in range(len(ca.symvar(m))):
    #     buf.set_arg(i, memoryview(args[i]))
    
    # res = np.asfortranarray(np.zeros(m.shape))

    # buf.set_res(0, memoryview(res))

    # f_eval()
    
    # print(res)

    # print(f'eval({d1}, x: 2, y: 5, z: -1) = {d1.eval({x: 2, y: 5, z: -1})}')

