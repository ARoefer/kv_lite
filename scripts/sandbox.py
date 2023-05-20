import kineverse as kv
import numpy     as np

if __name__ == '__main__':
    x, y, z = [kv.gm.Symbol(c) for c in 'xyz']

    print(f'x >= x: {x >= x}')
    print(f'x > y: {x > y}')
    print(f'x < y: {x < y}')
    print(f'x == x: {x == x}')
    print(f'x == y: {x == y}')
    print(f'x == new x: {x == kv.gm.Symbol("x")}')

    print(f'id(x) == id(new x): {id(x) == id(kv.gm.Symbol("x"))}')

    d = kv.gm.KVArray(np.diag([x, y, z]))

    symbols = kv.gm._get_symbols(d)

    print(f'{d}')
    print(f'Is symbolic: {d.is_symbolic}')
    print(f'Symbols: {d.symbols}')
