import casadi as ca
import numpy  as np

from functools import lru_cache
from .symbol import Symbol,       \
                    Position,     \
                    Velocity,     \
                    Acceleration, \
                    Jerk,         \
                    Snap


def _find_array_shape(nl):
    if isinstance(nl, list) or isinstance(nl, tuple):
        sub_shapes = {_find_array_shape(e) for e in nl}
        if len(sub_shapes) > 1:
            raise TypeError(f'Array dimensions must all have the same size.')
        return (len(nl), ) + sub_shapes.pop()
    return tuple()


def _is_symbolic(nl):
    if isinstance(nl, KVArray):
        return max(*[_is_symbolic(e) for e in nl])
    return isinstance(nl, Symbol)

def _get_symbols(nl):
    if isinstance(nl, KVArray):
        out = set()
        for e in nl:
            out.update(_get_symbols(e))
        return out
    return {nl} if isinstance(nl, Symbol) else set()


class KVExpr():
    pass


class KVArray(np.ndarray):
    def __new__(cls, input_array):
        # Input array is an already formed ndarray instance
        # We first cast to be our class type
        obj = np.asarray(input_array).view(cls)
        # add the new attribute to the created instance
        obj._symbols = None
        # Finally, we must return the newly created object:
        return obj

    def __array_finalize__(self, obj):
        # see InfoArray.__array_finalize__ for comments
        if obj is None:
            return
        self._symbols = None

    @property
    def symbols(self):
        if self._symbols is None:
            self._symbols = frozenset(_get_symbols(self))
        return self._symbols

    @property
    def is_symbolic(self):
        return len(self.symbols) > 0

    def __add__(self, other):
        if isinstance(other):
            return super().__add__(self, np.asarray([other]))
        return super().__add__(self, other)

    def __sub__(self, other):
        if isinstance(other):
            return super().__sub__(self, np.asarray([other]))
        return super().__sub__(self, other)

    def __mul__(self, other):
        if isinstance(other):
            return super().__mul__(self, np.asarray([other]))
        return super().__mul__(self, other)

    def __div__(self, other):
        if isinstance(other):
            return super().__div__(self, np.asarray([other]))
        return super().__div__(self, other)

    def __radd__(self, other):
        return self + other

    def __rsub__(self, other):
        return self - other

    def __rmul__(self, other):
        return self * other

    def __rdiv__(self, other):
        return self / other
