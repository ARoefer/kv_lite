import casadi as ca
import numpy  as np

from math      import prod

def _Matrix(data):
    try:
        return ca.SX(data)
    except NotImplementedError:
        if hasattr(data, u'shape'):
            m = ca.SX(*data.shape)
        else:
            x = len(data)
            if isinstance(data[0], list) or isinstance(data[0], tuple):
                y = len(data[0])
            else:
                y = 1
            m = ca.SX(x, y)
        for i in range(m.shape[0]):
            if y > 1:
                for j in range(m.shape[1]):
                    try:
                        m[i, j] = data[i][j]
                    except:
                        m[i, j] = data[i, j]
            else:
                if isinstance(data[i], list) or isinstance(data[i], tuple):
                    m[i] = data[i][0]
                else:
                    m[i] = data[i]
        return m


class EvaluationError(Exception):
    pass


def _speed_up(function, parameters, shape):
    params   = list(parameters)
    m_params = [p._ca_data for p in params]
    # print(f'Speed up function: {function}\nArgs: {m_params}')
    try:
        f = ca.Function('f', m_params, [ca.densify(function)])
    except:
        f = ca.Function('f', m_params, ca.densify(function))
    return _CompiledFunction(params, f, shape)


class _CompiledFunction():
    def __init__(self, params, fast_f, shape):
        self.params = params
        self.fast_f = fast_f
        self.shape  = shape
        self.buf, self.f_eval = fast_f.buffer()
        self.out = np.zeros(shape) # , order='F')
        self.buf.set_res(0, memoryview(self.out))

    def __call__(self, args : dict):
        try:
            filtered_args = [float(args[k]) for k in self.params]
            return self.call_unchecked(filtered_args)
        except KeyError as e:
            raise EvaluationError(f'Missing variable for evaluation: {e}')

    def call_unchecked(self, filtered_args):
        """
        :param filtered_args: parameter values in the same order as in self.str_params
        :type filtered_args: list
        :return:
        """
        filtered_args = np.array(filtered_args, dtype=float, order='F').reshape((len(filtered_args), 1))
        for i in range(len(self.params)):
            self.buf.set_arg(i, memoryview(filtered_args[i]))
        self.f_eval()
        return self.out # .reshape(self.shape)
    

class KVExpr():
    """Container wrapping CASADI expressions. 
       Mainly exists to avoid the nasty parts of CASADI expressions.
    """
    def __new__(cls, expr):
        # Singelton rule for Symbols
        if isinstance(expr, KVSymbol):
            return expr

        out = super().__new__(cls)
        out._symbols   = None
        out._o_symbols = None
        # Compiled function for evaluation
        out._function  = None

        # Straight copy
        if isinstance(expr, KVExpr):
            out._ca_data   = expr._ca_data
            out._symbols   = expr._symbols
            out._o_symbols = expr._o_symbols
        else: # New element
            out._ca_data = expr
        return out

    def __float__(self):
        if self.is_symbolic:
            raise RuntimeError('Expressions with symbols cannot be auto-converted to float.')
        return float(self._ca_data)

    def __iadd__(self, other):
        self._ca_data += other
        self._symbols  = None
        self._function = None
        return self

    def __isub__(self, other):
        self._ca_data -= other
        self._symbols  = None
        self._function = None
        return self

    def __imul__(self, other):
        self._ca_data *= other
        self._symbols  = None
        self._function = None
        return self

    def __itruediv__(self, other):
        return self.__idiv__(other)

    def __idiv__(self, other):
        self._ca_data /= other
        self._symbols  = None
        self._function = None
        return self

    def __neg__(self):
        return KVExpr(-self._ca_data)

    def __add__(self, other):
        if isinstance(other, KVExpr):
            return KVExpr(self._ca_data + other._ca_data)
        elif isinstance(other, np.ndarray):
            return other + self
        return KVExpr(self._ca_data + other)

    def __sub__(self, other):
        if isinstance(other, KVExpr):
            return KVExpr(self._ca_data - other._ca_data)
        elif isinstance(other, np.ndarray):
            return other.__rsub__(self)
        return KVExpr(self._ca_data - other)

    def __mul__(self, other):
        if isinstance(other, KVExpr):
            return KVExpr(self._ca_data * other._ca_data)
        elif isinstance(other, np.ndarray):
            return other * self
        return KVExpr(self._ca_data * other)

    def __truediv__(self, other):
        return self.__div__(other)

    def __div__(self, other):
        if isinstance(other, KVExpr):
            return KVExpr(self._ca_data / other._ca_data)
        elif isinstance(other, np.ndarray):
            return other.__rdiv__(self)
        return KVExpr(self._ca_data / other)

    def __radd__(self, other):
        if isinstance(other, KVExpr):
            return KVExpr(self._ca_data + other._ca_data)
        return KVExpr(self._ca_data + other)

    def __rsub__(self, other):
        if isinstance(other, KVExpr):
            return KVExpr(other._ca_data - self._ca_data)
        return KVExpr(other) - KVExpr(self._ca_data)

    def __rmul__(self, other):
        if isinstance(other, KVExpr):
            return KVExpr(self._ca_data * other._ca_data)
        return KVExpr(self._ca_data * other)

    def __rtruediv__(self, other):
        return self.__div__(other)

    def __rdiv__(self, other):
        if isinstance(other, KVExpr):
            return KVExpr(other._ca_data / self._ca_data)
        return KVExpr(other) / KVExpr(self._ca_data)

    def __pow__(self, other):
        if isinstance(other, KVExpr):
            return KVExpr(self._ca_data ** other._ca_data)
        elif isinstance(other, np.ndarray):
            return KVArray([self])**other
        return KVExpr(self._ca_data ** other)

    def __str__(self):
        return str(self._ca_data)
    
    def __repr__(self):
        return f'KV({self._ca_data})'

    @property
    def is_zero(self):
        return self._ca_data.is_zero()

    @property
    def is_one(self):
        return self._ca_data.is_one()

    @property
    def is_symbolic(self):
        return len(self.symbols) > 0

    @property
    def symbols(self):
        if self._symbols is None:
            if type(self._ca_data) in {ca.SX, ca.MX, ca.DM}:
                self._symbols = frozenset({KVSymbol(str(e)) for e in ca.symvar(self._ca_data)})
            else:
                self._symbols = frozenset()
        return self._symbols

    @property
    def ordered_symbols(self):
        if self._o_symbols is None:
            self._o_symbols = tuple(self.symbols)
        return self._o_symbols

    def jacobian(self, symbols):
        jac = ca.jacobian(self._ca_data, _Matrix([s._ca_data for s in symbols]))
        np_jac = KVArray(np.array([KVExpr(e) for e in jac.elements()]).reshape(jac.shape))
        return np_jac

    def tangent(self, symbols=None):
        """Generalized tangent expression of this expression:
            t(q, \dot{q}) = J(q) * \dot{dot}
         
           Does full tangent by default, but 'symbols' argument can
           be used to override the generation of derivatives. 
        """
        positions = list(symbols) if symbols is not None else list(self.symbols)
        J = self.jacobian(positions)
        return J.dot(KVArray([[p.derivative() for p in positions]]).T).item() # Result is 1x1

    def eval(self, args : dict = {}):
        if self._function is None:
            self._function = _speed_up(self._ca_data, self.ordered_symbols, (1,))
        return float(self._function(args))

    def unchecked_eval(self, args):
        if self._function is None:
            self._function = _speed_up(self._ca_data, self.ordered_symbols, (1,))
        return float(self._function.call_unchecked(args))

    def as_casadi(self):
        return self._ca_data

    def substitute(self, assignments : dict):
        return type(self)(ca.substitute(self._ca_data,
                                        KVArray(list(assignments.keys())).as_casadi(),
                                        KVArray(list(assignments.values())).as_casadi()))


class KVSymbol(KVExpr):
    _INSTANCES = {}

    TYPE_UNKNOWN  = 0
    TYPE_POSITION = 1
    TYPE_VELOCITY = 2
    TYPE_ACCEL    = 3
    TYPE_JERK     = 4
    TYPE_SNAP     = 5
    TYPE_SUFFIXES = {'UNKNOWN': TYPE_UNKNOWN,
                     'position': TYPE_POSITION,
                     'velocity': TYPE_VELOCITY,
                     'acceleration': TYPE_ACCEL,
                     'jerk': TYPE_JERK,
                     'snap': TYPE_SNAP}
    TYPE_SUFFIXES_INV = {v: k for k, v in TYPE_SUFFIXES.items()}

    def __new__(cls, name, typ=TYPE_UNKNOWN, prefix=None):
        if typ not in KVSymbol.TYPE_SUFFIXES_INV:
            raise KeyError(f'Unknown symbol type {typ}')
        
        full_name = f'{str(name).replace("/", "__")}__{KVSymbol.TYPE_SUFFIXES_INV[typ]}' if typ != KVSymbol.TYPE_UNKNOWN else name
        if prefix is not None:
            full_name = f'{prefix}__{full_name}'
        
        if full_name in KVSymbol._INSTANCES:
            return KVSymbol._INSTANCES[full_name]
        
        out = super().__new__(cls, ca.SX.sym(full_name))
        out.name = name
        out.type = typ
        out.prefix = prefix
        out._full_name = full_name
        out._symbols   = frozenset({out})
        KVSymbol._INSTANCES[full_name] = out
        return out
    
    def __eq__(self, other: object) -> bool:
        if isinstance(other, KVSymbol):
            return self._full_name == other._full_name
    
    def __hash__(self) -> int:
        return hash(self._full_name)

    def __lt__(self, other) -> bool:
        if not isinstance(other, KVSymbol):
            raise TypeError(f"< not supported between instances of '{type(other)}' and '{type(self)}'")
        return self._full_name < other._full_name
    
    def __gt__(self, other) -> bool:
        if not isinstance(other, KVSymbol):
            raise TypeError(f"> not supported between instances of '{type(other)}' and '{type(self)}'")
        return self._full_name > other._full_name
    
    def __le__(self, other) -> bool:
        if not isinstance(other, KVSymbol):
            raise TypeError(f"<= not supported between instances of '{type(other)}' and '{type(self)}'")
        return self._full_name <= other._full_name
    
    def __ge__(self, other) -> bool:
        if not isinstance(other, KVSymbol):
            raise TypeError(f">= not supported between instances of '{type(other)}' and '{type(self)}'")
        return self._full_name >= other._full_name

    # No in-place modification of symbols, as they are constant
    def __iadd__(self, other):
        return self + other

    def __isub__(self, other):
        return self - other

    def __imul__(self, other):
        return self * other

    def __idiv__(self, other):
        return self / other

    def derivative(self):
        if self.type == KVSymbol.TYPE_UNKNOWN:
            raise RuntimeError(f'Cannot differentiate symbol of unknown type.')
        if self.type == KVSymbol.TYPE_SNAP:
            raise RuntimeError(f'Cannot differentiate symbol beyond snap.')

        return KVSymbol(self.name, self.type + 1, self.prefix)
    
    def integral(self):
        if self.type == KVSymbol.TYPE_UNKNOWN:
            raise RuntimeError(f'Cannot integrate symbol of unknown type.')
        if self.type == KVSymbol.TYPE_POSITION:
            raise RuntimeError(f'Cannot integrate symbol beyond position.')

        return KVSymbol(self.name, self.type - 1, self.prefix)

    def eval(self, args : dict):
        if self in args:
            return args[self]
        raise EvaluationError()
    
    def unchecked_eval(self, args):
        return args[0]

    def substitute(self, assignments : dict):
        # There is only one possible substitution
        if self in assignments:
            return assignments[self]
        return self


def Position(name, prefix=None):
    return KVSymbol(name, KVSymbol.TYPE_POSITION, prefix)

def Velocity(name, prefix=None):
    return KVSymbol(name, KVSymbol.TYPE_VELOCITY, prefix)

def Acceleration(name, prefix=None):
    return KVSymbol(name, KVSymbol.TYPE_ACCEL, prefix)

def Jerk(name, prefix=None):
    return KVSymbol(name, KVSymbol.TYPE_JERK, prefix)

def Snap(name, prefix=None):
    return KVSymbol(name, KVSymbol.TYPE_SNAP, prefix)


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
    return isinstance(nl, KVSymbol)

def _get_symbols(nl):
    if isinstance(nl, KVArray):
        out = set()
        for e in nl:
            out.update(_get_symbols(e))
        return out
    return nl.symbols if isinstance(nl, KVExpr) else set()


_vec_is_zero = np.vectorize(lambda v: v.is_zero if isinstance(v, KVExpr) else v == 0)
_vec_is_one  = np.vectorize(lambda v: v.is_one  if isinstance(v, KVExpr) else v == 1)


class KVArray(np.ndarray):
    def __new__(cls, input_array):
        # Input array is an already formed ndarray instance
        # We first cast to be our class type
        obj = np.asarray(input_array).view(cls)
        # add the new attribute to the created instance
        obj._symbols   = None
        obj._o_symbols = None
        obj._function  = None
        # Finally, we must return the newly created object:
        return obj

    def __array_finalize__(self, obj):
        # see InfoArray.__array_finalize__ for comments
        if obj is None:
            return
        self._symbols   = None
        self._o_symbols = None
        self._function  = None

    @property
    def is_zero(self):
        return _vec_is_zero(self).min()

    @property
    def is_one(self):
        return _vec_is_one(self).min()

    @property
    def symbols(self):
        if self._symbols is None:
            self._symbols = frozenset(_get_symbols(self))
        return self._symbols

    @property
    def ordered_symbols(self):
        if self._o_symbols is None:
            self._o_symbols = tuple(self.symbols)
        return self._o_symbols

    @property
    def is_symbolic(self):
        return len(self.symbols) > 0

    def __add__(self, other):
        if isinstance(other, KVExpr):
            return super().__add__(np.asarray([other]))
        return super().__add__(other)

    def __sub__(self, other):
        if isinstance(other, KVExpr):
            return super().__sub__(np.asarray([other]))
        return super().__sub__(other)

    def __mul__(self, other):
        if isinstance(other, KVExpr):
            return super().__mul__(np.asarray([other]))
        return super().__mul__(other)

    def __truediv__(self, other):
        if isinstance(other, KVExpr):
            return super().__truediv__(np.asarray([other]))
        return super().__truediv__(other)

    def __radd__(self, other):
        return self + other

    def __rsub__(self, other):
        return KVArray([other]) - self

    def __rmul__(self, other):
        return self * other

    def __rtruediv__(self, other):
        return KVArray([other]) / self

    def __pow__(self, other):
        if isinstance(other, KVExpr):
            return super().__pow__(np.asarray([other]))
        return super().__pow__(other)

    def eval(self, args : dict):
        if self.dtype != object:
            return self.copy()

        if self._function is None:
            flat_f = [e._ca_data if isinstance(e, KVExpr) else e for e in self.flatten()]
            self._function = _speed_up(_Matrix(flat_f), self.ordered_symbols, self.shape)
        return self._function(args).copy()

    def unchecked_eval(self, args):
        if self.dtype != object:
            return self.copy()

        if self._function is None:
            flat_f = [e._ca_data if isinstance(e, KVExpr) else e for e in self.flatten()]
            self._function = _speed_up(_Matrix(flat_f), self.ordered_symbols, self.shape)
        return self._function.call_unchecked(args).copy()

    def jacobian(self, symbols):
        """Invokes jacobian() on all elements"""
        return vstack([e.jacobian(symbols) if isinstance(e, KVExpr) else zeros(len(symbols)) for e in self.flatten()]).reshape(self.shape + (len(symbols), )).squeeze()

    def tangent(self, symbols=None):
        """Invokes tangent() on all elements"""
        return KVArray([e.tangent(symbols) for e in self.flatten()]).reshape(self.shape)

    def as_casadi(self):
        if len(self.shape) > 2:
            raise RuntimeError(f'Casadi supports at most 2 dimensions. Shape of array is {self.shape}.')
        flat_f = [e._ca_data if isinstance(e, KVExpr) else e for e in self.flatten()]
        return _Matrix(flat_f).reshape(self.shape)

    def substitute(self, assignments : dict):
        return KVArray([e.substitute(assignments) if isinstance(e, KVExpr) else e for e in self.flatten()]).reshape(self.shape)


def array(a):
    return KVArray(np.array(a))

def asarray(a):
    return KVArray(np.asarray(a))

def diag(v, k=0):
    return KVArray(np.diag(v, k))

def eye(N, M=None, k=0):
    return KVArray(np.eye(N, M, k))

def ones(shape):
    return KVArray(np.ones(shape))

def zeros(shape):
    return KVArray(np.zeros(shape))

def tri(N, M=None, k=0):
    return KVArray(np.tri(N, M=M, k=k))

def hstack(tup):
    return KVArray(np.hstack(tup))

def vstack(tup):
    return KVArray(np.vstack(tup))

def stack(arrays, axis):
    return KVArray(np.stack(arrays, axis))


def wrap_array(f):
    def g(v):
        if isinstance(v, np.ndarray) or isinstance(v, list) or isinstance(v, tuple): # Containers need to be wrapped
            return f(KVArray(v))
        elif isinstance(v, KVArray):  # KVArrays are fine
            return f(v)
        return f(v).item()            # v is some kind of atom
    return g

def atan2(a, b):
    a = a._ca_data if isinstance(a, KVExpr) else a
    b = b._ca_data if isinstance(b, KVExpr) else b
    return KVExpr(ca.atan2(a, b))

def min(a, b):
    if isinstance(a, KVExpr):
        return KVExpr(ca.mmin(_Matrix([a._ca_data, b])))
    if isinstance(b, KVExpr):
        a = a if isinstance(a, KVExpr) else KVExpr(ca.SX(b))
        return KVExpr(ca.mmin(_Matrix([a, b._ca_data])))
    return np.min((a, b))

def max(a, b):
    if isinstance(a, KVExpr):
        return KVExpr(ca.mmax(_Matrix([a._ca_data, b])))
    if isinstance(b, KVExpr):
        a = a if isinstance(a, KVExpr) else KVExpr(ca.SX(b))
        return KVExpr(ca.mmax(_Matrix([a, b._ca_data])))
    return np.max((a, b))

sqrt = wrap_array(np.vectorize(lambda v: KVExpr(ca.sqrt(v._ca_data)) if isinstance(v, KVExpr) else np.sqrt(v)))
abs  = wrap_array(np.vectorize(lambda v: KVExpr(ca.sqrt(v._ca_data ** 2)) if isinstance(v, KVExpr) else np.abs(v)))

sin = wrap_array(np.vectorize(lambda v: KVExpr(ca.sin(v._ca_data)) if isinstance(v, KVExpr) else np.sin(v)))
cos = wrap_array(np.vectorize(lambda v: KVExpr(ca.cos(v._ca_data)) if isinstance(v, KVExpr) else np.cos(v)))

asin   = wrap_array(np.vectorize(lambda v: KVExpr(ca.asin(v._ca_data)) if isinstance(v, KVExpr) else np.arcsin(v)))
acos   = wrap_array(np.vectorize(lambda v: KVExpr(ca.acos(v._ca_data)) if isinstance(v, KVExpr) else np.arccos(v)))
arcsin = asin
arccos = acos

asinh   = wrap_array(np.vectorize(lambda v: KVExpr(ca.asinh(v._ca_data)) if isinstance(v, KVExpr) else np.arcsinh(v)))
acosh   = wrap_array(np.vectorize(lambda v: KVExpr(ca.acosh(v._ca_data)) if isinstance(v, KVExpr) else np.arccosh(v)))
arcsinh = asinh
arccosh = acosh

exp = wrap_array(np.vectorize(lambda v: KVExpr(ca.exp(v._ca_data)) if isinstance(v, KVExpr) else np.exp(v)))
log = wrap_array(np.vectorize(lambda v: KVExpr(ca.log(v._ca_data)) if isinstance(v, KVExpr) else np.log(v)))

tan    = wrap_array(np.vectorize(lambda v: KVExpr(ca.tan(v._ca_data)) if isinstance(v, KVExpr) else np.tan(v)))
atan   = wrap_array(np.vectorize(lambda v: KVExpr(ca.atan(v._ca_data)) if isinstance(v, KVExpr) else np.arctan(v)))
arctan = atan
tanh   = wrap_array(np.vectorize(lambda v: KVExpr(ca.tanh(v._ca_data)) if isinstance(v, KVExpr) else np.tanh(v)))
atanh  = wrap_array(np.vectorize(lambda v: KVExpr(ca.atanh(v._ca_data)) if isinstance(v, KVExpr) else np.arctanh(v)))
arctanh = atanh
