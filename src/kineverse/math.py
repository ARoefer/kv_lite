import casadi as ca
import numpy  as np

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


class KVExpr():
    """Container wrapping CASADI expressions. 
       Mainly exists to avoid the nasty parts of CASADI expressions.
    """
    def __new__(cls, expr):
        # Singelton rule for Symbols
        if isinstance(expr, KVSymbol):
            return expr

        out = super().__new__(cls)
        out._symbols = None

        # Straight copy
        if isinstance(expr, KVExpr):
            out._ca_data = expr._ca_data
            out._symbols = expr._symbols
        else: # New element
            out._ca_data = expr
        return out

    def __float__(self):
        if self.is_symbolic:
            raise RuntimeError('Expressions with symbols cannot be auto-converted to float.')
        return float(self._ca_data)

    def __iadd__(self, other):
        self._ca_data += other
        self._symbols = None
        return self

    def __isub__(self, other):
        self._ca_data -= other
        self._symbols = None
        return self

    def __imul__(self, other):
        self._ca_data *= other
        self._symbols = None
        return self

    def __idiv__(self, other):
        self._ca_data /= other
        self._symbols = None
        return self

    def __add__(self, other):
        if isinstance(other, KVExpr):
            return KVExpr(self._ca_data + other._ca_data)
        return KVExpr(self._ca_data + other)

    def __sub__(self, other):
        if isinstance(other, KVExpr):
            return KVExpr(self._ca_data - other._ca_data)
        return KVExpr(self._ca_data - other)

    def __mul__(self, other):
        if isinstance(other, KVExpr):
            return KVExpr(self._ca_data * other._ca_data)
        return KVExpr(self._ca_data * other)

    def __div__(self, other):
        if isinstance(other, KVExpr):
            return KVExpr(self._ca_data / other._ca_data)
        return KVExpr(self._ca_data / other)

    def __radd__(self, other):
        if isinstance(other, KVExpr):
            return KVExpr(self._ca_data + other._ca_data)
        return KVExpr(self._ca_data + other)

    def __rsub__(self, other):
        if isinstance(other, KVExpr):
            return KVExpr(self._ca_data - other._ca_data)
        return KVExpr(self._ca_data - other)

    def __rmul__(self, other):
        if isinstance(other, KVExpr):
            return KVExpr(self._ca_data * other._ca_data)
        return KVExpr(self._ca_data * other)

    def __rdiv__(self, other):
        if isinstance(other, KVExpr):
            return KVExpr(self._ca_data / other._ca_data)
        return KVExpr(self._ca_data / other)

    def __pow__(self, other):
        if isinstance(other, KVExpr):
            return KVExpr(self._ca_data ** other._ca_data)
        return KVExpr(self._ca_data ** other)

    def __str__(self):
        return str(self._ca_data)
    
    def __repr__(self):
        return f'KV({self._ca_data})'

    @property
    def is_symbolic(self):
        return len(self.symbols) > 0

    @property
    def symbols(self):
        if self._symbols is None:
            self._symbols = frozenset({KVSymbol(str(e)) for e in ca.symvar(self._ca_data)})
        return self._symbols

    def jacobian(self, symbols):
        jac = ca.jacobian(self._ca_data, _Matrix([s._ca_data for s in symbols]))
        np_jac = KVArray(np.array([KVExpr(e) for e in jac.elements()]).reshape(jac.shape))
        return np_jac


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
        
        full_name = f'{name}__{KVSymbol.TYPE_SUFFIXES_INV[typ]}' if typ != KVSymbol.TYPE_UNKNOWN else name
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

    def __div__(self, other):
        if isinstance(other, KVExpr):
            return super().__div__(np.asarray([other]))
        return super().__div__(other)

    def __radd__(self, other):
        return self + other

    def __rsub__(self, other):
        return self - other

    def __rmul__(self, other):
        return self * other

    def __rdiv__(self, other):
        return self / other

    def __pow__(self, other):
        if isinstance(other, KVExpr):
            return super().__pow__(np.asarray([other]))
        return super().__pow__(other)

def array(a):
    return KVArray(np.array(a))

def asarray(a):
    return KVArray(np.asarray(a))

def diag(v, k=0):
    return KVArray(np.diag(v, k))

def eye(N, M=None, k=0):
    return KVArray(np.eye(N, M, k))


sin = np.vectorize(lambda v: KVExpr(ca.sin(v._ca_data)) if isinstance(v, KVExpr) else np.sin(v))
cos = np.vectorize(lambda v: KVExpr(ca.cos(v._ca_data)) if isinstance(v, KVExpr) else np.cos(v))

asin   = np.vectorize(lambda v: KVExpr(ca.asin(v._ca_data)) if isinstance(v, KVExpr) else np.arcsin(v))
acos   = np.vectorize(lambda v: KVExpr(ca.acos(v._ca_data)) if isinstance(v, KVExpr) else np.arccos(v))
arcsin = asin
arccos = acos

asinh   = np.vectorize(lambda v: KVExpr(ca.asinh(v._ca_data)) if isinstance(v, KVExpr) else np.arcsinh(v))
acosh   = np.vectorize(lambda v: KVExpr(ca.acosh(v._ca_data)) if isinstance(v, KVExpr) else np.arccosh(v))
arcsinh = asinh
arccosh = acosh

exp = np.vectorize(lambda v: KVExpr(ca.exp(v._ca_data)) if isinstance(v, KVExpr) else np.exp(v))
log = np.vectorize(lambda v: KVExpr(ca.log(v._ca_data)) if isinstance(v, KVExpr) else np.log(v))

tan    = np.vectorize(lambda v: KVExpr(ca.tan(v._ca_data)) if isinstance(v, KVExpr) else np.tan(v))
atan   = np.vectorize(lambda v: KVExpr(ca.atan(v._ca_data)) if isinstance(v, KVExpr) else np.arctan(v))
arctan = atan
tanh   = np.vectorize(lambda v: KVExpr(ca.tanh(v._ca_data)) if isinstance(v, KVExpr) else np.tanh(v))
atanh  = np.vectorize(lambda v: KVExpr(ca.atanh(v._ca_data)) if isinstance(v, KVExpr) else np.arctanh(v))
arctanh = atanh
