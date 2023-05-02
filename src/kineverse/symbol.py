import casadi as ca

class Symbol():
    _INSTANCES = {}

    TYPE_UNKNOWN  = 0
    TYPE_POSITION = 1
    TYPE_VELOCITY = 2
    TYPE_ACCEL    = 3
    TYPE_JERK     = 4
    TYPE_SNAP     = 5
    TYPE_SUFFIXES = {'position': TYPE_POSITION,
                    'velocity': TYPE_VELOCITY,
                    'acceleration': TYPE_ACCEL,
                    'jerk': TYPE_JERK,
                    'snap': TYPE_SNAP}
    TYPE_SUFFIXES_INV = {v: k for k, v in TYPE_SUFFIXES.items()}

    def __new__(cls, name, typ=TYPE_UNKNOWN, prefix=None):
        if typ not in Symbol.TYPE_SUFFIXES_INV:
            raise KeyError(f'Unknown symbol type {typ}')
        
        full_name = f'{name}__{Symbol.TYPE_SUFFIXES_INV[typ]}' if typ != Symbol.TYPE_UNKNOWN else name
        if prefix is not None:
            full_name = f'{prefix}__{full_name}'
        
        if full_name in Symbol._INSTANCES:
            Symbol._INSTANCES[full_name]
        
        out = super().__new__(cls)
        out.name = name
        out.type = typ
        out.prefix = prefix
        out._full_name = full_name
        out._ca_symbol = ca.SX.sym(ca.SX(), full_name)
        return out
    
    def __eq__(self, other: object) -> bool:
        if isinstance(Symbol):
            return self._full_name == other._full_name
    
    def __hash__(self) -> int:
        return hash(self._full_name)

    def derivative(self):
        if self.type == Symbol.TYPE_UNKNOWN:
            raise RuntimeError(f'Cannot differentiate symbol of unknown type.')
        if self.type == Symbol.TYPE_SNAP:
            raise RuntimeError(f'Cannot differentiate symbol beyond snap.')

        return Symbol(self.name, self.type + 1, self.prefix)
    
    def integral(self):
        if self.type == Symbol.TYPE_UNKNOWN:
            raise RuntimeError(f'Cannot integrate symbol of unknown type.')
        if self.type == Symbol.TYPE_POSITION:
            raise RuntimeError(f'Cannot integrate symbol beyond position.')

        return Symbol(self.name, self.type - 1, self.prefix)

def Position(name, prefix=None):
    return Symbol(name, Symbol.TYPE_POSITION, prefix)

def Velocity(name, prefix=None):
    return Symbol(name, Symbol.TYPE_VELOCITY, prefix)

def Acceleration(name, prefix=None):
    return Symbol(name, Symbol.TYPE_ACCEL, prefix)

def Jerk(name, prefix=None):
    return Symbol(name, Symbol.TYPE_JERK, prefix)

def Snap(name, prefix=None):
    return Symbol(name, Symbol.TYPE_SNAP, prefix)
