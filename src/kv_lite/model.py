from dataclasses import dataclass, \
                        field
from typing      import Iterable, \
                        Union

from .graph import DirectedEdge

from . import spatial as gm
from .graph import Graph,        \
                   Frame,        \
                   DirectedEdge


class Constraint():
    def __init__(self, lb : gm.KVExpr, ub : gm.KVExpr, expr : gm.KVExpr) -> None:
        self.lb   = lb   if not isinstance(lb, gm.KVArray) else lb.item()
        self.ub   = ub   if not isinstance(ub, gm.KVArray) else ub.item()
        self.expr = expr if not isinstance(expr, gm.KVArray) else expr.item()
    
    def matrix(self):
        return gm.KVArray([self.lb, self.ub, self.expr])
    
    def __iter__(self):
        return iter((self.lb, self.ub, self.expr))

    def __unpack__(self):
        return iter((self.lb, self.ub, self.expr))

    @property
    def symbols(self):
        return self.expr.symbols
    
    def __str__(self):
        return f'C({self.lb} <= {self.expr} <= {self.ub})'
    
    def __repr__(self):
        return str(self)


class ConstrainedEdge(DirectedEdge):
    def __init__(self, parent, child, constraints=None) -> None:
        super().__init__(parent, child)
        self.constraints = constraints


class TransformEdge(DirectedEdge):
    def __init__(self, parent, child, tf : gm.Transform) -> None:
        super().__init__(parent, child)
        self._transform = tf

    def eval(self, graph, current_tf : gm.KVArray) -> gm.KVArray:
        return self._transform.dot(current_tf)


class ConstrainedTransformEdge(ConstrainedEdge):
    def __init__(self, parent, child, tf : gm.Transform, constraints=None) -> None:
        super().__init__(parent, child, constraints)
        self._transform = tf

    def eval(self, graph, current_tf : gm.KVArray) -> gm.KVArray:
        return self._transform.dot(current_tf)


class Model(Graph):
    def __init__(self) -> None:
        super().__init__()
        self._constraints = {}
        self._symbol_constraint_map = {}

    def add_edge(self, edge: DirectedEdge, name : str=None):
        out = super().add_edge(edge, name)

        if isinstance(edge, ConstrainedEdge):
            for cn, c in edge.constraints.items():
                self.add_constraint(cn, c)
        
        return out
    
    def remove_edge(self, edge: DirectedEdge):
        out = super().remove_edge(edge)

        if isinstance(edge, ConstrainedEdge):
            for cn in edge.constraints.keys():
                if cn in self._constraints:
                    self.remove_constraint(cn)

        return out

    def add_constraint(self, name : str, constraint : Constraint):
        if name in self._constraints:
            self.remove_constraint(name)
        
        for s in constraint.symbols:
            if s not in self._symbol_constraint_map:
                self._symbol_constraint_map[s] = set()
            
            self._symbol_constraint_map[s].add(name)
        
        self._constraints[name] = constraint

    def remove_constraint(self, name : str):
        if name not in self._constraints:
            raise KeyError(f'Unknown constraint {name}')
        
        for s in self._constraints[name].symbols:
            self._symbol_constraint_map[s].remove(name)
        
        del self._constraints[name]

    def get_constraints(self, symbols : Iterable[gm.Symbol]):
        out = {}
        for s in symbols:
            if s in self._symbol_constraint_map:
                for cn in self._symbol_constraint_map[s]:
                    out[cn] = self._constraints[cn]
        
        return out


@dataclass
class Inertial:
    origin  : gm.KVArray
    mass    : float
    moments : gm.KVArray


@dataclass
class Geometry:
    type      : str
    mesh_path : str
    dim_scale : gm.KVArray = field(default_factory=lambda: gm.vector3(1, 1, 1))
    origin    : gm.KVArray = field(default_factory=gm.Transform.identity)


class Body(Frame):
    def __init__(self, name, inertial : Inertial,
                       geom_collision : Iterable[Geometry],
                       geom_visual : Iterable[Geometry] = None) -> None:
        super().__init__(name)

        self.inertial       = inertial
        self.geom_collision = geom_collision
        self.geom_visual    = geom_visual if geom_visual is not None else geom_collision
