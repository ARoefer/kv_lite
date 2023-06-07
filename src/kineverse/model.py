from typing import Iterable

from . import spatial as gm
from .graph import Graph,       \
                   Frame,       \
                   DirectedEdge


class Constraint():
    def __init__(self, lb : gm.KVExpr, ub : gm.KVExpr, expr : gm.KVExpr) -> None:
        self.lb   = lb
        self.ub   = ub
        self.expr = expr
    
    def matrix(self):
        return gm.KVArray([self.lb, self.ub, self.expr])
    
    @property
    def symbols(self):
        return self.expr.symbols


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


class Model(Graph):
    def __init__(self) -> None:
        super().__init__()
        self._constraints = {}
        self._symbol_constraint_map = {}

    def add_constraint(self, name, constraint):
        if name in self._constraints:
            self.remove_constraint(name)
        
        for s in constraint.symbols:
            if s not in self._symbol_constraint_map:
                self._symbol_constraint_map[s] = set()
            
            self._symbol_constraint_map[s].add(name)
        
        self._constraints[name] = constraint

    def remove_constraint(self, name):
        if name not in self._constraints:
            raise KeyError(f'Unknown constraint {name}')
        
        for s in self._constraints[name].symbols:
            self._symbol_constraint_map[s].remove(name)
        
        del self._constraints[name]

    def get_constraints(self, symbols : Iterable):
        out = {}
        for s in symbols:
            if s in self._symbol_constraint_map:
                for cn in self._symbol_constraint_map[s]:
                    out[cn] = self._constraints[cn]
        
        return out



class Body(Frame):
    def __init__(name, ) -> None:
        super().__init__(name)
    
    