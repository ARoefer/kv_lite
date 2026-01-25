import numpy as np

from dataclasses import dataclass
from enum        import Enum

try:
    from qpsolvers   import solve_qp
except ModuleNotFoundError as NO_QPSOLVERS:
    solve_qp = None


from . import spatial as kv

class ObjectiveTypes(Enum):
    EQUALITY=0
    INEQUALITY=1


@dataclass
class Objective:
    type : ObjectiveTypes
    # Always a flattened vector
    expr : kv.KVArray
    gain : kv.KVArray
    weight : float | kv.KVArray | None=None

    def __post_init__(self):
        if not isinstance(self.expr, kv.KVArray):
            if isinstance(self.expr, np.ndarray):
                self.expr = kv.KVArray(self.expr)
            else:
                self.expr = kv.KVArray([self.expr])
        
        self.expr = self.expr.reshape((-1,))
        gain = kv.ones(self.expr.shape)
        self.gain   = gain * self.gain
        if self.weight is not None:
            if self.weight == 0:
                self.weight = None
            else:
                weight = kv.ones(self.expr.shape)
                self.weight = weight * self.weight

    @property
    def ndim(self) -> int:
        return len(self.expr)

    @property
    def is_soft(self) -> bool:
        return not self.is_hard

    @property
    def is_hard(self) -> bool:
        return self.weight is None



def _build_objective_matrix(objectives : dict[str, Objective],
                            control_symbols : list[kv.KVSymbol],
                            slack_padding : int=0) -> tuple[kv.KVArray, kv.KVArray, dict[str, tuple[int, int]], int]:
    # Putting soft constraints in a block makes matrix building easier
    objectives = dict([(n, o) for n, o in objectives.items() if o.is_hard] + [(n, o) for n, o in objectives.items() if o.is_soft])
    
    # Each dimension of each objective gets its own slack variable if it is soft
    soft_dims  = sum([o.ndim for o in objectives.values() if o.is_soft])

    left_col_blocks = kv.vstack([o.expr.jacobian(control_symbols) for o in objectives.values()])
    if slack_padding > 0:
        M = kv.hstack((left_col_blocks, kv.zeros((len(left_col_blocks), slack_padding))))
    else:
        M = left_col_blocks
    
    if soft_dims > 0:
        right_col = kv.vstack((kv.zeros((len(left_col_blocks) - soft_dims, soft_dims)),
                               kv.eye(sum([o.ndim for o in objectives.values() if o.is_soft]))))
        diag_weights = kv.hstack([o.weight for o in objectives.values() if o.is_soft])
        M = kv.hstack((M, right_col))
    else:
        diag_weights = None

    V = kv.hstack([o.gain for o in objectives.values()])

    objective_ranges = {}
    start = 0
    for on, o in objectives.items():
        objective_ranges[on] = (start, start + o.ndim - 1)
        start = start + o.ndim

    return M, V, objective_ranges, soft_dims, diag_weights


class QPController:
    def __init__(self, robot : kv.urdf.URDFObject,
                       objectives : dict[str, Objective],
                       control_symbols : list[kv.KVSymbol],
                       control_costs : dict[kv.KVSymbol],
                       default_control_cost=0.1,
                       dampening_factor : float=None,
                       position_integration_factor : float=None,
                       regularization_weight : dict[kv.Symbol, float] | float=None,
                       regularization_target : dict[kv.Symbol, float]=None,
                       solver : str='daqp'):
        if solve_qp is None:
            raise ImportError(f'You do not have the "qpsolvers" package installed. Original exception: {NO_QPSOLVERS}')

        self._robot  = robot
        self._solver = solver

        self._control_symbols = control_symbols
        self._eq_objectives   = {n: o for n, o in objectives.items() if o.type == ObjectiveTypes.EQUALITY}
        self._ineq_objectives = {n: o for n, o in objectives.items() if o.type == ObjectiveTypes.INEQUALITY}
        self._position_integration_factor = position_integration_factor

        self._dampening_factor = dampening_factor

        # 
        if len(self._eq_objectives) > 0:
            self._A, self._b, _, A_softdims, A_softweights = _build_objective_matrix(self._eq_objectives, control_symbols)
        else:
            self._A, self._b, A_softdims, A_softweights = None, None, 0, None

        if len(self._ineq_objectives) > 0:
            self._G, self._h, _, G_softdims, G_softweights = _build_objective_matrix(self._ineq_objectives, control_symbols, slack_padding=A_softdims)
        else:
            self._G, self._h, G_softdims, G_softweights = None, None, 0, None

        if self._A is not None and G_softdims > 0:
            self._A = kv.hstack((self._A, kv.zeros((len(self._A), G_softdims))))

        self._costs = kv.diag(kv.hstack(([control_costs.get(v, default_control_cost) for v in control_symbols],
                                         (A_softweights if A_softweights is not None else []),
                                         (G_softweights if G_softweights is not None else []),
                                        )))
        self._q_vel_limits = np.vstack([dict(zip(robot.q, robot.q_dot_limit)).get(s, [-1e6, 1e6]) for s in self._control_symbols])
        self._q_pos_limits = np.vstack([dict(zip(robot.q, robot.q_limit)).get(s, [-1e6, 1e6])     for s in self._control_symbols])
        self._x_limits     = np.vstack([self._q_vel_limits, [[-1e6, 1e6]]*(G_softdims + A_softdims)])

        self._regularization_weight = regularization_weight
        if regularization_weight is not None:
            # Holds this information for the entire x vector
            self._has_reg_target = np.zeros(len(self._x_limits), dtype=bool)
            if isinstance(regularization_weight, dict):
                self._regularization_weight = np.array([regularization_weight.get(c, 0) for c in self._control_symbols])
            else:
                self._regularization_weight = np.ones(len(self._control_symbols)) * regularization_weight

            if regularization_target is None:
                self._regularization_target = self._x_limits.sum(axis=-1)[:len(self._control_symbols)] * 0.5
                self._has_reg_target[:len(self._control_symbols)] = True
            else:
                self._regularization_target = np.zeros(len(self._control_symbols))
                for x, c in enumerate(self._control_symbols):
                    if c in regularization_target:
                        if isinstance(regularization_weight, dict) and c not in regularization_weight:
                            raise ValueError(f'{c} has regularization target but no weight.')
                        self._has_reg_target[x] = True
                        self._regularization_target[x] = regularization_target[c]

        self.reset()

    def reset(self):
        self._last_x_dot = None

    def eval(self, q_obs : dict[kv.Symbol, float]) -> dict[kv.Symbol, float]:
        # (J(EE_vel))
        if self._last_x_dot is None:
            self._last_x_dot = np.zeros(len(self._x_limits))

        q_pos = np.asarray([q_obs[c] for c in self._control_symbols])

        # TODO: ADD ACCELERATION BOUND
        q = np.zeros(self._costs.shape[0])
        if self._regularization_weight is not None:
            q[:len(q_pos)] = (q_pos - self._regularization_target) * self._regularization_weight
            q[~self._has_reg_target] = 0

        P = self._costs(q_obs)

        A = self._A(q_obs) if self._A is not None else None
        b = self._b(q_obs) if self._b is not None else None
        
        G = self._G(q_obs) if self._G is not None else None
        h = self._h(q_obs) if self._h is not None else None

        if self._dampening_factor is not None:
            limits = (self._x_limits * self._dampening_factor) + self._last_x_dot[..., None]
        else:
            limits = self._x_limits

        if self._position_integration_factor is not None:
            # Gap to the boundary scaled by temporal integration step
            pos_limit_gap = (self._q_pos_limits - q[:len(self._control_symbols),None]) / self._position_integration_factor
            limits.T[0, :len(self._control_symbols)] = np.max((limits.T[0, :len(self._control_symbols)], pos_limit_gap.T[0]), axis=0)
            limits.T[1, :len(self._control_symbols)] = np.min((limits.T[1, :len(self._control_symbols)], pos_limit_gap.T[1]), axis=0)

        x_dot = solve_qp(P=P, q=q,
                         A=A, b=b, G=G, h=h,
                         lb=limits.T[0],
                         ub=limits.T[1],
                         solver=self._solver)
        if x_dot is None:
            raise ValueError('Solver has not returned anything. Indicates that there is no solution.')
        self._last_x_dot = x_dot
        return dict(zip(self._control_symbols, x_dot))
    
    def eval_objectives(self, q_obs : dict[kv.Symbol, float]) -> dict[str, np.ndarray]:
        out = {}
        for n, o in list(self._eq_objectives.items()) + list(self._ineq_objectives.items()):
            out[n] = o.expr(q_obs)
        return out

    def is_satisfied(self, tol=1e-4) -> bool:
        if self._last_x_dot is None:
            return False
        return (np.abs(self._last_x_dot) <= tol).all()
