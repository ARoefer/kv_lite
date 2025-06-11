# Copyright (c) 2025 Adrian Röfer, Robot Learning Lab, University of Freiburg
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.

import numpy as np

from . import spatial as gm

from functools import cached_property


class VectorizedLayout():
    """Wrapper to create KOMO-style k-th order costs of expressions and compute the
       matching sparse Jacobian. Only supports constant and equal time steps.
    """
    def __init__(self, expr : gm.KVArray,
                       t_steps : list[int],
                       args : list[gm.KVSymbol],
                       delta_t : float=1.0,
                       order : int=0,
                       weights : float | np.ndarray=None,
                       bias    : float | np.ndarray=None,
                       value_offset : int=0,
                       arg_offset   : int=0):
        if order > 3:
            raise NotImplementedError(f'Currently only support order until 3. You gave {order}')
        
        self._t_steps = t_steps
        self._expr    = gm.VEval(expr.squeeze(), args)
        self._J       = gm.VEval(expr.squeeze().jacobian(args), args)
        self._delta_t = delta_t
        self._order   = order
        self._weights = weights
        self._bias    = bias
        self._required_steps = list(range(self._t_steps[0] - self._order, self._t_steps[0])) + self._t_steps
        self._pad_steps = sum([s < 0 for s in self._required_steps])
    
        self.layout(value_offset, arg_offset)

    def layout(self, value_offset : int, arg_offset : int):
        # Y coordinates: number of steps * dimensions * (order + 1) * J_width
        
        # Each time step creates the same number of values and is dependent on J-width times (order+1) vars
        xy_expected = len(self._t_steps) * self._expr.shape[0] * self._J.shape[1] * (self._order + 1)

        # Y coordinates
        y_coords = ((np.arange(len(self._t_steps) * self._expr.shape[0])[...,None]) + np.zeros(self._J.shape[1] * (self._order + 1))).flatten()
        # Checking that there are enough indices
        assert len(y_coords) == xy_expected
        # And that they do not leave the bounds of the series
        assert y_coords.min() == 0
        assert y_coords.max() == len(self._t_steps) * self._J.shape[0] - 1
        
        # X coordinates
        J_x_base_coords = (np.arange(self._J.shape[1] * (self._order + 1)) + np.zeros((self._J.shape[0], 1)))
        x_coords = (J_x_base_coords + ((np.arange(len(self._t_steps)) + self._required_steps[0]) * self._J.shape[1])[None,:,None,None]).flatten()
        # Checking that there are enough indices
        assert len(x_coords) == xy_expected
        # And that they do not leave the bounds of the arguments
        assert x_coords.min() == self._required_steps[0] * self._J.shape[1]
        assert x_coords.max() == (self._required_steps[-1] + 1) * self._J.shape[1] - 1
        coord_mask = (y_coords >= 0) & (x_coords >= 0)
        
        self._J_CACHE = np.empty((coord_mask.sum(), 3))
        self._J_CACHE[:, 0] = y_coords[coord_mask] + value_offset
        self._J_CACHE[:, 1] = x_coords[coord_mask] + arg_offset
        
        self._J_MASK = None
        if self._pad_steps > 0:
            self._J_MASK = np.ones((len(self._t_steps),
                                    self._J.shape[0],
                                    (self._order + 1),
                                    self._J.shape[1]), dtype=bool)
            for ps in range(self._pad_steps):
                self._J_MASK[ps,:,:self._pad_steps-ps] = False

    @cached_property
    def symbols(self) -> set[gm.KVSymbol]:
        return self._expr.symbols

    @cached_property
    def ordered_symbols(self) -> set[gm.KVSymbol]:
        return self._expr.ordered_symbols

    @cached_property
    def required_steps(self) -> list[int]:
        return self._required_steps[self._pad_steps:]

    @cached_property
    def dim(self) -> int:
        return (np.prod(self._expr.shape) if type(self._weights) != np.ndarray else self._weights.shape[0]) * len(self._t_steps)

    @cached_property
    def J_size(self) -> int:
        return self._J_CACHE.shape[0]

    def eval_expr(self, M : np.ndarray) -> np.ndarray:
        # Ensure that we evaluate exactly as much as we said we would
        assert np.prod(M.shape[:-1]) == len(self.required_steps)

        if self._pad_steps > 0:
            M_pad = np.empty((M.shape[0] + self._pad_steps, M.shape[1]))
            M_pad[:self._pad_steps] = M[0]
            M_pad[self._pad_steps:] = M
        else:
            M_pad = M

        e_out = self._expr(M_pad)
        
        match self._order:
            case 0:
                pass
            case 1:
                # Velocity approximation - backward difference
                e_out = (e_out[1:] - e_out[:-1]) / self._delta_t
            case 2:
                # Acceleration approximation - backward difference
                e_out = (e_out[2:] - 2 * e_out[1:-1] + e_out[:-2]) / (self._delta_t**2)
            case 3:
                # Jerk approximation - backward difference
                e_out = (e_out[3:] - 3 * e_out[2:-1] + 3 * e_out[1:-2] - e_out[:-3]) / (self._delta_t**3)

        if self._weights is None:
            out = e_out
        else:
            out = (self._weights @ e_out[...,None]).reshape((-1, self._weights.shape[-2])) if type(self._weights) == np.ndarray else self._weights * e_out
        if self._bias is None:
            return out
        return out + self._bias

    def eval_J(self, M : np.ndarray) -> np.ndarray:
        # Ensure that we evaluate exactly as much as we said we would
        assert np.prod(M.shape[:-1]) == len(self.required_steps)

        if self._pad_steps > 0:
            M_pad = np.empty((M.shape[0] + self._pad_steps, M.shape[1]))
            M_pad[:self._pad_steps] = M[0]
        else:
            M_pad = M

        # Dense representation of the Jacobian
        # (T, O+1, V, Q)
        # 
        # 0 J/x_{t-O}  J/x_{t-O+1} .... J/x_{t} 
        # 0 ... J/x_{t-O}  J/x_{t-O+1} .... J/x_{t} 
        J_dense = np.empty((len(self._t_steps),
                            self._J.shape[0],
                            (self._order + 1),
                            self._J.shape[1]))

        match self._order:
            case 0:
                # Dense Jacobian to return
                J_dense = self._J(M_pad)
            case 1:
                # J of velocity
                J_temp  = self._J(M_pad)
                J_temp /= self._delta_t
                J_dense[:,:,0] = -J_temp[:-1]
                J_dense[:,:,1] =  J_temp[1: ]
            case 2:
                # J of acceleration
                # (x_t - 2x_{t-1} + x_{t-2}) / dt^2
                J_temp  = self._J(M_pad)
                dt_sq = (self._delta_t**2)
                J_dense[:,:,0] = J_temp[:-2] / dt_sq
                J_dense[:,:,1] = (-2 * J_temp[1:-1]) / dt_sq
                J_dense[:,:,2] = J_temp[2:] / dt_sq
            case 3:
                # (x_T - 3x_{t-1} + 3x_{t-2} - x_{t-3}) / dt^3
                J_temp  = self._J(M_pad)
                dt_cube = (self._delta_t**3)
                J_dense[:,:,0] = -J_temp[:-3] / dt_cube
                J_dense[:,:,1] = (3 / dt_cube) * J_temp[1:-2]
                J_dense[:,:,2] = (-3 / dt_cube) * J_temp[2:-1]
                J_dense[:,:,3] = J_temp[3:] / dt_cube

        if self._weights is not None:
            J_dense = self._weights @ J_dense if type(self._weights) == np.ndarray else self._weights * J_dense
        
        if self._J_MASK is not None:
            self._J_CACHE[:, 2] = J_dense[self._J_MASK].flatten()
        else:
            self._J_CACHE[:, 2] = J_dense.flatten()
        return self._J_CACHE
    
class MacroLayout():
    def __init__(self, **components : VectorizedLayout):
        self._components = components
        all_steps = set(sum([c.required_steps for c in components.values()], []))
        if 0 not in all_steps:
            raise ValueError('Time step 0 is not referenced by components.')

        sorted_steps = np.asarray(sorted(all_steps))
        if sorted_steps.min() < 0:
            raise ValueError('Somehow a component is defining a cost for a negative time step.')
        
        if (sorted_steps[1:] - sorted_steps[:-1] != 1).any():
            raise ValueError(f'Given problem does not densly cover all timesteps. Steps:\n  {sorted_steps}')

        all_series_symbols = set(sum([c.ordered_symbols for c in self._components.values()], []))
        if len(a:=[c for c in self._components.values() if len(c.symbols) != len(all_series_symbols)]) > 0:
            raise ValueError(f'Non-overlapping series-symbols in {a}')
        
        self._n_series_steps = len(all_steps)

        series_width = len(all_series_symbols)
        component_arg_offsets   = [min(c.required_steps) * series_width for c in self._components.values()]
        self._component_value_offsets = [0]
        for c in self._components.values():
            # Apply layout
            c.layout(self._component_value_offsets[-1], 0)
            self._component_value_offsets.append(self._component_value_offsets[-1] + c.dim)
        del self._component_value_offsets[-1]

        self._component_J_offsets = [0]
        for c in self._components.values():
            # Apply layout
            self._component_J_offsets.append(self._component_J_offsets[-1] + c.J_size)
        del self._component_J_offsets[-1]
        self._J_size = sum([c.J_size for c in self._components.values()])

        self._x_shape = (len(sorted_steps), series_width)

    @property
    def series_symbols(self) -> list[gm.KVSymbol]:
        return next(iter(self._components.values())).ordered_symbols

    @property
    def n_series_steps(self) -> int:
        return self._n_series_steps

    @cached_property
    def in_dim(self) -> int:
        return len(self.series_symbols) * self._n_series_steps

    @cached_property
    def out_dim(self) -> int:
        return sum([c.dim for c in self._components.values()])

    def eval_expr(self, x : np.ndarray) -> np.ndarray:
        x = x.reshape(self._x_shape)
        out_expr = np.empty(self.out_dim, dtype=float)
        for c, offset in zip(self._components.values(),
                             self._component_value_offsets):
            out_expr[offset:offset+c.dim] = c.eval_expr(x[c.required_steps]).flatten()

        return out_expr

    def eval_J(self, x : np.ndarray) -> np.ndarray:
        x = x.reshape(self._x_shape)
        out_J    = np.empty((self._J_size, 3))
        for c, offset in zip(self._components.values(),
                             self._component_J_offsets):
            out_J[offset:offset+c.J_size] = c.eval_J(x[c.required_steps])

        if out_J[:,:2].min() < 0:
            raise ValueError(f'Sparse Jacobian coordinates are less than 0.')
        
        if ((self.out_dim, self.in_dim) - out_J[:,:2].max(axis=0) < 1).any():
            raise ValueError(f'Sparse Jacobian coordinates are out of limits. Limit: {self.out_dim - 1, self.in_dim - 1}, Given: {out_J[:,:2].max(axis=0)}.')

        return out_J

    def report(self, x : np.ndarray) -> dict[str, float]:
        out_x = self.eval_expr(x)
        out_d = {}
        for (cn, c), offset in zip(self._components.items(),
                                   self._component_value_offsets):
            out_d[cn] = out_x[offset:offset+c.dim].reshape((len(c._t_steps), -1))
        return out_d
