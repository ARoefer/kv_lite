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
                       delta_t : float,
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
        
        # Y coordinates: number of steps * dimensions * (order + 1) * J_width
        y_coords = (value_offset + (np.arange(len(self._required_steps) * self._expr.shape[0])[...,None] - self._pad_steps * self._expr.shape[0]) + np.zeros(self._J.shape[1] * (self._order + 1))).flatten()
        # X coordinates
        x_coords = (arg_offset + (np.arange(self._J.shape[1] * (self._order + 1)) + np.zeros((self._J.shape[0], 1))) + (np.arange(self._required_steps[0], self._required_steps[-1] + 1) * self._J.shape[1])[None,:,None,None]).flatten()
        coord_mask = (y_coords >= 0) & (x_coords >= 0)
        
        self._J_CACHE = np.empty((coord_mask.sum(), 3))
        self._J_CACHE[:, 0] = y_coords[coord_mask]
        self._J_CACHE[:, 1] = x_coords[coord_mask]

    @cached_property
    def required_steps(self) -> list[int]:
        return self._required_steps[self._pad_steps:]

    @cached_property
    def height(self) -> int:
        return (np.prod(self._expr.shape) if type(self._weights) != np.ndarray else self._weights.shape[0]) * self._t_steps

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
                            (self._order + 1),
                            self._J.shape[0],
                            self._J.shape[1]))

        match self._order:
            case 0:
                # Dense Jacobian to return
                J_dense = self._J(M_pad)
            case 1:
                # J of velocity
                J_temp  = self._J(M_pad)
                J_temp /= self._delta_t
                J_dense[:,0] = -J_temp[:-1]
                J_dense[:,1] =  J_temp[1: ]
            case 2:
                # J of acceleration
                # (x_t - 2x_{t-1} + x_{t-2}) / dt^2
                J_temp  = self._J(M_pad)
                dt_sq = (self._delta_t**2)
                J_dense[:,0] = J_temp[:-2] / dt_sq
                J_dense[:,1] = (-2 * J_temp[1:-1]) / dt_sq
                J_dense[:,2] = J_temp[2:] / dt_sq
            case 3:
                # (x_T - 3x_{t-1} + 3x_{t-2} - x_{t-3}) / dt^3
                J_temp  = self._J(M_pad)
                dt_cube = (self._delta_t**3)
                J_dense[:,0] = -J_temp[:-3] / dt_cube
                J_dense[:,1] = (3 / dt_cube) * J_temp[1:-2]
                J_dense[:,2] = (-3 / dt_cube) * J_temp[2:-1]
                J_dense[:,3] = J_temp[3:] / dt_cube

        if self._weights is not None:
            J_dense = self._weights @ J_dense if type(self._weights) == np.ndarray else self._weights * J_dense
        
        self._J_CACHE[:, 2] = J_dense.flatten()[-len(self._J_CACHE):]
        return self._J_CACHE