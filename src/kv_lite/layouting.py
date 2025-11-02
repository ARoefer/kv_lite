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

       The vectorized layout differentiates between per-timestep symbols and shared
       symbols. If more than one consistent symbol stamp is used, the stamps != None
       identify symbols from a series. The distance between the different stamps identifies
       their delta in time. The reference is only ever backwards, so tagging a a_t=0 and a_t=5
       leads to the eval handler requesting the time step t-5 for the current step t.
    """
    def __init__(self, expr : gm.KVArray,
                       t_steps : list[int],
                       args_series : list[gm.KVSymbol],
                       args_shared : list[gm.KVSymbol],
                       delta_t : float=1.0,
                       order : int=0,
                       weights : float | np.ndarray=None,
                       bias    : float | np.ndarray=None,
                       diff_symbols : set[gm.KVSymbol]=None,
                       value_offset : int=0,
                       arg_offset   : int=0):
        if order > 3:
            raise NotImplementedError(f'Currently only support order until 3. You gave {order}')
        
        if type(args_series) not in {list, tuple, np.ndarray, gm.KVArray}:
            raise ValueError(f'Arguments need to be provided as an ordered type. Type "{type(args_series)}" is unordered.')
        
        if type(args_shared) not in {list, tuple, np.ndarray, gm.KVArray}:
            raise ValueError(f'Arguments need to be provided as an ordered type. Type "{type(args_shared)}" is unordered.')

        args_shared = gm.array([s.set_stamp(None) for s in args_shared])
        args_series = gm.array([s.set_stamp(None) for s in args_series])

        if len(intersection:=set(args_shared) & set(args_series)) > 0:
            raise ValueError(f'Intersection of shared args and series args {intersection}')

        self._original_diff_symbols = diff_symbols
        diff_symbols = diff_symbols if diff_symbols is not None else set(args_shared) | set(args_series)
        self._t_steps = t_steps
        self._order   = order
        self._required_steps = list(range(self._t_steps[0] - self._order, self._t_steps[0])) + self._t_steps
        self._series_symbols = args_series
        self._shared_symbols = args_shared
        self._n_shared_syms  = len(self._shared_symbols)
        self._n_shared_diffs = len(diff_symbols & set(self._shared_symbols))

        # TODO: Add explicit marking of shared and series symbols

        stamped_symbols   = {s.set_stamp(None) for s in expr.symbols if s.stamp is not None}
        unstamped_symbols = {s for s in expr.symbols if s.stamp is None}

        if len(undefined:=stamped_symbols - set(self._series_symbols)) > 0:
            raise ValueError(f'Some symbols in expression are stamped but not identified as series symbols: {undefined}')
        
        if len(undefined:=unstamped_symbols - set(self._series_symbols) - set(self._shared_symbols)) > 0:
            raise ValueError(f'Some symbols in expression are unstamped but not identified as arguments: {undefined}')

        stamps = {v.stamp for v in expr.symbols}
        if len(stamps) > 1:
            if None in stamps:
                stamps.remove(None)

            max_stamp = max(stamps)
            sorted_stamps = list(sorted(stamps))
            relative_steps = np.asarray([s - max_stamp for s in sorted_stamps if s is not None])
            self._step_reorder = np.asarray(self._required_steps)[:,None] + relative_steps[None]
            self._required_steps = sorted(set(self._step_reorder.flatten()))
            self._step_reorder -= self._step_reorder.min()
            J_coord_offsets = (0, relative_steps.min() * len(self._series_symbols))
            if (relative_steps[relative_steps != 0] >= -self._order).any():
                raise ValueError(f'Stride overlaps with higher-order. This is not permitted.')

            # Generate the arguments needed for evaluation. We stack them from lowest to highest t.
            # We always lead with the shared arguments            
            self._eval_args = gm.hstack([self._shared_symbols] + [self._series_symbols.set_stamp(stamp) for stamp in sorted_stamps])
            if diff_symbols is None:
                self._syms_derivative = self._eval_args
            else:
                self._syms_derivative = gm.hstack([s for s in self._shared_symbols if s in diff_symbols] + 
                                                  sum([[s.set_stamp(stamp) for s in self._series_symbols if s in diff_symbols] for stamp in sorted_stamps], []))
        else:
            if next(iter(stamps)) is not None:
                expr = expr.set_stamp(None)

            self._step_reorder = None
            J_coord_offsets = (0, 0)
            self._eval_args = gm.hstack([self._shared_symbols, self._series_symbols])
            self._syms_derivative = self._eval_args if diff_symbols is None else self._eval_args[np.isin(self._eval_args, list(diff_symbols))]

        self._unstamped_diff_symbols = {s.set_stamp(None) for s in self._syms_derivative}
        self._width_step_derivative  = len(self._unstamped_diff_symbols) - self._n_shared_diffs
        self._expr    = gm.VEval(expr.reshape((-1,)), self._eval_args)

        self._J_coords, self._J_sparse = expr.squeeze().jacobian(self._syms_derivative).reshape((-1, len(self._syms_derivative))).to_coo()
        self._J_shared = self._J_coords[:, 1] < self._n_shared_diffs
        self._J_coords[~self._J_shared] += J_coord_offsets
        self._J_eval  = gm.VEval(self._J_sparse, self._eval_args)
        self._delta_t = delta_t
        self._weights = weights
        self._bias    = bias
        self._pad_steps = sum([s < 0 for s in self._required_steps])
    
        self.layout(value_offset, arg_offset)

    def layout(self, value_offset : int, arg_offset : int):
        macro_block_offsets = np.asarray([(0, -x) for x in range(self._order + 1)])[::-1] * self._width_step_derivative #  - self._n_shared_diffs)

        # Coordinates of the Jacobian of an entire time step (E, J_W * (order+1))
        # Ordered t-o, t-o+1, ..., t
        step_J_coords = (macro_block_offsets[:,None] + self._J_coords[None]).reshape((-1, 2))

        t_block_offsets = np.asarray([(x * self._expr.shape[0],
                                       x * (self._width_step_derivative)) for x in range(len(self._t_steps))])

        full_J_coords_steps = t_block_offsets[:,None] + step_J_coords[None]
        # We're resetting the horizontal offsets of all shared vars
        shared_mask = np.hstack([self._J_shared] * (self._order + 1))
        full_J_coords_steps += (value_offset, arg_offset)
        # Resetting the x locations of all shared symbols
        if shared_mask.any():
            full_J_coords_steps[:, shared_mask, 1] = np.hstack([self._J_coords[self._J_shared, 1]] * (self._order + 1))
        full_J_coords = full_J_coords_steps.reshape((-1, 2))

        # We cannot have multiple connections of a shared var to a step,
        # so we exclude the ones introduced by the higher-order estimates.
        keep_shared_mask = shared_mask.copy()
        keep_shared_mask[len(self._J_shared):] = False

        # TODO: Figure out J-mask and C offsets
        self._J_MASK  = ((full_J_coords_steps[..., 1] >= self._J_shared.sum()) | keep_shared_mask) & (full_J_coords_steps[..., 0] >= 0)
        self._J_MASK  = self._J_MASK.reshape((-1,))
        self._J_CACHE = np.empty((full_J_coords.shape[0], 3))
        self._J_CACHE[:, :2] = full_J_coords
        
        self._J_DATA_VIEW = np.lib.stride_tricks.as_strided(self._J_CACHE[0, 2:],
                                                            (len(self._t_steps), self._order + 1, len(self._J_coords)),
                                                            ((self._order + 1) * len(self._J_coords) * self._J_CACHE.strides[0],
                                                             len(self._J_coords) * self._J_CACHE.strides[0],
                                                             self._J_CACHE.strides[0]))

    def reorder_symbols(self, new_series_order : list[gm.KVSymbol],
                              new_shared_order : list[gm.KVSymbol],
                              value_offset : int=0, arg_offset : int=0) -> "VectorizedLayout":
        return VectorizedLayout(self._expr._e,
                                self._t_steps,
                                new_series_order,
                                new_shared_order,
                                self._delta_t,
                                self._order,
                                self._weights,
                                self._bias,
                                self._original_diff_symbols,
                                value_offset,
                                arg_offset)

    @property
    def diff_symbols(self) -> set[gm.KVSymbol]:
        """Symbols that a derivative is computed for."""
        return self._unstamped_diff_symbols

    @property
    def series_symbols(self) -> gm.KVArray[gm.KVSymbol]:
        """Symbols that exist for each time step."""
        return self._series_symbols

    @property
    def shared_symbols(self) -> gm.KVArray[gm.KVSymbol]:
        """Symbols that are shared across time steps."""
        return self._shared_symbols

    @cached_property
    def symbols(self) -> set[gm.KVSymbol]:
        """All symbols of the cost term."""
        return self._expr.symbols

    @cached_property
    def ordered_symbols(self) -> set[gm.KVSymbol]:
        """All symbols of the cost term in their vectorized evaluation order."""
        return self._expr.ordered_symbols

    @cached_property
    def unstamped_symbols(self) -> set[gm.KVSymbol]:
        """All symbols but without any time stamps."""
        return {s.set_stamp(None) for s in self.symbols}

    @cached_property
    def required_steps(self) -> list[int]:
        """List of step indices that this layout needs for evaluation."""
        return self._required_steps[self._pad_steps:]

    @cached_property
    def dim(self) -> int:
        """Size of the evaluated vector."""
        return (np.prod(self._expr.shape) if type(self._weights) != np.ndarray else self._weights.shape[0]) * len(self._t_steps)

    @cached_property
    def J_size(self) -> int:
        """Number of non-zero entries in the Jacobian."""
        return self._J_MASK.sum()

    @cached_property
    def pad_size(self) -> tuple[int, int]:
        """Shape of the needed-padding values, meaning the required arguments for time steps < 0."""
        return (self._pad_steps, len(self.series_symbols))

    @property
    def t_steps(self) -> list[int]:
        """Time steps that values are computed for."""
        return self._t_steps

    def eval_expr(self, series : np.ndarray, shared : np.ndarray=None, series_pad_values : np.ndarray=None) -> np.ndarray:
        """Evaluates the function at a specific point. The values of the series and
           the shared variables are provided separately as (T, N) and (M) arrays.
           It is also possible to provide (P, N) pad values or a shape/value
           that broadcasts to (P, N).

        Args:
            series (np.ndarray): Series values (T, N).
            shared (np.ndarray, optional): Shared values (M). Defaults to None.
            series_pad_values (np.ndarray, optional): Padding values broadcasting to (P, N). Defaults to None.

        Returns:
            np.ndarray: Evaluated vector (Z).
        """
        return self._eval_expr(self._make_M(series, shared, series_pad_values))

    def eval_J(self, series : np.ndarray, shared : np.ndarray=None, series_pad_values : np.ndarray=None) -> np.ndarray:
        """Evaluates the Jacobian at a specific point. The values of the series and
           the shared variables are provided separately as (T, N) and (M) arrays.
           It is also possible to provide (P, N) pad values or a shape/value
           that broadcasts to (P, N).

        Args:
            series (np.ndarray): Series values (T, N).
            shared (np.ndarray, optional): Shared values (M). Defaults to None.
            series_pad_values (np.ndarray, optional): Padding values broadcasting to (P, N). Defaults to None.

        Returns:
            np.ndarray: Evaluated sparse Jacobian (3, *) as (row, column, value).
        """
        return self._eval_J(self._make_M(series, shared, series_pad_values))
    
    def eval_all(self, series : np.ndarray, shared : np.ndarray=None, series_pad_values : np.ndarray=None) -> tuple[np.ndarray, np.ndarray]:
        """Evaluates function and Jacobian simultaneously at a point. Slightly faster than
           individual evaluation.

        Args:
            series (np.ndarray): Series values (T, N).
            shared (np.ndarray, optional): Shared values (M). Defaults to None.
            series_pad_values (np.ndarray, optional): Padding values broadcasting to (P, N). Defaults to None.

        Returns:
            tuple[np.ndarray, np.ndarray]: Evaluated function as (Z), evaluated sparse Jacobian (3, *).
        """
        big_M = self._make_M(series, shared, series_pad_values)
        return self._eval_expr(big_M), self._eval_J(big_M)

    def _make_M(self, series : np.ndarray, shared : np.ndarray=None, series_pad_values : np.ndarray=None) -> np.ndarray:
        assert np.prod(series.shape[:-1]) == len(self.required_steps)

        if self._pad_steps > 0:
            series_pad = np.empty((series.shape[0] + self._pad_steps, series.shape[1]))
            series_pad[:self._pad_steps] = series[0] if series_pad_values is None else series_pad_values
            series_pad[self._pad_steps:] = series
        else:
            series_pad = series

        if self._step_reorder is not None:
            series_pad = series_pad[self._step_reorder].reshape((self._step_reorder.shape[0], -1))

        big_M = np.empty((series_pad.shape[0],
                          len(self._eval_args)))
        big_M[:, :self._n_shared_syms] = shared
        big_M[:, self._n_shared_syms:] = series_pad

        return big_M
    
    def _eval_expr(self, big_M : np.ndarray) -> np.ndarray:
        e_out = self._expr(big_M)
        
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
            return out.reshape((-1,))
        return (out + self._bias).reshape((-1,))
    
    def _eval_J(self, big_M : np.ndarray) -> np.ndarray:
        # Dense representation of the Jacobian
        # (T, O+1, V, Q)
        # 
        # 0 J/x_{t-O}  J/x_{t-O+1} .... J/x_{t} 
        # 0 ... J/x_{t-O}  J/x_{t-O+1} .... J/x_{t} 

        match self._order:
            case 0:
                # Dense Jacobian to return
                self._J_DATA_VIEW[:] = self._J_eval(big_M).reshape(self._J_DATA_VIEW.shape)
            case 1:
                # J of velocity
                J_temp  = self._J_eval(big_M)
                J_temp /= self._delta_t
                self._J_DATA_VIEW[:, 0] = -J_temp[:-1].reshape(s:=self._J_DATA_VIEW[:, 0].shape)
                self._J_DATA_VIEW[:, 1] =  J_temp[1: ].reshape(s)
            case 2:
                # J of acceleration
                # (x_t - 2x_{t-1} + x_{t-2}) / dt^2
                J_temp  = self._J_eval(big_M)
                dt_sq = (self._delta_t**2)
                self._J_DATA_VIEW[:, 2] = J_temp[:-2].reshape(s:=self._J_DATA_VIEW[:, 2].shape) / dt_sq
                self._J_DATA_VIEW[:, 1] = (-2 * J_temp[1:-1].reshape(s)) / dt_sq
                self._J_DATA_VIEW[:, 0] = J_temp[2:].reshape(s) / dt_sq
            case 3:
                # (x_T - 3x_{t-1} + 3x_{t-2} - x_{t-3}) / dt^3
                J_temp  = self._J_eval(big_M)
                dt_cube = (self._delta_t**3)
                self._J_DATA_VIEW[:, 3] = -J_temp[:-3].reshape(s:=self._J_DATA_VIEW[:, 3].shape) / dt_cube
                self._J_DATA_VIEW[:, 2] = (3 / dt_cube) * J_temp[1:-2].reshape(s)
                self._J_DATA_VIEW[:, 1] = (-3 / dt_cube) * J_temp[2:-1].reshape(s)
                self._J_DATA_VIEW[:, 0] = J_temp[3:].reshape(s) / dt_cube

        # Summing the step-wise impact of shared vars into the first slot.
        if self._order > 0 and self._n_shared_diffs > 0:
            self._J_DATA_VIEW[:, 0, :self._n_shared_diffs] = self._J_DATA_VIEW[:, :, :self._n_shared_diffs].sum(axis=1)

        if self._weights is not None:
            self._J_DATA_VIEW *= self._weights
        
        if self._J_MASK is not None:
            return self._J_CACHE[self._J_MASK]
        return self._J_CACHE


class MacroLayout():
    """Builds a consistent evaluation layout for multiple VectorizedLayouts and
       manages their evaluation.
    """
    def __init__(self, components : dict[str, VectorizedLayout],
                 bounds : dict[gm.KVSymbol, tuple[float, float]]=None,
                 default_bound=1e6):
        self._components = components
        all_steps = set(sum([c.required_steps for c in components.values()], []))
        if 0 not in all_steps:
            raise ValueError('Time step 0 is not referenced by components.')

        sorted_steps = np.asarray(sorted(all_steps))
        if sorted_steps.min() < 0:
            raise ValueError('Somehow a component is defining a cost for a negative time step.')
        
        if (sorted_steps[1:] - sorted_steps[:-1] != 1).any():
            raise ValueError(f'Given problem does not densely cover all timesteps. Steps:\n  {sorted_steps}')

        # We don't neeed this anymore. There's logic for merging the symbols below
        # all_series_symbols = set(sum([list(c.unstamped_symbols) for c in self._components.values()], []))
        # if len(a:=[c for c in self._components.values() if len(c.unstamped_symbols) != len(all_series_symbols)]) > 0:
        #     raise ValueError(f'Non-overlapping series-symbols in {a}')
        
        self._n_series_steps = len(all_steps)

        series_symbols = set()
        shared_symbols = set()
        for x, (n, c) in enumerate(components.items()):
            series_symbols |= set(c.series_symbols)
            if c.shared_symbols is not None:
                shared_symbols |= set(c.shared_symbols)
            if len(intersect:=(shared_symbols & series_symbols)) > 0:
                conflicts = []
                for n_p, c_p in list(components.items())[:x]:
                    if len(set(c_p.series_symbols) & intersect) > 0:
                        conflicts.append(n_p)
                raise ValueError(f'Processing {n} led to an overlap of series symbols and shared symbols: {intersect}\n  These were identified as series symbols by: {", ".join(conflicts)}')

        self._shared_symbols = list(shared_symbols)
        self._series_symbols = list(series_symbols)
        self._diff_symbols   = frozenset(sum([list(c.diff_symbols) for c in self._components.values()], []))
        updated_components = {}

        series_width = len(self.active_series_symbols)
        component_arg_offsets   = [min(c.t_steps) * series_width for c in self._components.values()]
        self._component_value_offsets = [0]
        self._component_J_offsets     = [0]
        for (n, c), arg_offset in zip(self._components.items(), component_arg_offsets):
            # Apply layout
            updated_components[n] = c.reorder_symbols(self._series_symbols,
                                                      self._shared_symbols,
                                                      self._component_value_offsets[-1],
                                                      arg_offset)
            self._component_value_offsets.append(self._component_value_offsets[-1] + updated_components[n].dim)
            self._component_J_offsets.append(self._component_J_offsets[-1] + updated_components[n].J_size)
        del self._component_value_offsets[-1]
        del self._component_J_offsets[-1]
        self._components = updated_components

        bounds = {} if bounds is None else bounds
        self._bounds = np.array([bounds.get(s, [-default_bound, default_bound]) for s in self._series_symbols if s in self._diff_symbols] * self._n_series_steps)
        if len(set(self._diff_symbols) & set(self._shared_symbols)) > 0:
            self._bounds = np.vstack(([bounds.get(s, [-default_bound, default_bound]) for s in self._shared_symbols if s in self._diff_symbols], 
                                      self._bounds))

        self._J_size = sum([c.J_size for c in self._components.values()])
        self._x_shape = (len(sorted_steps), series_width)

    @property
    def bounds(self) -> np.ndarray:
        """Bounds of variables that are selected for optimization in their evaluation order. (V, 2) as (Low, High)."""
        return self._bounds

    @cached_property
    def diff_mask(self) -> np.ndarray:
        """Masks in all symbols the ones which a derivative is computed for."""
        return np.hstack(([s in self._diff_symbols for s in self._shared_symbols] + self._n_series_steps * [s in self._diff_symbols for s in self._series_symbols]))

    @cached_property
    def diff_symbols(self) -> gm.KVArray:
        """Symbols that a derivative is computed for."""
        return self.in_symbols[self.diff_mask]

    @property
    def active_symbols(self) -> frozenset[gm.KVSymbol]:
        """Synonymous with `set(self.diff_symbols)`."""
        return self._diff_symbols

    @cached_property
    def active_shared_symbols(self) -> frozenset[gm.KVSymbol]:
        """Active symbols shared across time steps."""
        return frozenset({s for s in self._shared_symbols if s in self.active_symbols})

    @cached_property
    def active_series_symbols(self) -> frozenset[gm.KVSymbol]:
        """Active symbols that belong to a specific time step."""
        return frozenset({s for s in self._series_symbols if s in self.active_symbols})

    @property
    def series_symbols(self) -> gm.KVArray:
        """Symbols belonging to the time series in vector order."""
        return self._series_symbols
    
    @property
    def shared_symbols(self) -> gm.KVArray:
        """Symbols belonging shared across time steps in vector order."""
        return self._shared_symbols

    @property
    def n_series_steps(self) -> int:
        """Number of time series steps."""
        return self._n_series_steps

    @cached_property
    def in_dim(self) -> int:
        """Size of the full vector for evaluation."""
        return len(self._shared_symbols) + len(self._series_symbols) * self._n_series_steps

    @cached_property
    def out_dim(self) -> int:
        """Size of the full computed residual."""
        return sum([c.dim for c in self._components.values()])

    @cached_property
    def in_symbols(self) -> gm.KVArray:
        """All symbols needed to evaluate a point."""
        return gm.hstack([self.shared_symbols] + [self.series_symbols] * self.n_series_steps)

    @cached_property
    def pad_size(self) -> tuple[int, int]:
        """Shape of the needed time series padding."""
        return np.vstack([c.pad_size for c in self._components.values()]).max(axis=0)

    def _make_M_and_S(self, x : np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return x[:len(self._shared_symbols)], x[len(self._shared_symbols):].reshape((-1, len(self._series_symbols)))

    def eval_all(self, x : np.ndarray, pads : dict[str, np.ndarray]=None) -> tuple[np.ndarray, np.ndarray]:
        """Evaluates both value and Jacobian at a point. The point must be presented
           as a stacked vector following `self.in_symbols` in order. Padding can be
           overridden, but is given as a dictionary of values that broadcast to shape P.

        Args:
            x (np.ndarray): Point to evaluate (*).
            pads (dict[str, np.ndarray], optional): Padding values to use {s: (->P)}. Defaults to None.

        Returns:
            tuple[np.ndarray, np.ndarray]: Evaluated function as (Z), evaluated sparse Jacobian (3, *).
        """
        x_shared, x_series = self._make_M_and_S(x)
        out_expr = np.empty(self.out_dim, dtype=float)
        out_J = np.empty((self._J_size, 3))
        for (n, c), v_offset, j_offset in zip(self._components.items(),
                                              self._component_value_offsets,
                                              self._component_J_offsets):
            out_expr[v_offset:v_offset+c.dim], out_J[j_offset:j_offset+c.J_size] = c.eval_all(x_series[c.required_steps],
                                                                                              x_shared,
                                                                                              pads)

        if out_J[:,:2].min() < 0:
            raise ValueError(f'Sparse Jacobian coordinates are less than 0.')
        
        if ((self.out_dim, self.diff_mask.sum()) - out_J[:,:2].max(axis=0) < 1).any():
            raise ValueError(f'Sparse Jacobian coordinates are out of limits. Limit: {self.out_dim - 1, self.in_dim - 1}, Given: {out_J[:,:2].max(axis=0)}.')
        return out_expr, out_J

    def eval_expr(self, x : np.ndarray, pads : dict[str, np.ndarray]=None) -> np.ndarray:
        return self._eval_expr(*self._make_M_and_S(x), pads)

    def eval_J(self, x : np.ndarray, pads : dict[str, np.ndarray]=None) -> np.ndarray:
        return self._eval_J(*self._make_M_and_S(x), pads)

    def _eval_expr(self, x_shared : np.ndarray, x_series: np.ndarray, pads : np.ndarray=None) -> np.ndarray:
        out_expr = np.empty(self.out_dim, dtype=float)
        for (n, c), offset in zip(self._components.items(),
                                  self._component_value_offsets):
            out_expr[offset:offset+c.dim] = c.eval_expr(x_series[c.required_steps],
                                                        x_shared,
                                                        pads).flatten()
        return out_expr
    
    def _eval_J(self, x_shared : np.ndarray, x_series: np.ndarray, pads : np.ndarray=None) -> np.ndarray:
        out_J = np.empty((self._J_size, 3))
        for (n, c), offset in zip(self._components.items(),
                                  self._component_J_offsets):
            out_J[offset:offset+c.J_size] = c.eval_J(x_series[c.required_steps],
                                                     x_shared,
                                                     pads)

        if out_J[:,:2].min() < 0:
            raise ValueError(f'Sparse Jacobian coordinates are less than 0.')
        
        if ((self.out_dim, self.in_dim) - out_J[:,:2].max(axis=0) < 1).any():
            raise ValueError(f'Sparse Jacobian coordinates are out of limits. Limit: {self.out_dim - 1, self.in_dim - 1}, Given: {out_J[:,:2].max(axis=0)}.')
        return out_J

    def report(self, x : np.ndarray) -> dict[str, np.ndarray]:
        """Reports for each sub-layout its value at the given point `x` which
           has to follow the value order given by `self.in_symbols`.

        Args:
            x (np.ndarray): Encoded point to evaluate (*).

        Returns:
            dict[str, float]: Values of sub-layouts at x (*).
        """
        out_x = self.eval_expr(x)
        out_d = {}
        for (cn, c), offset in zip(self._components.items(),
                                   self._component_value_offsets):
            out_d[cn] = out_x[offset:offset+c.dim].reshape((len(c._t_steps), -1))
        return out_d


import robotic as ry
from robotic import nlp
SolverObjectives = ry.OT


class RAI_NLP(nlp.NLP):
    """Low-level bridge to the NLP solver interface of ry/rai/komo.
    """
    def __init__(self, objectives : dict[str, tuple[ry.OT, VectorizedLayout]],
                       bounds : dict[gm.KVSymbol, tuple[float, float]],
                       default_bound=1e6,
                       constants : np.ndarray | dict[gm.KVSymbol, float | np.ndarray]=None,
                       pads : np.ndarray | dict[gm.KVSymbol, float]=None,
                       init : np.ndarray | dict[gm.KVSymbol, float]=None):
        """Instantiates a new NLP problem. Objectives are given as named layouts
           accompanied by the objective type. If the layouts do not differentiate
           fully against all their variables, then this interface requires a value
           assignment for these constant values. This can be overriden later as well.

        Args:
            objectives (dict[str, tuple[ry.OT, VectorizedLayout]]): Objectives given as Layouts and objective type.
            bounds (dict[gm.KVSymbol, tuple[float, float]]): Bounds for optimization variables.
            default_bound (float, optional): Default bound which is applied to all variables
                                             that do not have a custom one. Defaults to 1e6.
            constants (np.ndarray | dict[gm.KVSymbol, float | np.ndarray], optional): Value of variables that are constants. Defaults to None.
            pads (np.ndarray | dict[gm.KVSymbol, float], optional): Padding values for the time series. Defaults to None.
            init (np.ndarray | dict[gm.KVSymbol, float], optional): Initial value to propose to the solver if it asks. Defaults to None.
        """

        self._layout = MacroLayout({n: v for n, (_, v) in objectives.items()},
                                   bounds,
                                   default_bound)
        
        self._features = sum([[o] * v.dim for o, v in objectives.values()], [])

        self._X_CACHE = np.empty(self._layout.in_dim)
        if not self._layout.diff_mask.all():
            if constants is None:
                raise ValueError(f'Expected {(~self._layout.diff_mask).sum()} constant values.')
            self.set_constants(constants)

        self._logging_active = False
        self._log = None
        
        self._pads = None
        if pads is not None:
            self.set_pads(pads)
        
        self._init = None
        if init is not None:
            self.set_init(init)

    @property
    def active_symbols(self) -> frozenset[gm.KVSymbol]:
        """Symbols that are being optimized."""
        return self._layout.active_symbols

    @property
    def active_shared_symbols(self) -> frozenset[gm.KVSymbol]:
        """Symbols that are being optimized and shared across time steps."""
        return self._layout.active_shared_symbols

    @property
    def active_series_symbols(self) -> frozenset[gm.KVSymbol]:
        """Symbols that are being optimized and identify a time step."""
        return self._layout.active_series_symbols

    @property
    def series_symbols(self) -> gm.KVArray:
        """All symbols of the time series."""
        return self._layout.series_symbols
    
    @property
    def shared_symbols(self) -> gm.KVArray:
        """All symbols shared across time steps."""
        return self._layout.shared_symbols

    @property
    def n_series_steps(self) -> int:
        """Number of time steps in the series."""
        return self._layout.n_series_steps

    def deactivate_logging(self):
        """Turn off logging."""
        self._logging_active = False

    def reset_log(self):
        """Clear the last log and activate the logging feature."""
        self._logging_active = True
        self._log = None

    @property
    def log(self) -> dict[str, np.ndarray]:
        """If logging is active, this log holds the stacked layout
           outputs of all evaluations since the last call of `self.reset_log()`.

        Returns:
            dict[str, np.ndarray]: Layout outputs {str: (E, *)} where is the number of calls to `evaluate`.
        """
        return self._log

    def set_init(self, new_init : np.ndarray | dict[gm.KVSymbol, float | np.ndarray]):
        """Sets a new initial sample that the optimizer can query."""
        if isinstance(new_init, dict):
            self._init = self._init if self._init is not None else np.zeros(self.getDimension())
            for s, v in new_init.items():
                self._init[np.isin(self._layout.diff_symbols, [s])] = v
        else:
            self._init = new_init

    def set_constants(self, new_constants : np.ndarray | dict[gm.KVSymbol, float | np.ndarray]):
        """Sets new values for the constants in the problem."""
        if isinstance(new_constants, dict):
            for s, v in new_constants.items():
                self._X_CACHE[np.isin(self._layout.in_symbols, [s]) & (~self._layout.diff_mask)] = v
        else:
            self._X_CACHE[~self._layout.diff_mask] = new_constants
    
    def set_pads(self, new_pads : np.ndarray | dict[gm.KVSymbol, float | np.ndarray]):
        """Sets new padding values for the series."""
        if isinstance(new_pads, dict):
            self._pads = self._pads if self._pads is not None else np.zeros(self._layout.pad_size)
            for s, v in new_pads.items():
                self._pads.T[np.isin(self._layout.series_symbols, [s])] = v
        else:
            self._pads = new_pads

    def evaluate(self, x: np.ndarray):
        """Evaluates the problem at point x and returns value and sparse Jacobian."""
        self._X_CACHE[self._layout.diff_mask] = x
        phi, J = self._layout.eval_all(self._X_CACHE, self._pads)
        if self._logging_active:
            log = self._layout.report(self._X_CACHE)
            if self._log is None:
                self._log = log
            else:
                for k, v in log.items():
                    self._log[k] = np.vstack((self._log[k], v))
        return phi, J
    
    def objectives_report(self, x : np.ndarray) -> dict[str, np.ndarray]:
        """Generates a report for the point `x`.

        Args:
            x (np.ndarray): Point to evaluate (`self.getDimension()`).

        Returns:
            dict[str, np.ndarray]: Non-squared value of all objectives at `x`.
        """
        x_copy = self._X_CACHE.copy()
        x_copy[self._layout.diff_mask] = x
        return self._layout.report(x_copy)

    def f(self, x: np.ndarray):
        raise NotImplementedError()

    def getFHessian(self, x):
        return []

    def getDimension(self) -> int:
        return self._layout.diff_mask.sum()

    def getFeatureTypes(self):
        return self._features

    def getInitializationSample(self):
        return self._init

    def getBounds(self):
        return self._layout.bounds.T

    def report(self, verbose):
        return "RAI NLP Layout"

    def make_full_solution(self, x : np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Decodes a point `x` into a 1D block of shared symbols and a (T, N) block of series symbols.
           Note: These blocks include the values of constants.
        """
        x_copy = self._X_CACHE.copy()
        x_copy[self._layout.diff_mask] = x
        return x_copy[:len(self._layout.shared_symbols)], x_copy[len(self._layout.shared_symbols):].reshape((-1, len(self._layout.series_symbols)))


class RAI_NLPSolver():
    """Wraps the interaction with the rai/ry/komo NLP-solver.
       Internally builds the `RAI_NLP` for the given objectives and
       manages the instantiation and operation of the solver and decodes
       its output.
    """
    def __init__(self, objectives : dict[str, tuple[ry.OT, VectorizedLayout]],
                       bounds : dict[gm.KVSymbol, tuple[float, float]]=None,
                       default_bound=1e6,
                       constants : np.ndarray=None,
                       pads : np.ndarray=None,
                       init : np.ndarray=None):
        """Instantiates a new NLP Solver. Objectives are given as named layouts
           accompanied by the objective type. If the layouts do not differentiate
           fully against all their variables, then this interface requires a value
           assignment for these constant values. This can be overriden later as well.

        Args:
            objectives (dict[str, tuple[ry.OT, VectorizedLayout]]): Objectives given as Layouts and objective type.
            bounds (dict[gm.KVSymbol, tuple[float, float]]): Bounds for optimization variables.
            default_bound (float, optional): Default bound which is applied to all variables
                                             that do not have a custom one. Defaults to 1e6.
            constants (np.ndarray | dict[gm.KVSymbol, float | np.ndarray], optional): Value of variables that are constants. Defaults to None.
            pads (np.ndarray | dict[gm.KVSymbol, float], optional): Padding values for the time series. Defaults to None.
            init (np.ndarray | dict[gm.KVSymbol, float], optional): Initial value to propose to the solver if it asks. Defaults to None.
        """
        self._nlp = RAI_NLP(objectives, bounds, default_bound, constants, pads, init)
    
    def solve(self, /,
                    init_sample=None,
                    constants=None,
                    pads=None,
                    stepMax=0.5,
                    damping=1e-4,
                    stopEvals=500,
                    verbose=1,
                    logging=False) -> tuple[np.ndarray, np.ndarray, ry.SolverReturn]:
        """Invokes the solver. Provides many options for overriding the initials,
           constants and padding values for this run. Also exposes some of the
           internal solver options. Note that this *always* uses the AuLa solver.

        Args:
            init_sample (np.ndarray, optional): Override the starting point of the optimization. Defaults to None.
            constants (np.ndarray | dict[gm.KVSymbol, float | np.ndarray], optional): Override the constants in the problem. Defaults to None.
            pads (np.ndarray | dict[gm.KVSymbol, float | np.ndarray], optional): Override the padding of the time series. Defaults to None.
            stepMax (float, optional): Max step size of the solver. Defaults to 0.5.
            damping (float, optional): _description_. Defaults to 1e-4.
            stopEvals (int, optional): Max number of evals the solver can do. Defaults to 500.
            verbose (int, optional): Verbosity of the solver. Defaults to 1.
            logging (bool, optional): Activates logging of the inner NLP.
                                      The resulting log is available under `self.log`. Defaults to False.

        Returns:
            tuple[np.ndarray, np.ndarray, ry.SolverReturn]: Shared symbols, series symbols, inner solver return.
        """
        if pads is not None:
            self._nlp.set_pads(pads)
        if constants is not None:
            self._nlp.set_constants(constants)
        
        if logging:
            self._nlp.reset_log()
        else:
            self._nlp.deactivate_logging()

        solver = ry.NLP_Solver()
        solver.setPyProblem(self._nlp)
        solver.setSolver(ry.OptMethod.augmentedLag)
        solver.setInitialization(init_sample)
        solver.setOptions(stepMax=stepMax, damping=damping, stopEvals=stopEvals, verbose=verbose)
        solver_return = solver.solve(0, verbose=verbose)

        return self._nlp.make_full_solution(solver_return.x) + (solver_return,)

    @property
    def series_symbols(self) -> gm.KVArray:
        return self._nlp.series_symbols
    
    @property
    def shared_symbols(self) -> gm.KVArray:
        return self._nlp.shared_symbols

    def report(self, x : np.ndarray) -> dict[str, np.ndarray]:
        return self._nlp.objectives_report(x)
    
    def set_pads(self, new_pads : np.ndarray):
        self._nlp.set_pads(new_pads)

    @property
    def log(self):
        return self._nlp.log

    @property
    def bounds(self) -> np.ndarray:
        return self._nlp.getBounds()