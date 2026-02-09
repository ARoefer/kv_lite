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


def interpolate_cspline(t : float | np.ndarray,
                        stamps     : np.ndarray,
                        positions  : np.ndarray,
                        velocities : np.ndarray=None) -> np.ndarray:
    t = np.clip(t, stamps.min(), stamps.max())

    idx_t1 = np.argmax(t < stamps[:,None], axis=0)
    idx_t0 = idx_t1 - 1

    p0 = positions[idx_t0]
    p1 = positions[idx_t1]

    if velocities is None:
        v0 = np.zeros_like(p0)
        v1 = np.zeros_like(p1)
    else:
        v0 = velocities[idx_t0]
        v1 = velocities[idx_t1]
    
    t0 = stamps[idx_t0]
    dt = stamps[idx_t1] - t0
    f  = ((t - t0) / dt).reshape((-1, 1))

    return (2*f**3 - 3*f**2 + 1) * p0 + (f**3-2*f**2+f) * v0 + (-2*f**3 + 3*f**2) * p1 + (f**3-f**2) * v1

def retime_spline(spline : np.ndarray,
                  vel_limits : np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Retimes a spline such that all velocities within all segments
       never exceed the given limits. Returns the re-timed and limited spline.

    Args:
        spline (np.ndarray): Spline as (T, N, 2) where the last dimension is position and velocity.
        vel_limits (np.ndarray): Velocity limits to enforce (N).
                                 Limits are assumed to be symmetrical. Need to be positive.

    Returns:
        np.ndarray: _description_
    """
    if (vel_limits <= 0).any():
        raise ValueError(f'Velocity limits contain 0 or negative numbers. Min: {vel_limits.min()}')

    pos  = spline[..., 0]
    # First we ensure that all velocity control-points are within the specified limits
    vels = np.clip(spline[..., 1], -vel_limits, vel_limits)

    # We convert the first order derivative of the spline into the form ax**2+bx+c
    # by computing a, b, and c for every time-step
    a = 6 * pos[:-1] - 6 * pos[1:] + 3 * vels[1:]
    b = -6 * pos[:-1] + 6 * pos[1:] - vels[:-1] - 2 * vels[1:]
    c = vels[:-1]

    # Per-segment roots, assuming a is not 0
    roots_1 = -(np.sqrt((b**2)-4*a*c) + b) / (2*a)
    roots_2 = (np.sqrt((b**2)-4*a*c) - b) / (2*a)

    # These peaks can be outside of a segment
    # All peaks can only ever be within a segment
    t_peaks = np.clip((roots_1 + roots_2) / 2, 0, 1)
    # If a is zero, there's a second solution
    t_peaks[a == 0] = np.clip((-c / b)[a == 0], 0, 1)
    t_peaks[np.isnan(t_peaks)] = 0

    # Evaluating the peak_vels in each interval
    peak_vels = t_peaks**2*a + t_peaks*b + vels[:-1]

    # Evaluating how close the peak velocities are to the limits
    # Values below 1 indicate that joints can go faster, above
    # 1 shows that they are going too fast
    vel_frac = np.abs(peak_vels / vel_limits)
    
    # The fastest joint limits the segment
    seg_lens = np.max(vel_frac, axis=-1)

    new_stamps = np.hstack(([0], np.cumsum(seg_lens)))

    return new_stamps, np.stack((pos, vels), axis=-1)


try:
    import robotic as ry
    from robotic import nlp


    class PathRetimeProblem(nlp.NLP):
        """Low-acceleration retiming as per Section IV.B of https://arxiv.org/pdf/2203.05390"""
        def __init__(self, path : np.ndarray, vel_limits : np.ndarray, alpha=0.1):
            # (K+1)
            self._path   = path
            self._alpha  = alpha

            K = self._path.shape[0] - 1
            N = self._path.shape[1]
            self._full_limits = np.zeros((K, 1+N, 2))
            self._full_limits[:, 1:, 0] = -vel_limits
            self._full_limits[:, 1:, 1] =  vel_limits
            # No Segment is shorter than 100ms or longer than 100s
            self._full_limits[:,  0, :] = (1e-1, 100)
            # Cutting of the last v_K we do not have
            self._full_limits = self._full_limits.reshape((-1, 2))[:(K * (1+N)) - N]

            # We construct a sparse Jacobian to fill later
            # We will have K tau-s, K-1 v-s. Each tau has influence on 2N losses, each v on 4.
            # Additionally each tau has an influence on the overall loss, which we add as an individual loss to 
            self._J_CACHE = -np.ones(((K - 1) * N * 4 + K * (2 * N + 1), 3))
            # All x-indices of optimization vars including the unavailable v_K
            var_indices = np.arange(K * (1 + N)).reshape((K, -1))
            tau_indices = var_indices[:, 0]
            # All indices of velocity decision variables, excluding v_0 and v_K
            vel_indices = var_indices[:-1, 1:].flatten()
            # Number of losses in a segment
            step_width = N * 2
            # Diagonal for activation loss
            offset = len(tau_indices)
            self._tau_offset = offset
            self._J_CACHE[:offset,  1] = tau_indices
            self._J_CACHE[:offset,  0] = np.arange(len(tau_indices))
            self._J_CACHE[:offset, -1] = 1
            # Setting repeating ranges of tau-s along X
            self._J_CACHE[offset:offset + offset*2*N, 1] = (np.zeros(step_width)[None] + tau_indices[:,None]).flatten()
            # Setting the y-coordinates of the losses. They just count up.
            self._J_CACHE[offset:offset + offset*step_width, 0] = offset + np.arange(step_width*offset)
            # Moving to coords of v0
            offset = offset + offset*step_width
            self._v0_offset = offset
            # Doubling each v index to compactly write dD and dV together
            self._J_CACHE[offset:offset + len(vel_indices)*2, 1] = (np.zeros(2)[None] + vel_indices[:,None]).flatten()
            #                                                      V0 does not exist for k=1
            self._J_CACHE[offset:offset + len(vel_indices)*2, 0] = self._tau_offset + step_width + ((np.arange(K-1) * step_width)[:,None] + np.arange(step_width).reshape((2, -1)).T.flatten()[None]).flatten()
            # Moving to coords of v1
            offset = offset + len(vel_indices)*2
            self._v1_offset = offset
            self._J_CACHE[offset:offset + len(vel_indices)*2, 1] = (np.zeros(2)[None] + vel_indices[:,None]).flatten()
            self._J_CACHE[offset:offset + len(vel_indices)*2, 0] = self._tau_offset + ((np.arange(K-1) * step_width)[:,None] + np.arange(step_width).reshape((2, -1)).T.flatten()[None]).flatten()

        def evaluate(self, x : np.ndarray):
            """x is assumed to be a stack of delta-timings and velocities
            though we assume v_0 and v_K to be zero.
            This leaves us with K-1 velocities and tau_K segment lengths.
            The packing is (tau_1, v_1, ..., tau_K-1, v_K-1, tau_K)

            Args:
                x (np.ndarray): Packed values.
            
            Returns:
                (np.ndarray, np.ndarray): Linear cost vector (K*2N) and sparse Jacobian (K, (K-1)*(N+1)+1)
            """
            # Expecting K-1 velocities and tau-s and one K-th tau
            assert np.prod(x.shape) == (self._path.shape[0] - 1) * (self._path.shape[1] + 1) - self._path.shape[1]
            # Grabbing all tau (K)
            taus = x[::self._path.shape[1]+1]
            # Tacking the start and end velocities onto the decision variables
            # (K+1, N)
            vels = np.vstack((np.zeros(self._path.shape[1]),
                            x[:-1].reshape((-1, self._path.shape[1] + 1))[:, 1:],
                            np.zeros(self._path.shape[1])))
            
            # Computing segment costs. Section IV.B of https://arxiv.org/pdf/2203.05390
            # Eq. (15) - (K, N)
            V = vels[1:] - vels[:-1]
            # (K, N)
            D = self._path[1:] - self._path[:-1] - (taus[..., None] / 2) * (vels[:-1] + vels[1:])
            # Eq. (16)
            D_bar_fac = np.sqrt(12) * taus**(-1.5)
            D_bar     = D_bar_fac[..., None] * D
            V_bar_fac = taus**(-0.5)
            V_bar     = V_bar_fac[..., None] * V
            # We are not applying the squaring or summing from Eq. (16)
            # as the solver does internally.
            # We stack the costs so that the flattened cost for each segment is
            # always one consecutive block.
            # (K, 2N)
            cost = np.hstack((D_bar, V_bar))

            expander = np.zeros((1, D.shape[1]))

            # DERIVATIVES
            # (K, N)
            D_bar_d_tau = (np.sqrt(3) * (-6 * self._path[1:] + 6 * self._path[:-1] + taus[..., None] * (vels[:-1] + vels[1:]))) / (2*taus[..., None]**(2.5))
            # Need to force an expansion back to full dimension
            D_bar_d_v0 = (-D_bar_fac * (taus/2))[..., None] + expander
            D_bar_d_v1 = -D_bar_d_v0

            # (K, N)
            V_bar_d_v0  = -V_bar_fac[..., None] + expander
            V_bar_d_v1  = -V_bar_d_v0
            V_bar_d_tau = -V / (2*taus[..., None]**(1.5))

            # Setting Jac for tau
            J_tau = np.hstack((D_bar_d_tau, V_bar_d_tau)).flatten()
            self._J_CACHE[self._tau_offset:self._tau_offset+len(J_tau), -1] = self._alpha * J_tau

            # First row is of constant velocity
            J_v0 = np.stack((D_bar_d_v0, V_bar_d_v0), axis=-1)[1:].flatten()
            self._J_CACHE[self._v0_offset:self._v0_offset+len(J_v0), -1] = self._alpha * J_v0

            # Last row is of constant velocity
            J_v1 = np.stack((D_bar_d_v1, V_bar_d_v1), axis=-1)[:-1].flatten()
            self._J_CACHE[self._v1_offset:self._v1_offset+len(J_v1), -1] = self._alpha * J_v1

            return np.hstack((taus, self._alpha * cost.flatten())), self._J_CACHE

        def getInitializationSample(self):
            # K velocities
            init = np.ones((self._path.shape[0] - 1,  self._path.shape[1] + 1))
            low_vel_bounds, high_vel_bounds = self._full_limits.T[:, 1:1+self._path.shape[1]]
            init[:, 1:] = np.clip(self._path[1:] - self._path[:-1], low_vel_bounds, high_vel_bounds)

            return init.flatten()[:-self._path.shape[1]] # Removing v_K

        def f(self, x: np.ndarray):
            raise NotImplementedError()

        def getFHessian(self, x):
            return []

        def getDimension(self) -> int:
            return (self._path.shape[0] - 1) * (1 + self._path.shape[1]) - self._path.shape[1]
        
        def getFeatureTypes(self):
            return [ry.OT.sos] * ((self._path.shape[0] - 1) * (2 * self._path.shape[1] + 1))
        
        def getBounds(self):
            return self._full_limits.T
        
        def report(self, verbose):
            return "RAI Path retiming"
        

    def retime_path(path : np.ndarray, limits : np.ndarray, alpha=0.1, iter_limit=200, verbose=0) -> tuple[np.ndarray, np.ndarray]:
        problem = PathRetimeProblem(path, limits, alpha)

        solver = ry.NLP_Solver()
        solver.setPyProblem(problem)
        solver.setSolver(ry.OptMethod.augmentedLag)
        solver.setOptions(stepMax=iter_limit, verbose=verbose)
        solver_return = solver.solve(0, verbose=verbose)
        
        taus = solver_return.x[::path.shape[1] + 1]
        stamps = np.hstack(((0.0,), np.cumsum(taus)))
        inner_velocities = solver_return.x[:-1].reshape((path.shape[0] - 2, -1))[:, 1:]
        full_velocities  = np.vstack((np.zeros(path.shape[1]),
                                    inner_velocities,
                                    np.zeros(path.shape[1])))
        return stamps, np.stack((path, full_velocities), axis=-1)

except (ModuleNotFoundError, ImportError) as e:
    def retime_path(path : np.ndarray, limits : np.ndarray, alpha=0.1, iter_limit=200, verbose=0) -> tuple[np.ndarray, np.ndarray]:
        raise ModuleNotFoundError(f'Cannot use RAI because you are missing dependencies. Use "pip install kineverse[rai]" to install them. Original exception: {e}')
