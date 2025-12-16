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
    t_peaks[(a == 0) & (b == 0)] = 0

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


