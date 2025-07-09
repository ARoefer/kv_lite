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
