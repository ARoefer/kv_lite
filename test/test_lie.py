# Copyright (c) 2024 Adrian Röfer, Robot Learning Lab, University of Freiburg
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
import gtsam

import kv_lite as kv

if __name__ == '__main__':
    print('TESTING SO3')
    test_so3 = (np.random.random((10000, 3)) - 0.5) * 2 * np.pi

    # Setting up test expressions
    # SO3 Expmap
    rv  = kv.array([kv.Symbol(f'r_{x}') for x in 'xyz'])
    expr_SO3 = kv.SO3.expmap(rv, epsilon=0)
    expr_SO3.set_symbol_order(rv)
    
    # SO3 Logmap
    SO3 = kv.array([[kv.Symbol(f'm_{x}{y}') for x in 'xyz'] for y in 'xyz'])
    expr_so3 = kv.SO3.logmap(SO3, epsilon=0)
    expr_so3.set_symbol_order(SO3.flatten())

    kv_SO3    = expr_SO3(test_so3)
    gtsam_SO3 = np.array([gtsam.Rot3.Expmap(v).matrix() for v in test_so3])
    delta_SO3 = gtsam_SO3 - kv_SO3

    print(f'Expmap Delta:\n== Mean:\n{delta_SO3.mean(axis=0)}\n== Std:\n{delta_SO3.std(axis=0)}')

    kv_so3    = expr_so3(kv_SO3.reshape((-1, 9)))
    gtsam_so3 = np.array([gtsam.Rot3.Logmap(gtsam.Rot3(R)) for R in gtsam_SO3])
    # print(f'GTSAM-GT-Delta: {np.abs(test_so3 - gtsam_so3).max()}')

    delta_so3 = gtsam_so3 - kv_so3
    print(f'Logmap Delta:\n== Mean:\n{np.abs(delta_so3).mean(axis=0)}\n== Std:\n{delta_so3.std(axis=0)}')

    # ==========================================================
    #                          SE3
    # ==========================================================
    print('TESTING SE3')
    test_se3 = (np.random.random((10000, 6)) - 0.5) * 2 * np.pi

    # Setting up test expressions
    # SE3 Expmap
    tv  = kv.array([kv.Symbol(x) for x in 'wx wy wz rx ry rz'.split(' ')])
    expr_SE3 = kv.SE3.expmap(tv[:3], tv[-3:], epsilon=0)
    expr_SE3.set_symbol_order(tv)
    
    # SE3 Logmap
    SE3 = kv.array([[kv.Symbol(f'm_{x}{y}') for x in 'xyzw'] for y in 'xyzw'])
    expr_se3 = kv.SE3.logmap(SE3, epsilon=0)
    expr_se3.set_symbol_order(SE3.flatten())

    kv_SE3    = expr_SE3(test_se3)
    gtsam_SE3 = np.array([gtsam.Pose3.Expmap(v).matrix() for v in test_se3])
    delta_SE3 = gtsam_SE3 - kv_SE3

    print(f'Expmap Delta:\n== Mean:\n{delta_SE3.mean(axis=0)}\n== Std:\n{delta_SE3.std(axis=0)}')

    kv_se3    = expr_se3(kv_SE3[:, :3].reshape((-1, 12)))
    gtsam_se3 = np.array([gtsam.Pose3.Logmap(gtsam.Pose3(M)) for M in gtsam_SE3])
    # print(f'GTSAM-GT-Delta: {np.abs(test_se3 - gtsam_so3).max()}')

    delta_se3 = gtsam_se3 - kv_se3
    print(f'Logmap Delta:\n== Mean:\n{delta_se3.mean(axis=0)}\n== Std:\n{delta_se3.std(axis=0)}')
