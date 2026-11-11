# Copyright (c) 2026 Adrian Röfer, Robot Learning Lab, University of Freiburg
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

"""Position-limit handling of ``QPController`` (``position_integration_factor``).

A two-joint arm whose objective asks every joint for a fixed velocity: the command that
comes back is then exactly the velocity bound the controller derived for that joint.
"""

import numpy  as np
import pytest

import kv_lite as kv

from kv_lite.control import QPController, Objective, ObjectiveTypes


URDF = """<robot name="arm">
  <link name="base"/><link name="l1"/><link name="l2"/>
  <joint name="j1" type="revolute"><parent link="base"/><child link="l1"/><axis xyz="0 0 1"/>
    <limit lower="-1.0" upper="1.0" velocity="2.0" effort="1"/></joint>
  <joint name="j2" type="revolute"><parent link="l1"/><child link="l2"/><origin xyz="1 0 0"/><axis xyz="0 0 1"/>
    <limit lower="-0.5" upper="0.5" velocity="2.0" effort="1"/></joint>
</robot>"""

HORIZON = 0.1    # [s] position_integration_factor
V_MAX   = 2.0    # [rad/s] velocity limit of both joints
PUSH    = 10.0   # [rad/s] requested velocity, beyond V_MAX so that the bounds always bind


def _controller(goal_vel : np.ndarray, weight : float | None=1000.0) -> tuple[QPController, list]:
    """Controller asking each joint for ``goal_vel``, soft by default so the bounds win."""
    robot = kv.urdf.load_urdf(kv.Model(), URDF)
    q     = robot.q
    track = Objective(ObjectiveTypes.EQUALITY, expr=kv.array(q), gain=np.asarray(goal_vel, float), weight=weight)
    return QPController(robot, {'track': track}, control_symbols=q, control_costs={},
                        position_integration_factor=HORIZON), q


def _command(ctrl : QPController, q : list, positions : list[float]) -> np.ndarray:
    cmd = ctrl.eval(dict(zip(q, positions)))
    return np.array([cmd[s] for s in q])


def test_velocity_is_capped_to_reach_the_limit_within_the_horizon():
    ctrl, q = _controller([PUSH, -PUSH])
    cmd     = _command(ctrl, q, [0.95, -0.45])
    np.testing.assert_allclose(cmd, [0.05 / HORIZON, -0.05 / HORIZON], atol=1e-6)


def test_far_from_the_limits_the_velocity_limits_apply():
    ctrl, q = _controller([PUSH, -PUSH])
    np.testing.assert_allclose(_command(ctrl, q, [0.0, 0.0]), [V_MAX, -V_MAX], atol=1e-6)


def test_velocity_limits_are_not_overwritten_by_a_solve():
    ctrl, q = _controller([PUSH, PUSH])
    before  = ctrl._x_limits.copy()
    _command(ctrl, q, [0.99, 0.49])
    np.testing.assert_array_equal(ctrl._x_limits, before)
    np.testing.assert_allclose(_command(ctrl, q, [0.0, 0.0]), [V_MAX, V_MAX], atol=1e-6)


@pytest.mark.parametrize('weight', [1000.0, None])
def test_past_a_limit_the_qp_stays_feasible(weight):
    """A joint far past its limit may not move further out, and is not forced back."""
    ctrl, q = _controller([0.0, 0.0], weight=weight)
    np.testing.assert_allclose(_command(ctrl, q, [1.5, -1.0]), [0.0, 0.0], atol=1e-6)


def test_past_a_limit_the_joint_may_move_back_but_not_further_out():
    ctrl, q = _controller([PUSH, -PUSH])                 # both pushing further out
    np.testing.assert_allclose(_command(ctrl, q, [1.5, -1.0]), [0.0, 0.0], atol=1e-6)

    ctrl, q = _controller([-PUSH, PUSH])                 # both heading back inside
    np.testing.assert_allclose(_command(ctrl, q, [1.5, -1.0]), [-V_MAX, V_MAX], atol=1e-6)
