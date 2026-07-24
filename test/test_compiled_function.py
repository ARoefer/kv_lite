import casadi as ca
import numpy  as np
import pytest

from kv_lite.math import (_CompiledFunction, _speed_up,
                           Position, EvaluationError)


class TestSpeedUp:
    def test_returns_compiled_function(self):
        x = Position('cf_su_x')
        assert isinstance(_speed_up(x._ca_data * 2, [x], (1,)), _CompiledFunction)

    def test_scalar_output_shape(self):
        x = Position('cf_su_sx')
        f = _speed_up(x._ca_data * 2, [x], (1,))
        assert f.shape == (1,)

    def test_vector_output_shape(self):
        x, y = Position('cf_su_vx'), Position('cf_su_vy')
        f    = _speed_up(ca.vertcat(x._ca_data, y._ca_data), [x, y], (2,))
        assert f.shape == (2,)


class TestCallWithDict:
    def test_correct_value(self):
        x = Position('cf_ev_x')
        f = _speed_up(x._ca_data * 3, [x], (1,))
        assert f({x: 4.0}) == pytest.approx(12.0)

    def test_missing_key_raises(self):
        x = Position('cf_miss_x')
        f = _speed_up(x._ca_data, [x], (1,))
        with pytest.raises(EvaluationError):
            f({})

    def test_extra_keys_are_ignored(self):
        x, y = Position('cf_extra_x'), Position('cf_extra_y')
        f    = _speed_up(x._ca_data * 2, [x], (1,))
        assert f({x: 3.0, y: 99.0}) == pytest.approx(6.0)

    def test_argument_order_from_params_not_dict(self):
        x, y = Position('cf_ord_x'), Position('cf_ord_y')
        # Compile x - y with params in order [y, x]
        # __call__ must feed args as [y_val, x_val] regardless of dict order
        g = _speed_up(x._ca_data - y._ca_data, [y, x], (1,))
        assert g({x: 5.0, y: 2.0}) == pytest.approx(3.0)


class TestCallUnchecked:
    def test_single_input(self):
        x = Position('cf_uc_x')
        f = _speed_up(x._ca_data * 2, [x], (1,))
        np.testing.assert_allclose(f.call_unchecked(np.array([3.0])), [6.0])

    def test_batched_input(self):
        x      = Position('cf_uc_bx')
        f      = _speed_up(x._ca_data * 2, [x], (1,))
        result = f.call_unchecked(np.array([[3.0], [5.0]]))
        np.testing.assert_allclose(result, [[6.0], [10.0]])

    def test_multi_param_single_input(self):
        x, y = Position('cf_uc_mx'), Position('cf_uc_my')
        f    = _speed_up(x._ca_data + y._ca_data, [x, y], (1,))
        np.testing.assert_allclose(f.call_unchecked(np.array([3.0, 4.0])), [7.0])

    def test_multi_param_batched(self):
        x, y   = Position('cf_uc_mbx'), Position('cf_uc_mby')
        f      = _speed_up(x._ca_data + y._ca_data, [x, y], (1,))
        result = f.call_unchecked(np.array([[1.0, 2.0], [3.0, 4.0]]))
        np.testing.assert_allclose(result, [[3.0], [7.0]])

    def test_vector_output(self):
        x, y   = Position('cf_uc_vx'), Position('cf_uc_vy')
        f      = _speed_up(ca.vertcat(x._ca_data * 2, y._ca_data * 3), [x, y], (2,))
        result = f.call_unchecked(np.array([1.0, 2.0]))
        np.testing.assert_allclose(result, [2.0, 6.0])

    def test_vector_output_batched(self):
        x, y   = Position('cf_uc_vbx'), Position('cf_uc_vby')
        f      = _speed_up(ca.vertcat(x._ca_data * 2, y._ca_data * 3), [x, y], (2,))
        result = f.call_unchecked(np.array([[1.0, 2.0], [3.0, 1.0]]))
        np.testing.assert_allclose(result, [[2.0, 6.0], [6.0, 3.0]])

    def test_output_shape_single(self):
        x = Position('cf_sh_x')
        f = _speed_up(x._ca_data * 2, [x], (1,))
        assert f.call_unchecked(np.array([1.0])).shape == (1,)

    def test_output_shape_batched(self):
        x = Position('cf_sh_bx')
        f = _speed_up(x._ca_data * 2, [x], (1,))
        assert f.call_unchecked(np.array([[1.0], [2.0], [3.0]])).shape == (3, 1)

    def test_leading_dims_broadcast(self):
        x      = Position('cf_ld_x')
        f      = _speed_up(x._ca_data * 2, [x], (1,))
        result = f.call_unchecked(np.array([[1.0], [2.0], [3.0]]))
        np.testing.assert_allclose(result, [[2.0], [4.0], [6.0]])
