import numpy  as np
import pytest

import kv_lite.math as kvm

from kv_lite.math import (KVArray, KVExpr, Position,
                           sin,  cos,  tan,
                           asin, acos, atan,  arcsin, arccos, arctan,
                           asinh, acosh, tanh, atanh,  arcsinh, arccosh, arctanh,
                           sqrt, exp, log, atan2)
from kv_lite.math import abs as kv_abs   # shadows builtin otherwise
from kv_lite.math import min as kv_min
from kv_lite.math import max as kv_max


# ─── Wrapped unary functions ──────────────────────────────────────────────────

_UNARY = [
    (sin,   np.pi / 2,  1.0,         'sin'),
    (cos,   0.0,        1.0,         'cos'),
    (tan,   np.pi / 4,  1.0,         'tan'),
    (asin,  1.0,        np.pi / 2,   'asin'),
    (acos,  1.0,        0.0,         'acos'),
    (atan,  1.0,        np.pi / 4,   'atan'),
    (asinh, 0.0,        0.0,         'asinh'),
    (acosh, 1.0,        0.0,         'acosh'),
    (tanh,  0.0,        0.0,         'tanh'),
    (atanh, 0.0,        0.0,         'atanh'),
    (sqrt,  4.0,        2.0,         'sqrt'),
    (kv_abs, -3.0,      3.0,         'abs'),
    (exp,   0.0,        1.0,         'exp'),
    (log,   1.0,        0.0,         'log'),
]
_UNARY_IDS = [t[3] for t in _UNARY]


class TestUnaryNumericInput:
    @pytest.mark.parametrize('fn,val,expected,_', _UNARY, ids=_UNARY_IDS)
    def test_float_input_returns_correct_value(self, fn, val, expected, _):
        assert fn(val) == pytest.approx(expected, abs=1e-9)

    @pytest.mark.parametrize('fn,val,expected,_', _UNARY, ids=_UNARY_IDS)
    def test_float_input_returns_scalar(self, fn, val, expected, _):
        result = fn(val)
        assert not isinstance(result, (np.ndarray, KVExpr))


class TestUnarySymbolicInput:
    @pytest.mark.parametrize('fn,val,expected,name', _UNARY, ids=_UNARY_IDS)
    def test_kvexpr_input_returns_kvexpr(self, fn, val, expected, name):
        x = Position(f'mf_{name}_x')
        assert isinstance(fn(x), KVExpr)

    @pytest.mark.parametrize('fn,val,expected,name', _UNARY, ids=_UNARY_IDS)
    def test_kvexpr_input_evaluates_correctly(self, fn, val, expected, name):
        x      = Position(f'mf_{name}_ev')
        result = fn(x)
        assert result.eval({x: val}) == pytest.approx(expected, abs=1e-9)


class TestUnaryArrayInput:
    def test_list_input_returns_kvarray(self):
        assert isinstance(sin([0.0, np.pi / 2]), KVArray)

    def test_ndarray_input_returns_kvarray(self):
        assert isinstance(sin(np.array([0.0, np.pi / 2])), KVArray)

    def test_kvarray_numeric_returns_kvarray(self):
        assert isinstance(sin(KVArray([0.0, np.pi / 2])), KVArray)

    def test_kvarray_symbolic_returns_kvarray(self):
        x = Position('mf_arr_x')
        assert isinstance(sin(KVArray([x])), KVArray)

    def test_list_input_correct_values(self):
        np.testing.assert_allclose(sin([0.0, np.pi / 2]), [0.0, 1.0], atol=1e-9)

    def test_kvarray_symbolic_evaluates_correctly(self):
        x, y = Position('mf_arr_ex'), Position('mf_arr_ey')
        result = sin(KVArray([x, y]))
        np.testing.assert_allclose(result.eval({x: 0.0, y: np.pi / 2}), [0.0, 1.0], atol=1e-9)


class TestAliases:
    def test_arcsin_is_asin(self):
        assert arcsin is asin

    def test_arccos_is_acos(self):
        assert arccos is acos

    def test_arctan_is_atan(self):
        assert arctan is atan

    def test_arcsinh_is_asinh(self):
        assert arcsinh is asinh

    def test_arccosh_is_acosh(self):
        assert arccosh is acosh

    def test_arctanh_is_atanh(self):
        assert arctanh is atanh


# ─── atan2 ────────────────────────────────────────────────────────────────────

class TestAtan2:
    def test_two_floats(self):
        assert float(atan2(1.0, 1.0)) == pytest.approx(np.pi / 4)

    def test_returns_kvexpr(self):
        assert isinstance(atan2(1.0, 1.0), KVExpr)

    def test_kvexpr_kvexpr(self):
        y, x = Position('mf_a2_y'), Position('mf_a2_x')
        assert atan2(y, x).eval({y: 1.0, x: 1.0}) == pytest.approx(np.pi / 4)

    def test_kvexpr_float(self):
        y = Position('mf_a2_ky')
        assert atan2(y, 1.0).eval({y: 1.0}) == pytest.approx(np.pi / 4)

    def test_float_kvexpr(self):
        x = Position('mf_a2_kx')
        assert atan2(1.0, x).eval({x: 1.0}) == pytest.approx(np.pi / 4)

    def test_quadrant(self):
        y, x = Position('mf_a2_qy'), Position('mf_a2_qx')
        assert atan2(y, x).eval({y: -1.0, x: -1.0}) == pytest.approx(-3 * np.pi / 4)


# ─── min / max ────────────────────────────────────────────────────────────────

class TestMin:
    def test_numeric_array_full_reduction(self):
        assert kv_min(np.array([3.0, 1.0, 2.0])) == pytest.approx(1.0)

    def test_symbolic_full_reduction(self):
        x, y = Position('mf_min_x'), Position('mf_min_y')
        result = kv_min(KVArray([x, y]))
        assert result.eval({x: 3.0, y: 1.0}) == pytest.approx(1.0)
        assert result.eval({x: 1.0, y: 3.0}) == pytest.approx(1.0)

    def test_numeric_array_axis(self):
        arr = np.array([[1.0, 4.0], [3.0, 2.0]])
        np.testing.assert_allclose(kv_min(arr, axis=0), [1.0, 2.0])

    def test_numeric_array_keepdims(self):
        arr = np.array([3.0, 1.0, 2.0])
        assert kv_min(arr, keepdims=True).shape == (1,)


class TestMinHighDim:
    # ── Numeric (goes through numpy — ground truth for expected values) ───────

    def test_3d_axis0_shape(self):
        assert kv_min(np.arange(24.0).reshape(2, 3, 4), axis=0).shape == (3, 4)

    def test_3d_axis1_shape(self):
        assert kv_min(np.arange(24.0).reshape(2, 3, 4), axis=1).shape == (2, 4)

    def test_3d_axis2_shape(self):
        assert kv_min(np.arange(24.0).reshape(2, 3, 4), axis=2).shape == (2, 3)

    def test_3d_axis1_values(self):
        arr = np.arange(24.0).reshape(2, 3, 4)
        np.testing.assert_allclose(kv_min(arr, axis=1), np.min(arr, axis=1))

    def test_3d_multi_axis_shape(self):
        assert kv_min(np.arange(24.0).reshape(2, 3, 4), axis=(0, 2)).shape == (3,)

    def test_3d_multi_axis_values(self):
        arr = np.arange(24.0).reshape(2, 3, 4)
        np.testing.assert_allclose(kv_min(arr, axis=(0, 2)), np.min(arr, axis=(0, 2)))

    def test_3d_keepdims_axis1_shape(self):
        assert kv_min(np.arange(24.0).reshape(2, 3, 4), axis=1, keepdims=True).shape == (2, 1, 4)

    def test_3d_keepdims_multi_axis_shape(self):
        assert kv_min(np.arange(24.0).reshape(2, 3, 4), axis=(0, 2), keepdims=True).shape == (1, 3, 1)

    # ── Symbolic (goes through _pooling_helper) ───────────────────────────────

    def test_2d_symbolic_axis0_shape(self):
        x00, x01 = Position('mnh_mn_x00'), Position('mnh_mn_x01')
        x10, x11 = Position('mnh_mn_x10'), Position('mnh_mn_x11')
        arr = KVArray([[x00, x01], [x10, x11]])
        assert kv_min(arr, axis=0).shape == (2,)

    def test_2d_symbolic_axis0_values(self):
        x00, x01 = Position('mnh_mv_x00'), Position('mnh_mv_x01')
        x10, x11 = Position('mnh_mv_x10'), Position('mnh_mv_x11')
        arr  = KVArray([[x00, x01], [x10, x11]])
        args = {x00: 3.0, x01: 1.0, x10: 2.0, x11: 4.0}
        np.testing.assert_allclose(kv_min(arr, axis=0).eval(args), [2.0, 1.0])

    def test_2d_symbolic_axis1_shape(self):
        x00, x01 = Position('mnh_m1s_x00'), Position('mnh_m1s_x01')
        x10, x11 = Position('mnh_m1s_x10'), Position('mnh_m1s_x11')
        arr = KVArray([[x00, x01], [x10, x11]])
        assert kv_min(arr, axis=1).shape == (2,)

    def test_2d_symbolic_axis1_values(self):
        x00, x01 = Position('mnh_m1v_x00'), Position('mnh_m1v_x01')
        x10, x11 = Position('mnh_m1v_x10'), Position('mnh_m1v_x11')
        arr  = KVArray([[x00, x01], [x10, x11]])
        args = {x00: 3.0, x01: 1.0, x10: 2.0, x11: 4.0}
        np.testing.assert_allclose(kv_min(arr, axis=1).eval(args), [1.0, 2.0])

    def test_2d_symbolic_keepdims_axis0_shape(self):
        x00, x01 = Position('mnh_mk0_x00'), Position('mnh_mk0_x01')
        x10, x11 = Position('mnh_mk0_x10'), Position('mnh_mk0_x11')
        arr = KVArray([[x00, x01], [x10, x11]])
        assert kv_min(arr, axis=0, keepdims=True).shape == (1, 2)

    def test_2d_symbolic_keepdims_axis1_shape(self):
        x00, x01 = Position('mnh_mk1_x00'), Position('mnh_mk1_x01')
        x10, x11 = Position('mnh_mk1_x10'), Position('mnh_mk1_x11')
        arr = KVArray([[x00, x01], [x10, x11]])
        assert kv_min(arr, axis=1, keepdims=True).shape == (2, 1)

    def test_3d_symbolic_axis1_shape(self):
        # 2×3×2 = 12 symbols; reduce axis=1 → expected shape (2, 2)
        syms = [Position(f'mnh_m3_{i}') for i in range(12)]
        arr  = KVArray(np.array(syms, dtype=object).reshape(2, 3, 2))
        assert kv_min(arr, axis=1).shape == (2, 2)

    def test_3d_symbolic_axis1_values(self):
        syms = [Position(f'mnh_m3v_{i}') for i in range(12)]
        arr  = KVArray(np.array(syms, dtype=object).reshape(2, 3, 2))
        # row 0: [[0,1],[2,3],[4,5]] → min over axis=1 per column: [0,1]
        # row 1: [[6,7],[8,9],[10,11]] → min over axis=1 per column: [6,7]
        args = {s: float(i) for i, s in enumerate(syms)}
        np.testing.assert_allclose(kv_min(arr, axis=1).eval(args), [[0.0, 1.0], [6.0, 7.0]])


class TestMax:
    def test_numeric_array_full_reduction(self):
        assert kv_max(np.array([3.0, 1.0, 2.0])) == pytest.approx(3.0)

    def test_symbolic_full_reduction(self):
        x, y = Position('mf_max_x'), Position('mf_max_y')
        result = kv_max(KVArray([x, y]))
        assert result.eval({x: 3.0, y: 1.0}) == pytest.approx(3.0)
        assert result.eval({x: 1.0, y: 3.0}) == pytest.approx(3.0)

    def test_numeric_array_axis(self):
        arr = np.array([[1.0, 4.0], [3.0, 2.0]])
        np.testing.assert_allclose(kv_max(arr, axis=0), [3.0, 4.0])

    def test_numeric_array_keepdims(self):
        arr = np.array([3.0, 1.0, 2.0])
        assert kv_max(arr, keepdims=True).shape == (1,)


class TestMaxHighDim:
    # ── Numeric ───────────────────────────────────────────────────────────────

    def test_3d_axis0_shape(self):
        assert kv_max(np.arange(24.0).reshape(2, 3, 4), axis=0).shape == (3, 4)

    def test_3d_axis1_shape(self):
        assert kv_max(np.arange(24.0).reshape(2, 3, 4), axis=1).shape == (2, 4)

    def test_3d_axis1_values(self):
        arr = np.arange(24.0).reshape(2, 3, 4)
        np.testing.assert_allclose(kv_max(arr, axis=1), np.max(arr, axis=1))

    def test_3d_multi_axis_shape(self):
        assert kv_max(np.arange(24.0).reshape(2, 3, 4), axis=(0, 2)).shape == (3,)

    def test_3d_multi_axis_values(self):
        arr = np.arange(24.0).reshape(2, 3, 4)
        np.testing.assert_allclose(kv_max(arr, axis=(0, 2)), np.max(arr, axis=(0, 2)))

    def test_3d_keepdims_axis1_shape(self):
        assert kv_max(np.arange(24.0).reshape(2, 3, 4), axis=1, keepdims=True).shape == (2, 1, 4)

    def test_3d_keepdims_multi_axis_shape(self):
        assert kv_max(np.arange(24.0).reshape(2, 3, 4), axis=(0, 2), keepdims=True).shape == (1, 3, 1)

    # ── Symbolic ──────────────────────────────────────────────────────────────

    def test_2d_symbolic_axis0_shape(self):
        x00, x01 = Position('mxh_mn_x00'), Position('mxh_mn_x01')
        x10, x11 = Position('mxh_mn_x10'), Position('mxh_mn_x11')
        arr = KVArray([[x00, x01], [x10, x11]])
        assert kv_max(arr, axis=0).shape == (2,)

    def test_2d_symbolic_axis0_values(self):
        x00, x01 = Position('mxh_mv_x00'), Position('mxh_mv_x01')
        x10, x11 = Position('mxh_mv_x10'), Position('mxh_mv_x11')
        arr  = KVArray([[x00, x01], [x10, x11]])
        args = {x00: 3.0, x01: 1.0, x10: 2.0, x11: 4.0}
        np.testing.assert_allclose(kv_max(arr, axis=0).eval(args), [3.0, 4.0])

    def test_2d_symbolic_axis1_shape(self):
        x00, x01 = Position('mxh_m1s_x00'), Position('mxh_m1s_x01')
        x10, x11 = Position('mxh_m1s_x10'), Position('mxh_m1s_x11')
        arr = KVArray([[x00, x01], [x10, x11]])
        assert kv_max(arr, axis=1).shape == (2,)

    def test_2d_symbolic_axis1_values(self):
        x00, x01 = Position('mxh_m1v_x00'), Position('mxh_m1v_x01')
        x10, x11 = Position('mxh_m1v_x10'), Position('mxh_m1v_x11')
        arr  = KVArray([[x00, x01], [x10, x11]])
        args = {x00: 3.0, x01: 1.0, x10: 2.0, x11: 4.0}
        np.testing.assert_allclose(kv_max(arr, axis=1).eval(args), [3.0, 4.0])

    def test_2d_symbolic_keepdims_axis0_shape(self):
        x00, x01 = Position('mxh_mk0_x00'), Position('mxh_mk0_x01')
        x10, x11 = Position('mxh_mk0_x10'), Position('mxh_mk0_x11')
        arr = KVArray([[x00, x01], [x10, x11]])
        assert kv_max(arr, axis=0, keepdims=True).shape == (1, 2)

    def test_2d_symbolic_keepdims_axis1_shape(self):
        x00, x01 = Position('mxh_mk1_x00'), Position('mxh_mk1_x01')
        x10, x11 = Position('mxh_mk1_x10'), Position('mxh_mk1_x11')
        arr = KVArray([[x00, x01], [x10, x11]])
        assert kv_max(arr, axis=1, keepdims=True).shape == (2, 1)

    def test_3d_symbolic_axis1_shape(self):
        syms = [Position(f'mxh_m3_{i}') for i in range(12)]
        arr  = KVArray(np.array(syms, dtype=object).reshape(2, 3, 2))
        assert kv_max(arr, axis=1).shape == (2, 2)

    def test_3d_symbolic_axis1_values(self):
        syms = [Position(f'mxh_m3v_{i}') for i in range(12)]
        arr  = KVArray(np.array(syms, dtype=object).reshape(2, 3, 2))
        # row 0: [[0,1],[2,3],[4,5]] → max over axis=1 per column: [4,5]
        # row 1: [[6,7],[8,9],[10,11]] → max over axis=1 per column: [10,11]
        args = {s: float(i) for i, s in enumerate(syms)}
        np.testing.assert_allclose(kv_max(arr, axis=1).eval(args), [[4.0, 5.0], [10.0, 11.0]])
