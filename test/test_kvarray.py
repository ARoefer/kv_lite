import casadi as ca
import numpy  as np
import pytest

from kv_lite.math import (KVArray, KVExpr, KVSymbol, EvaluationError,
                           Position, Velocity)


class TestConstruction:
    def test_1d_shape(self):
        assert KVArray([1.0, 2.0, 3.0]).shape == (3,)

    def test_2d_shape(self):
        assert KVArray([[1.0, 2.0], [3.0, 4.0]]).shape == (2, 2)

    def test_from_kvexpr_list_is_object_dtype(self):
        x, y = Position('kac_x'), Position('kac_y')
        arr  = KVArray([x, y])
        assert arr.dtype == object

    def test_is_ndarray_subclass(self):
        assert isinstance(KVArray([1, 2, 3]), np.ndarray)


class TestIsZeroIsOne:
    def test_is_zero_all_zero(self):
        assert KVArray([0.0, 0.0]).is_zero

    def test_is_zero_not_all_zero(self):
        assert not KVArray([0.0, 1.0]).is_zero

    def test_is_one_all_one(self):
        assert KVArray([1.0, 1.0]).is_one

    def test_is_one_not_all_one(self):
        assert not KVArray([1.0, 0.0]).is_one

    def test_is_zero_symbolic_zero(self):
        x   = Position('iz_x')
        arr = KVArray([x * 0, KVExpr(ca.SX(0))])
        assert arr.is_zero

    def test_is_zero_symbolic_nonzero(self):
        x   = Position('iz_nx')
        arr = KVArray([x, KVExpr(ca.SX(0))])
        assert not arr.is_zero


class TestSymbols:
    def test_collects_all_free_symbols(self):
        x, y = Position('ks_x'), Position('ks_y')
        assert KVArray([x, y, x * 2]).symbols == frozenset({x, y})

    def test_empty_for_numeric_array(self):
        assert KVArray([1.0, 2.0]).symbols == frozenset()

    def test_is_symbolic_true(self):
        assert KVArray([Position('ks_sx')]).is_symbolic

    def test_is_symbolic_false(self):
        assert not KVArray([1.0, 2.0]).is_symbolic

    def test_ordered_symbols_matches_symbols(self):
        x, y = Position('ks_ox'), Position('ks_oy')
        arr  = KVArray([x, y])
        assert frozenset(arr.ordered_symbols) == arr.symbols


class TestSymbolOrder:
    def test_fixes_order(self):
        x, y = Position('kso_x'), Position('kso_y')
        arr  = KVArray([x, y])
        arr.set_symbol_order([y, x])
        assert list(arr.ordered_symbols) == [y, x]

    def test_accepts_superset(self):
        x, y, z = Position('kso_sx'), Position('kso_sy'), Position('kso_sz')
        arr     = KVArray([x, y])
        arr.set_symbol_order([z, y, x])
        assert list(arr.ordered_symbols) == [y, x]

    def test_incomplete_raises(self):
        x, y = Position('kso_ex'), Position('kso_ey')
        with pytest.raises(ValueError):
            KVArray([x, y]).set_symbol_order([x])

    def test_resets_function_cache(self):
        x, y = Position('kso_cx'), Position('kso_cy')
        arr  = KVArray([x, y])
        _    = arr.eval({x: 1.0, y: 2.0})   # compile
        arr.set_symbol_order([y, x])
        assert arr._function is None


class TestArithmetic:
    @pytest.fixture(autouse=True)
    def syms(self):
        self.x    = Position('kar_x')
        self.y    = Position('kar_y')
        self.z    = Position('kar_z')
        self.arr  = KVArray([self.x, self.y])
        self.args = {self.x: 3.0, self.y: 4.0, self.z: 10.0}

    def test_add_scalar(self):
        np.testing.assert_allclose((self.arr + 1).eval(self.args), [4.0, 5.0])

    def test_sub_scalar(self):
        np.testing.assert_allclose((self.arr - 1).eval(self.args), [2.0, 3.0])

    def test_mul_scalar(self):
        np.testing.assert_allclose((self.arr * 2).eval(self.args), [6.0, 8.0])

    def test_div_scalar(self):
        np.testing.assert_allclose((self.arr / 2).eval(self.args), [1.5, 2.0])

    def test_pow_scalar(self):
        np.testing.assert_allclose((self.arr ** 2).eval(self.args), [9.0, 16.0])

    def test_add_kvexpr_broadcasts(self):
        np.testing.assert_allclose((self.arr + self.z).eval(self.args), [13.0, 14.0])

    def test_mul_kvexpr_broadcasts(self):
        np.testing.assert_allclose((self.arr * self.z).eval(self.args), [30.0, 40.0])

    def test_radd_scalar(self):
        np.testing.assert_allclose((1 + self.arr).eval(self.args), [4.0, 5.0])

    def test_rsub_scalar(self):
        np.testing.assert_allclose((10 - self.arr).eval(self.args), [7.0, 6.0])

    def test_rmul_scalar(self):
        np.testing.assert_allclose((2 * self.arr).eval(self.args), [6.0, 8.0])

    def test_rtruediv_scalar(self):
        np.testing.assert_allclose((12 / self.arr).eval(self.args), [4.0, 3.0])

    def test_add_two_kvarrays(self):
        other = KVArray([self.y, self.x])
        np.testing.assert_allclose((self.arr + other).eval(self.args), [7.0, 7.0])

    def test_result_is_kvarray(self):
        assert isinstance(self.arr * 2, KVArray)


class TestEval:
    def test_symbolic_1d(self):
        x, y = Position('ke_x'), Position('ke_y')
        arr  = KVArray([x * 2, y * 3])
        np.testing.assert_allclose(arr.eval({x: 1.0, y: 2.0}), [2.0, 6.0])

    def test_symbolic_2d(self):
        x   = Position('ke_2x')
        arr = KVArray([[x, x * 2], [x * 3, x * 4]])
        np.testing.assert_allclose(arr.eval({x: 1.0}), [[1.0, 2.0], [3.0, 4.0]])

    def test_numeric_returns_copy(self):
        arr    = KVArray([1.0, 2.0, 3.0])
        result = arr.eval({})
        np.testing.assert_allclose(result, [1.0, 2.0, 3.0])
        assert result is not arr

    def test_missing_symbol_raises(self):
        x = Position('ke_mx')
        with pytest.raises(EvaluationError):
            KVArray([x]).eval({})

    def test_unchecked_eval_single(self):
        x, y = Position('ke_us_x'), Position('ke_us_y')
        arr  = KVArray([x * 2, y * 3])
        arr.set_symbol_order([x, y])
        np.testing.assert_allclose(arr.unchecked_eval(np.array([1.0, 2.0])), [2.0, 6.0])

    def test_unchecked_eval_batched(self):
        x, y = Position('ke_ub_x'), Position('ke_ub_y')
        arr  = KVArray([x * 2, y * 3])
        arr.set_symbol_order([x, y])
        result = arr.unchecked_eval(np.array([[1.0, 2.0], [3.0, 1.0]]))
        np.testing.assert_allclose(result, [[2.0, 6.0], [6.0, 3.0]])

    def test_call_with_dict_dispatches_to_eval(self):
        x = Position('ke_cd')
        np.testing.assert_allclose(KVArray([x, x * 2])({x: 5.0}), [5.0, 10.0])

    def test_call_with_array_dispatches_to_unchecked_eval(self):
        x   = Position('ke_ca')
        arr = KVArray([x, x * 2])
        arr.set_symbol_order([x])
        np.testing.assert_allclose(arr(np.array([5.0])), [5.0, 10.0])


class TestJacobian:
    def test_1d_two_symbols(self):
        x, y = Position('kj_x'), Position('kj_y')
        arr  = KVArray([2 * x, 3 * y])
        jac  = arr.jacobian([x, y])
        np.testing.assert_allclose(jac.eval({x: 0.0, y: 0.0}), [[2.0, 0.0], [0.0, 3.0]])

    def test_nonlinear(self):
        x   = Position('kj_nx')
        arr = KVArray([x ** 2, x ** 3])
        jac = arr.jacobian([x])
        # Single-symbol jacobian is squeezed from (N,1) → (N,)
        np.testing.assert_allclose(jac.eval({x: 2.0}), [4.0, 12.0])

    def test_constant_element_gives_zero_row(self):
        x   = Position('kj_cx')
        arr = KVArray([x, KVExpr(ca.SX(5.0))])
        jac = arr.jacobian([x])
        np.testing.assert_allclose(jac.eval({x: 1.0}), [1.0, 0.0])

    def test_result_is_kvarray(self):
        x = Position('kj_rx')
        assert isinstance(KVArray([x]).jacobian([x]), KVArray)


class TestTangent:
    def test_1d_two_symbols(self):
        x, y  = Position('kt_x'),  Position('kt_y')
        x_dot = x.derivative()
        y_dot = y.derivative()
        arr   = KVArray([2 * x, 3 * y])
        tan   = arr.tangent([x, y])
        args  = {x: 0.0, y: 0.0, x_dot: 1.0, y_dot: 2.0}
        np.testing.assert_allclose(tan.eval(args), [2.0, 6.0])

    def test_result_is_kvarray(self):
        x = Position('kt_rx')
        assert isinstance(KVArray([x]).tangent([x]), KVArray)


class TestAsCasadi:
    def test_1d_becomes_row_vector(self):
        x, y = Position('csd_x'), Position('csd_y')
        assert KVArray([x, y]).as_casadi().shape == (1, 2)

    def test_2d_preserves_shape(self):
        x = Position('csd_2x')
        assert KVArray([[x, x], [x, x]]).as_casadi().shape == (2, 2)

    def test_3d_raises(self):
        with pytest.raises(RuntimeError):
            KVArray(np.zeros((2, 2, 2))).as_casadi()


class TestSubstitute:
    def test_replaces_symbol_in_all_elements(self):
        x, y = Position('ksub_x'), Position('ksub_y')
        arr  = KVArray([x, x * 2])
        assert arr.substitute({x: y}).symbols == frozenset({y})

    def test_non_matching_symbol_is_noop(self):
        x, y = Position('ksub_ax'), Position('ksub_ay')
        arr  = KVArray([x])
        np.testing.assert_allclose(arr.substitute({y: ca.SX(99.0)}).eval({x: 1.0}), [1.0])

    def test_non_kvexpr_elements_unchanged(self):
        x   = Position('ksub_nx')
        arr = KVArray([x, KVExpr(ca.SX(5.0))])
        assert float(arr.substitute({x: ca.SX(3.0)})[1]) == pytest.approx(5.0)


class TestSetStamp:
    def test_stamps_all_symbols(self):
        x, y = Position('kss_x'), Position('kss_y')
        result = KVArray([x, y]).set_stamp(3)
        assert all(s.stamp == 3 for s in result.symbols)

    def test_stamped_array_evaluates_correctly(self):
        x      = Position('kss_ex')
        result = KVArray([x * 2, x * 3]).set_stamp(1)
        x_t1   = Position('kss_ex', stamp=1)
        np.testing.assert_allclose(result.eval({x_t1: 2.0}), [4.0, 6.0])


class TestToCoo:
    def test_1d_count_of_nonzero(self):
        x   = Position('coo_x')
        arr = KVArray([x, KVExpr(ca.SX(0)), x * 2])
        _, vals = arr.to_coo()
        assert len(vals) == 2

    def test_1d_correct_coordinates(self):
        x      = Position('coo_cx')
        arr    = KVArray([x, KVExpr(ca.SX(0)), x * 2])
        coords, _ = arr.to_coo()
        np.testing.assert_array_equal(coords, [[0], [2]])

    def test_2d_correct_coordinates(self):
        x      = Position('coo_2x')
        arr    = KVArray([[x,               KVExpr(ca.SX(0))],
                          [KVExpr(ca.SX(0)), x]])
        coords, _ = arr.to_coo()
        np.testing.assert_array_equal(coords, [[0, 0], [1, 1]])

    def test_numeric_array(self):
        arr    = KVArray([[1.0, 0.0], [0.0, 2.0]])
        coords, vals = arr.to_coo()
        assert len(vals) == 2
        np.testing.assert_array_equal(coords, [[0, 0], [1, 1]])

    def test_all_zero_returns_empty(self):
        arr    = KVArray([KVExpr(ca.SX(0)), KVExpr(ca.SX(0))])
        _, vals = arr.to_coo()
        assert len(vals) == 0


class TestMinMaxMethods:
    def test_numeric_min_scalar(self):
        assert KVArray([3.0, 1.0, 2.0]).min() == pytest.approx(1.0)

    def test_numeric_min_axis(self):
        arr = KVArray([[1.0, 4.0], [3.0, 2.0]])
        np.testing.assert_allclose(arr.min(axis=0), [1.0, 2.0])

    def test_numeric_min_keepdims(self):
        assert KVArray([3.0, 1.0, 2.0]).min(keepdims=True).shape == (1,)

    def test_numpy_min_on_numeric_kvarray(self):
        assert np.min(KVArray([3.0, 1.0, 2.0])) == pytest.approx(1.0)

    def test_symbolic_min_raises(self):
        with pytest.raises(ValueError, match='kv_lite.math.min'):
            KVArray([Position('mm_min_x')]).min()

    def test_numpy_min_on_symbolic_raises(self):
        with pytest.raises(ValueError, match='kv_lite.math.min'):
            np.min(KVArray([Position('mm_npmin_x')]))

    def test_numeric_max_scalar(self):
        assert KVArray([3.0, 1.0, 2.0]).max() == pytest.approx(3.0)

    def test_numeric_max_axis(self):
        arr = KVArray([[1.0, 4.0], [3.0, 2.0]])
        np.testing.assert_allclose(arr.max(axis=0), [3.0, 4.0])

    def test_numeric_max_keepdims(self):
        assert KVArray([3.0, 1.0, 2.0]).max(keepdims=True).shape == (1,)

    def test_numpy_max_on_numeric_kvarray(self):
        assert np.max(KVArray([3.0, 1.0, 2.0])) == pytest.approx(3.0)

    def test_symbolic_max_raises(self):
        with pytest.raises(ValueError, match='kv_lite.math.max'):
            KVArray([Position('mm_max_x')]).max()

    def test_numpy_max_on_symbolic_raises(self):
        with pytest.raises(ValueError, match='kv_lite.math.max'):
            np.max(KVArray([Position('mm_npmax_x')]))
