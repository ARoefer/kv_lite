import operator

import casadi as ca
import numpy  as np
import pytest

from kv_lite.math import (KVArray, KVExpr, KVSymbol, EvaluationError,
                           Position, Velocity)


class TestConstruction:
    def test_wrapping_symbol_returns_symbol(self):
        s = Position('ce_q')
        assert KVExpr(s) is s

    def test_wrapping_kvexpr_copies_data(self):
        x    = Position('ce_x')
        e    = x * 2
        copy = KVExpr(e)
        assert copy._ca_data is e._ca_data

    def test_wrapping_kvexpr_copies_symbols(self):
        x    = Position('ce_xs')
        e    = x * 2
        _    = e.symbols        # populate cache
        copy = KVExpr(e)
        assert copy._symbols == e._symbols

    def test_float_conversion_on_constant(self):
        assert float(KVExpr(ca.SX(7.0))) == pytest.approx(7.0)

    def test_float_conversion_raises_on_symbolic(self):
        with pytest.raises(RuntimeError):
            float(Position('ce_sym'))


class TestStr:
    def test_str_does_not_raise(self):
        str(Position('str_x') * 2)

    def test_repr_does_not_raise(self):
        repr(Position('repr_x') * 2)


class TestBinaryArithmetic:
    @pytest.fixture(autouse=True)
    def syms(self):
        self.a    = Position('ba_a')
        self.b    = Position('ba_b')
        self.args = {self.a: 5.0, self.b: 2.0}

    @pytest.mark.parametrize('op,expected', [
        (operator.add,      7.0),
        (operator.sub,      3.0),
        (operator.mul,     10.0),
        (operator.truediv,  2.5),
        (operator.pow,     25.0),
    ])
    def test_kvexpr_op_kvexpr(self, op, expected):
        assert op(self.a, self.b).eval(self.args) == pytest.approx(expected)

    @pytest.mark.parametrize('op,expected', [
        (operator.add,      7.0),
        (operator.sub,      3.0),
        (operator.mul,     10.0),
        (operator.truediv,  2.5),
        (operator.pow,     25.0),
    ])
    def test_kvexpr_op_scalar(self, op, expected):
        assert op(self.a, 2.0).eval(self.args) == pytest.approx(expected)

    @pytest.mark.parametrize('op,expected', [
        (operator.add,      7.0),
        (operator.sub,     -3.0),
        (operator.mul,     10.0),
        (operator.truediv,  0.4),
    ])
    def test_scalar_op_kvexpr(self, op, expected):
        assert op(2.0, self.a).eval(self.args) == pytest.approx(expected)

    def test_neg(self):
        assert (-self.a).eval(self.args) == pytest.approx(-5.0)

    def test_result_is_kvexpr(self):
        assert isinstance(self.a + self.b, KVExpr)


class TestInPlaceArithmetic:
    def _make_expr(self, name):
        x = Position(name)
        return x * 2, x  # expr = 2*x

    @pytest.mark.parametrize('op,scalar,expected', [
        (operator.iadd,      3.0, 13.0),
        (operator.isub,      3.0,  7.0),
        (operator.imul,      3.0, 30.0),
        (operator.itruediv,  2.0,  5.0),
    ])
    def test_inplace_op_correct_result(self, op, scalar, expected):
        expr, x = self._make_expr(f'ip_{op.__name__}_x')
        expr     = op(expr, scalar)
        assert expr.eval({x: 5.0}) == pytest.approx(expected)

    def test_inplace_resets_symbols_cache(self):
        expr, _ = self._make_expr('ip_cache_x')
        _       = expr.symbols   # populate cache
        expr   += 1
        assert expr._symbols is None

    def test_inplace_resets_function_cache(self):
        expr, x = self._make_expr('ip_fcache_x')
        _       = expr.eval({x: 1.0})   # compile and cache
        expr   += 1
        assert expr._function is None


class TestProperties:
    def test_is_zero_true(self):
        assert KVExpr(ca.SX(0)).is_zero

    def test_is_zero_false(self):
        assert not KVExpr(ca.SX(1)).is_zero

    def test_is_one_true(self):
        assert KVExpr(ca.SX(1)).is_one

    def test_is_one_false(self):
        assert not KVExpr(ca.SX(0)).is_one

    def test_is_symbolic_true(self):
        assert Position('prop_x').is_symbolic

    def test_is_symbolic_false(self):
        assert not KVExpr(ca.SX(3.0)).is_symbolic

    def test_symbols_contains_all_free_symbols(self):
        x, y = Position('prop_sx'), Position('prop_sy')
        assert (x + y).symbols == frozenset({x, y})

    def test_symbols_empty_for_constant(self):
        assert KVExpr(ca.SX(3.0)).symbols == frozenset()

    def test_ordered_symbols_matches_symbols(self):
        x, y = Position('prop_ox'), Position('prop_oy')
        expr = x + y
        assert frozenset(expr.ordered_symbols) == expr.symbols


class TestSymbolOrder:
    def test_set_symbol_order_fixes_order(self):
        x, y = Position('so_x'), Position('so_y')
        expr = x + y
        expr.set_symbol_order([y, x])
        assert list(expr.ordered_symbols) == [y, x]

    def test_set_symbol_order_accepts_superset(self):
        x, y, z = Position('so_sx'), Position('so_sy'), Position('so_sz')
        expr = x + y
        expr.set_symbol_order([z, y, x])   # z not in expr — should be filtered out
        assert list(expr.ordered_symbols) == [y, x]

    def test_set_symbol_order_incomplete_raises(self):
        x, y = Position('so_ex'), Position('so_ey')
        expr = x + y
        with pytest.raises(ValueError):
            expr.set_symbol_order([x])     # missing y

    def test_set_symbol_order_resets_function_cache(self):
        x, y = Position('so_cx'), Position('so_cy')
        expr = x + y
        _    = expr.eval({x: 1.0, y: 1.0})  # compile
        expr.set_symbol_order([y, x])
        assert expr._function is None


class TestEval:
    def test_eval_correct_value(self):
        x = Position('ev_x')
        assert (x * 3).eval({x: 4.0}) == pytest.approx(12.0)

    def test_eval_caches_function(self):
        x    = Position('ev_cx')
        expr = x * 3
        _    = expr.eval({x: 4.0})
        assert expr._function is not None

    def test_eval_missing_symbol_raises(self):
        x = Position('ev_mx')
        with pytest.raises(EvaluationError):
            (x * 3).eval({})

    def test_call_with_dict_dispatches_to_eval(self):
        x = Position('ev_dc')
        assert (x * 3)({x: 4.0}) == pytest.approx(12.0)

    def test_unchecked_eval_single(self):
        x    = Position('ev_us')
        expr = x * 2
        expr.set_symbol_order([x])
        result = expr.unchecked_eval(np.array([5.0]))
        assert result == pytest.approx([10.0])

    def test_unchecked_eval_batched(self):
        x    = Position('ev_ub')
        expr = x * 2
        expr.set_symbol_order([x])
        result = expr.unchecked_eval(np.array([[5.0], [3.0]]))
        np.testing.assert_allclose(result, [[10.0], [6.0]])

    def test_call_with_array_dispatches_to_unchecked_eval(self):
        x    = Position('ev_ac')
        expr = x * 2
        expr.set_symbol_order([x])
        result = expr(np.array([5.0]))
        assert result == pytest.approx([10.0])


class TestJacobian:
    def test_linear_single_symbol(self):
        x   = Position('jac_x')
        jac = (2 * x).jacobian([x])
        np.testing.assert_allclose(jac.eval({x: 0.0}), [[2.0]])

    def test_linear_two_symbols(self):
        x, y = Position('jac_mx'), Position('jac_my')
        jac  = (2 * x + 3 * y).jacobian([x, y])
        np.testing.assert_allclose(jac.eval({x: 0.0, y: 0.0}), [[2.0, 3.0]])

    def test_nonlinear(self):
        x   = Position('jac_nx')
        jac = (x ** 2).jacobian([x])
        np.testing.assert_allclose(jac.eval({x: 3.0}), [[6.0]])

    def test_returns_kvarray(self):
        x = Position('jac_rx')
        assert isinstance((2 * x).jacobian([x]), KVArray)


class TestTangent:
    def test_linear_single_symbol(self):
        x     = Position('tan_x')
        x_dot = x.derivative()
        tan   = (2 * x).tangent([x])
        assert tan.eval({x: 1.0, x_dot: 3.0}) == pytest.approx(6.0)

    def test_linear_two_symbols(self):
        x, y   = Position('tan_mx'), Position('tan_my')
        x_dot  = x.derivative()
        y_dot  = y.derivative()
        tan    = (x + y).tangent([x, y])
        assert tan.eval({x: 0.0, y: 0.0, x_dot: 2.0, y_dot: 3.0}) == pytest.approx(5.0)

    def test_defaults_to_all_symbols(self):
        x     = Position('tan_dx')
        x_dot = x.derivative()
        tan   = (2 * x).tangent()
        assert tan.eval({x: 1.0, x_dot: 1.0}) == pytest.approx(2.0)

    def test_tangent_raises_on_unknown_type(self):
        from kv_lite.math import Symbol
        x = Symbol('tan_unk')
        with pytest.raises(RuntimeError):
            (2 * x).tangent()


class TestSubstitute:
    def test_substitute_symbol_with_value(self):
        x      = Position('sub_x')
        result = (x * 3).substitute({x: ca.SX(2.0)})
        assert float(result) == pytest.approx(6.0)

    def test_substitute_symbol_with_symbol(self):
        x, y   = Position('sub_sx'), Position('sub_sy')
        result = (x + 1).substitute({x: y})
        assert result.symbols == frozenset({y})

    def test_substitute_absent_symbol_is_noop(self):
        x, y   = Position('sub_ax'), Position('sub_ay')
        expr   = x + 1
        result = expr.substitute({y: ca.SX(99.0)})
        assert result.eval({x: 2.0}) == pytest.approx(3.0)


class TestSetStamp:
    def test_stamps_all_symbols(self):
        x, y   = Position('ss_x'), Position('ss_y')
        expr   = x + y
        result = expr.set_stamp(5)
        stamped_syms = result.symbols
        assert all(s.stamp == 5 for s in stamped_syms)

    def test_stamped_result_evaluates_correctly(self):
        x       = Position('ss_ex')
        result  = (x * 2).set_stamp(1)
        x_t1    = Position('ss_ex', stamp=1)
        assert result.eval({x_t1: 3.0}) == pytest.approx(6.0)

    def test_set_stamp_subset_of_symbols(self):
        x, y   = Position('ss_px'), Position('ss_py')
        expr   = x + y
        result = expr.set_stamp(3, symbols={x})
        syms   = result.symbols
        assert Position('ss_px', stamp=3) in syms
        assert y in syms
