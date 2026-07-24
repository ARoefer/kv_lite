import pytest
import numpy as np

from kv_lite.math import (KVSymbol, KVArray, EvaluationError,
                           Symbol, Position, Velocity, Acceleration, Jerk, Snap)


class TestSingleton:
    def test_same_name_same_instance(self):
        assert Symbol('sg_q') is Symbol('sg_q')

    def test_same_name_diff_type_diff_instance(self):
        assert Symbol('sg_q') is not Position('sg_q')

    def test_diff_name_diff_instance(self):
        assert Symbol('sg_q1') is not Symbol('sg_q2')

    def test_stamp_differentiates(self):
        assert Position('sg_q_stamp') is not Position('sg_q_stamp', stamp=0)

    def test_prefix_differentiates(self):
        assert Position('sg_q_pfx') is not Position('sg_q_pfx', prefix='p')


class TestFullName:
    @pytest.mark.parametrize('factory,suffix', [
        (Position,     'position'),
        (Velocity,     'velocity'),
        (Acceleration, 'acceleration'),
        (Jerk,         'jerk'),
        (Snap,         'snap'),
    ])
    def test_typed_full_name(self, factory, suffix):
        assert factory('fn_q')._full_name == f'fn_q__{suffix}'

    def test_unknown_type_full_name(self):
        assert Symbol('fn_q_unk')._full_name == 'fn_q_unk'

    def test_prefix_in_full_name(self):
        assert Position('fn_q_pfx', prefix='p')._full_name == 'p__fn_q_pfx__position'

    def test_stamp_in_full_name(self):
        assert Position('fn_q_stamp', stamp=3)._full_name == 'fn_q_stamp__position__t3'

    def test_prefix_and_stamp_in_full_name(self):
        assert Position('fn_q_both', prefix='p', stamp=3)._full_name == 'p__fn_q_both__position__t3'

    def test_slash_replacement_in_typed_symbol(self):
        assert Position('/a/b')._full_name == '__a__b__position'

    def test_slash_not_replaced_in_unknown_type(self):
        assert Symbol('/a/b')._full_name == '/a/b'


class TestValidation:
    def test_invalid_type_raises(self):
        with pytest.raises(KeyError):
            KVSymbol('val_q', typ=99)

    def test_float_stamp_raises(self):
        with pytest.raises(ValueError):
            Position('val_q', stamp=1.5)

    def test_string_stamp_raises(self):
        with pytest.raises(ValueError):
            Position('val_q', stamp='0')


class TestAttributes:
    def test_name(self):
        assert Position('attr_q').name == 'attr_q'

    def test_type(self):
        assert Position('attr_q').type == KVSymbol.TYPE_POSITION

    def test_prefix_none_by_default(self):
        assert Position('attr_q').prefix is None

    def test_prefix_set(self):
        assert Position('attr_q', prefix='p').prefix == 'p'

    def test_stamp_none_by_default(self):
        assert Position('attr_q').stamp is None

    def test_stamp_set(self):
        assert Position('attr_q', stamp=1).stamp == 1


class TestSymbolsProperty:
    def test_symbols_contains_only_self(self):
        s = Position('syms_q')
        assert s.symbols == frozenset({s})

    def test_is_symbolic(self):
        assert Position('syms_q').is_symbolic is True


class TestDerivativeChain:
    @pytest.mark.parametrize('start,end', [
        (Position,     Velocity),
        (Velocity,     Acceleration),
        (Acceleration, Jerk),
        (Jerk,         Snap),
    ])
    def test_derivative_step(self, start, end):
        assert start('dc_q').derivative() is end('dc_q')

    def test_derivative_preserves_prefix(self):
        assert Position('dc_q', prefix='p').derivative() is Velocity('dc_q', prefix='p')

    def test_derivative_preserves_stamp(self):
        assert Position('dc_q', stamp=0).derivative() is Velocity('dc_q', stamp=0)

    def test_derivative_unknown_raises(self):
        with pytest.raises(RuntimeError):
            Symbol('dc_q_unk').derivative()

    def test_derivative_snap_raises(self):
        with pytest.raises(RuntimeError):
            Snap('dc_q_snap').derivative()


class TestIntegralChain:
    @pytest.mark.parametrize('start,end', [
        (Snap,         Jerk),
        (Jerk,         Acceleration),
        (Acceleration, Velocity),
        (Velocity,     Position),
    ])
    def test_integral_step(self, start, end):
        assert start('ic_q').integral() is end('ic_q')

    def test_integral_preserves_prefix(self):
        assert Velocity('ic_q', prefix='p').integral() is Position('ic_q', prefix='p')

    def test_integral_preserves_stamp(self):
        assert Velocity('ic_q', stamp=0).integral() is Position('ic_q', stamp=0)

    def test_integral_unknown_raises(self):
        with pytest.raises(RuntimeError):
            Symbol('ic_q_unk').integral()

    def test_integral_position_raises(self):
        with pytest.raises(RuntimeError):
            Position('ic_q_pos').integral()


class TestComparisons:
    def test_eq_same_symbol(self):
        assert Position('cmp_q') == Position('cmp_q')

    def test_eq_diff_type(self):
        assert not (Position('cmp_q') == Velocity('cmp_q'))

    def test_hash_consistent(self):
        assert hash(Position('cmp_q')) == hash(Position('cmp_q'))

    def test_usable_in_set(self):
        s = {Position('cmp_q_set'), Position('cmp_q_set'), Velocity('cmp_q_set')}
        assert len(s) == 2

    def test_lt_ordering(self):
        a, b = sorted([Symbol('cmp_zzz'), Symbol('cmp_aaa')])
        assert a._full_name == 'cmp_aaa'
        assert b._full_name == 'cmp_zzz'

    @pytest.mark.parametrize('op', ['__lt__', '__gt__', '__le__', '__ge__'])
    def test_ordering_with_non_symbol_raises(self, op):
        with pytest.raises(TypeError):
            getattr(Symbol('cmp_q_type'), op)(42)


class TestInPlaceOperators:
    def test_iadd_returns_expr_not_symbol(self):
        s  = Position('ip_q_add')
        s += 1
        assert not isinstance(s, KVSymbol)

    def test_isub_returns_expr_not_symbol(self):
        s  = Position('ip_q_sub')
        s -= 1
        assert not isinstance(s, KVSymbol)

    def test_imul_returns_expr_not_symbol(self):
        s  = Position('ip_q_mul')
        s *= 2
        assert not isinstance(s, KVSymbol)

    def test_idiv_returns_expr_not_symbol(self):
        s  = Position('ip_q_div')
        s /= 2
        assert not isinstance(s, KVSymbol)

    def test_original_singleton_unchanged(self):
        original = Position('ip_q_orig')
        s  = Position('ip_q_orig')
        s += 1
        assert Position('ip_q_orig') is original


class TestEval:
    def test_eval_returns_value(self):
        s = Position('ev_q')
        assert s.eval({s: 3.14}) == pytest.approx(3.14)

    def test_eval_missing_symbol_raises(self):
        s = Position('ev_q_miss')
        with pytest.raises(EvaluationError):
            s.eval({})

    def test_unchecked_eval_returns_first_arg(self):
        s = Position('ev_q_unc')
        assert s.unchecked_eval(np.array([42.0])) == pytest.approx(42.0)


class TestSubstitute:
    def test_substitute_present_key(self):
        s      = Position('sub_q')
        result = s.substitute({s: 5.0})
        assert result == 5.0

    def test_substitute_absent_key_returns_self(self):
        s  = Position('sub_q_abs')
        s2 = Velocity('sub_q_abs')
        assert s.substitute({s2: 5.0}) is s


class TestSetStamp:
    def test_set_stamp_returns_stamped_singleton(self):
        assert Position('ss_q').set_stamp(2) is Position('ss_q', stamp=2)

    def test_set_stamp_preserves_name(self):
        assert Position('ss_q').set_stamp(1).name == 'ss_q'

    def test_set_stamp_preserves_type(self):
        assert Position('ss_q').set_stamp(1).type == KVSymbol.TYPE_POSITION

    def test_set_stamp_preserves_prefix(self):
        assert Position('ss_q', prefix='p').set_stamp(1).prefix == 'p'


class TestLike:
    def test_1d_shape(self):
        assert KVSymbol.like(np.zeros(3)).shape == (3,)

    def test_2d_shape(self):
        assert KVSymbol.like(np.zeros((2, 3))).shape == (2, 3)

    def test_elements_are_symbols(self):
        result = KVSymbol.like(np.zeros(3))
        assert all(isinstance(e, KVSymbol) for e in result.flatten())

    def test_custom_prefix(self):
        result = KVSymbol.like(np.zeros(2), prefix='lk')
        assert all(e._full_name.startswith('lk_') for e in result)

    def test_non_array_returns_prefix(self):
        assert KVSymbol.like('not_an_array', prefix='fallback') == 'fallback'


class TestSetPrefix:
    def test_returns_singleton_with_new_prefix(self):
        assert Position('sp_q').set_prefix('new') is Position('sp_q', prefix='new')

    def test_preserves_name(self):
        assert Position('sp_q').set_prefix('p').name == 'sp_q'

    def test_preserves_type(self):
        assert Position('sp_q').set_prefix('p').type == KVSymbol.TYPE_POSITION

    def test_preserves_stamp(self):
        assert Position('sp_q', stamp=2).set_prefix('p').stamp == 2

    def test_replaces_existing_prefix(self):
        assert Position('sp_q', prefix='old').set_prefix('new') is Position('sp_q', prefix='new')

    def test_removes_prefix_with_none(self):
        assert Position('sp_q', prefix='p').set_prefix(None) is Position('sp_q')
