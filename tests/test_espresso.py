import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from espresso import parse_input, minimize, format_sop, conv_to_int, to_positional


def write_tmp(content):
    f = tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False)
    f.write(content)
    f.close()
    return f.name


# ── parse_input ──────────────────────────────────────────────────────────────

def test_parse_input_minterms_only():
    path = write_tmp("f(3) = E(1,3,5,7) + d()")
    no_of_ip, ip_list = parse_input(path)
    os.unlink(path)
    assert no_of_ip == 3
    assert ip_list == [1, 3, 5, 7]


def test_parse_input_with_dont_cares():
    path = write_tmp("f(4) = E(0,2,6,8,10,11,14) + d(9,15)")
    no_of_ip, ip_list = parse_input(path)
    os.unlink(path)
    assert no_of_ip == 4
    assert 0 in ip_list and 9 in ip_list and 15 in ip_list


# ── to_positional ─────────────────────────────────────────────────────────────

def test_to_positional_zero():
    # minterm 0 with 2 inputs → both complemented: [[T,F],[T,F]]
    result = to_positional([0], 2)
    assert result == [[[True, False], [True, False]]]


def test_to_positional_three():
    # minterm 3 with 2 inputs → both uncomplemented: [[F,T],[F,T]]
    result = to_positional([3], 2)
    assert result == [[[False, True], [False, True]]]


# ── minimize + format_sop ─────────────────────────────────────────────────────

def test_minimize_returns_nonempty():
    path = write_tmp("f(4) = E(0,2,6,8,10,11,14) + d(9,15)")
    result = minimize(path)
    os.unlink(path)
    assert len(result) > 0


def test_format_sop_structure():
    path = write_tmp("f(4) = E(0,2,6,8,10,11,14) + d(9,15)")
    result = minimize(path)
    os.unlink(path)
    sop = format_sop(result)
    assert sop.startswith("( ")
    assert sop.endswith(" )")
    assert "X" in sop


def test_minimize_all_ones():
    # f(2) = E(0,1,2,3): function is always 1, should minimize to empty SOP or tautology
    path = write_tmp("f(2) = E(0,1,2,3) + d()")
    result = minimize(path)
    os.unlink(path)
    # result should be valid (non-None); exact form is algorithm-dependent
    assert result is not None


def test_minimize_single_minterm():
    # f(2) = E(0): only minterm 0 → (~X0)(~X1)
    path = write_tmp("f(2) = E(0) + d()")
    result = minimize(path)
    os.unlink(path)
    sop = format_sop(result)
    assert "(~X0)" in sop
    assert "(~X1)" in sop


if __name__ == "__main__":
    test_parse_input_minterms_only()
    test_parse_input_with_dont_cares()
    test_to_positional_zero()
    test_to_positional_three()
    test_minimize_returns_nonempty()
    test_format_sop_structure()
    test_minimize_all_ones()
    test_minimize_single_minterm()
    print("All tests passed.")
