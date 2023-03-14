#!/usr/bin/env python3
"""
ESPRESSO: Heuristic Boolean logic minimizer.

Implements the ESPRESSO algorithm for two-level logic minimization.
Reads a Boolean function and outputs a minimized Sum-of-Products (SOP) expression.

Input file format:
    f(n) = E(m1,m2,...) + d(dc1,dc2,...)

  n        – number of inputs
  E(...)   – on-set minterms
  d(...)   – don't-care minterms (use d() for none)

Usage:
    python espresso.py [input_file]      # default: Equation.txt
    python espresso.py my_function.txt
"""

import argparse
import numpy as np


# ---------------------------------------------------------------------------
# Input parsing
# ---------------------------------------------------------------------------

def parse_input(filepath):
    """
    Parse a Boolean function description file.
    Returns (no_of_ip, ip_list) where ip_list combines minterms and don't-cares.
    """
    with open(filepath, 'r') as f:
        res = [line.split() for line in f]

    no_of_ip = 0
    str_ip = ""

    for row in res:
        for j, token in enumerate(row):
            if j == 0:                          # f(n)
                k = 2
                s = ""
                while token[k] != ')':
                    s += token[k]
                    k += 1
                no_of_ip = int(s)

            if j == 2:                          # E(m1,m2,...)
                k = 2
                while token[k] != ')':
                    str_ip += token[k]
                    k += 1
                str_ip += ','

            if j == 4:                          # d(dc1,dc2,...)
                k = 2
                has_dc = False
                while token[k] != ')':
                    has_dc = True
                    str_ip += token[k]
                    k += 1
                if has_dc:
                    str_ip += ','

    ip_list = []
    cur = ""
    for ch in str_ip:
        if ch != ',':
            cur += ch
        else:
            ip_list.append(int(cur))
            cur = ""

    return no_of_ip, ip_list


# ---------------------------------------------------------------------------
# Positional cube representation
#   [True, False]  – literal appears complemented
#   [False, True]  – literal appears uncomplemented
#   [True, True]   – don't-care (literal absent from term)
# ---------------------------------------------------------------------------

def to_positional(index_list, no_of_ip):
    """Convert a list of minterm indices to positional cube notation."""
    order = 2 ** no_of_ip
    pos_list = []
    for idx in index_list:
        n = order // 2
        num = idx
        pos_dash = []
        while n != 0:
            pos_dash.append([False, True] if num // n == 1 else [True, False])
            num = num % n
            n = n // 2
        pos_list.append(pos_dash)
    return pos_list


def conv_to_int(posA, no_of_ip):
    """Convert positional cube list to integer (0/1) representation."""
    result = []
    for i in range(len(posA)):
        row = []
        for j in range(no_of_ip):
            p = posA[i][j]
            if   p == [True, False]: row.append([1, 0])
            elif p == [False, True]: row.append([0, 1])
            elif p == [True, True]:  row.append([1, 1])
            else:                    row.append([0, 0])
        result.append(row)
    return result


# ---------------------------------------------------------------------------
# Weight-based sorting
# ---------------------------------------------------------------------------

def min_weights(num, no_of_ip):
    """Compute per-term weights based on column occurrence counts."""
    flat = []
    for i in range(len(num)):
        row = []
        for j in range(no_of_ip):
            row.append(num[i][j][0])
            row.append(num[i][j][1])
        flat.append(row)

    col_weight = []
    for i in range(len(flat)):
        for j in range(len(flat[0])):
            if i == 0:
                col_weight.append(1 if flat[i][j] == 1 else 0)
            else:
                if flat[i][j] == 1:
                    col_weight[j] += 1

    impl_weight = []
    for i in range(len(num)):
        wt = 0
        k = 0
        for j in range(no_of_ip):
            wt += num[i][j][0] * col_weight[k]
            wt += num[i][j][1] * col_weight[k + 1]
            k += 2
        impl_weight.append(wt)
    return impl_weight


def sort_ascending(impl_weight, pos_list):
    """Return pos_list sorted by ascending weight."""
    unique_weights = np.unique(sorted(impl_weight))
    ordered = []
    for w in unique_weights:
        for idx, v in enumerate(impl_weight):
            if v == w:
                ordered.append(idx)
    return [pos_list[i] for i in ordered]


def sort_descending(impl_weight, pos_list):
    """Return pos_list sorted by descending weight."""
    unique_weights = -np.sort(-np.unique(impl_weight))
    ordered = []
    for w in unique_weights:
        for idx, v in enumerate(impl_weight):
            if v == w:
                ordered.append(idx)
    return [pos_list[i] for i in ordered]


# ---------------------------------------------------------------------------
# Core ESPRESSO operations
# ---------------------------------------------------------------------------

def are_disjoint(posa, posb, no_of_ip):
    """Return True if cubes posa and posb have no common minterm."""
    for i in range(no_of_ip):
        v1 = posa[i][0] and posb[i][0]
        v2 = posa[i][1] and posb[i][1]
        if v1 == v2 and v1 != True:
            return True
    return False


def count_disjoint_positions(posa, posb, no_of_ip):
    """Count positions where posa and posb have no common assignment."""
    count = 0
    for i in range(no_of_ip):
        v1 = posa[i][0] and posb[i][0]
        v2 = posa[i][1] and posb[i][1]
        if v1 == v2 and v1 != True:
            count += 1
    return count


def irredundancy(posA, np_pos, term_count, no_of_ip):
    """Remove terms from np_pos that are covered by the expanded cube posA."""
    new_pos = []
    index = []
    for i in range(term_count):
        if i != 0:
            new_pos.append(np_pos[i - 1])
    new_pos.append(posA)
    for i in range(len(np_pos)):
        if i > term_count - 2:
            if are_disjoint(posA, np_pos[i], no_of_ip):
                new_pos.append(np_pos[i])
            else:
                index.append(i)
    return np.array(new_pos), index, term_count + 1


def expand(np_pos, np_bar_pos, posA, term_count, red_term, no_of_ip):
    """Expand cube posA as far as possible without intersecting the off-set."""
    index = []
    for i in range(len(posA)):
        if i != red_term:
            posA[i][0], posA[i][1] = posA[i][1], posA[i][0]
            intersects_off_set = any(
                not are_disjoint(posA, np_bar_pos[j], no_of_ip)
                for j in range(len(np_bar_pos))
            )
            if not intersects_off_set:
                posA[i][0] = True
                posA[i][1] = True
            else:
                posA[i][0], posA[i][1] = posA[i][1], posA[i][0]
                index.append(i)
    new_np_pos, index, term_count_val = irredundancy(posA, np_pos, term_count, no_of_ip)
    return posA, new_np_pos, index, term_count_val


def consensus(funcA, no_of_ip):
    """Remove redundant prime implicants using the consensus method."""
    new_funcA = []
    for i in range(len(funcA)):
        immfunc = []
        eflag = 0
        for j in range(len(funcA)):
            if i != j and not are_disjoint(funcA[i], funcA[j], no_of_ip):
                immfunc.append(funcA[j].tolist())
        if len(immfunc) < 2:
            new_funcA.append(funcA[i].tolist())
            eflag = 1
        if eflag == 0:
            np_immfunc = np.array(immfunc)
            for k in range(len(np_immfunc)):
                vF = 0
                for l in range(k + 1, len(np_immfunc)):
                    if count_disjoint_positions(np_immfunc[k], np_immfunc[l], no_of_ip) != 1:
                        new_funcA.append(funcA[i].tolist())
                        vF = 1
                        break
                if vF == 1:
                    break
        else:
            eflag = 0
    return np.array(new_funcA)


def cost_function(funcA_int):
    """Count total literals in a cover (lower is better)."""
    cost = 0
    for term in funcA_int:
        for lit in term:
            if lit != [1, 1]:
                cost += 1
    return cost


def reduction_order(funcTerm):
    """Return which positions are expanded (don't-care) and their bit patterns."""
    input_no = []
    for i, lit in enumerate(funcTerm):
        if lit[0] == lit[1] == True:
            input_no.append(i)
    redcF = len(input_no)
    order_list = [bin(i)[2:].zfill(redcF) for i in range(2 ** redcF)]
    return redcF, order_list, input_no


def reduce(func_np, np_bar_pos, no_of_ip):
    """Reduction step: shrink each expanded cube where doing so lowers cost."""
    red_impl_weight = min_weights(conv_to_int(func_np.tolist(), no_of_ip), no_of_ip)
    funcA_list = sort_descending(red_impl_weight, func_np.tolist())
    new_np_pos = np.array(funcA_list)
    np_funcA = np.copy(new_np_pos)
    term_count_val = len(np_funcA)
    new_np_funcA_int = conv_to_int(np_funcA.tolist(), no_of_ip)

    for i in range(len(np_funcA)):
        _, order_list, input_no = reduction_order(np_funcA[i])
        usedlist = [0] * len(input_no)
        for j in range(len(usedlist)):
            if usedlist[j] == 0:
                usedlist[j] = 1
                for k in range(len(order_list)):
                    for l in range(len(order_list[k])):
                        if order_list[k][l] == '0':
                            np_funcA[i][input_no[l]][0] = True
                            np_funcA[i][input_no[l]][1] = False
                        else:
                            np_funcA[i][input_no[l]][0] = False
                            np_funcA[i][input_no[l]][1] = True
                    _, new_np_funcA, _, _ = expand(
                        np_funcA, np_bar_pos, np_funcA[i], term_count_val, input_no[j], no_of_ip
                    )
                    np_funcA = np.copy(new_np_pos)
                    org_cf = cost_function(conv_to_int(np_funcA.tolist(), no_of_ip))
                    new_cf = cost_function(conv_to_int(new_np_funcA.tolist(), no_of_ip))
                    if new_cf < org_cf:
                        new_np_funcA_int = conv_to_int(new_np_funcA.tolist(), no_of_ip)
    return new_np_funcA_int


# ---------------------------------------------------------------------------
# Top-level pipeline
# ---------------------------------------------------------------------------

def minimize(filepath):
    """
    Run the full ESPRESSO pipeline on the function described in filepath.
    Returns the minimized cover as a list of terms in integer positional notation.
    """
    no_of_ip, ip_list = parse_input(filepath)
    order = 2 ** no_of_ip

    f_bar = [i for i in range(order) if i not in ip_list]

    pos_list = to_positional(ip_list, no_of_ip)
    weights = min_weights(conv_to_int(pos_list, no_of_ip), no_of_ip)
    f_list = sort_ascending(weights, pos_list)
    f_bar_list = to_positional(f_bar, no_of_ip)

    np_pos = np.array(f_list)
    np_bar_pos = np.array(f_bar_list)

    # Expansion + irredundancy loop
    k = len(np_pos)
    new_np_pos = np.copy(np_pos)
    term_count_val = 1
    no_iter = 0

    while k != 0:
        _, new_np_pos, index, term_count_val = expand(
            new_np_pos, np_bar_pos, np_pos[0], term_count_val, -1, no_of_ip
        )
        p = np_pos.tolist()
        for i in range(len(index)):
            index[i] -= no_iter
        no_iter += 1
        if len(np_pos) > len(new_np_pos):
            for idx in sorted(index, reverse=True):
                del p[idx]
        else:
            for idx in sorted(index, reverse=True):
                if 0 <= idx < len(np_pos):
                    del p[idx]
        np_pos = np.array(p)
        k = len(np_pos)

    new_np_pos = consensus(new_np_pos, no_of_ip)
    return reduce(new_np_pos, np_bar_pos, no_of_ip)


def format_sop(result_int):
    """Format the minimized cover as a human-readable SOP string."""
    terms = []
    for term in result_int:
        literals = []
        for j, lit in enumerate(term):
            if lit == [1, 0]:
                literals.append(f'(~X{j})')
            elif lit == [0, 1]:
                literals.append(f'(X{j})')
        terms.append(''.join(literals))
    return '( ' + ' + '.join(terms) + ' )'


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="ESPRESSO heuristic Boolean logic minimizer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Input file format:  f(n) = E(m1,m2,...) + d(dc1,dc2,...)"
    )
    parser.add_argument(
        "input_file",
        nargs="?",
        default="Equation.txt",
        help="path to input file (default: Equation.txt)"
    )
    args = parser.parse_args()

    result = minimize(args.input_file)
    print(format_sop(result))
