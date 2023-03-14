# ESPRESSO Boolean Logic Minimizer

A Python implementation of the **ESPRESSO** heuristic algorithm for two-level Boolean logic minimization. Given a Boolean function as a set of minterms and don't-cares, it produces a minimized Sum-of-Products (SOP) expression.

---

## What is ESPRESSO?

ESPRESSO is a heuristic logic minimization algorithm developed at UC Berkeley. Unlike exact minimizers (Quine–McCluskey), it uses three iterative operations — **expand**, **irredundancy**, and **reduce** — to find a near-minimal cover efficiently. It is widely used in synthesis tools (e.g., ABC, SIS) to simplify combinational logic before mapping to gates.

---

## Input Format

Functions are described in a plain text file:

```
f(n) = E(m1,m2,...) + d(dc1,dc2,...)
```

| Field | Meaning |
|-------|---------|
| `n` | number of input variables |
| `E(...)` | on-set: minterm indices where the function is 1 |
| `d(...)` | don't-care set: use `d()` if there are none |

Example (`Equation.txt`):
```
f(4) = E(0,2,6,8,10,11,14) + d(9,15)
```

---

## Usage

```bash
# Install dependencies
pip install -r requirements.txt

# Run with the sample input
python3 espresso.py

# Run with a custom input file
python3 espresso.py my_function.txt
```

### Example output

```
( (~X1)(~X3) + (X2)(~X3) + (X0)(~X1) + (X0)(X2) )
```

Variables are zero-indexed (`X0`, `X1`, …). `~Xi` denotes the complement of `Xi`.

---

## Algorithm

The minimizer runs three phases iteratively:

1. **Expand** — grows each cube in the on-set as far as possible without covering any off-set minterm.
2. **Irredundancy** — removes cubes made redundant by the expansion.
3. **Reduce** — shrinks each expanded cube where doing so lowers the literal count (cost function), preparing for the next expansion.

A **consensus** step removes any remaining redundant prime implicants before the final result is returned.

---

## Running Tests

```bash
python3 tests/test_espresso.py
```

---

## Dependencies

- Python 3.7+
- [NumPy](https://numpy.org/)

---

## License

MIT — see [LICENSE](LICENSE).
