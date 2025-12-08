import math
import time
from typing import Iterable, List, Dict

from pyeda.inter import exprvars, truthtable
from pyeda.boolalg.minimization import espresso_tts
from pyeda.boolalg.expr import OrOp, AndOp  # for type checks used internally


def extract_sop_terms(expr) -> List[List]:
    """
    Given a minimized PyEDA expression, return a list of product terms in
    Sum-of-Products (SOP) form. Each term is returned as a list of literals.

    Example output:
        [[x0, ~x1, x2], [~x0, x3], ...]

    NOTE:
        We first convert to DNF (OR of ANDs). Then we parse the top level:
        - If it's an OR node → each child is one product term.
        - If it's not OR       → it must be a single term.
    """

    # Convert to Disjunctive Normal Form (OR of ANDs). PyEDA does this automatically.
    dnf = expr.to_dnf()

    # Constant 0 → EMPTY SOP (no product terms)
    if dnf.is_zero():
        return []

    # If expression is a disjunction, split its children; otherwise treat as a lone term.
    if isinstance(dnf, OrOp):      # dnf is: (term1) OR (term2) OR ...
        sum_terms = list(dnf.xs)   # dnf.xs holds the term nodes
    else:                          # single product such as x0 & ~x1 & x2
        sum_terms = [dnf]

    # Build list of lists of literals
    terms: List[List] = []
    for term in sum_terms:
        # Each product term can be AndOp (multiple literals) or a single literal
        if isinstance(term, AndOp):
            terms.append(list(term.xs))
        else:
            terms.append([term])
    return terms


def build_tt_from_on_off_sets(
    n_vars: int,
    on_set: Iterable[int],
    off_set: Iterable[int],
    default: str = "-"
) -> str:
    """
    Construct a truth table of length 2^n_vars (lexicographic order) using:

        f = 1  for indices in `on_set`
        f = 0  for indices in `off_set`
        f = '-' (don't-care) for unspecified indices

    This is useful for incomplete truth tables. Espresso can choose the best
    minimization for don't-cares automatically.

    Parameters
    ----------
    n_vars : int
        Number of input variables in Boolean function.
    on_set : iterable of int
        Indices where f = 1.
    off_set : iterable of int
        Indices where f = 0.
    default : str, optional
        Character for unspecified entries ('-' recommended).
    """

    size = 2 ** n_vars            # total minterms
    tt = [default] * size         # initialize all entries as don't-cares

    # Convert to set for safety and check conflicts
    on_set, off_set = set(on_set), set(off_set)
    overlap = on_set & off_set
    if overlap:
        raise ValueError(f"ON and OFF sets overlap at indices: {overlap}")

    # Fill ON minterms with 1
    for idx in on_set:
        if not (0 <= idx < size):
            raise ValueError(f"ON index {idx} out of range for n_vars={n_vars}")
        tt[idx] = "1"

    # Fill OFF minterms with 0
    for idx in off_set:
        if not (0 <= idx < size):
            raise ValueError(f"OFF index {idx} out of range for n_vars={n_vars}")
        tt[idx] = "0"

    return "".join(tt)


def minimize_truth_table_espresso(tt: str, verbose: bool = True):
    """
    Minimize a Boolean function using the Espresso heuristic (via PyEDA),
    given its truth-table string in lexicographic order.

    Accepted characters:
        '1'  → ON-set
        '0'  → OFF-set
        '-' or 'x' → Don't-care (Espresso decides best)

    Returns
    -------
    f_min : PyEDA expression
        Minimized SOP (DNF) Boolean expression.
    stats  : dict
        Contains:
            'n_vars', 'variables', 'truth_table',
            'num_terms', 'literal_counts', 'total_literals', 'elapsed'

    NOTE:
        verbose=True prints the expression and statistics.
        verbose=False returns values silently (useful for experiments).
    """

    # Validate that the truth table length is a power of 2
    L = len(tt)
    if L == 0 or (L & (L - 1)) != 0:
        raise ValueError("Truth table length must be 2^n")

    # Number of variables = log2(length)
    n_vars = int(math.log2(L))
    X = exprvars("x", n_vars)       # Creates x0, x1, ..., x_{n-1}

    # Build truth table object (PyEDA accepts '-' as don't-care)
    f_tt = truthtable(X, tt)

    # Run Espresso and measure runtime
    t0 = time.perf_counter()
    f_min, = espresso_tts(f_tt)     # espresso_tts returns a 1-element tuple here
    elapsed = time.perf_counter() - t0

    # Compute statistics
    terms = extract_sop_terms(f_min)
    literal_counts = [len(t) for t in terms]

    stats: Dict = {
        "n_vars": n_vars,
        "variables": list(X),
        "truth_table": tt,
        "num_terms": len(terms),
        "literal_counts": literal_counts,
        "total_literals": sum(literal_counts),
        "elapsed": elapsed,
    }

    # Print details if requested
    if verbose:
        print("\n=== Espresso Minimization ===")
        print("f =", f_min)
        print("\nStatistics:", stats, "\n")

    return f_min, stats