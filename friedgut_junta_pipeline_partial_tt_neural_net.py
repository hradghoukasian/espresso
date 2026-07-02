"""
friedgut_junta_pipeline_partial_tt_neural_net.py

Train Algorithm 3 to learn a multi-stage XOR-of-Espresso-circuits model, then
compile the learned Espresso formulas into an exact ReLU-style neural-network
evaluator by replacing each Boolean gate with its corresponding gate network:

    NOT(a) = ReLU(-a + 1)
    AND(a_1,...,a_r) = ReLU(sum_j a_j - (r-1))
    OR(a_1,...,a_r)  = ReLU(1 - ReLU(1 - sum_j a_j))
    XOR(a_1,...,a_r) = ReLU(ReLU(s) + 2 * sum_{k=1}^{r-1} (-1)^k ReLU(s-k)),
                       where s = sum_j a_j

What this script does:
1) Run Algorithm 3 and learn the Espresso-stage predictor.
2) Compute train/test accuracy of the original learned circuit model.
3) Parse each learned Espresso expression into an AST.
4) Replace each NOT/AND/OR gate by its exact ReLU counterpart.
5) Combine stage outputs by the exact multi-input XOR ReLU network.
6) Compute train/test accuracy of the compiled neural-network evaluator.
7) Verify prediction agreement between the original circuit model and compiled NN.

This script is intentionally explicit and prints the procedure so it is easy to
check correctness step by step.

Requires:
  pip install pyeda
"""

from __future__ import annotations
import random
import re
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple, Union


# ============================================================
# PyEDA import
# ============================================================

PYEDA_OK = True
ESPRESSO_EXPR_AVAILABLE = True
PYEDA_IMPORT_ERR = None

try:
    from pyeda.inter import exprvars, Or, And
    try:
        from pyeda.inter import espresso_exprs
    except Exception:
        espresso_exprs = None
        ESPRESSO_EXPR_AVAILABLE = False
except Exception as e:
    PYEDA_OK = False
    ESPRESSO_EXPR_AVAILABLE = False
    PYEDA_IMPORT_ERR = e
    exprvars = None
    Or = None
    And = None
    espresso_exprs = None


# ============================================================
# Bit utilities
# ============================================================

def proj_index(x: int, cols: Sequence[int]) -> int:
    idx = 0
    for j, bit in enumerate(cols):
        if (x >> bit) & 1:
            idx |= 1 << j
    return idx


def bit_of_int(x: int, bit: int) -> int:
    return (x >> bit) & 1


# ============================================================
# Synthetic target function: S-junta
# ============================================================

@dataclass(frozen=True)
class SJunta:
    B: int
    junta_bits: List[int]
    junta_tt: List[int]

    def __call__(self, x: int) -> int:
        u = proj_index(x, self.junta_bits)
        return self.junta_tt[u]


def make_random_s_junta(B: int, S: int, rng: random.Random) -> SJunta:
    junta_bits = sorted(rng.sample(range(B), S))
    junta_tt = [rng.randint(0, 1) for _ in range(1 << S)]
    return SJunta(B=B, junta_bits=junta_bits, junta_tt=junta_tt)


def sample_dataset(f: SJunta, T: int, rng: random.Random) -> Tuple[List[int], List[int]]:
    xs = rng.sample(range(1 << f.B), T)
    ys = [f(x) for x in xs]
    return xs, ys


# ============================================================
# Espresso stage representation
# ============================================================

@dataclass
class Stage:
    bits: List[int]
    tt_full: List[int]
    expr_str: str

    def predict_one(self, x: int) -> int:
        k = len(self.bits)
        if k == 0:
            return 0
        u = proj_index(x, self.bits)
        return self.tt_full[u]


# ============================================================
# Original learned circuit model = XOR of stages
# ============================================================

def predict_model(stages: List[Stage], x: int) -> int:
    y = 0
    for st in stages:
        y ^= st.predict_one(x)
    return y


def accuracy_from_predictor(predict_fn, xs: Sequence[int], ys: Sequence[int]) -> float:
    correct = 0
    for x, y in zip(xs, ys):
        if int(round(predict_fn(x))) == y:
            correct += 1
    return correct / max(1, len(xs))


def accuracy(stages: List[Stage], xs: Sequence[int], ys: Sequence[int]) -> float:
    return accuracy_from_predictor(lambda x: predict_model(stages, x), xs, ys)


# ============================================================
# Influence estimation from observed neighbor pairs
# ============================================================

def influences_from_dataset_pairs(
    B: int,
    x_train: Sequence[int],
    residual_map: Dict[int, int],
) -> Tuple[List[float], List[int]]:
    xset = set(x_train)
    influences = [0.0] * B
    pair_counts = [0] * B

    for i in range(B):
        mask = 1 << i
        mismatches = 0
        cnt = 0
        for x in x_train:
            xn = x ^ mask
            if xn in xset:
                cnt += 1
                if residual_map[x] != residual_map[xn]:
                    mismatches += 1
        pair_counts[i] = cnt
        influences[i] = (mismatches / cnt) if cnt > 0 else 0.0

    return influences, pair_counts


# ============================================================
# Select J_t
# ============================================================

def select_topK_with_threshold(
    influences: Sequence[float],
    K: int,
    tau: float,
    rng: random.Random,
) -> List[int]:
    candidates = [i for i, inf in enumerate(influences) if inf > tau]
    rng.shuffle(candidates)
    candidates.sort(key=lambda i: influences[i], reverse=True)
    return candidates[: min(K, len(candidates))]


# ============================================================
# Build projected surrogate with don't cares
# ============================================================

def build_projected_majority_tt_with_dc(
    x_train: Sequence[int],
    r_train: Sequence[int],
    J: Sequence[int],
    rng: random.Random,
) -> Tuple[List[str], int]:
    k = len(J)
    if k == 0:
        c1 = sum(r_train)
        c0 = len(r_train) - c1
        if c1 > c0:
            return ["1"], 0
        if c0 > c1:
            return ["0"], 0
        return [str(rng.randint(0, 1))], 0

    size = 1 << k
    count0 = [0] * size
    count1 = [0] * size

    for x, r in zip(x_train, r_train):
        u = proj_index(x, J)
        if r == 1:
            count1[u] += 1
        else:
            count0[u] += 1

    tt_list: List[str] = []
    for u in range(size):
        c0, c1 = count0[u], count1[u]
        if c0 == 0 and c1 == 0:
            tt_list.append("-")
        elif c1 > c0:
            tt_list.append("1")
        elif c0 > c1:
            tt_list.append("0")
        else:
            tt_list.append(str(rng.randint(0, 1)))

    return tt_list, k


def format_selected_influences(J: List[int], infl: Sequence[float], max_show: int = 20) -> str:
    pairs = [(i, infl[i]) for i in J]
    pairs.sort(key=lambda p: p[1], reverse=True)
    pairs = pairs[:max_show]
    return ", ".join([f"{i}:{v:.4f}" for i, v in pairs])


# ============================================================
# Pretty-print learned expressions with global bit indices
# ============================================================

def expr_uses_local_pyeda_vars(expr_str: str) -> bool:
    return re.search(r"\bx\[\d+\]", expr_str) is not None


def convert_expr_to_global_bits(expr_str: str, bits: Sequence[int]) -> str:
    def repl(match: re.Match) -> str:
        local_idx = int(match.group(1))
        if 0 <= local_idx < len(bits):
            return f"x_{bits[local_idx]}"
        return match.group(0)

    return re.sub(r"\bx\[(\d+)\]", repl, expr_str)


def pretty_stage_expression(expr_str: str, bits: Sequence[int]) -> str:
    if expr_uses_local_pyeda_vars(expr_str):
        return convert_expr_to_global_bits(expr_str, bits)
    return expr_str


def print_learned_circuits(stages: List[Stage]) -> None:
    print("\nLearned stage circuits:")
    for t, st in enumerate(stages, start=1):
        expr_global = pretty_stage_expression(st.expr_str, st.bits)
        print(f"Stage {t}:")
        print(f"  selected bits (global) J_{t} = {st.bits}")
        print(f"  F_{t}(x) = {expr_global}")

    print("\nFinal learned model:")
    if stages:
        print("  H(x) = " + " XOR ".join([f"(F_{t}(x))" for t in range(1, len(stages) + 1)]))
    else:
        print("  H(x) = 0")


# ============================================================
# Espresso minimization
# ============================================================

def learn_espresso_stage_from_tt(
    tt_list: List[str],
    k: int,
    rng: random.Random,
) -> Tuple[List[int], str]:
    if k == 0:
        val = 1 if tt_list[0] == "1" else 0
        return [val], tt_list[0]

    tt_filled = [ch if ch != "-" else str(rng.randint(0, 1)) for ch in tt_list]
    tt_full_from_fill = [1 if ch == "1" else 0 for ch in tt_filled]

    if (not PYEDA_OK) or (not ESPRESSO_EXPR_AVAILABLE) or (espresso_exprs is None):
        reason = "PyEDA/espresso_exprs unavailable; using filled table (no minimization)"
        return tt_full_from_fill, reason

    X = exprvars("x", k)
    on_terms = []
    for u, ch in enumerate(tt_filled):
        if ch != "1":
            continue
        lits = [(X[j] if ((u >> j) & 1) else ~X[j]) for j in range(k)]
        on_terms.append(And(*lits))

    if not on_terms:
        return [0] * (1 << k), "0"

    expr = Or(*on_terms)

    try:
        minimized = espresso_exprs(expr)[0]
    except Exception as e:
        reason = f"espresso_exprs failed ({type(e).__name__}: {e}); using filled table (no minimization)"
        return tt_full_from_fill, reason

    tt_full: List[int] = []
    for u in range(1 << k):
        assignment = {X[j]: ((u >> j) & 1) for j in range(k)}
        v = minimized.restrict(assignment)
        tt_full.append(1 if v.is_one() else 0)

    return tt_full, str(minimized)


# ============================================================
# Algorithm 3 training
# ============================================================

def train_multistage_xor_espresso(
    B: int,
    K: int,
    m: int,
    tau: float,
    x_train: List[int],
    y_train: List[int],
    x_test: Optional[List[int]] = None,
    y_test: Optional[List[int]] = None,
    seed: int = 0,
    verbose: bool = True,
) -> List[Stage]:
    rng = random.Random(seed)
    stages: List[Stage] = []

    if verbose and (not PYEDA_OK):
        print(f"[warn] PyEDA import failed: {PYEDA_IMPORT_ERR}")
        print("[warn] Will run WITHOUT Espresso minimization (still trains/predicts).")

    if verbose and PYEDA_OK and (not ESPRESSO_EXPR_AVAILABLE):
        print("[warn] pyeda.inter.espresso_exprs not available in your PyEDA build.")
        print("[warn] Will run WITHOUT Espresso minimization (still trains/predicts).")

    for t in range(1, m + 1):
        r_train: List[int] = []
        residual_map: Dict[int, int] = {}
        for x, y in zip(x_train, y_train):
            ht = predict_model(stages, x)
            r = y ^ ht
            r_train.append(r)
            residual_map[x] = r

        infl, pair_counts = influences_from_dataset_pairs(B, x_train, residual_map)
        J = select_topK_with_threshold(infl, K=K, tau=tau, rng=rng)
        tt_list, k = build_projected_majority_tt_with_dc(x_train, r_train, J, rng=rng)
        tt_full, expr_str = learn_espresso_stage_from_tt(tt_list, k=k, rng=rng)
        stages.append(Stage(bits=list(J), tt_full=tt_full, expr_str=expr_str))

        if verbose:
            train_acc = accuracy(stages, x_train, y_train)
            msg = f"[stage {t:02d}] |J|={len(J)} tau={tau:.4g} train_acc={train_acc:.4f}"
            nonzero_pairs = sum(1 for c in pair_counts if c > 0)
            avg_pairs = sum(pair_counts) / max(1, B)
            msg += f"  pairbits_nonzero={nonzero_pairs}/{B} avg_|P_i|={avg_pairs:.2f}"

            if len(J) > 0:
                size = 1 << len(J)
                seen = sum(1 for ch in tt_list if ch != "-")
                msg += f"  proj_seen={seen}/{size}"
                msg += f"  sel_infl=[{format_selected_influences(J, infl)}]"
            else:
                msg += "  sel_infl=[]"

            if x_test is not None and y_test is not None:
                test_acc_now = accuracy(stages, x_test, y_test)
                msg += f"  test_acc={test_acc_now:.4f}"

            if ("no minimization" in expr_str) or ("failed" in expr_str):
                msg += "  [espresso:fallback]"

            print(msg)

    return stages


# ============================================================
# AST for learned Espresso expressions
# ============================================================

@dataclass(frozen=True)
class ConstNode:
    value: int


@dataclass(frozen=True)
class VarNode:
    local_idx: int


@dataclass(frozen=True)
class NotNode:
    child: "ExprNode"


@dataclass(frozen=True)
class AndNode:
    children: List["ExprNode"]


@dataclass(frozen=True)
class OrNode:
    children: List["ExprNode"]


ExprNode = Union[ConstNode, VarNode, NotNode, AndNode, OrNode]


class ExprParser:
    def __init__(self, s: str):
        self.s = s.replace(" ", "")
        self.n = len(self.s)
        self.i = 0

    def parse(self) -> ExprNode:
        node = self._parse_expr()
        if self.i != self.n:
            raise ValueError(f"Unexpected trailing text in expression: {self.s[self.i:]}")
        return node

    def _peek(self, text: str) -> bool:
        return self.s.startswith(text, self.i)

    def _consume(self, text: str) -> None:
        if not self._peek(text):
            raise ValueError(f"Expected '{text}' at position {self.i} in {self.s}")
        self.i += len(text)

    def _parse_expr(self) -> ExprNode:
        if self._peek("Or("):
            return self._parse_nary("Or")
        if self._peek("And("):
            return self._parse_nary("And")
        if self._peek("~"):
            self._consume("~")
            child = self._parse_atom()
            return NotNode(child)
        return self._parse_atom()

    def _parse_atom(self) -> ExprNode:
        if self._peek("x["):
            self._consume("x[")
            start = self.i
            while self.i < self.n and self.s[self.i].isdigit():
                self.i += 1
            if start == self.i:
                raise ValueError(f"Missing index after x[ in {self.s}")
            idx = int(self.s[start:self.i])
            self._consume("]")
            return VarNode(idx)
        if self._peek("0"):
            self._consume("0")
            return ConstNode(0)
        if self._peek("1"):
            self._consume("1")
            return ConstNode(1)
        if self._peek("Or("):
            return self._parse_nary("Or")
        if self._peek("And("):
            return self._parse_nary("And")
        raise ValueError(f"Cannot parse atom at position {self.i} in {self.s}")

    def _parse_nary(self, op_name: str) -> ExprNode:
        self._consume(op_name + "(")
        children: List[ExprNode] = []
        while True:
            child = self._parse_expr()
            children.append(child)
            if self._peek(","):
                self._consume(",")
                continue
            break
        self._consume(")")
        if op_name == "Or":
            return OrNode(children)
        if op_name == "And":
            return AndNode(children)
        raise ValueError(f"Unknown n-ary operator {op_name}")


# ============================================================
# Exact ReLU gate counterparts
# ============================================================

def relu(v: float) -> float:
    return v if v > 0.0 else 0.0


def nn_not(a: float) -> float:
    return relu((-1.0) * a + 1.0)


def nn_and(inputs: Sequence[float]) -> float:
    r = len(inputs)
    if r == 0:
        return 1.0
    s = sum(inputs)
    return relu(s - (r - 1))


def nn_or(inputs: Sequence[float]) -> float:
    if len(inputs) == 0:
        return 0.0
    s = sum(inputs)
    return relu(1.0 - relu(1.0 - s))


def nn_xor(inputs: Sequence[float]) -> float:
    r = len(inputs)
    if r == 0:
        return 0.0
    s = sum(inputs)
    val = relu(s)
    for k in range(1, r):
        val += 2.0 * ((-1.0) ** k) * relu(s - k)
    return relu(val)




def tt_full_to_exact_dnf_expr(tt_full: Sequence[int], k: int) -> str:
    """
    Build an exact (not minimized) DNF expression over local variables x[0],...,x[k-1]
    from the fully specified truth table tt_full.
    """
    if k == 0:
        return "1" if int(tt_full[0]) == 1 else "0"

    terms: List[str] = []
    for u, val in enumerate(tt_full):
        if int(val) != 1:
            continue
        lits: List[str] = []
        for j in range(k):
            lits.append(f"x[{j}]" if ((u >> j) & 1) else f"~x[{j}]")
        terms.append(f"And({', '.join(lits)})")

    if not terms:
        return "0"
    if len(terms) == 1:
        return terms[0]
    return "Or(" + ", ".join(terms) + ")"

# ============================================================
# Compile/evaluate learned Espresso expressions as a ReLU network
# ============================================================

def evaluate_expr_as_relu_nn(node: ExprNode, x: int, stage_bits: Sequence[int]) -> float:
    if isinstance(node, ConstNode):
        return float(node.value)

    if isinstance(node, VarNode):
        global_bit = stage_bits[node.local_idx]
        return float(bit_of_int(x, global_bit))

    if isinstance(node, NotNode):
        child_val = evaluate_expr_as_relu_nn(node.child, x, stage_bits)
        return nn_not(child_val)

    if isinstance(node, AndNode):
        vals = [evaluate_expr_as_relu_nn(ch, x, stage_bits) for ch in node.children]
        return nn_and(vals)

    if isinstance(node, OrNode):
        vals = [evaluate_expr_as_relu_nn(ch, x, stage_bits) for ch in node.children]
        return nn_or(vals)

    raise TypeError(f"Unknown node type: {type(node)}")


def count_gate_nodes(node: ExprNode) -> Dict[str, int]:
    out = {"NOT": 0, "AND": 0, "OR": 0}
    _count_gate_nodes_rec(node, out)
    return out


def _count_gate_nodes_rec(node: ExprNode, out: Dict[str, int]) -> None:
    if isinstance(node, (ConstNode, VarNode)):
        return
    if isinstance(node, NotNode):
        out["NOT"] += 1
        _count_gate_nodes_rec(node.child, out)
        return
    if isinstance(node, AndNode):
        out["AND"] += 1
        for ch in node.children:
            _count_gate_nodes_rec(ch, out)
        return
    if isinstance(node, OrNode):
        out["OR"] += 1
        for ch in node.children:
            _count_gate_nodes_rec(ch, out)
        return
    raise TypeError(type(node))


@dataclass
class CompiledStageNN:
    bits: List[int]
    expr_str: str
    ast: ExprNode

    def predict_one(self, x: int) -> int:
        val = evaluate_expr_as_relu_nn(self.ast, x, self.bits)
        return int(round(val))


@dataclass
class CompiledFullNN:
    stage_nets: List[CompiledStageNN]

    def predict_one(self, x: int) -> int:
        stage_outputs = [float(st.predict_one(x)) for st in self.stage_nets]
        val = nn_xor(stage_outputs)
        return int(round(val))


def compile_stage_to_relu_nn(stage: Stage) -> CompiledStageNN:
    expr = stage.expr_str.strip()
    if ("no minimization" in expr) or ("failed" in expr):
        expr = tt_full_to_exact_dnf_expr(stage.tt_full, len(stage.bits))
    parser = ExprParser(expr)
    ast = parser.parse()
    return CompiledStageNN(bits=list(stage.bits), expr_str=expr, ast=ast)


def compile_full_model_to_relu_nn(stages: List[Stage]) -> CompiledFullNN:
    return CompiledFullNN([compile_stage_to_relu_nn(st) for st in stages])


# ============================================================
# Verification helpers
# ============================================================

def verify_stage_nn_matches_table(stage: Stage, compiled_stage: CompiledStageNN) -> Tuple[bool, Optional[Tuple[int, int, int]]]:
    k = len(stage.bits)
    for u in range(1 << k):
        x = 0
        for j, bit in enumerate(stage.bits):
            if (u >> j) & 1:
                x |= (1 << bit)
        table_val = stage.tt_full[u]
        nn_val = compiled_stage.predict_one(x)
        if table_val != nn_val:
            return False, (u, table_val, nn_val)
    return True, None


def agreement_rate(pred1, pred2, xs: Sequence[int]) -> float:
    same = 0
    for x in xs:
        if int(round(pred1(x))) == int(round(pred2(x))):
            same += 1
    return same / max(1, len(xs))


# ============================================================
# Main experiment
# ============================================================

def main():
    B = 15
    S = 8
    T_train = 2 ** B
    T_test = 2 ** B
    K = 6
    m = 5
    tau = 0.02
    seed = 42

    verbose_training = False
    print_circuits = True

    rng = random.Random(seed)
    f = make_random_s_junta(B=B, S=S, rng=rng)
    x_train, y_train = sample_dataset(f, T_train, rng=rng)
    x_test, y_test = sample_dataset(f, T_test, rng=rng)

    print("=" * 90)
    print("STEP 0: DATA / TARGET")
    print(f"seed = {seed}")
    print(f"B = {B}, S = {S}, T_train = {T_train}, T_test = {T_test}, K = {K}, m = {m}, tau = {tau}")
    print(f"True junta bits: {f.junta_bits}")

    print("\n" + "=" * 90)
    print("STEP 1: RUN ALGORITHM 3 AND LEARN THE ESPRESSO MODEL")
    t0 = time.perf_counter()
    stages = train_multistage_xor_espresso(
        B=B,
        K=K,
        m=m,
        tau=tau,
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
        seed=seed,
        verbose=verbose_training,
    )
    t1 = time.perf_counter()
    print(f"Learned {len(stages)} stages in {t1 - t0:.2f} seconds.")

    if print_circuits:
        print_learned_circuits(stages)

    espresso_train_acc = accuracy(stages, x_train, y_train)
    espresso_test_acc = accuracy(stages, x_test, y_test)

    print("\nOriginal learned Espresso-circuit predictor:")
    print(f"  train accuracy = {espresso_train_acc:.6f}")
    print(f"  test  accuracy = {espresso_test_acc:.6f}")

    print("\n" + "=" * 90)
    print("STEP 2: COMPILE EACH LEARNED ESPRESSO FORMULA INTO ITS EXACT RELU-GATE COUNTERPART")
    print("Procedure:")
    print("  - Parse each stage expression into an AST with nodes {Var, NOT, AND, OR}.")
    print("  - Evaluate NOT nodes via     ReLU(-a + 1).")
    print("  - Evaluate AND nodes via     ReLU(sum(inputs) - (r-1)).")
    print("  - Evaluate OR nodes via      ReLU(1 - ReLU(1 - sum(inputs))).")
    print("  - Combine stage outputs via  multi-input XOR parity ReLU network.")
    print("  - Compare the compiled NN predictor against the original circuit predictor.")

    compiled_nn = compile_full_model_to_relu_nn(stages)

    total_counts = {"NOT": 0, "AND": 0, "OR": 0, "XOR": 1 if len(stages) > 0 else 0}
    for t, (stage, compiled_stage) in enumerate(zip(stages, compiled_nn.stage_nets), start=1):
        ok, mismatch = verify_stage_nn_matches_table(stage, compiled_stage)
        counts = count_gate_nodes(compiled_stage.ast)
        total_counts["NOT"] += counts["NOT"]
        total_counts["AND"] += counts["AND"]
        total_counts["OR"] += counts["OR"]

        source_expr = compiled_stage.expr_str
        expr_global = pretty_stage_expression(source_expr, stage.bits)
        print(f"\nStage {t}:")
        print(f"  selected bits = {stage.bits}")
        if source_expr == stage.expr_str:
            print(f"  source        = Espresso minimized expression")
        else:
            print(f"  source        = exact DNF fallback from stage.tt_full (PyEDA unavailable in this environment)")
        print(f"  expression    = {expr_global}")
        print(f"  gate counts   = NOT:{counts['NOT']}  AND:{counts['AND']}  OR:{counts['OR']}")
        if ok:
            print("  verification  = PASS (compiled stage NN matches stage.tt_full on all local assignments)")
        else:
            u, table_val, nn_val = mismatch
            print("  verification  = FAIL")
            print(f"  mismatch at local assignment u={u}: tt_full={table_val}, nn={nn_val}")
            raise RuntimeError("Compiled stage NN does not match stage truth table.")

    print("\nFull compiled NN gate summary:")
    print(f"  total NOT gates = {total_counts['NOT']}")
    print(f"  total AND gates = {total_counts['AND']}")
    print(f"  total OR gates  = {total_counts['OR']}")
    print(f"  final XOR gate  = {total_counts['XOR']} (multi-input XOR over stage outputs)")

    print("\n" + "=" * 90)
    print("STEP 3: EVALUATE THE COMPILED RELU NETWORK AND COMPARE WITH THE ORIGINAL CIRCUIT MODEL")

    nn_train_acc = accuracy_from_predictor(compiled_nn.predict_one, x_train, y_train)
    nn_test_acc = accuracy_from_predictor(compiled_nn.predict_one, x_test, y_test)

    train_agree = agreement_rate(lambda x: predict_model(stages, x), compiled_nn.predict_one, x_train)
    test_agree = agreement_rate(lambda x: predict_model(stages, x), compiled_nn.predict_one, x_test)

    print("Compiled ReLU-network predictor:")
    print(f"  train accuracy = {nn_train_acc:.6f}")
    print(f"  test  accuracy = {nn_test_acc:.6f}")
    print()
    print("Agreement between original Espresso predictor and compiled ReLU network:")
    print(f"  train agreement = {train_agree:.6f}")
    print(f"  test  agreement = {test_agree:.6f}")
    print()
    print("Difference (should be zero if the implementation is correct):")
    print(f"  |train acc diff| = {abs(espresso_train_acc - nn_train_acc):.12f}")
    print(f"  |test  acc diff| = {abs(espresso_test_acc - nn_test_acc):.12f}")

    if abs(espresso_train_acc - nn_train_acc) > 1e-12 or abs(espresso_test_acc - nn_test_acc) > 1e-12:
        print("\n[warning] Accuracy mismatch detected.")
    else:
        print("\n[success] The compiled ReLU network exactly matches the learned Espresso-circuit predictor on the datasets.")

    print("\nDone.")


if __name__ == "__main__":
    main()
