from __future__ import annotations

import math
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
# AST for learned expressions
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
        return OrNode(children) if op_name == "Or" else AndNode(children)


# ============================================================
# Exact ReLU gate formulas
# ============================================================

def relu(v: float) -> float:
    return v if v > 0.0 else 0.0



def nn_not(a: float) -> float:
    return relu(1.0 - a)



def nn_and(inputs: Sequence[float]) -> float:
    r = len(inputs)
    if r == 0:
        return 1.0
    return relu(sum(inputs) - (r - 1))



def nn_or_lemma5(inputs: Sequence[float]) -> float:
    s = sum(inputs)
    return relu(s) - relu(s - 1.0)



def nn_or_lemma6(inputs: Sequence[float]) -> float:
    s = sum(inputs)
    return relu(1.0 - relu(1.0 - s))



def nn_xor_lemma7(inputs: Sequence[float]) -> float:
    r = len(inputs)
    if r == 0:
        return 0.0
    s = sum(inputs)
    val = relu(s)
    for k in range(1, r):
        val += 2.0 * ((-1.0) ** k) * relu(s - k)
    return relu(val)



def nn_xor_lemma8(inputs: Sequence[float]) -> float:
    r = len(inputs)
    if r == 0:
        return 0.0
    if r == 1:
        return inputs[0]
    k = math.ceil(math.log2(r))
    x = sum(inputs) / (2 ** k)
    for _ in range(k):
        u = relu(x)
        v = relu(x - 0.5)
        x = relu(2.0 * u - 4.0 * v)
    return x


# ============================================================
# DNF fallback and SOP normalization
# ============================================================

def tt_full_to_exact_dnf_expr(tt_full: Sequence[int], k: int) -> str:
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


Literal = Tuple[str, int]  # ("pos"|"neg", global_bit)


@dataclass
class StageSOP:
    terms: List[List[Literal]]
    expr_str: str

    def kind(self) -> str:
        if len(self.terms) == 0:
            return "const0"
        if len(self.terms) == 1 and len(self.terms[0]) == 0:
            return "const1"
        if len(self.terms) == 1 and len(self.terms[0]) == 1:
            return "literal"
        if len(self.terms) == 1:
            return "and"
        return "sop"


def ast_literal_to_global(node: ExprNode, bits: Sequence[int]) -> Literal:
    if isinstance(node, VarNode):
        return ("pos", bits[node.local_idx])
    if isinstance(node, NotNode) and isinstance(node.child, VarNode):
        return ("neg", bits[node.child.local_idx])
    raise ValueError(f"Expected literal, got {type(node)}")



def ast_to_sop(node: ExprNode, bits: Sequence[int]) -> StageSOP:
    if isinstance(node, ConstNode):
        return StageSOP(terms=[[]] if node.value == 1 else [], expr_str="")
    if isinstance(node, VarNode):
        return StageSOP(terms=[[ast_literal_to_global(node, bits)]], expr_str="")
    if isinstance(node, NotNode):
        return StageSOP(terms=[[ast_literal_to_global(node, bits)]], expr_str="")
    if isinstance(node, AndNode):
        return StageSOP(terms=[[ast_literal_to_global(ch, bits) for ch in node.children]], expr_str="")
    if isinstance(node, OrNode):
        terms: List[List[Literal]] = []
        for ch in node.children:
            child_sop = ast_to_sop(ch, bits)
            if child_sop.kind() == "const1":
                return StageSOP(terms=[[]], expr_str="")
            terms.extend(child_sop.terms)
        return StageSOP(terms=terms, expr_str="")
    raise TypeError(type(node))



def stage_to_sop(stage: Stage) -> StageSOP:
    expr = stage.expr_str.strip()
    if ("no minimization" in expr) or ("failed" in expr):
        expr = tt_full_to_exact_dnf_expr(stage.tt_full, len(stage.bits))
    sop = ast_to_sop(ExprParser(expr).parse(), stage.bits)
    sop.expr_str = expr
    return sop


# ============================================================
# Compiled network objects and evaluation
# ============================================================

@dataclass
class ArchitectureStats:
    input_width: int
    layer_widths: List[int]  # non-input ReLU widths only

    @property
    def depth(self) -> int:
        return len(self.layer_widths)

    @property
    def total_neurons(self) -> int:
        return sum(self.layer_widths)

    @property
    def max_width(self) -> int:
        return max(self.layer_widths) if self.layer_widths else 0


@dataclass
class CompiledVariant:
    name: str
    or_choice: str
    xor_choice: str
    B: int
    stages: List[Stage]
    sops: List[StageSOP]
    stats: ArchitectureStats

    def predict_one(self, x: int) -> int:
        pos = [float(bit_of_int(x, i)) for i in range(self.B)]
        neg = [nn_not(v) for v in pos]

        stage_vals: List[float] = []
        for sop in self.sops:
            k = sop.kind()
            if k == "const0":
                val = 0.0
            elif k == "const1":
                val = 1.0
            elif k == "literal":
                sign, bit = sop.terms[0][0]
                val = pos[bit] if sign == "pos" else neg[bit]
            elif k == "and":
                vals = [(pos[b] if s == "pos" else neg[b]) for s, b in sop.terms[0]]
                val = nn_and(vals)
            else:
                term_vals = []
                for term in sop.terms:
                    vals = [(pos[b] if s == "pos" else neg[b]) for s, b in term]
                    term_vals.append(nn_and(vals))
                val = nn_or_lemma5(term_vals) if self.or_choice == "lemma5" else nn_or_lemma6(term_vals)
            stage_vals.append(val)

        out = nn_xor_lemma7(stage_vals) if self.xor_choice == "lemma7" else nn_xor_lemma8(stage_vals)
        return int(round(out))


# ============================================================
# Layer-count compiler
# ============================================================

def build_architecture_stats(B: int, sops: List[StageSOP], or_choice: str, xor_choice: str) -> ArchitectureStats:
    """
    Count strict layer-by-layer ReLU widths.

    Every non-input layer may only read from the immediately previous layer.
    Therefore, whenever a signal would otherwise skip a layer (for example a
    literal feeding an OR block, or an AND-only stage feeding the XOR block),
    we insert an identity ReLU carry neuron in each skipped hidden layer.
    """
    m = len(sops)
    widths: List[int] = [2 * B]  # literal layer

    # AND / carry layer:
    # - const/literal/and stages contribute one stage signal here
    # - each SOP stage contributes one signal per product term; a single-literal
    #   term is carried by identity, while a longer term is formed by Lemma 4.
    and_or_carry_width = 0
    n_sop = 0
    for sop in sops:
        k = sop.kind()
        if k == "sop":
            n_sop += 1
            and_or_carry_width += len(sop.terms)
        else:
            and_or_carry_width += 1
    widths.append(and_or_carry_width)

    # OR / stage-alignment layers:
    # every stage must survive to the common stage-output depth using only
    # adjacent-layer connections.
    if or_choice == "lemma5":
        # Each genuine OR stage uses the width-2 Lemma 5 gadget in one ReLU layer.
        # Every non-OR stage is carried through that same layer by one identity neuron.
        widths.append(2 * n_sop + (m - n_sop))
    elif or_choice == "lemma6":
        # Each genuine OR stage uses two width-1 ReLU layers.
        # Every non-OR stage is carried through both layers by one identity neuron.
        widths.append(m)
        widths.append(m)
    else:
        raise ValueError(or_choice)

    # Final XOR block.
    if xor_choice == "lemma7":
        if m >= 2:
            widths.extend([m, 1])
        elif m == 1:
            widths.append(1)
    elif xor_choice == "lemma8":
        if m >= 2:
            k = math.ceil(math.log2(m))
            for _ in range(k):
                widths.extend([2, 1])
        elif m == 1:
            widths.append(1)
    else:
        raise ValueError(xor_choice)

    return ArchitectureStats(input_width=B, layer_widths=widths)



def compile_variant(stages: List[Stage], B: int, or_choice: str, xor_choice: str) -> CompiledVariant:
    sops = [stage_to_sop(st) for st in stages]
    stats = build_architecture_stats(B=B, sops=sops, or_choice=or_choice, xor_choice=xor_choice)
    return CompiledVariant(
        name=f"{or_choice}_{xor_choice}",
        or_choice=or_choice,
        xor_choice=xor_choice,
        B=B,
        stages=stages,
        sops=sops,
        stats=stats,
    )


# ============================================================
# Verification helpers
# ============================================================

def verify_variant_matches_model(variant: CompiledVariant, xs: Sequence[int], stages: List[Stage]) -> Tuple[bool, Optional[Tuple[int, int, int]]]:
    for x in xs:
        circuit_val = predict_model(stages, x)
        nn_val = variant.predict_one(x)
        if circuit_val != nn_val:
            return False, (x, circuit_val, nn_val)
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
    B = 20
    S = 8
    T_train = 4000
    T_test = 4000
    K = 6
    m = 100
    tau = 0.02
    seed = 0

    verbose_training = False
    print_circuits = True

    rng = random.Random(seed)
    f = make_random_s_junta(B=B, S=S, rng=rng)
    x_train, y_train = sample_dataset(f, T=T_train, rng=rng)
    x_test, y_test = sample_dataset(f, T=T_test, rng=rng)

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
    print("STEP 2: COMPILE 4 FEEDFORWARD RELU NETWORKS")
    variants = [
        compile_variant(stages, B=B, or_choice="lemma5", xor_choice="lemma7"),
        compile_variant(stages, B=B, or_choice="lemma5", xor_choice="lemma8"),
        compile_variant(stages, B=B, or_choice="lemma6", xor_choice="lemma7"),
        compile_variant(stages, B=B, or_choice="lemma6", xor_choice="lemma8"),
    ]

    all_ok = True
    for variant in variants:
        train_ok, train_mis = verify_variant_matches_model(variant, x_train, stages)
        test_ok, test_mis = verify_variant_matches_model(variant, x_test, stages)
        nn_train_acc = accuracy_from_predictor(variant.predict_one, x_train, y_train)
        nn_test_acc = accuracy_from_predictor(variant.predict_one, x_test, y_test)
        train_agree = agreement_rate(lambda x: predict_model(stages, x), variant.predict_one, x_train)
        test_agree = agreement_rate(lambda x: predict_model(stages, x), variant.predict_one, x_test)
        stats = variant.stats

        print("\n" + "-" * 90)
        print(f"Variant: {variant.name}")
        print(f"  OR implementation  = {variant.or_choice}")
        print(f"  XOR implementation = {variant.xor_choice}")
        print(f"  verified on train  = {'YES' if train_ok else 'NO'}")
        print(f"  verified on test   = {'YES' if test_ok else 'NO'}")
        print(f"  train agreement    = {train_agree:.6f}")
        print(f"  test agreement     = {test_agree:.6f}")
        print(f"  train accuracy     = {nn_train_acc:.6f}")
        print(f"  test accuracy      = {nn_test_acc:.6f}")
        print(f"  layers (depth)     = {stats.depth}")
        print(f"  total neurons      = {stats.total_neurons}")
        print(f"  max width          = {stats.max_width}")
        print(f"  non-input widths   = {stats.layer_widths}")

        if not train_ok:
            print(f"  first train mismatch = x={train_mis[0]}, circuit={train_mis[1]}, nn={train_mis[2]}")
            all_ok = False
        if not test_ok:
            print(f"  first test mismatch  = x={test_mis[0]}, circuit={test_mis[1]}, nn={test_mis[2]}")
            all_ok = False

    print("\n" + "=" * 90)
    if all_ok:
        print("[success] All 4 compiled feedforward ReLU networks match the learned Algorithm 3 circuit on train and test data.")
    else:
        print("[warning] At least one compiled network did not match the learned circuit.")


if __name__ == "__main__":
    main()
