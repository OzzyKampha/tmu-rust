"""Check TMFuzzyPatternClassifier against the reference Fuzzy-Pattern TM.

A line-by-line Python transliteration of check_clause, vote and feedback! from
BooBSD/FuzzyPatternTM's src/FuzzyPatternTM.jl (MIT License, Copyright (c) 2025
Artem Hnilov), kept in its data layout (separate positive/negative clause sets,
separate literal and inverted-literal states). It draws its random numbers from
tmu-rs's SplitMix64 in the reference's order, so it must end in exactly the
states tmu-rs reaches. Each class's feedback touches only its own clauses, so
the reference's shuffled class order and its unsynchronised threads are left
out; samples are visited in tmu-rs's shuffled order.

    cargo build --release --example fuzzy_pattern_dump
    python3 scripts/check_fuzzy_pattern.py target/release/examples/fuzzy_pattern_dump
"""
import json, sys, subprocess
M = (1 << 64) - 1
G = 0x9E3779B97F4A7C15
class Rng:
    def __init__(s, seed): s.x = (seed + G) & M
    def u64(s):
        s.x = (s.x + G) & M; z = s.x
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & M
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & M
        return z ^ (z >> 31)
    def f64(s): return (s.u64() >> 11) * (1.0 / (1 << 53))
    def below(s, n): return s.u64() % n   # stands in for Julia's rand(1:n) (0-based here)

def check_clause(x, literals, literals_inverted, LF):
    length_literals = len(literals) + len(literals_inverted)
    c = length_literals if 0 < length_literals < LF else LF
    for i in literals:
        if c <= 0: return 0
        c -= (not x[i])
    for i in literals_inverted:
        if c <= 0: return 0
        c -= x[i]
    return c

class TATeam:
    def __init__(s, size, half, limit):
        s.size = size
        s.pos = [[limit - 1] * size for _ in range(half)]; s.neg = [[limit - 1] * size for _ in range(half)]
        s.pos_inv = [[limit - 1] * size for _ in range(half)]; s.neg_inv = [[limit - 1] * size for _ in range(half)]
        s.pl = [[] for _ in range(half)]; s.nl = [[] for _ in range(half)]
        s.pli = [[] for _ in range(half)]; s.nli = [[] for _ in range(half)]

def vote(ta, x, LF):
    pos = sum(check_clause(x, ta.pl[i], ta.pli[i], LF) for i in range(len(ta.pl)))
    neg = sum(check_clause(x, ta.nl[i], ta.nli[i], LF) for i in range(len(ta.nl)))
    return pos, neg

def feedback(tm, ta, rng, x, clauses1, clauses_inverted1, clauses2, clauses_inverted2, literals1, literals_inverted1, literals2, literals_inverted2, positive):
    T, LF, L, s, limit = tm["T"], tm["LF"], tm["L"], tm["s"], tm["limit"]
    p, n = vote(ta, x, LF); v = max(-T, min(T, p - n))
    update = ((T - v) if positive else (T + v)) / (T * 2)
    state_max, state_min = 255, 0
    for j in range(len(clauses1)):           # Feedback 1
        c, ci = clauses1[j], clauses_inverted1[j]
        if rng.f64() < update:
            if check_clause(x, literals1[j], literals_inverted1[j], LF) > 0:
                if len(literals1[j]) + len(literals_inverted1[j]) <= L:
                    for i in range(ta.size):
                        if x[i] == 1 and c[i] < state_max: c[i] += 1
                        if x[i] == 0 and ci[i] < state_max: ci[i] += 1
                for i in range(ta.size):
                    if x[i] == 0 and c[i] < limit and c[i] > state_min: c[i] -= 1
                    if x[i] == 1 and ci[i] < limit and ci[i] > state_min: ci[i] -= 1
            else:
                for _ in range(s):
                    i = rng.below(ta.size)
                    if c[i] > state_min: c[i] -= 1
                    i = rng.below(ta.size)
                    if ci[i] > state_min: ci[i] -= 1
            literals1[j] = [i for i in range(ta.size) if c[i] >= limit]
            literals_inverted1[j] = [i for i in range(ta.size) if ci[i] >= limit]
    for j in range(len(clauses2)):           # Feedback 2
        c, ci = clauses2[j], clauses_inverted2[j]
        if rng.f64() < update:
            if check_clause(x, literals2[j], literals_inverted2[j], LF) > 0:
                for i in range(ta.size):
                    if x[i] == 0 and c[i] < limit: c[i] += 1
                    if x[i] == 1 and ci[i] < limit: ci[i] += 1
                literals2[j] = [i for i in range(ta.size) if c[i] >= limit]
                literals_inverted2[j] = [i for i in range(ta.size) if ci[i] >= limit]

def train_one(tm, teams, rngs, x, y):
    # Each class's feedback reads and writes only its own team, so the
    # reference's shuffled class order does not change the result.
    for cls, ta in enumerate(teams):
        if cls == y:
            feedback(tm, ta, rngs[cls], x, ta.pos, ta.pos_inv, ta.neg, ta.neg_inv, ta.pl, ta.pli, ta.nl, ta.nli, True)
        else:
            feedback(tm, ta, rngs[cls], x, ta.neg, ta.neg_inv, ta.pos, ta.pos_inv, ta.nl, ta.nli, ta.pl, ta.pli, False)

def main(binpath, seed, epochs, nf, cl, T, S, L, LF, limit):
    out = json.loads(subprocess.run([binpath, *map(str, (seed, epochs, nf, cl, T, S, L, LF, limit))], capture_output=True, text=True, check=True).stdout)
    X, Y = out["X"], out["y"]
    s = round(nf / S)  # Python's round is ties-to-even, as Julia's
    assert s == out["s_draws"], (s, out["s_draws"])
    tm = dict(T=T, LF=LF, L=L, s=s, limit=limit)
    half = cl // 2
    teams = [TATeam(nf, half, limit) for _ in range(3)]
    rngs = [Rng(seed ^ ((G * (c + 1)) & M)) for c in range(3)]
    shuffle = Rng(seed)
    for _ in range(epochs):
        order = list(range(len(Y)))
        for i in range(len(Y) - 1, 0, -1):
            k = shuffle.below(i + 1); order[i], order[k] = order[k], order[i]
        for i in order:
            train_one(tm, teams, rngs, X[i], Y[i])
    # tmu-rs layout: clause 2j = positive j, 2j+1 = negative j; literals: positive then inverted.
    diffs = 0
    for c, ta in enumerate(teams):
        for j in range(half):
            for k, a in enumerate((ta.pos[j] + ta.pos_inv[j], ta.neg[j] + ta.neg_inv[j])):
                r = out["states"][c][2 * j + k]
                diffs += sum(1 for u, w in zip(a, r) if u != w)
    pred = []
    for x in X:
        votes = [(lambda pn: pn[0] - pn[1])(vote(ta, x, LF)) for ta in teams]
        pred.append(max(range(3), key=lambda c: (votes[c], -c)))
    mism = sum(p != q for p, q in zip(pred, out["pred"]))
    acc = sum(p == y for p, y in zip(pred, Y)) / len(Y)
    states = sum(len(r) for cls in out["states"] for r in cls)
    print(f"seed {seed} epochs {epochs} nf {nf} clauses {cl} T {T} S {S} (s {s}) L {L} LF {LF} limit {limit}: "
          f"{diffs} of {states} states differ, {mism} of {len(Y)} predictions differ, train acc {acc:.3f}")
    return diffs == 0 and mism == 0

if __name__ == "__main__":
    b = sys.argv[1]
    ok = all([main(b, 1, 3, 16, 6, 5, 3.0, 5, 2, 128), main(b, 7, 5, 16, 10, 8, 5.0, 3, 1, 128),
              main(b, 3, 4, 20, 8, 6, 2.5, 16, 4, 220), main(b, 11, 6, 9, 4, 3, 2.0, 2, 3, 200)])
    print("ALL IDENTICAL" if ok else "MISMATCH")
