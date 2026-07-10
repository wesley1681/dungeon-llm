"""Empirical team-strength currency for balanced-but-diverse matchmaking.

WHY empirical (see project memory / Phase 1 calibration, scripts/exp_matchup_calib.py):
a naive Σ-of-levels prior is provably useless for balance — at *equal* Σlevel the
scripted win rate swings 0.11↔0.94, driven by (a) action economy (more bodies win
super-linearly) and (b) class-feature breakpoints (a champion's strength CLIFFS at
L5 Extra Attack: champ@L4 never beats champ@L5). No closed-form Σ+exponent captures
that, so strength is LEARNED from actual game outcomes.

Model (generalised Bradley-Terry / team-Elo, linear in logits):

    T(team) = Σ_i  r[atom_i]  +  g[|team|]
    P(A beats B) = σ( T(A) − T(B) )

  • ``r[atom]`` — a per-ATOM rating in logit units. An atom is (class, level) for a
    standard class (so L4 and L5 champions are DIFFERENT atoms → breakpoints are
    representable) or the monster id for a monster (its statblock is its level).
  • ``g[k]`` — a body-count term (action economy) beyond the linear sum; g[0]=g[1]=0
    anchor, g[k] free for k≥2.

Both r and g are fit from observed (teamA, teamB, outcome) triples — offline by
replaying scripted games (validation) or ONLINE, piggybacking on the training loop
(each rollout episode is one such triple). Only DIFFERENCES of T matter, so the
overall additive gauge is free (predictions are gauge-invariant).
"""
from __future__ import annotations
import json
import math
from collections import Counter
from pathlib import Path


def atom_key(arch: str, level: int, is_monster: bool) -> str:
    """Canonical atom id. Monsters key on id alone (statblock == level); classes
    key on (class, level) so a feature breakpoint at some level is its own atom."""
    return f"mon:{arch}" if is_monster else f"{arch}@L{int(level)}"


def _sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


class StrengthModel:
    """Learned team-strength currency. Keys: atom_key(...) strings."""

    def __init__(self, k_max: int = 8):
        self.k_max = k_max
        self.r: dict[str, float] = {}
        # g[0]=g[1]=0 anchored; g[k] learnable for k>=2.
        self.g: list[float] = [0.0] * (k_max + 1)
        # Agent-seat skill offset: P(agent wins) = σ(T(A)−T(B)+seat_bias). Zero
        # (inert) unless training turns it on. Needed because side A is a LEARNING
        # policy and side B scripts/snapshots — the per-atom terms are seat-
        # symmetric and can't express "the agent seat is currently weaker by s".
        # Training it makes the matchmaker a self-balancing CURRICULUM: a weak
        # policy is handed weaker opponents; as it improves, seat_bias rises and
        # opponents scale up — every fight stays ~50/50 for the current policy.
        self.seat_bias: float = 0.0

    # ── strength / prediction ────────────────────────────────────────────────
    def team_strength(self, atoms: list[str]) -> float:
        k = len(atoms)
        gk = self.g[k] if k <= self.k_max else self.g[self.k_max]
        return sum(self.r.get(a, 0.0) for a in atoms) + gk

    def win_prob(self, a_atoms: list[str], b_atoms: list[str],
                 seat_bias: float | None = None) -> float:
        """P(team A beats team B). The seat term (0 unless training set it) offsets
        for A being the learning agent seat vs a scripted/snapshot B. Pass an
        explicit ``seat_bias`` to override — e.g. 0 for a SELF-PLAY opponent whose
        skill tracks the agent (no persistent skill gap), vs ``self.seat_bias``
        for a fixed-skill SCRIPT opponent (the gap the agent outgrows)."""
        sb = self.seat_bias if seat_bias is None else seat_bias
        return _sigmoid(self.team_strength(a_atoms) - self.team_strength(b_atoms)
                        + sb)

    # ── online / offline fitting (Elo-style logistic SGD) ────────────────────
    def update(self, a_atoms: list[str], b_atoms: list[str], y: float,
               lr: float = 0.05, update_seat_bias: bool = False,
               lr_seat: float = 0.15) -> float:
        """One gradient step for outcome ``y`` (1 A-win / 0.5 draw / 0 A-loss).
        Returns the PRE-update predicted P(A wins) (for calibration logging).

        d/dθ of log-loss for logistic: error e = (y − p) flows +1 onto every A
        feature, −1 onto every B feature. Features = per-atom counts and the
        body-count g[k] indicators. Shared atoms across the two teams cancel,
        exactly as they should (a mirror match teaches nothing)."""
        p = self.win_prob(a_atoms, b_atoms)
        e = (y - p) * lr
        # net per-atom occurrence (A − B); shared atoms cancel to 0.
        net: Counter = Counter(a_atoms)
        net.subtract(Counter(b_atoms))
        for atom, c in net.items():
            if c:
                self.r[atom] = self.r.get(atom, 0.0) + e * c
        ka, kb = len(a_atoms), len(b_atoms)
        if ka != kb:
            if 2 <= ka <= self.k_max:
                self.g[ka] += e
            if 2 <= kb <= self.k_max:
                self.g[kb] -= e
        if update_seat_bias:      # A is always the agent seat in training
            self.seat_bias += (y - p) * lr_seat
        return p

    # ── cold-start prior ─────────────────────────────────────────────────────
    def bootstrap(self, class_ids, levels, monster_equiv: dict[str, float],
                  party_monster_equiv: dict[str, float] | None = None,
                  logit_per_level: float = 0.6, body_bonus: float = 0.4,
                  party_size: int = 3) -> None:
        """Rough prior so early matchmaking spreads games over all regions; the
        online fit then corrects it (Phase 1 proved the prior itself is crude).
        r ≈ (equiv-)level × logit_per_level; g[k] ≈ body_bonus × (k−1).

        ``monster_equiv`` are 1v1-equiv levels: a monster there is worth ONE class
        body at that level, so r = eq × logit_per_level.

        ``party_monster_equiv`` are 1vN party-content monsters (1v1-equiv is inf
        but their FAIR fight is a ``party_size``-body party at the measured equiv
        level). Their SOLO rating must therefore be that whole party's strength —
        ``party_size`` bodies plus the body-count term — NOT one class at eq, or
        the additive currency underrates them ~party_size× and every early boss-
        vs-party proposal is wrong."""
        for c in class_ids:
            for lv in levels:
                self.r.setdefault(atom_key(c, lv, False), lv * logit_per_level)
        for mid, eq in monster_equiv.items():
            if math.isfinite(eq):
                self.r.setdefault(atom_key(mid, 0, True), eq * logit_per_level)
        for mid, eq in (party_monster_equiv or {}).items():
            if math.isfinite(eq):
                self.r.setdefault(
                    atom_key(mid, 0, True),
                    party_size * (eq * logit_per_level)
                    + body_bonus * (party_size - 1))
        for k in range(2, self.k_max + 1):
            if self.g[k] == 0.0:
                self.g[k] = body_bonus * (k - 1)

    # ── persistence ──────────────────────────────────────────────────────────
    def to_dict(self) -> dict:
        return {"k_max": self.k_max, "r": self.r, "g": self.g,
                "seat_bias": self.seat_bias}

    @classmethod
    def from_dict(cls, d: dict) -> "StrengthModel":
        m = cls(k_max=int(d["k_max"]))
        m.r = dict(d["r"])
        m.g = list(d["g"])
        m.seat_bias = float(d.get("seat_bias", 0.0))
        return m

    def save(self, path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def load(cls, path) -> "StrengthModel":
        return cls.from_dict(json.loads(Path(path).read_text()))


# ── matchmaker: turn the currency into balanced-but-diverse encounters ─────────
# A "spec" is (arch, level, is_monster) — the tuple env.reset needs per entity
# (arch -> agent_archs, level -> agent_levels). atom_key(*spec) is its rating id.

def sample_imbalance_target(rng, fair_band: float = 0.08,
                            tail_frac: float = 0.3) -> float:
    """Target P(side-A wins) for one encounter. Mostly fair (~0.5 ± fair_band);
    a ``tail_frac`` minority is deliberately lopsided so the policy also trains
    'win when ahead' and 'don't throw when behind'. Symmetric about 0.5."""
    if rng.random() < tail_frac:
        return rng.uniform(0.15, 0.85)
    return rng.uniform(0.5 - fair_band, 0.5 + fair_band)


def gap_for_target(target: float, fair_gap: int = 1, tail_gap: int = 2,
                   thresh: float = 0.15) -> int:
    """Max |na−nb| allowed for an encounter, chosen from its balance target.

    Fair fights (|target−0.5| ≤ thresh) stay tight — the additive currency is
    accurate near parity, and the Nv1-swarm undervaluation (see
    project_matchup_balance) only bites at wide body gaps. Deliberately lopsided
    tail fights may widen to ``tail_gap`` so 1-vs-many 'boss vs party' encounters
    (a single party-content monster vs a 3-body team) can appear — there the
    currency is rough, but the fight is imbalanced BY DESIGN and the online fit
    plus the party-equiv bootstrap carry it. Boundary is exclusive: |Δ| must
    strictly EXCEED thresh to widen."""
    return tail_gap if abs(target - 0.5) > thresh else fair_gap


def propose_matchup(model: StrengthModel, rng, draw_identity, *,
                    sizes=(1, 2, 3), n_candidates: int = 32, target=None,
                    max_size_gap: int | None = None, seat_bias: float | None = None):
    """Build a diverse, ~target-balanced encounter using ONLY the currency (no
    games). Side A is drawn free; side B is the closest-to-target of
    ``n_candidates`` random draws. Returns (A_specs, B_specs, p_model, target).

    ``draw_identity(rng) -> (arch, level, is_monster)`` samples the identity
    space (classes×levels, monsters, later grafts) — injected so the matchmaker
    stays decoupled from the registries. ``max_size_gap`` caps |na−nb|: the
    currency undervalues extreme Nv1 swarms (additive-model limit, see
    project_matchup_balance), so bounding the body-count gap keeps proposals in
    the region where the model is accurate; the online fit owns wider gaps."""
    if target is None:
        target = sample_imbalance_target(rng)
    na = rng.choice(sizes)
    a_specs = [draw_identity(rng) for _ in range(na)]
    a_atoms = [atom_key(*s) for s in a_specs]

    b_sizes = [k for k in sizes
               if max_size_gap is None or abs(k - na) <= max_size_gap]
    best = None
    for _ in range(n_candidates):
        nb = rng.choice(b_sizes)
        b_specs = [draw_identity(rng) for _ in range(nb)]
        p = model.win_prob(a_atoms, [atom_key(*s) for s in b_specs],
                           seat_bias=seat_bias)
        err = abs(p - target)
        if best is None or err < best[0]:
            best = (err, b_specs, p)
    return a_specs, best[1], best[2], target
