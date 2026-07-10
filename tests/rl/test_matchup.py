import math
import random

from trpg.rl.matchup import (StrengthModel, atom_key, propose_matchup,
                             sample_imbalance_target, gap_for_target)


def test_atom_key_class_vs_monster():
    # class atoms are per-(class,level) so a feature breakpoint is its own atom
    assert atom_key("champion", 4, False) != atom_key("champion", 5, False)
    # monster atoms key on id alone (statblock == level)
    assert atom_key("ogre", 3, True) == atom_key("ogre", 7, True) == "mon:ogre"


def test_mirror_is_even():
    m = StrengthModel()
    a = [atom_key("champion", 5, False)]
    assert m.win_prob(a, a) == 0.5


def test_team_strength_is_sum_plus_body_term():
    m = StrengthModel()
    m.r[atom_key("champion", 5, False)] = 1.0
    m.g[2] = 0.5
    two = [atom_key("champion", 5, False)] * 2
    assert m.team_strength(two) == 2.0 + 0.5


def test_update_moves_prediction_toward_outcome():
    m = StrengthModel()
    a = [atom_key("champion", 6, False)]
    b = [atom_key("champion", 4, False)]
    p0 = m.win_prob(a, b)
    for _ in range(100):
        m.update(a, b, 1.0, lr=0.05)
    assert m.win_prob(a, b) > p0 + 0.3       # climbs toward an A-win


def test_shared_atoms_cancel_in_update():
    # a true mirror teaches nothing: identical teams -> no rating change
    m = StrengthModel()
    a = [atom_key("champion", 5, False)]
    m.update(a, list(a), 1.0, lr=0.1)
    assert m.r.get(atom_key("champion", 5, False), 0.0) == 0.0
    assert all(x == 0.0 for x in m.g)


def test_body_count_term_learns_action_economy():
    # 2-body repeatedly beating 1-body raises g[2] (net atom cancels the shared id)
    m = StrengthModel()
    x = [atom_key("champion", 3, False)] * 2
    y = [atom_key("champion", 3, False)]
    for _ in range(100):
        m.update(x, y, 1.0, lr=0.05)
    assert m.g[2] > 0.5


def test_bootstrap_monotone_and_skips_inf_monsters():
    m = StrengthModel()
    m.bootstrap(["champion"], [3, 5, 8],
                {"goblin": 0.5, "ogre": 4.2, "tarrasque": math.inf})
    r = m.r
    assert (r[atom_key("champion", 3, False)]
            < r[atom_key("champion", 5, False)]
            < r[atom_key("champion", 8, False)])
    assert atom_key("goblin", 0, True) in r
    assert atom_key("ogre", 0, True) in r
    assert atom_key("tarrasque", 0, True) not in r      # inf -> untrainable, skipped
    # body term monotone increasing
    assert m.g[2] < m.g[3] < m.g[4]


def test_bootstrap_party_content_monster_rated_as_full_party():
    # A party-content monster (1v1-equiv inf, but a MEASURED 3-party equiv) must
    # be seeded as a whole party's strength, NOT one class at that equiv level —
    # else the additive currency badly underrates it and every early matchmaking
    # proposal involving it is wrong.
    m = StrengthModel(k_max=4)
    m.bootstrap(["champion"], [3, 8], {"goblin": 0.5},
                party_monster_equiv={"hill_giant": 3.2},
                logit_per_level=0.6, body_bonus=0.4, party_size=3)
    hg = m.r[atom_key("hill_giant", 0, True)]
    naive = 3.2 * 0.6                      # the WRONG single-body seeding
    assert hg > 3 * naive - 1e-9           # ~a 3-body party, not one body
    assert abs(hg - (3 * naive + 0.4 * 2)) < 1e-9   # party sum + g[3] prior
    # a party boss out-rates a single top-of-band class atom
    assert hg > m.r[atom_key("champion", 8, False)]
    # inf party-equiv (true apex) is skipped just like inf 1v1 monsters
    m.bootstrap(["champion"], [3], {}, party_monster_equiv={"tarrasque": math.inf})
    assert atom_key("tarrasque", 0, True) not in m.r


def test_gap_for_target_widens_only_on_lopsided():
    # near-fair fights stay tight (currency accurate near parity); deliberately
    # lopsided tail fights may widen so 1-vs-many boss-vs-party can appear.
    assert gap_for_target(0.50, fair_gap=1, tail_gap=2, thresh=0.15) == 1
    assert gap_for_target(0.55, fair_gap=1, tail_gap=2, thresh=0.15) == 1
    assert gap_for_target(0.20, fair_gap=1, tail_gap=2, thresh=0.15) == 2
    assert gap_for_target(0.80, fair_gap=1, tail_gap=2, thresh=0.15) == 2
    # threshold behaves either side of the band (avoid the exact 0.15 boundary,
    # which is not float-representable via 0.5±0.15)
    assert gap_for_target(0.64, thresh=0.15) == 1            # |Δ|=0.14 < 0.15
    assert gap_for_target(0.66, thresh=0.15) == 2            # |Δ|=0.16 > 0.15


def _champ_draw(rng):
    return ("champion", rng.randint(2, 8), False)


def _fitted_champ_model():
    # a monotone champion rating with an L5 breakpoint + body term (like the fit)
    m = StrengthModel(k_max=4)
    for lv, r in zip(range(2, 9), [0.0, 0.5, 0.65, 2.5, 2.9, 3.3, 3.3]):
        m.r[atom_key("champion", lv, False)] = r
    m.g = [0.0, 0.0, 1.2, 2.6, 3.7]
    return m


def test_imbalance_target_mostly_fair_and_symmetric():
    rng = random.Random(0)
    ts = [sample_imbalance_target(rng, fair_band=0.08, tail_frac=0.3)
          for _ in range(4000)]
    fair = [t for t in ts if abs(t - 0.5) <= 0.08]
    assert 0.6 < len(fair) / len(ts) < 0.8            # ~70% fair
    assert abs(sum(ts) / len(ts) - 0.5) < 0.03        # symmetric about 0.5


def test_propose_matchup_hits_target_and_respects_sizes():
    m = _fitted_champ_model()
    rng = random.Random(1)
    errs = []
    for _ in range(200):
        tgt = sample_imbalance_target(rng)
        a, b, p, t = propose_matchup(m, rng, _champ_draw, sizes=(1, 2, 3),
                                     n_candidates=48, target=tgt)
        assert 1 <= len(a) <= 3 and 1 <= len(b) <= 3
        assert t == tgt
        errs.append(abs(p - tgt))
    # with 48 candidates the currency should get close to the target on average
    assert sum(errs) / len(errs) < 0.08


def test_propose_matchup_respects_max_size_gap():
    m = _fitted_champ_model()
    rng = random.Random(2)
    for _ in range(200):
        a, b, p, t = propose_matchup(m, rng, _champ_draw, sizes=(1, 2, 3),
                                     max_size_gap=1)
        assert abs(len(a) - len(b)) <= 1


def test_seat_bias_inert_by_default():
    # offline / untrained: seat_bias stays 0, win_prob unchanged, mirror even
    m = StrengthModel()
    a = [atom_key("champion", 5, False)]
    for _ in range(50):
        m.update(a, [atom_key("champion", 3, False)], 0.0)   # no update_seat_bias
    assert m.seat_bias == 0.0


def test_seat_bias_absorbs_agent_seat_disadvantage():
    # a learning agent that always loses at 'fair' composition drives seat_bias
    # negative, so the matchmaker will compensate with weaker opponents.
    m = StrengthModel()
    a = [atom_key("champion", 5, False)]
    b = [atom_key("champion", 5, False)]           # identical comp: T(A)=T(B)
    for _ in range(60):
        m.update(a, b, 0.0, update_seat_bias=True)  # agent always loses
    assert m.seat_bias < -1.0
    # identical-comp fight is now predicted as an agent loss
    assert m.win_prob(a, b) < 0.35
    m.save  # ensure attr present for persistence
    d = m.to_dict()
    assert d["seat_bias"] == m.seat_bias
    assert StrengthModel.from_dict(d).seat_bias == m.seat_bias


def test_save_load_roundtrip(tmp_path):
    m = StrengthModel(k_max=5)
    m.r[atom_key("champion", 5, False)] = 1.23
    m.g[3] = 0.7
    m.seat_bias = -2.0
    p = tmp_path / "sm.json"
    m.save(p)
    m2 = StrengthModel.load(p)
    assert m2.k_max == 5
    assert m2.r[atom_key("champion", 5, False)] == 1.23
    assert m2.g[3] == 0.7
    assert m2.seat_bias == -2.0
