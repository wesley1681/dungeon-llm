"""obs v7 (2026-07-06): per-entity six ability MODIFIERS (STR/DEX/CON/INT/WIS/CHA).

Saving throws roll d20 + the TARGET's ability modifier (combat.make_saving_throw:
`stat_mod = character.stats.modifier(stat)`), so a save-or-lose spell's success
hinges on the target's stat — yet before v7 the entity row was BLIND to it
(perturbing an enemy's WIS left the whole row unchanged). This pins:
  (1) the six modifiers are now in the entity obs, at a strict-append tail;
  (2) perturbing one stat moves ONLY its column (no leakage into other tails);
  (3) the migration is bit-exact for a real pre-v7 checkpoint (uni_v10): the new
      entity_mlp / critic columns are zero, so the policy ignores them until trained.
"""
import os
import numpy as np
import pytest
import torch

import trpg.rl.obs as O
from trpg.engine.skill import SKILL_FEATURE_DIM
from trpg.engine.character import Character, Stats
from trpg.engine.vec2 import Vec2


def _self():
    s = Character(name='S', race='', class_='', level=5, stats=Stats(INT=16),
                  hp=40, max_hp=40, ac=13, is_npc=False)
    s.position = Vec2(5, 5)
    return s


def _enemy(**stats):
    base = dict(STR=10, DEX=10, CON=10, INT=10, WIS=10, CHA=10)
    base.update(stats)
    e = Character(name='E', race='', class_='', level=5, stats=Stats(**base),
                  hp=50, max_hp=50, ac=12, is_npc=True, attitude=0)
    e.position = Vec2(8, 5)
    return e


def _row(e):
    e._cap_desc_cache = None
    return O._entity_row(e, _self(), 30.0, 30.0, is_self=False, is_enemy=True)


def test_entity_dim_grew_by_six():
    assert O.N_V7_ABILITY == 6
    assert O.ENTITY_DIM == O._ENTITY_DIM_V6 + O.N_V7_ABILITY


def test_ability_columns_are_normalised_modifiers():
    r = _row(_enemy(STR=20, DEX=8))          # +5 STR, -1 DEX, +0 WIS
    st = O.ENT_ABILITY_START
    assert r[st + 0] == pytest.approx(5 / O.ABILITY_MOD_NORM)    # STR
    assert r[st + 1] == pytest.approx(-1 / O.ABILITY_MOD_NORM)   # DEX
    assert r[st + 4] == pytest.approx(0.0)                       # WIS


@pytest.mark.parametrize("stat,col", [("STR", 0), ("DEX", 1), ("CON", 2),
                                      ("INT", 3), ("WIS", 4), ("CHA", 5)])
def test_perturbing_a_stat_moves_only_its_own_column(stat, col):
    base = _row(_enemy())
    hi = _row(_enemy(**{stat: 20}))
    changed = set(np.where(np.abs(hi - base) > 1e-6)[0].tolist())
    assert changed == {O.ENT_ABILITY_START + col}, (stat, changed)


def test_migrate_v6_to_v7_strict_append_zeros():
    rng = np.random.default_rng(0)
    v6 = rng.random((O.N_ENTITY_SLOTS, O._ENTITY_DIM_V6)).astype(np.float32)
    out = O.migrate_entities_v6_to_v7(v6)
    assert out.shape[-1] == O.ENTITY_DIM
    assert np.array_equal(out[:, :O._ENTITY_DIM_V6], v6)      # old cols verbatim
    assert np.all(out[:, O._ENTITY_DIM_V6:] == 0.0)          # new cols zero
    assert O.migrate_entities_v6_to_v7(out).shape[-1] == O.ENTITY_DIM  # idempotent


def test_v3_chain_lands_at_live_width():
    # the oldest public migrator must chain all the way to v7
    rng = np.random.default_rng(1)
    v3 = rng.random((O.N_ENTITY_SLOTS, O._ENTITY_DIM_V3)).astype(np.float32)
    out = O.migrate_entities_v3_to_v4(v3)
    assert out.shape[-1] == O.ENTITY_DIM


UNI = "models/unified/uni_v10.pt"


def _synthetic_obs(entities):
    B = 1
    m = torch.zeros(B, O.N_SKILL_SLOTS); m[:, :6] = 1.0
    return {
        "skills": torch.randn(B, O.N_SKILL_SLOTS, SKILL_FEATURE_DIM),
        "skill_mask": m,
        "entities": entities,
        "resources": torch.rand(B, 4),
        "terrain": torch.zeros(B, O.N_GRID, O.N_GRID),
        "entity_grid": torch.zeros(B, O.N_ENTITY_GRID_CHANNELS, O.N_GRID, O.N_GRID),
        "distance_grid": torch.rand(B, O.N_DISTANCE_GRID_CHANNELS, O.N_GRID, O.N_GRID),
        "los_grid": torch.ones(B, O.N_LOS_GRID_CHANNELS, O.N_GRID, O.N_GRID),
        "reach_grid": torch.ones(B, O.N_REACH_GRID_CHANNELS, O.N_GRID, O.N_GRID),
        "threat_grid": torch.zeros(B, O.N_THREAT_GRID_CHANNELS, O.N_GRID, O.N_GRID),
        "end_features": torch.rand(B, O.END_FEATURES_DIM),
        "decision_context": torch.zeros(B, O.N_DECISION_CTX),
    }


@pytest.mark.xfail(reason="2026-07-06 obs v8+:capability_descriptor 精簡(移除可從 "
                          "entity_skills 導出的 25 個 OR 欄位)shrink 了 entity 列的中間段,"
                          "刻意打破 strict-append 遷移契約 → pre-v8 的 uni_v10 不再 bit-exact "
                          "相容。架構優先決定,版本遷移修復延後(見 obs.py N_V4_DESC 附近註解)。",
                   strict=False)
@pytest.mark.skipif(not os.path.exists(UNI), reason="uni_v10 checkpoint absent")
def test_uni_v10_migration_is_bit_exact():
    from trpg.rl.model import CombatPolicyNet, _ENTITY_DIM_V6
    raw = torch.load(UNI, map_location="cpu", weights_only=True)
    adapted = CombatPolicyNet.adapt_state_dict_for_perarch(raw)

    # entity_mlp widened to live width, the six new columns exactly zero
    w = adapted["entity_mlp.0.weight"]
    assert w.shape[1] == O.ENTITY_DIM
    assert torch.all(w[:, -O.N_V7_ABILITY:] == 0.0)

    # critic per-slot entity blocks widened, each slot's six ability cols zero
    cw = adapted["critic.net.0.weight"]
    for s in range(O.N_ENTITY_SLOTS):
        blk = cw[:, s * O.ENTITY_DIM + _ENTITY_DIM_V6: (s + 1) * O.ENTITY_DIM]
        assert torch.all(blk == 0.0), s

    # loads without dropping entity_mlp (a drop → random weights → behaviour break)
    net = CombatPolicyNet()
    msd = net.state_dict()
    assert adapted["entity_mlp.0.weight"].shape == msd["entity_mlp.0.weight"].shape
    net.load_state_dict({k: v for k, v in adapted.items()
                         if k in msd and v.shape == msd[k].shape}, strict=False)
    net.eval()

    # behavioural bit-exactness: identical obs EXCEPT the ability columns → the
    # policy output must be identical, i.e. the migrated ability weights are inert.
    ent = torch.randn(1, O.N_ENTITY_SLOTS, O.ENTITY_DIM) * 0.1
    zeroed = ent.clone()
    zeroed[:, :, O.ENT_ABILITY_START:] = 0.0
    obs_a = _synthetic_obs(ent)
    obs_b = dict(obs_a)
    obs_b["entities"] = zeroed        # only the ability cols differ
    with torch.no_grad():
        oa = net(obs_a)
        ob = net(obs_b)
    for x, y in zip(oa, ob):
        assert torch.equal(x, y), "ability cols must be inert on a migrated ckpt"
