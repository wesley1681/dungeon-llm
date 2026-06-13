"""Behavioural tests for abilities/mechanics that were encoded-but-not-executable
(engine_ready=False) and are being wired up so every class kit is complete
before the integration training wave.

Each block pins (1) the action executes and deals the right effect, (2) the
resource cost is paid, (3) the descriptor's EV matches what the engine actually
deals (no descriptor/engine divergence), and where relevant (4) a save / miss
path behaves correctly.
"""
from unittest.mock import patch

import pytest

from trpg.engine.character import Character, Stats, CombatState
from trpg.engine.vec2 import Vec2
from trpg.engine.world_state import WorldState
from trpg.engine.items import WEAPON_DEFS
from trpg.engine.combat import execute_action
from trpg.engine.skill import available_skills


def _cleric(level=5, **kw):
    c = Character(name="C", race="", class_="牧師", level=level,
                  stats=Stats(WIS=16), hp=40, max_hp=40, ac=16, is_npc=False,
                  spell_slots={1: 3, 2: 2}, **kw)
    c.spellcasting_ability = "WIS"
    c.position = Vec2(5, 5)
    return c


def _dummy(hp=100, ac=10, **kw):
    e = Character(name="E", race="", class_="", level=1, stats=Stats(),
                  hp=hp, max_hp=hp, ac=ac, is_npc=True, attitude=0, **kw)
    e.position = Vec2(6, 5)
    return e


def _world(caster, *others):
    chars = {"c": caster}
    for i, o in enumerate(others):
        chars[f"e{i}"] = o
    ws = WorldState(characters=chars, scene="", dungeon_map=None)
    ws.party_ids = ["c"]
    ws.combat = CombatState(initiative_order=list(chars.keys()))
    return ws


# ── guiding_bolt (cleric life / war): SPELL_ATTACK 4d6 radiant + on-hit mark ──

def test_guiding_bolt_hits_deals_flat_4d6_and_consumes_slot():
    cleric = _cleric()
    enemy = _dummy(hp=100, ac=10)
    ws = _world(cleric, enemy)
    # roll("4d6") is mocked to one call → 6. The assertion that matters is
    # that NO spellcasting mod (+WIS=3) is added on top (flat-damage spell).
    with patch("trpg.engine.combat.roll_d20", return_value=18), \
         patch("trpg.engine.combat.roll", return_value=6):
        res = execute_action({
            "type": "SPELL_ATTACK", "caster": "c", "target": "e0",
            "spell_name": "引導光彈", "damage_dice": "4d6", "damage_type": "光耀",
            "range_m": 36.0, "slot_level": 1, "add_spell_mod": False,
            "on_hit_status": "distracted", "consumes": ["action"],
            "skill_id": "guiding_bolt_life",
        }, ws)
    assert res["hit"] is True
    assert res["damage"] == 6            # rolled 4d6, no WIS added
    assert cleric.spell_slots[1] == 2    # one L1 slot spent
    assert enemy.has_status("distracted")  # next attacker gets advantage


def test_guiding_bolt_miss_spends_slot_no_mark():
    cleric = _cleric()
    enemy = _dummy(hp=100, ac=25)
    ws = _world(cleric, enemy)
    with patch("trpg.engine.combat.roll_d20", return_value=3):
        res = execute_action({
            "type": "SPELL_ATTACK", "caster": "c", "target": "e0",
            "spell_name": "引導光彈", "damage_dice": "4d6", "damage_type": "光耀",
            "range_m": 36.0, "slot_level": 1, "add_spell_mod": False,
            "on_hit_status": "distracted", "consumes": ["action"],
            "skill_id": "guiding_bolt_life",
        }, ws)
    assert res["hit"] is False
    assert res["damage"] == 0
    assert cleric.spell_slots[1] == 2          # slot spent even on a miss (RAW)
    assert not enemy.has_status("distracted")  # no mark without a hit


def test_guiding_bolt_no_slot_is_rejected():
    cleric = _cleric()
    cleric.spell_slots[1] = 0
    enemy = _dummy()
    ws = _world(cleric, enemy)
    res = execute_action({
        "type": "SPELL_ATTACK", "caster": "c", "target": "e0",
        "spell_name": "引導光彈", "damage_dice": "4d6", "damage_type": "光耀",
        "range_m": 36.0, "slot_level": 1, "add_spell_mod": False,
        "skill_id": "guiding_bolt_life",
    }, ws)
    assert res["type"] == "ERROR"


def test_spiritual_weapon_attack_still_adds_wis_mod():
    """Regression: the add_spell_mod flag defaults True so the existing
    spiritual-weapon spell attack keeps adding WIS (1d8+WIS) bit-for-bit."""
    cleric = _cleric()
    enemy = _dummy(hp=100, ac=10)
    ws = _world(cleric, enemy)
    with patch("trpg.engine.combat.roll_d20", return_value=18), \
         patch("trpg.engine.combat.roll", return_value=5):
        res = execute_action({
            "type": "SPELL_ATTACK", "caster": "c", "target": "e0",
            "spell_name": "精神武器", "damage_dice": "1d8", "damage_type": "力場",
            "range_m": 18.0, "consumes": ["bonus_action"],
            "skill_id": "spiritual_weapon_attack_life",
        }, ws)
    assert res["hit"] is True
    assert res["damage"] == 5 + 3        # 1d8(5) + WIS(+3) = 8


def test_guiding_bolt_available_when_cleric_has_slot():
    cleric = _cleric()
    cleric.known_abilities = ["guiding_bolt_life"]
    enemy = _dummy()
    ws = _world(cleric, enemy)
    ids = [s.skill_id for s in available_skills(cleric, ws)]
    assert "guiding_bolt_life" in ids
    cleric.spell_slots[1] = 0
    ids = [s.skill_id for s in available_skills(cleric, ws)]
    assert "guiding_bolt_life" not in ids   # gated by slot availability


# ── wrathful_smite (devotion paladin): weapon swing + 1d6 psychic + frighten ──

def _paladin(level=3, **kw):
    p = Character(name="P", race="", class_="聖騎士", level=level,
                  stats=Stats(STR=16, CHA=16), hp=30, max_hp=30, ac=18,
                  is_npc=False, weapons=[WEAPON_DEFS["長劍"]],
                  spell_slots={1: 2}, **kw)
    p.spellcasting_ability = "CHA"
    p.position = Vec2(5, 5)
    p.known_abilities = ["wrathful_smite_dev"]
    return p


def test_wrathful_smite_deals_weapon_plus_psychic_and_frightens():
    pal = _paladin()
    enemy = _dummy(hp=100, ac=10)
    ws = _world(pal, enemy)
    from trpg.engine.abilities import ABILITY_REGISTRY
    built = ABILITY_REGISTRY["wrathful_smite_dev"].build_action("c", "e0", None, char=pal)
    # WIS save fails (enemy rolls low) → frightened; weapon 1d8(6)+STR(3)=9,
    # rider 1d6(4) 精神 = 13 total. roll() calls: weapon dmg, rider dmg, save.
    with patch("trpg.engine.combat.roll_d20", return_value=18), \
         patch("trpg.engine.combat.roll", side_effect=[6, 4]), \
         patch("trpg.engine.combat.make_saving_throw", return_value=(False, 5)):
        res = execute_action(built, ws)
    assert res["hit"] is True
    assert res.get("rider_damage") == 4              # 1d6 精神 rider landed
    assert res["damage"] == 9 + 4                     # weapon(9) + psychic(4)
    assert res.get("rider_status_applied") == "frightened"
    assert enemy.has_status("frightened")
    assert pal.spell_slots[1] == 1                    # one L1 slot spent
    assert pal.concentrating_on == "憤怒打擊"          # concentration registered


def test_wrathful_smite_save_success_no_fear_but_damage_lands():
    pal = _paladin()
    enemy = _dummy(hp=100, ac=10)
    ws = _world(pal, enemy)
    from trpg.engine.abilities import ABILITY_REGISTRY
    built = ABILITY_REGISTRY["wrathful_smite_dev"].build_action("c", "e0", None, char=pal)
    with patch("trpg.engine.combat.roll_d20", return_value=18), \
         patch("trpg.engine.combat.roll", side_effect=[6, 4]), \
         patch("trpg.engine.combat.make_saving_throw", return_value=(True, 20)):
        res = execute_action(built, ws)
    assert res["damage"] == 9 + 4                     # psychic still lands
    assert not enemy.has_status("frightened")         # WIS save negates fear


def test_wrathful_smite_no_slot_rejected():
    pal = _paladin()
    pal.spell_slots[1] = 0
    enemy = _dummy()
    ws = _world(pal, enemy)
    from trpg.engine.abilities import ABILITY_REGISTRY
    built = ABILITY_REGISTRY["wrathful_smite_dev"].build_action("c", "e0", None, char=pal)
    with patch("trpg.engine.combat.roll_d20", return_value=18):
        res = execute_action(built, ws)
    assert res["type"] == "ERROR"


# ── scorching_ray (evocation wizard): 3 independent attack rolls, 2d6 火 each ──

def _wizard(level=5, **kw):
    w = Character(name="W", race="", class_="法師", level=level,
                  stats=Stats(INT=16), hp=30, max_hp=30, ac=13, is_npc=False,
                  spell_slots={1: 3, 2: 3}, **kw)
    w.spellcasting_ability = "INT"
    w.position = Vec2(5, 5)
    w.known_abilities = ["scorching_ray_ev"]
    return w


def _build_scorch(wiz, target_str):
    from trpg.engine.abilities import ABILITY_REGISTRY
    return ABILITY_REGISTRY["scorching_ray_ev"].build_action(
        "c", target_str, None, char=wiz)


def test_scorching_ray_three_rays_one_target_flat_2d6():
    wiz = _wizard()
    enemy = _dummy(hp=100, ac=5)            # low AC → all rays hit
    ws = _world(wiz, enemy)
    built = _build_scorch(wiz, "e0")
    assert len(built["ray_targets"]) == 3   # 1 target → 3 rays pile on it
    # 3 rays, each 2d6 mocked to one roll() = 7, no INT mod added.
    with patch("trpg.engine.combat.roll_d20", return_value=15), \
         patch("trpg.engine.combat.roll", return_value=7):
        res = execute_action(built, ws)
    assert res["type"] == "MULTI_SPELL_ATTACK"
    assert len(res["rays"]) == 3
    assert all(r["hit"] for r in res["rays"])
    assert res["total_damage"] == 21        # 3 × 7, flat (no INT)
    assert wiz.spell_slots[2] == 2          # one L2 slot spent


def test_scorching_ray_splits_across_targets_independent_rolls():
    wiz = _wizard()
    e0 = _dummy(hp=100, ac=5)
    e1 = _dummy(hp=100, ac=99)              # unhittable
    e1.position = Vec2(7, 5)
    ws = _world(wiz, e0, e1)
    built = _build_scorch(wiz, "e0,e1")
    # rays cycle [e0, e1, e0]. d20=10: e0 hits (AC5), e1 misses (AC99).
    with patch("trpg.engine.combat.roll_d20", return_value=10), \
         patch("trpg.engine.combat.roll", return_value=7):
        res = execute_action(built, ws)
    hits = [r for r in res["rays"] if r["hit"]]
    assert len(hits) == 2                   # both e0 rays hit, e1 ray misses
    assert res["total_damage"] == 14


def test_scorching_ray_no_l2_slot_rejected():
    wiz = _wizard()
    wiz.spell_slots[2] = 0
    enemy = _dummy(ac=5)
    ws = _world(wiz, enemy)
    built = _build_scorch(wiz, "e0")
    res = execute_action(built, ws)
    assert res["type"] == "ERROR"


def test_scorching_ray_available_gated_by_l2_slot():
    wiz = _wizard()
    enemy = _dummy()
    ws = _world(wiz, enemy)
    ids = [s.skill_id for s in available_skills(wiz, ws)]
    assert "scorching_ray_ev" in ids
    wiz.spell_slots[2] = 0
    ids = [s.skill_id for s in available_skills(wiz, ws)]
    assert "scorching_ray_ev" not in ids


# ── multi-target heal: mass_cure_wounds (per-target dice) + preserve_life (pool) ──

def _ally(hp, max_hp, **kw):
    a = Character(name="A", race="", class_="戰士", level=3, stats=Stats(),
                  hp=hp, max_hp=max_hp, ac=14, is_npc=False, **kw)
    a.position = Vec2(5, 6)
    return a


def test_mass_cure_wounds_heals_each_target_and_spends_l5_slot():
    cleric = _cleric(level=9)
    cleric.spell_slots = {5: 1}
    cleric.known_abilities = ["mass_cure_wounds_life"]
    a1 = _ally(hp=5, max_hp=40)
    a2 = _ally(hp=10, max_hp=40)
    a2.position = Vec2(6, 6)
    ws = _world(cleric, a1, a2)
    from trpg.engine.abilities import ABILITY_REGISTRY
    built = ABILITY_REGISTRY["mass_cure_wounds_life"].build_action(
        "c", "e0,e1", None, char=cleric)
    assert built["dice"] == "3d8+3"         # 3d8 + WIS(+3) folded into the dice
    # roll("3d8+3") mocked to one call → 13 healed per target.
    with patch("trpg.engine.combat.roll", return_value=13):
        res = execute_action(built, ws)
    assert res["type"] == "MULTI_HEAL"
    assert len(res["targets"]) == 2
    assert a1.hp == 5 + 13
    assert a2.hp == 10 + 13
    assert cleric.spell_slots[5] == 0


def test_preserve_life_pool_distributes_most_wounded_first_capped_half():
    cleric = _cleric(level=3)            # pool = 5 × 3 = 15
    a1 = _ally(hp=2, max_hp=40)          # very wounded, half-cap = 20
    a2 = _ally(hp=30, max_hp=40)         # already above half (20) → no heal
    a2.position = Vec2(6, 6)
    ws = _world(cleric, a1, a2)
    from trpg.engine.abilities import ABILITY_REGISTRY
    built = ABILITY_REGISTRY["channel_divinity_preserve_life"].build_action(
        "c", "e0,e1", None, char=cleric)
    res = execute_action(built, ws)
    # All 15 goes to a1 (most wounded), capped at half max (20): 2 → 17.
    assert a1.hp == 17
    assert a2.hp == 30                    # already past half max → gets nothing
    assert res["total_healed"] == 15


def test_preserve_life_cannot_exceed_half_max_hp():
    cleric = _cleric(level=10)           # pool = 50, large
    a1 = _ally(hp=18, max_hp=40)         # half-cap = 20 → only 2 HP of room
    ws = _world(cleric, a1)
    from trpg.engine.abilities import ABILITY_REGISTRY
    built = ABILITY_REGISTRY["channel_divinity_preserve_life"].build_action(
        "c", "e0", None, char=cleric)
    res = execute_action(built, ws)
    assert a1.hp == 20                    # capped at half max, not full
    assert res["total_healed"] == 2


def test_mass_cure_revives_dying_ally():
    cleric = _cleric(level=9)
    cleric.spell_slots = {5: 1}
    cleric.known_abilities = ["mass_cure_wounds_life"]
    downed = _ally(hp=0, max_hp=40)
    downed.death_saves = {"successes": 1, "failures": 2}
    ws = _world(cleric, downed)
    from trpg.engine.abilities import ABILITY_REGISTRY
    built = ABILITY_REGISTRY["mass_cure_wounds_life"].build_action(
        "c", "e0", None, char=cleric)
    with patch("trpg.engine.combat.roll", return_value=13):
        res = execute_action(built, ws)
    assert downed.hp == 13
    assert res["targets"][0]["revived"] is True


# ── pushing_attack (battle master): weapon hit + STR save → shove 4.5m ────────

def _bm_fighter(level=3, **kw):
    f = Character(name="F", race="", class_="戰士", level=level,
                  stats=Stats(STR=16), hp=30, max_hp=30, ac=16, is_npc=False,
                  weapons=[WEAPON_DEFS["長劍"]], **kw)
    f.position = Vec2(5, 5)
    f.known_abilities = ["pushing_attack"]
    return f


def _world_bf(caster, *others):
    """World with a real battlefield so knockback can clamp to bounds."""
    from trpg.engine.vec2 import Battlefield
    ws = _world(caster, *others)
    ws.combat.battlefield = Battlefield(width=30, height=30)
    return ws


def test_pushing_attack_shoves_target_on_failed_save():
    bm = _bm_fighter()
    enemy = _dummy(hp=100, ac=5)
    enemy.position = Vec2(6, 5)          # 1m east of the fighter
    ws = _world_bf(bm, enemy)
    from trpg.engine.abilities import ABILITY_REGISTRY
    built = ABILITY_REGISTRY["pushing_attack"].build_action("c", "e0", None, char=bm)
    before_x = enemy.position.x
    with patch("trpg.engine.combat.roll_d20", return_value=18), \
         patch("trpg.engine.combat.roll", return_value=6), \
         patch("trpg.engine.combat.make_saving_throw", return_value=(False, 4)):
        res = execute_action(built, ws)
    assert res["hit"] is True
    assert "knockback" in res
    assert res["knockback"]["distance"] == 4.5
    assert enemy.position.x > before_x + 4.0   # shoved straight east (away)


def test_pushing_attack_no_shove_on_successful_save():
    bm = _bm_fighter()
    enemy = _dummy(hp=100, ac=5)
    enemy.position = Vec2(6, 5)
    ws = _world_bf(bm, enemy)
    from trpg.engine.abilities import ABILITY_REGISTRY
    built = ABILITY_REGISTRY["pushing_attack"].build_action("c", "e0", None, char=bm)
    with patch("trpg.engine.combat.roll_d20", return_value=18), \
         patch("trpg.engine.combat.roll", return_value=6), \
         patch("trpg.engine.combat.make_saving_throw", return_value=(True, 20)):
        res = execute_action(built, ws)
    assert res["push_save_success"] is True
    assert "knockback" not in res
    assert enemy.position.x == 6           # stayed put


def test_pushing_attack_clamps_at_battlefield_edge():
    bm = _bm_fighter()
    bm.position = Vec2(27, 5)
    enemy = _dummy(hp=100, ac=5)
    enemy.position = Vec2(28, 5)            # near the east wall (width=30)
    ws = _world_bf(bm, enemy)
    from trpg.engine.abilities import ABILITY_REGISTRY
    built = ABILITY_REGISTRY["pushing_attack"].build_action("c", "e0", None, char=bm)
    with patch("trpg.engine.combat.roll_d20", return_value=18), \
         patch("trpg.engine.combat.roll", return_value=6), \
         patch("trpg.engine.combat.make_saving_throw", return_value=(False, 4)):
        res = execute_action(built, ws)
    # Pushed east toward the wall but clamped inside bounds (< full 4.5m).
    assert enemy.position.x <= 30
    assert res.get("knockback", {}).get("distance", 4.5) < 4.5


def test_pushing_attack_use_pool_tracked():
    bm = _bm_fighter()
    enemy = _dummy(ac=5)
    enemy.position = Vec2(6, 5)
    ws = _world_bf(bm, enemy)
    from trpg.engine.abilities import ABILITY_REGISTRY
    built = ABILITY_REGISTRY["pushing_attack"].build_action("c", "e0", None, char=bm)
    with patch("trpg.engine.combat.roll_d20", return_value=18), \
         patch("trpg.engine.combat.roll", return_value=6), \
         patch("trpg.engine.combat.make_saving_throw", return_value=(False, 4)):
        execute_action(built, ws)
    # max_uses=4 → execute_action deducts one use (superiority die pool).
    assert bm.ability_uses.get("pushing_attack", 4) == 3


# ── bear_totem (passive): raging resists ALL damage except psychic ────────────

def _barbarian(level=3, totem=False, **kw):
    b = Character(name="B", race="", class_="野蠻人", level=level,
                  stats=Stats(STR=16), hp=40, max_hp=40, ac=14, is_npc=False,
                  weapons=[WEAPON_DEFS["長劍"]], **kw)
    b.position = Vec2(5, 5)
    b.known_abilities = ["rage"] + (["bear_totem"] if totem else [])
    return b


def _rage(barb, ws):
    from trpg.engine.abilities import ABILITY_REGISTRY
    return execute_action(
        ABILITY_REGISTRY["rage"].build_action("c", None, None, char=barb), ws)


def test_plain_rage_resists_only_physical():
    barb = _barbarian(totem=False)
    ws = _world(barb)
    _rage(barb, ws)
    from trpg.engine.combat import apply_damage
    # physical halved, fire NOT.
    barb.hp = 40
    apply_damage(barb, 10, dtype="斬擊", world_state=ws)
    assert barb.hp == 35                   # 10 → 5 (resisted)
    barb.hp = 40
    apply_damage(barb, 10, dtype="火", world_state=ws)
    assert barb.hp == 30                   # 10 full (no resist)


def test_bear_totem_rage_resists_all_except_psychic():
    barb = _barbarian(totem=True)
    ws = _world(barb)
    _rage(barb, ws)
    from trpg.engine.combat import apply_damage
    barb.hp = 40
    apply_damage(barb, 10, dtype="火", world_state=ws)
    assert barb.hp == 35                   # fire now halved (bear totem)
    barb.hp = 40
    apply_damage(barb, 10, dtype="斬擊", world_state=ws)
    assert barb.hp == 35                   # physical still halved
    barb.hp = 40
    apply_damage(barb, 10, dtype="精神", world_state=ws)
    assert barb.hp == 30                   # psychic is the lone exception


# ── precision_attack / guided_strike: post-roll boost flips a miss to a hit ───

def test_precision_attack_die_flips_miss_to_hit():
    bm = _bm_fighter()
    bm.known_abilities = ["precision_attack"]
    enemy = _dummy(hp=100, ac=20)
    enemy.position = Vec2(6, 5)
    ws = _world(bm, enemy)
    built = {"type": "ATTACK", "attacker": "c", "target": "e0",
             "weapon": "長劍", "consumes": ["action"], "skill_id": "weapon:長劍"}
    # d20=15 → roll_total = 15 + STR(3) + prof(2) = 20? STR16→+3, L3 prof+2 → 20.
    # Set AC so it just misses, then a +1d8 die flips it. Use d20=12 → 17 vs AC20
    # (gap 3); precision die rolls 6 → 23 ≥ 20 hit. weapon dmg roll then.
    with patch("trpg.engine.combat.roll_d20", return_value=12), \
         patch("trpg.engine.combat.roll", side_effect=[6, 5]):
        res = execute_action(built, ws)
    assert res.get("roll_boost", {}).get("skill_id") == "precision_attack"
    assert res["hit"] is True
    assert bm.ability_uses.get("precision_attack", 4) == 3   # one die spent


def test_guided_strike_plus10_flips_miss_and_spends_channel_divinity():
    cleric = _cleric(level=5)
    cleric.known_abilities = ["channel_divinity_guided_strike"]
    cleric.weapons = [WEAPON_DEFS["長劍"]]
    enemy = _dummy(hp=100, ac=25)
    enemy.position = Vec2(6, 5)
    ws = _world(cleric, enemy)
    built = {"type": "ATTACK", "attacker": "c", "target": "e0",
             "weapon": "長劍", "consumes": ["action"], "skill_id": "weapon:長劍"}
    # d20=17 → 17 + STR(0) + prof(3) = 20 vs AC25 (gap 5 ≤ 10) → +10 = 30 hit.
    with patch("trpg.engine.combat.roll_d20", return_value=17), \
         patch("trpg.engine.combat.roll", return_value=5):
        res = execute_action(built, ws)
    assert res.get("roll_boost", {}) == {"skill_id": "channel_divinity_guided_strike",
                                         "amount": 10}
    assert res["hit"] is True
    assert cleric.ability_uses.get("channel_divinity_guided_strike", 1) == 0


def test_post_roll_boost_not_spent_when_hit_already_lands():
    bm = _bm_fighter()
    bm.known_abilities = ["precision_attack"]
    enemy = _dummy(hp=100, ac=5)           # easy hit, no boost needed
    enemy.position = Vec2(6, 5)
    ws = _world(bm, enemy)
    built = {"type": "ATTACK", "attacker": "c", "target": "e0",
             "weapon": "長劍", "consumes": ["action"], "skill_id": "weapon:長劍"}
    with patch("trpg.engine.combat.roll_d20", return_value=15), \
         patch("trpg.engine.combat.roll", return_value=5):
        res = execute_action(built, ws)
    assert "roll_boost" not in res
    assert bm.ability_uses.get("precision_attack", 4) == 4   # die preserved


def test_post_roll_boost_cannot_save_a_natural_one():
    bm = _bm_fighter()
    bm.known_abilities = ["precision_attack"]
    enemy = _dummy(hp=100, ac=12)
    enemy.position = Vec2(6, 5)
    ws = _world(bm, enemy)
    built = {"type": "ATTACK", "attacker": "c", "target": "e0",
             "weapon": "長劍", "consumes": ["action"], "skill_id": "weapon:長劍"}
    with patch("trpg.engine.combat.roll_d20", return_value=1):
        res = execute_action(built, ws)
    assert "roll_boost" not in res          # nat 1 always misses
    assert res["hit"] is False
    assert bm.ability_uses.get("precision_attack", 4) == 4


# ── summon/spawn: add creatures to the fight mid-combat on the summoner's side ─

def _summon_world(summoner, summoner_id, *others_kv, party=()):
    from trpg.engine.vec2 import Battlefield
    chars = {summoner_id: summoner}
    for cid, c in others_kv:
        chars[cid] = c
    ws = WorldState(characters=chars, scene="", dungeon_map=None)
    ws.party_ids = list(party)
    ws.combat = CombatState(initiative_order=list(chars.keys()))
    ws.combat.battlefield = Battlefield(width=30, height=30)
    return ws


def test_summon_npc_minions_join_summoner_side_and_initiative():
    from trpg.scenarios.monsters import make_monster
    from trpg.rl.obs import partition_entities
    boss = make_monster("ogre"); boss.is_npc = True; boss.position = Vec2(20, 15)
    pc = _bm_fighter(); pc.position = Vec2(5, 5)
    ws = _summon_world(boss, "boss", ("pc", pc), party=["pc"])
    from trpg.engine.abilities import ABILITY_REGISTRY
    built = ABILITY_REGISTRY["summon_wolf_pack"].build_action("boss", None, None, char=boss)
    res = execute_action(built, ws)
    assert res["type"] == "SUMMON"
    assert res["count"] == 2
    new_ids = res["summoned_ids"]
    assert all(nid in ws.characters for nid in new_ids)
    # Summoned by an NPC → not in party_ids → enemies from the PC's POV.
    allies, enemies = partition_entities(ws, "pc")
    assert all(nid in enemies for nid in new_ids)
    # Inserted into initiative right after the summoner.
    order = ws.combat.initiative_order
    assert order.index(new_ids[0]) == order.index("boss") + 1
    # Placed inside the battlefield bounds.
    for nid in new_ids:
        p = ws.characters[nid].position
        assert 0 <= p.x <= 30 and 0 <= p.y <= 30


def test_summon_by_party_member_creates_allies():
    from trpg.rl.obs import partition_entities
    fighter = _bm_fighter(); fighter.position = Vec2(5, 5)
    enemy = _dummy(); enemy.position = Vec2(25, 25)
    ws = _summon_world(fighter, "pc", ("foe", enemy), party=["pc"])
    from trpg.engine.abilities import ABILITY_REGISTRY
    built = ABILITY_REGISTRY["summon_wolf_pack"].build_action("pc", None, None, char=fighter)
    res = execute_action(built, ws)
    new_ids = res["summoned_ids"]
    # Summoned by a party member → added to party_ids → allies from PC's POV.
    assert all(nid in ws.party_ids for nid in new_ids)
    allies, enemies = partition_entities(ws, "pc")
    assert all(nid in allies for nid in new_ids)


def test_summon_unknown_monster_rejected():
    boss = _dummy(); boss.position = Vec2(10, 10)
    ws = _summon_world(boss, "boss")
    res = execute_action({"type": "SUMMON", "summoner": "boss",
                          "monster_id": "nonexistent_beast", "count": 1,
                          "skill_id": "summon_test", "consumes": ["action"]}, ws)
    assert res["type"] == "ERROR"


def test_summoned_creatures_get_unique_ids():
    from trpg.scenarios.monsters import make_monster
    boss = make_monster("ogre"); boss.is_npc = True; boss.position = Vec2(15, 15)
    ws = _summon_world(boss, "boss")
    from trpg.engine.abilities import ABILITY_REGISTRY
    b1 = ABILITY_REGISTRY["summon_wolf_pack"].build_action("boss", None, None, char=boss)
    ids1 = execute_action(b1, ws)["summoned_ids"]
    boss.ability_uses["summon_wolf_pack"] = 1   # refill for a second cast
    b2 = ABILITY_REGISTRY["summon_wolf_pack"].build_action("boss", None, None, char=boss)
    ids2 = execute_action(b2, ws)["summoned_ids"]
    assert len(set(ids1) | set(ids2)) == 4      # no id collisions across casts


# ── lair actions: once-per-round environmental effect (init count 20) ─────────

def _lair_world(round_number=1):
    from trpg.engine.vec2 import Battlefield
    from trpg.scenarios.monsters import make_monster
    boss = make_monster("ogre"); boss.is_npc = True; boss.position = Vec2(10, 10)
    boss.lair_options = [{"ability": "lair_crushing_rocks", "ev": 7.0}]
    pc = _bm_fighter(); pc.position = Vec2(12, 10); pc.hp = 50; pc.max_hp = 50
    ws = WorldState(characters={"boss": boss, "pc": pc}, scene="", dungeon_map=None)
    ws.party_ids = ["pc"]
    ws.combat = CombatState(initiative_order=["boss", "pc"], round_number=round_number)
    ws.combat.battlefield = Battlefield(width=30, height=30)
    return ws, boss, pc


def test_lair_action_fires_once_and_hits_nearest_enemy():
    from trpg.engine.combat_policy import run_lair_actions
    ws, boss, pc = _lair_world()
    with patch("trpg.engine.combat.roll", return_value=8):
        events = run_lair_actions(ws, round_num=1)
    assert len(events) == 1
    assert events[0]["type"] == "LAIR_ACTION"
    assert pc.hp == 42                       # 2d6(8) auto-hit landed on the PC
    assert boss.lair_acted_round == 1


def test_lair_action_not_repeated_in_same_round():
    from trpg.engine.combat_policy import run_lair_actions
    ws, boss, pc = _lair_world()
    with patch("trpg.engine.combat.roll", return_value=8):
        run_lair_actions(ws, round_num=1)
        again = run_lair_actions(ws, round_num=1)   # same round → no-op
    assert again == []
    assert pc.hp == 42                       # only one hit total
    # A new round fires it again.
    with patch("trpg.engine.combat.roll", return_value=8):
        nxt = run_lair_actions(ws, round_num=2)
    assert len(nxt) == 1
    assert pc.hp == 34


def test_lair_decider_can_decline():
    from trpg.engine.combat_policy import run_lair_actions
    ws, boss, pc = _lair_world()
    ws.lair_decider = lambda ctx: None        # model/decider forgoes it
    events = run_lair_actions(ws, round_num=1)
    assert events == []
    assert pc.hp == 50                        # untouched
    assert boss.lair_acted_round == 1         # but the once-per-round attempt is spent


def test_lair_action_piggybacks_on_legendary_hook():
    from trpg.engine.combat_policy import run_legendary_actions
    ws, boss, pc = _lair_world()
    # boss has NO legendary budget, only lair_options — the shared hook still
    # fires the lair action (zero new driver wiring).
    with patch("trpg.engine.combat.roll", return_value=8):
        events = run_legendary_actions(ws, ended_char_id="pc", round_num=1)
    assert any(e["type"] == "LAIR_ACTION" for e in events)
    assert pc.hp == 42


def test_no_lair_options_is_noop():
    from trpg.engine.combat_policy import run_lair_actions
    ws, boss, pc = _lair_world()
    boss.lair_options = []
    events = run_lair_actions(ws, round_num=1)
    assert events == []
    assert pc.hp == 50


# ── resource-use tracking: limited-use abilities can't be spammed ─────────────

def test_action_surge_use_is_deducted():
    from trpg.engine.abilities import ABILITY_REGISTRY
    f = _bm_fighter(level=4)
    enemy = _dummy(); enemy.position = Vec2(6, 5)
    ws = _world(f, enemy)
    built = ABILITY_REGISTRY["action_surge"].build_action("c", None, None, char=f)
    res = execute_action(built, ws)
    assert res["type"] == "ACTION_SURGE"
    # max_uses=1 → one use spent, so it's exhausted and gated out of the kit.
    assert f.ability_uses.get("action_surge", 1) == 0
    f.known_abilities = ["action_surge"]
    assert "action_surge" not in [s.skill_id for s in available_skills(f, ws)]
