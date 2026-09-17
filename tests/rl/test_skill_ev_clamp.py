"""OOD-graft → dodge-collapse 根因守門（2026-07-02，bug_miner 生成式挖出）。

根因：skill_proj 是未正規化的 nn.Linear，訓練從沒見過 expected_damage>66 的技能；
任何 EV 破百的嫁接技（swallow/breath/boss 大招）會扭曲 pooled 技能表徵方向 → trunk
隱狀態 ‖h‖ 塌 3× → 策略退回消極龜縮。修法＝obs 邊界把 expected_damage 夾在
SKILL_EV_OBS_CAP，落在空隙 (66, 85.5)：對所有訓練身份位元級不變、只把 OOD 嫁接技
夾回流形。這組測試盯死兩件事不許回歸：
  (1) CAP 仍在空隙內、且夾 clamp 對每個訓練身份都是 no-op（bit-exact、免重訓）；
  (2) 嫁接一個 ultra-EV 技能後，obs 的 expected_damage 欄確實被夾到 CAP。
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "scripts"))

from trpg.scenarios.monsters import register_monsters, onev1_viable_monsters  # noqa: E402
register_monsters()
from trpg.engine.abilities import ABILITY_REGISTRY  # noqa: E402
from trpg.engine.skill import (  # noqa: E402
    SKILL_EV_OBS_CAP, I_SKILL_EXPECTED_DAMAGE, available_skills)
from trpg.scenarios.archetypes import STANDARD_ARCHETYPES, CLASS_DEFS  # noqa: E402


def _kit_ids(ident):
    cd = CLASS_DEFS.get(ident)
    return [g.skill_id for g in cd.skills] if cd else []


def _trained_identities():
    # Everything the policy actually trained on: the standard class roster + the
    # 1v1/party monster band (CR gate that fed population/monster-actor training).
    ids = list(STANDARD_ARCHETYPES)
    ids += list(onev1_viable_monsters(12.0))
    return sorted(set(ids))


def _make_owner(idn):
    from trpg.scenarios.archetypes import make_character
    from trpg.scenarios.monsters import make_monster, MONSTER_DEFS
    if idn in MONSTER_DEFS:
        return make_monster(idn)
    return make_character(idn, level=20)


def _materialized_evs():
    """{skill_id: materialized expected_damage} via each ability's roster owner.

    expected_damage is now DERIVED on materialize (dice → EV), not stored on the
    static features, so the gap/boss invariants must be measured on the
    materialized value the obs actually sees. Weapon-riding EV needs the real
    owner's weapon (e.g. behir's jaw for swallow), so instantiate real owners."""
    out = {}
    for idn in list(STANDARD_ARCHETYPES) + list(MONSTER_DEFS_KEYS):
        try:
            ch = _make_owner(idn)
        except Exception:
            continue
        for sk in available_skills(ch, None):
            # keep the max seen (a boss jaw beats a fallback weapon)
            ev = float(sk.features.expected_damage)
            if ev > out.get(sk.skill_id, -1.0):
                out[sk.skill_id] = ev
    return out


from trpg.scenarios.monsters import MONSTER_DEFS as _MDEFS  # noqa: E402
MONSTER_DEFS_KEYS = list(_MDEFS.keys())


def test_cap_sits_in_the_ev_gap():
    # CAP must be above every trained-kit EV (so clamping is a no-op there) and
    # below the lowest OOD boss ultra (so those DO get reined in). The data-proven
    # gap is (66, 85.5). Measured on the MATERIALIZED EV the obs sees.
    evs = sorted(_materialized_evs().values())
    below = max(e for e in evs if e <= SKILL_EV_OBS_CAP)
    above = min(e for e in evs if e > SKILL_EV_OBS_CAP)
    assert below <= 66.0, f"something ≤CAP has EV {below} > 66 — gap moved"
    assert above >= 85.0, f"an ability just above CAP has EV {above} < 85 — gap moved"
    assert 66.0 < SKILL_EV_OBS_CAP < 85.5


def test_clamp_is_noop_on_every_trained_identity():
    # No trained identity carries a skill whose expected_damage exceeds CAP, so
    # the obs clamp never fires on a real scenario = bit-exact, retrain-free.
    over = []
    for idn in _trained_identities():
        try:
            ch = _make_owner(idn)
        except Exception:
            continue
        for sk in available_skills(ch, None):
            ev = float(sk.features.expected_damage)
            if ev > SKILL_EV_OBS_CAP:
                over.append((idn, sk.skill_id, ev))
    assert not over, (
        "trained identities carry >CAP skills — clamp would alter their obs "
        f"(not bit-exact): {over}")


def test_only_boss_ultras_exceed_cap():
    # Documents the exact OOD boundary the clamp reins in. If a NEW high-EV
    # ability is added, this test forces a conscious decision about the cap.
    over = {sid for sid, ev in _materialized_evs().items()
            if ev > SKILL_EV_OBS_CAP}
    assert over == {"swallow", "fire_breath_ancient",
                    "kraken_swallow", "tarrasque_swallow"}, over


def test_grafted_ultra_ev_is_clamped_in_obs():
    # End-to-end: graft a swallow onto a class and confirm the obs skill matrix
    # caps its expected_damage column at CAP (the fix is live in skills_obs).
    from gui_identity import build_identity
    from trpg.rl.env_v2 import CombatEnvV2
    gid = build_identity("battle_master", ["tarrasque_swallow"])
    env = CombatEnvV2(seed=1, n_agents=1, n_opps=1)
    env.reset(agent_archs=[gid], opp_archs=["battle_master"],
              level=5, opp_level=5)
    from trpg.rl.obs import skills_obs
    aid = env.agent_ids[0]
    mat, mask = skills_obs(env.ws, aid)
    col = mat[mask > 0.5, I_SKILL_EXPECTED_DAMAGE]
    assert col.size, "agent has no skills?"
    assert float(col.max()) <= SKILL_EV_OBS_CAP + 1e-4, (
        f"a skill row exceeded CAP in obs: max={col.max()}")
    # and the swallow really is present pre-clamp (materialized EV > CAP) so the
    # clamp genuinely fired, not a vacuous pass. Read the DERIVED EV the grafted
    # agent actually carries (materialize, not the now-computed-only features).
    sw = next(s for s in available_skills(env.ws.characters[aid], env.ws)
              if s.skill_id == "tarrasque_swallow")
    assert sw.features.expected_damage > SKILL_EV_OBS_CAP
