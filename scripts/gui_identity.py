"""Shared, tkinter-free identity assembler — SINGLE SOURCE OF TRUTH for the
identity combos a user can construct.

Design law (user mandate 2026-07-02): "只要 GUI 能湊出來的組合，bug_miner 就要
能湊出來。" To make that true *by construction* rather than by luck, both the GUI
selection panel (scripts/play_gui.py) and bug_miner's graft mutation build every
identity through :func:`build_identity` here — they physically cannot drift
because they share this code. Add a new GUI knob → add it here → the miner covers
it automatically. No tkinter import, so the miner stays importable headless.
"""
from __future__ import annotations

import dataclasses

from trpg.scenarios.archetypes import (
    CLASS_DEFS, ARCHETYPE_FACTORIES, ARCHETYPE_ROLES, STANDARD_ARCHETYPES,
    SkillGrant, TraitGrant, _factory,
)
from trpg.scenarios.monsters import MONSTER_DEFS

# GUI knob domains (mirrors the IdentityPanel widgets) — kept here so the miner
# samples exactly the same option sets the human sees in the panel.
DMG_TYPES = ["火", "冰", "閃電", "毒", "光耀", "強酸", "黯蝕", "雷鳴",
             "斬擊", "穿刺", "鈍擊"]
RESIST_MODES = {"免疫 (×0)": 0.0, "抗性 (×0.5)": 0.5,
                "易傷 (×2)": 2.0, "無": None}
LAYOUTS = ["open", "walls", "pillar", "difficult", "lava"]

# Boolean-trait knobs (checkbox label → TraitGrant recipe). Adding a checkbox to
# the GUI means adding an entry here; the miner then grafts it automatically.
BOOL_TRAITS = {
    "pack_tactics": ("pack_tactics", None),
    "regeneration": ("regeneration", {"amount": 10, "blocked_by": ["火", "強酸"]}),
    "undead_fortitude": ("undead_fortitude", None),
    "extra_attack": ("extra_attack", {"attacks": 2}),
}

_custom_counter = [0]


def resolve_identity(base_id: str, extra_skill_ids, trait_grants) -> str:
    """若沒有任何加掛，直接用 base_id；否則複製 ClassDef 追加 skills/traits 後
    註冊一個臨時 id 並回傳。怪物的 MonsterDef 經 dataclasses.replace 仍保留
    cr / natural_level 等欄位。"""
    if not extra_skill_ids and not trait_grants:
        return base_id
    base = CLASS_DEFS[base_id]
    _custom_counter[0] += 1
    new_id = f"_gui_{base_id}_{_custom_counter[0]}"
    cd = dataclasses.replace(
        base,
        archetype_id=new_id,
        skills=tuple(base.skills) +
        tuple(SkillGrant(s, min_level=1) for s in extra_skill_ids),
        traits=tuple(base.traits) + tuple(trait_grants),
    )
    CLASS_DEFS[new_id] = cd
    ARCHETYPE_FACTORIES[new_id] = _factory(new_id)
    ARCHETYPE_ROLES[new_id] = cd.role
    return new_id


def build_identity(base_id, extra_skill_ids=(), dmg_type=None, resist_mult=None,
                   bool_traits=()):
    """Assemble every GUI-constructable trait/skill combo through one door.

    Args mirror the IdentityPanel knobs exactly:
      extra_skill_ids : ability ids to graft (the multi-select skill list)
      dmg_type/resist_mult : the 抗性天賦 combobox (mult None = 「無」)
      bool_traits : subset of BOOL_TRAITS keys (the checkboxes)
    """
    traits = []
    if dmg_type is not None and resist_mult is not None:
        traits.append(TraitGrant("damage_table",
                                 params={"multipliers": {dmg_type: resist_mult}}))
    for key in bool_traits:
        name, params = BOOL_TRAITS[key]
        traits.append(TraitGrant(name, params=params) if params
                      else TraitGrant(name))
    return resolve_identity(base_id, list(extra_skill_ids), traits)


def identity_catalog():
    """回傳 [(label, id, is_monster, natural_level)]，職業在前、怪物在後
    （含全部怪物——與 GUI 下拉選單同源）。"""
    out = []
    for aid in STANDARD_ARCHETYPES:
        cd = CLASS_DEFS[aid]
        out.append((f"[職業] {aid} — {cd.default_name}", aid, False, 5))
    for mid, md in MONSTER_DEFS.items():
        out.append((f"[怪物] {mid} — {md.default_name} (CR{md.cr:g})",
                    mid, True, md.natural_level))
    return out
