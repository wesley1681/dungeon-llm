"""被動天賦盲區的 headroom 探針（開刀前的數據門檻，CLAUDE.md：先證明盲區有代價）。

三問：
  Q1 載荷事實——自身 typed-resist 是否「已經」在 obs 自身列？（若是，則該補的隱形
     天賦是 regeneration，不是抗性。capability_descriptor 讀 char.damage_multipliers
     並接在每一列含 row0=self。）
  Q2 盲區確認——regen amount 0 vs 60，自身 capability_descriptor 是否位元相同？
     （證明模型完全看不到再生。）
  Q3 headroom——troll 在 3v1 真實壓力（vs 校準隊伍）下，sweep regen∈{0,10,30,60}：
     冠軍 net 與腳本（寫死進攻）各自的 WR + 進攻率（用咬/爪的回合占比）。
     若腳本 WR 隨 regen 陡升、冠軍卻平（進攻率不隨 regen 變）＝模型沒在利用再生
     ＝盲區有代價＝obs 手術有獎賞。反之（冠軍也陡升）＝HP 反應已替代＝無 headroom。

  另含 low-HP 變體：troll 起始 HP 壓低 → 逼出「低血該不該續戰」的 turn-1 抉擇，
  此抉擇發生在任何 regen tick 被觀測到之前，HP 反應無法替代。

Usage: python scripts/probe_regen_headroom.py --games 40
"""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding="utf-8")
import warnings; warnings.filterwarnings("ignore")

import argparse
import random
from collections import Counter
from zlib import crc32

import numpy as np
import torch

from trpg.scenarios.monsters import MONSTER_DEFS, register_monsters
register_monsters()

from trpg.rl.env_v2 import CombatEnvV2
from trpg.rl.action import encode_action
from trpg.rl.model import apply_resource_mask, apply_entity_mask, pick_action
from trpg.rl.obs import (capability_descriptor, entities_obs, I_DESC_RESIST,
                         N_DAMAGE_TYPES)
from trpg.engine.skill import available_skills
from trpg.engine.combat_policy import make_archetype_policy

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from distill_routed import load_student
from train_population import blind_np_single

PARTY = ("battle_master", "life", "evocation")
TROLL_WEAPON = "weapon:巨魔之爪"
BLOCKED = ("火", "強酸")


def set_regen(ch, amount):
    ch.regeneration = None if amount <= 0 else {"amount": int(amount),
                                                "blocked_by": BLOCKED}


def episode(opp_archs, mon_lvl, opp_lvl, ep_key, net, regen, hp0=None):
    """troll 在 agent 席 vs opp_archs。net=None→腳本臂。回傳
    (won, atk_turns, troll_turns, turns)。regen=注入再生量；hp0=起始 HP 覆寫。"""
    k = crc32(ep_key.encode())
    random.seed(k)
    env = CombatEnvV2(seed=k ^ 0x5A5A5A, n_agents=1, n_opps=len(opp_archs))
    obs, _ = env.reset(agent_archs=["troll"], opp_archs=list(opp_archs),
                       level=mon_lvl, opp_level=opp_lvl)
    aid = env.agent_ids[0]
    troll = env.ws.characters[aid]
    set_regen(troll, regen)
    if hp0 is not None:
        troll.hp = min(troll.max_hp, hp0)
    script = None if net is not None else make_archetype_policy("troll")
    atk_turns = 0
    troll_turns = 0
    turns = 0
    done = False
    while not done:
        actor = env.current_agent_id
        ch = env.ws.characters[actor]
        is_troll = actor == aid
        if net is not None:
            ob = blind_np_single(obs)
            ot = {kk: torch.from_numpy(v).unsqueeze(0) for kk, v in ob.items()}
            with torch.no_grad():
                el, s, e, g = net(ot)
            s = apply_resource_mask(s, env.resources, env.ws, actor)
            e = apply_entity_mask(e, ot, env.ws, actor)
            act = list(pick_action(el[0], s[0], e[0], g[0], ws=env.ws,
                                   agent_id=actor))
        else:
            dec = script.decide(actor, ch, env.ws, env.resources,
                                env.ws.combat.round_number)
            if dec.action is None or getattr(dec, "fled", False):
                act = [0, 0, 0]
            else:
                try:
                    act = list(encode_action(dec.action, env.ws, actor))
                except Exception:
                    act = [0, 0, 0]
        if is_troll:
            sks = available_skills(ch, env.ws)
            if act[0] > 0 and act[0] < len(sks):
                if sks[act[0]].skill_id == TROLL_WEAPON:
                    atk_turns += 1
            # 一個 troll「回合」可能含多個 sub-action（移動+攻擊）；用是否消耗
            # action 粗略界定回合：這裡簡化為每個 troll 決策計一次，attack 率＝
            # 咬爪決策數 / troll 決策數，足以看出進攻傾向是否隨 regen 變。
            troll_turns += 1
        turns += 1
        obs, _, term, trunc, _ = env.step(act)
        done = term or trunc
    won = (env.ws.characters[aid].is_alive()
           and not any(env.ws.characters[o].is_alive() for o in env.opp_ids))
    return won, atk_turns, troll_turns, turns


def sweep(net, label, opp_lvl, games, regens, hp0=None):
    print(f"\n  [{label}]  troll vs {'+'.join(PARTY)} @L{opp_lvl}"
          f"{'' if hp0 is None else f', troll 起始HP={hp0}'}  "
          f"({games} games/cell)")
    print(f"    {'regen':>6} | {'script WR':>9} {'sc-atk%':>7} "
          f"| {'model WR':>9} {'md-atk%':>7}")
    for rg in regens:
        row = {}
        for arm, drv in (("script", None), ("model", net)):
            w = at = tt = 0
            for g in range(games):
                key = f"rgh|{label}|{rg}|{opp_lvl}|{g}"
                won, a, t, _ = episode(PARTY, MONSTER_DEFS["troll"].natural_level,
                                       opp_lvl, key, drv, rg, hp0=hp0)
                w += int(won); at += a; tt += t
            row[arm] = (w / games, at / max(1, tt))
        print(f"    {rg:>6} | {row['script'][0]:>8.1%} {row['script'][1]:>6.1%} "
              f"| {row['model'][0]:>8.1%} {row['model'][1]:>6.1%}")


def obs_assertions():
    print("== Q1/Q2 obs 載荷事實 ==")
    # Q1: 自身 typed-resist 是否在 obs 自身列？拿一隻有 damage_multipliers 的怪。
    resist_mon = None
    for cid, md in MONSTER_DEFS.items():
        for t in md.traits:
            if t.trait_id == "damage_table" and t.params.get("multipliers"):
                resist_mon = cid
                break
        if resist_mon:
            break
    env = CombatEnvV2(seed=1, n_agents=1, n_opps=1)
    env.reset(agent_archs=[resist_mon], opp_archs=["commoner"],
              level=MONSTER_DEFS[resist_mon].natural_level, opp_level=1)
    aid = env.agent_ids[0]
    ent = entities_obs(env.ws, aid)
    resist_slice = ent[0, I_DESC_RESIST:I_DESC_RESIST + N_DAMAGE_TYPES]
    nz = int(np.count_nonzero(resist_slice))
    print(f"  Q1 自身列 typed-resist（{resist_mon}）非零欄位數 = {nz}  "
          f"→ {'已在 obs（抗性不是盲區）' if nz > 0 else '不在 obs'}")
    print(f"     mult={dict(env.ws.characters[aid].damage_multipliers)}")

    # Q2: regen amount 0 vs 60，troll 自身 descriptor 是否位元相同？
    env2 = CombatEnvV2(seed=1, n_agents=1, n_opps=1)
    env2.reset(agent_archs=["troll"], opp_archs=["commoner"],
               level=MONSTER_DEFS["troll"].natural_level, opp_level=1)
    tch = env2.ws.characters[env2.agent_ids[0]]
    set_regen(tch, 0); tch._cap_desc_cache = None
    d0 = capability_descriptor(tch).copy()
    set_regen(tch, 60); tch._cap_desc_cache = None
    d60 = capability_descriptor(tch).copy()
    same = bool(np.array_equal(d0, d60))
    print(f"  Q2 troll descriptor regen0==regen60 ? {same}  "
          f"→ {'再生完全不在 obs（真盲區）' if same else '再生在 obs'}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="models/mon_actor1/ma_u0032.pt")
    p.add_argument("--games", type=int, default=40)
    args = p.parse_args()

    obs_assertions()

    net = load_student(args.ckpt); net.eval()
    print(f"\n== Q3 headroom：ckpt={args.ckpt} ==")
    regens = [0, 10, 30, 60]
    # 全 HP 真實 3v1 壓力（troll party3-equiv≈4.5 → L4/L5 是真戰）
    sweep(net, "fullHP@L4", 4, args.games, regens)
    sweep(net, "fullHP@L5", 5, args.games, regens)
    # low-HP turn-1 抉擇（逼出「低血續戰 vs 退縮」，HP 反應無法替代）
    sweep(net, "lowHP25@L4", 4, args.games, regens, hp0=25)


if __name__ == "__main__":
    main()
