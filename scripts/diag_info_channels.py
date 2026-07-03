"""統一逐通道因果自檢：模型有沒有在「看」敵人每一條資訊子通道？

對每條 obs 敵方描述子子通道，在**同一個真實戰鬥狀態**下比較兩個貪心動作：
  on : 正常 obs
  off: 只把該子通道的欄位在 ENEMY 列歸零
翻轉率 = 敵資訊改變時，模型選的動作跟著變的比例。關鍵：只統計**該子通道本來
就非零**的決策(該敵真的帶這條資訊)——否則歸零不改變 obs、翻轉恆 0 是「沒信號」
不是「瞎」。所以：
  flip% 高  = 模型讀這條、且會據此改行為 (綠)
  flip% ≈0 且 present 樣本多 = 通道帶了值但行為惰性 = 沒在看 (紅)
  present=0 = 這批對手沒有這條資訊、測不出 (N/A)

這是 probe_descriptor_ab 的逐通道推廣(那個只整塊歸零)。self 描述子保留(模型要
靠它知道自己的 kit)。全域骰配對，同狀態雙視圖 = 純「敵資訊翻不翻動作」。

用法: python scripts/diag_info_channels.py models/unified/uni_v9.pt --games 3
"""
from __future__ import annotations
import sys, os, argparse
from collections import defaultdict
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout.reconfigure(encoding="utf-8")
import warnings; warnings.filterwarnings("ignore")

import numpy as np
import torch

from trpg.rl.env_v2 import CombatEnvV2
from trpg.rl.model import apply_resource_mask, apply_entity_mask, pick_action
from trpg.rl.obs import (ENEMY_SLOT_START, ENT_DESC_START, N_V4_DESC,
                         N_V5_TRAIT, N_V6_CIMMUN, I_DESC_RESIST, I_DESC_CIMMUN,
                         ENT_TRAIT_START, I_ENT_LEVEL, I_ENT_MAXHP, I_ENT_AC,
                         I_ENT_DYING)
from trpg.engine.skill import N_STATUS_SLOTS, N_SAVE_STATS
from trpg.engine.damage import N_DAMAGE_TYPES
from trpg.scenarios.monsters import (register_monsters, onev1_viable_monsters,
                                     EQUIV_LEVEL_1V1, MONSTER_DEFS)
register_monsters()
from distill_routed import load_student
from synth_identity import STANDARD_IDS
from eval_routed import stable_seed

# (name, start_col, length) — every slice lives on the ENEMY rows.
_APPLIES = ENT_DESC_START + 9
_SAVES = _APPLIES + N_STATUS_SLOTS
CHANNELS = [
    ("cap_core(dmg/heal/rng/aoe/atk/dc)", ENT_DESC_START, 6),
    ("is_caster",          ENT_DESC_START + 6, 1),
    ("applies_status",     _APPLIES, N_STATUS_SLOTS),
    ("save_union",         _SAVES, N_SAVE_STATS),
    ("typed_resist",       I_DESC_RESIST, N_DAMAGE_TYPES),
    ("condition_immunity", I_DESC_CIMMUN, N_V6_CIMMUN),
    ("passive_traits",     ENT_TRAIT_START, N_V5_TRAIT),
    ("enemy_level",        I_ENT_LEVEL, 1),
    ("enemy_maxhp",        I_ENT_MAXHP, 1),
    ("enemy_ac",           I_ENT_AC, 1),
    ("enemy_dying",        I_ENT_DYING, 1),
    ("WHOLE_desc(sanity)", ENT_DESC_START, N_V4_DESC + N_V5_TRAIT + N_V6_CIMMUN),
]


def blind_self_one_hot(obs):
    from distill_routed import _ARCH_OH_START, _END_ARCH_START
    from trpg.rl.obs import N_ARCHETYPES
    out = dict(obs)
    ent = obs["entities"].copy()
    ent[0, _ARCH_OH_START:_ARCH_OH_START + N_ARCHETYPES] = 0.0
    out["entities"] = ent
    ef = obs["end_features"].copy()
    ef[_END_ARCH_START:_END_ARCH_START + N_ARCHETYPES] = 0.0
    out["end_features"] = ef
    return out


def zero_slice(obs, start, length):
    out = dict(obs)
    ent = out["entities"].copy()
    ent[ENEMY_SLOT_START:, start:start + length] = 0.0
    out["entities"] = ent
    return out


def greedy(net, ob, env, actor):
    ot = {k: torch.from_numpy(v).unsqueeze(0) for k, v in ob.items()}
    with torch.no_grad():
        el, s, e, g = net(ot)
    s = apply_resource_mask(s, env.resources, env.ws, actor)
    e = apply_entity_mask(e, ot, env.ws, actor)
    return tuple(pick_action(el[0], s[0], e[0], g[0], ws=env.ws, agent_id=actor))


def play(net, agent, opp, seed, level, opp_level, tally):
    env = CombatEnvV2(seed=seed, n_agents=1, n_opps=1)
    obs, _ = env.reset(agent_archs=[agent], opp_archs=[opp],
                       level=level, opp_level=opp_level)
    done = False
    while not done:
        actor = env.current_agent_id
        ob_on = blind_self_one_hot(obs)
        if actor in env.agent_ids:
            a_on = greedy(net, ob_on, env, actor)
            for name, start, length in CHANNELS:
                ob_off = zero_slice(ob_on, start, length)
                # only count if the ablation actually changed the enemy rows
                if np.array_equal(ob_on["entities"], ob_off["entities"]):
                    continue
                a_off = greedy(net, ob_off, env, actor)
                t = tally[name]
                t["present"] += 1
                if a_on != a_off:
                    t["flip"] += 1
            obs, _, term, trunc, _ = env.step(list(a_on))
        else:
            obs, _, term, trunc, _ = env.step([0, 0, 0])
        done = term or trunc


def main():
    p = argparse.ArgumentParser()
    p.add_argument("ckpt")
    p.add_argument("--games", type=int, default=3)
    p.add_argument("--mon_max_level", type=float, default=8.0)
    args = p.parse_args()

    net = load_student(args.ckpt); net.eval()
    mon_pool = onev1_viable_monsters(args.mon_max_level)
    tally = defaultdict(lambda: {"present": 0, "flip": 0})
    print(f"ckpt={args.ckpt}  agents={len(STANDARD_IDS)} classes  "
          f"enemies={len(mon_pool)} monsters  games/pairing={args.games}\n",
          flush=True)
    for mon in mon_pool:
        lvl = max(1, round(EQUIV_LEVEL_1V1[mon]))
        m_lvl = MONSTER_DEFS[mon].natural_level
        for a in STANDARD_IDS:
            for k in range(args.games):
                play(net, a, mon, stable_seed(f"info_{a}_{mon}_{k}"),
                     lvl, m_lvl, tally)

    print(f"{'channel':34s} {'present':>8s} {'flip%':>8s}   verdict")
    print("-" * 68)
    for name, _, _ in CHANNELS:
        t = tally[name]
        pr, fl = t["present"], t["flip"]
        if pr == 0:
            verdict = "N/A (該批敵無此資訊)"
            rate = 0.0
        else:
            rate = fl / pr
            if rate >= 0.05:
                verdict = "綠 讀取"
            elif pr >= 100:
                verdict = "紅 惰性(帶值不讀)"
            else:
                verdict = "灰 樣本少"
        print(f"{name:34s} {pr:8d} {rate:8.1%}   {verdict}")


if __name__ == "__main__":
    main()
