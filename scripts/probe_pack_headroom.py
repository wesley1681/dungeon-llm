"""pack_tactics 盲區 headroom 探針——純行為兌現天賦的「必定增益」測試。

pack_tactics（combat.py:1000）：攻擊「有己方盟友在 1.5m 內」的目標時得優勢。
零被動成分：狼群散開＝零優勢，集火貼同一目標＝優勢。要兌現此天賦，policy 必須
主動把身體擺到「盟友鄰接的目標」旁。

盲區確認：char.pack_tactics 是 bool，capability_descriptor 不讀 → 不在 obs。

headroom 檢定（模型控制狼群 vs 對手）：
  pack_active_rate = 狼決策時，其最近的可近戰敵人是否同時被「另一隻狼」鄰接
                     （即 _ally_adjacent_to 為真＝此刻攻擊會帶優勢）的回合占比。
  比較 pack_tactics ON vs OFF：
    - 盲模型 → ON==OFF（看不到天賦，走位不隨之變）＝盲區證據。
    - 若 ON 時 WR 顯著高於 OFF（優勢確實值錢）但模型沒多貼鄰 → 有 headroom。
  腳本臂作參照（GenericMonsterPolicy 狼）。

Usage: python scripts/probe_pack_headroom.py --games 40 --n_wolves 3 --opp battle_master --opp_lvl 3
"""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding="utf-8")
import warnings; warnings.filterwarnings("ignore")

import argparse
import random
from zlib import crc32

import torch

from trpg.scenarios.monsters import MONSTER_DEFS, register_monsters
register_monsters()

from trpg.rl.env_v2 import CombatEnvV2
from trpg.rl.action import encode_action
from trpg.rl.model import apply_resource_mask, apply_entity_mask, pick_action
from trpg.engine.skill import available_skills
from trpg.engine.combat import _ally_adjacent_to
from trpg.engine.combat_policy import make_archetype_policy

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from distill_routed import load_student
from train_population import blind_np_single

WOLF = "wolf"


def nearest_live_enemy(ws, wolf_id, opp_ids):
    wp = ws.characters[wolf_id].position
    live = [o for o in opp_ids if ws.characters[o].is_alive()]
    if not live:
        return None
    return min(live, key=lambda o: wp.distance_to(ws.characters[o].position))


def episode(n_wolves, opp_archs, wolf_lvl, opp_lvl, ep_key, net, pack_on):
    k = crc32(ep_key.encode())
    random.seed(k)
    env = CombatEnvV2(seed=k ^ 0x33CC33, n_agents=n_wolves, n_opps=len(opp_archs))
    obs, _ = env.reset(agent_archs=[WOLF] * n_wolves, opp_archs=list(opp_archs),
                       level=wolf_lvl, opp_level=opp_lvl)
    for aid in env.agent_ids:
        env.ws.characters[aid].pack_tactics = bool(pack_on)
    scripts = ({aid: make_archetype_policy(WOLF) for aid in env.agent_ids}
               if net is None else None)
    pack_active = 0
    wolf_steps = 0
    done = False
    while not done:
        actor = env.current_agent_id
        ch = env.ws.characters[actor]
        # 量測：此狼最近的活敵，是否被另一隻狼鄰接（此刻攻擊會帶優勢）
        tgt = nearest_live_enemy(env.ws, actor, env.opp_ids)
        if tgt is not None:
            wolf_steps += 1
            t = env.ws.characters[tgt]
            wolf_can_melee = ch.position.distance_to(t.position) <= 1.5 + 1e-6
            if wolf_can_melee and _ally_adjacent_to(env.ws, ch, t):
                pack_active += 1
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
            dec = scripts[actor].decide(actor, ch, env.ws, env.resources,
                                        env.ws.combat.round_number)
            if dec.action is None or getattr(dec, "fled", False):
                act = [0, 0, 0]
            else:
                try:
                    act = list(encode_action(dec.action, env.ws, actor))
                except Exception:
                    act = [0, 0, 0]
        obs, _, term, trunc, _ = env.step(act)
        done = term or trunc
    won = (any(env.ws.characters[a].is_alive() for a in env.agent_ids)
           and not any(env.ws.characters[o].is_alive() for o in env.opp_ids))
    return won, pack_active, wolf_steps


def run_arm(label, net, n_wolves, opp_archs, wolf_lvl, opp_lvl, games, pack_on):
    w = pa = ws = 0
    for g in range(games):
        key = f"pack|{label}|{pack_on}|{g}"
        won, p, s = episode(n_wolves, opp_archs, wolf_lvl, opp_lvl, key,
                            net, pack_on)
        w += int(won); pa += p; ws += s
    return w / games, pa / max(1, ws)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="models/mon_actor1/ma_u0032.pt")
    p.add_argument("--games", type=int, default=40)
    p.add_argument("--n_wolves", type=int, default=3)
    p.add_argument("--opp", default="battle_master")
    p.add_argument("--n_opp", type=int, default=2)
    p.add_argument("--opp_lvl", type=int, default=3)
    args = p.parse_args()

    net = load_student(args.ckpt); net.eval()
    opp_archs = [args.opp] * args.n_opp
    wolf_lvl = MONSTER_DEFS[WOLF].natural_level
    print(f"ckpt={args.ckpt}")
    print(f"{args.n_wolves}×wolf(L{wolf_lvl}) vs {args.n_opp}×{args.opp}"
          f"(L{args.opp_lvl}), {args.games} games/cell\n")
    print(f"{'arm':<8} {'pack':>5} | {'WR':>7} {'pack_active%':>13}")
    for arm, drv in (("model", net), ("script", None)):
        for pack_on in (False, True):
            wr, par = run_arm(arm, drv, args.n_wolves, opp_archs, wolf_lvl,
                             args.opp_lvl, args.games, pack_on)
            print(f"{arm:<8} {str(pack_on):>5} | {wr:>6.1%} {par:>12.1%}")
    print("\n讀法：模型 pack=False vs True 的 pack_active% 若相同＝走位不隨天賦變"
          "（盲）；WR(True)−WR(False) 即優勢的價值＝盲區代價。")


if __name__ == "__main__":
    main()
