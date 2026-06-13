"""pack_tactics 因果驗證（obs v5 目標的決定性交付）。

對每隻 pack-tactics 怪（wolf=訓練；kobold/dire_wolf=held-out 零樣本），模型控制
一個狼群 vs AoE 隊伍，量三條件下的 flank 率與 WR：

  ON   引擎 pack_tactics 開、obs 通道誠實開（=可貼鄰取優勢，且模型看得到）
  OFF  引擎 pack_tactics 關、obs 通道誠實關（=貼鄰無優勢，模型也知道）
  ABL  引擎 pack_tactics 開、但 obs 的 I_TRAIT_PACK 欄「歸零」
       （優勢仍可兌現，但模型被蒙住看不到天賦）

三條件共用同一場種骰，唯一差異就是天賦/通道。判讀：
  • flank(ON) ≫ flank(OFF)  → 模型依天賦改行為（讀到就貼鄰）
  • flank(ON) ≫ flank(ABL)  → 行為是「讀通道」造成的（蒙住就退回）＝因果
  • WR(ON) > WR(OFF)        → 貼鄰確實把優勢換成勝率
  • kobold/dire_wolf 同樣式  → 學到的是「pack_tactics 天賦→貼鄰」的通則，
                               非「wolf→貼鄰」死記（held-out 零樣本泛化）

Usage:
  python scripts/verify_pack_causal.py --ckpt models/pack_v5/ma_u0032.pt --games 60
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
from trpg.rl.model import apply_resource_mask, apply_entity_mask, pick_action
from trpg.rl.obs import I_TRAIT_PACK
from trpg.engine.combat import _ally_adjacent_to

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from distill_routed import load_student
from train_population import blind_np_single

OPPS = ("evocation", "battle_master")
CREATURES = ("wolf", "kobold", "dire_wolf")   # wolf trained; rest held-out
TRAINED = {"wolf"}


def episode(creature, n_pack, opp_lvl, ep_key, net, engine_pack_on, ablate):
    k = crc32(ep_key.encode())
    random.seed(k)
    env = CombatEnvV2(seed=k ^ 0x33CC33, n_agents=n_pack, n_opps=len(OPPS))
    obs, _ = env.reset(agent_archs=[creature] * n_pack, opp_archs=list(OPPS),
                       level=MONSTER_DEFS[creature].natural_level,
                       opp_level=opp_lvl)
    for aid in env.agent_ids:
        env.ws.characters[aid].pack_tactics = bool(engine_pack_on)
    pack_active = wolf_steps = 0
    done = False
    while not done:
        actor = env.current_agent_id
        ch = env.ws.characters[actor]
        live = [o for o in env.opp_ids if env.ws.characters[o].is_alive()]
        if live:
            tgt = min(live, key=lambda o: ch.position.distance_to(
                env.ws.characters[o].position))
            t = env.ws.characters[tgt]
            if ch.position.distance_to(t.position) <= 1.5 + 1e-6:
                wolf_steps += 1
                if _ally_adjacent_to(env.ws, ch, t):
                    pack_active += 1
        ob = blind_np_single(obs)
        ot = {kk: torch.from_numpy(v).unsqueeze(0) for kk, v in ob.items()}
        if ablate:
            ot["entities"][:, :, I_TRAIT_PACK] = 0.0   # blind model to its trait
        with torch.no_grad():
            el, s, e, g = net(ot)
        s = apply_resource_mask(s, env.resources, env.ws, actor)
        e = apply_entity_mask(e, ot, env.ws, actor)
        act = list(pick_action(el[0], s[0], e[0], g[0], ws=env.ws,
                               agent_id=actor))
        obs, _, term, trunc, _ = env.step(act)
        done = term or trunc
    won = (any(env.ws.characters[a].is_alive() for a in env.agent_ids)
           and not any(env.ws.characters[o].is_alive() for o in env.opp_ids))
    return won, pack_active, wolf_steps


def run_condition(net, creature, n_pack, opp_lvl, games, engine_pack_on, ablate):
    w = pa = ws = 0
    for g in range(games):
        # SAME key across conditions → identical dice; only trait/channel differ
        key = f"vpc|{creature}|{n_pack}|{opp_lvl}|{g}"
        won, p, s = episode(creature, n_pack, opp_lvl, key, net,
                            engine_pack_on, ablate)
        w += int(won); pa += p; ws += s
    return w / games, pa / max(1, ws)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="models/pack_v5/ma_u0032.pt")
    p.add_argument("--games", type=int, default=60)
    p.add_argument("--n_pack", type=int, default=3)
    p.add_argument("--opp_lvl", type=int, default=3)
    args = p.parse_args()

    net = load_student(args.ckpt); net.eval()
    print(f"ckpt={args.ckpt}  {args.n_pack}×creature vs {OPPS}@L{args.opp_lvl}, "
          f"{args.games} games/cell")
    print(f"{'creature':<12} {'(role)':<10} "
          f"{'flankON':>8} {'flankOFF':>9} {'flankABL':>9}  "
          f"{'WR_ON':>6} {'WR_OFF':>7} {'WR_ABL':>7}")
    for c in CREATURES:
        role = "trained" if c in TRAINED else "held-out"
        wr_on, fl_on = run_condition(net, c, args.n_pack, args.opp_lvl,
                                     args.games, True, False)
        wr_off, fl_off = run_condition(net, c, args.n_pack, args.opp_lvl,
                                       args.games, False, False)
        wr_abl, fl_abl = run_condition(net, c, args.n_pack, args.opp_lvl,
                                       args.games, True, True)
        flag = ""
        if fl_on - fl_off > 0.10 and fl_on - fl_abl > 0.10:
            flag = "  <- channel-conditioned (causal)"
        print(f"{c:<12} {('('+role+')'):<10} "
              f"{fl_on:>7.1%} {fl_off:>8.1%} {fl_abl:>8.1%}  "
              f"{wr_on:>5.1%} {wr_off:>6.1%} {wr_abl:>6.1%}{flag}")
    print("\n判讀：flankON≫flankOFF＝依天賦改行為；flankON≫flankABL＝行為由讀通道"
          "驅動（蒙住即退回）＝因果；held-out 同樣式＝trait→行為泛化。")


if __name__ == "__main__":
    main()
