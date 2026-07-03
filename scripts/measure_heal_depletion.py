"""直接回答「滿血浪費治療,會不會害你之後真正需要時沒得補」。

每場跑兩遍(同種骰):
  A = 原模型(可能滿血浪費 second_wind/cure 等一次性治療)
  B = 把『滿血純治療』那一手遮掉重選(=把該治療資源留到之後)
只在『A 真的發生過滿血浪費』的子集上比 A_win vs B_win。
若 B 在這子集明顯多贏 → 資源耗盡真的害到、是 WR-BUG。
若打平 → 留著也沒在「真正需要時」救回來 → 確認對勝率 0 影響。

用法: python scripts/measure_heal_depletion.py [模型] [--games N]
"""
from __future__ import annotations
import sys, os, argparse, random
from zlib import crc32
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding="utf-8")
import warnings; warnings.filterwarnings("ignore")
import torch
from trpg.scenarios.monsters import register_monsters, EQUIV_LEVEL_1V1
register_monsters()
from trpg.rl.env_v2 import CombatEnvV2
from trpg.rl.model import apply_resource_mask, apply_entity_mask, pick_action
from trpg.engine.skill import available_skills
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_population import blind_np_single
try:
    from chimera_defs import register_chimeras; register_chimeras()
except Exception:
    pass


def load(path):
    from distill_routed import load_student
    return load_student(path)


def _is_pure_heal_at_full(sks, idx, ch, ws, agent_ids):
    if not (0 < idx < len(sks)):
        return False
    sk = sks[idx]
    if getattr(sk.features, "expected_healing", 0) <= 0:
        return False
    allies = [ch] + [ws.characters[a] for a in agent_ids
                     if ws.characters[a] is not ch and ws.characters[a].is_alive()]
    return all(a.hp >= a.max_hp * 0.95 for a in allies)


def run(net, ident, opp_archs, lvl, opp_lvl, ep_key, gate):
    """回傳 (win, wasted_heal_happened)。"""
    k = crc32(ep_key.encode()); random.seed(k)
    env = CombatEnvV2(seed=k ^ 0x5A5A5A, n_agents=1, n_opps=len(opp_archs))
    obs, _ = env.reset(agent_archs=[ident], opp_archs=list(opp_archs),
                       level=lvl, opp_level=opp_lvl)
    aid = env.agent_ids[0]; done = False; wasted = False
    while not done:
        actor = env.current_agent_id
        ch = env.ws.characters[actor]
        if actor in env.agent_ids:
            ob = blind_np_single(obs)
            ot = {kk: torch.from_numpy(v).unsqueeze(0) for kk, v in ob.items()}
            with torch.no_grad():
                el, s, e, g = net(ot)
            s = apply_resource_mask(s, env.resources, env.ws, actor)
            e = apply_entity_mask(e, ot, env.ws, actor)
            act = list(pick_action(el[0], s[0], e[0], g[0], ws=env.ws, agent_id=actor))
            sks = available_skills(ch, env.ws)
            if _is_pure_heal_at_full(sks, act[0], ch, env.ws, env.agent_ids):
                wasted = True
                if gate:                       # 遮掉滿血補,留住資源,重選
                    guard = 0
                    while _is_pure_heal_at_full(sks, act[0], ch, env.ws,
                                                env.agent_ids) and guard < 8:
                        s[0, act[0]] = -1e9
                        act = list(pick_action(el[0], s[0], e[0], g[0],
                                               ws=env.ws, agent_id=actor))
                        guard += 1
            obs, _, term, trunc, _ = env.step(act)
        else:
            obs, _, term, trunc, _ = env.step([0, 0, 0])
        done = term or trunc
    win = (env.ws.characters[aid].is_alive()
           and not any(env.ws.characters[o].is_alive() for o in env.opp_ids))
    return win, wasted


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt", nargs="?", default="models/unified/uni_v8.pt")
    ap.add_argument("--games", type=int, default=80)
    args = ap.parse_args()
    net = load(args.ckpt)
    # 會滿血補的職業(有一次性治療)
    IDENTS = ["champion", "battle_master", "war", "devotion", "life", "vengeance"]
    OPP = ["battle_master", "champion", "vengeance", "evocation"]
    print(f"模型={os.path.basename(args.ckpt)} games={args.games}/身分\n")
    sub_a = sub_b = sub_n = 0   # 子集:A 發生過滿血浪費的對局
    allA = allB = alln = 0
    for ident in IDENTS:
        eq = EQUIV_LEVEL_1V1.get(ident, 5)
        lvl = 8 if eq == float("inf") else max(1, int(round(eq)))
        for gi in range(args.games):
            opp = [OPP[gi % len(OPP)]]; key = f"{ident}|{gi}"
            aw, wasted = run(net, ident, opp, lvl, lvl, key, gate=False)
            bw, _ = run(net, ident, opp, lvl, lvl, key, gate=True)
            allA += int(aw); allB += int(bw); alln += 1
            if wasted:                       # 只看「真的浪費過」的對局
                sub_a += int(aw); sub_b += int(bw); sub_n += 1
    print(f"全部對局:        A贏 {allA}/{alln}   B贏 {allB}/{alln}   Δ={allB-allA:+d}")
    print(f"『A浪費過治療』子集: A贏 {sub_a}/{sub_n}   B贏 {sub_b}/{sub_n}   Δ={sub_b-sub_a:+d}")
    print(f"\n解讀: 子集 Δ>0 = 留住資源在這些局真的多贏 = 滿血補是 WR-BUG;"
          f" Δ≈0 = 留著也沒救回來 = 對勝率無影響。")


if __name__ == "__main__":
    main()
