"""孤立 reach 通道手術：根治用戶原始 bug（模型當敵人停在觸及外、永不接戰）。

根因(數據)：grid-head 是平滑距離特徵的線性點積，表徵不出 1.5m 觸及硬門檻 → 學會停在
~1.72m、主動避開貼近敵人的格 → 再多移動也只挑回原格 → dodge 迴圈、永不攻擊。

修法：obs 新增 reach_grid 通道(逐格「在我觸及內嗎」)；本手術**凍結整個 uni_v7，只解凍
grid_query 投影的 reach 那一列(weight 末列 + bias 末項)**，用 oracle 的移動目標(收口到敵人
1m=觸及內)做 BC。只有 reach 列可塑 → 只能改「移動目標選擇」→ 結構上不可能侵蝕 skill/
entity/end/timing(全凍結)。zero-init → 起點位元級=uni_v7。

用法: python scripts/surgery_reach.py --warm models/unified/uni_v7.pt \
        --rounds 6 --roll_eps 120 --steps_per_round 300 --out_dir models/unified_reach
"""
from __future__ import annotations
import sys, os, argparse, random
from pathlib import Path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout.reconfigure(encoding="utf-8")
import warnings; warnings.filterwarnings("ignore")
import numpy as np
import torch

from trpg.scenarios.monsters import register_monsters, EQUIV_LEVEL_1V1, MONSTER_DEFS
register_monsters()
from trpg.rl.env_v2 import CombatEnvV2
from trpg.rl.action import encode_action
from trpg.rl.model import apply_resource_mask, apply_entity_mask, pick_action
from trpg.rl.train_bc import bc_loss_step
from trpg.engine.skill import available_skills
from trpg.engine.combat_policy import make_archetype_policy
from train_population import blind_np_single
from synth_identity import STANDARD_IDS
from distill_routed import load_student

_EQUIV = dict(EQUIV_LEVEL_1V1)
try:
    from chimera_defs import register_chimeras, CHIMERA_IDS
    register_chimeras(); CLS_CHIM = list(CHIMERA_IDS)
except Exception:
    CLS_CHIM = []
try:
    from chimera_monsters import register_chimera_monsters, CHIMERA_EQUIV_1V1
    register_chimera_monsters(); _EQUIV.update(CHIMERA_EQUIV_1V1)
    MON_CHIM = list(CHIMERA_EQUIV_1V1)
except Exception:
    MON_CHIM = []
MON = ["kobold", "goblin", "orc", "ogre", "wolf", "dire_wolf", "ghoul",
       "owlbear", "shadow", "wight", "skeleton", "zombie", "gargoyle",
       "bandit", "basilisk", "manticore"]
IDENTS = list(STANDARD_IDS) + MON + CLS_CHIM + MON_CHIM


def _only_last_row(grad):
    g = torch.zeros_like(grad)
    g[-1:] = grad[-1:]          # keep only the reach row/element
    return g


def freeze_all_but_reach(net):
    """凍結全網，只留 grid_query 投影的 reach 列(weight 末列 + bias 末項)可訓練。"""
    for p in net.parameters():
        p.requires_grad_(False)
    n_train = 0
    for name, p in net.named_parameters():
        base = name.rsplit(".", 1)[0]
        if (base == "grid_query_proj" or base.startswith("grid_query_projs.")) \
                and (name.endswith(".weight") or name.endswith(".bias")):
            p.requires_grad_(True)
            p.register_hook(_only_last_row)
            n_train += 1
    return n_train


def collect(net, n_eps, seed, rng):
    """Roll 盲化模型；記錄每個 agent 決策狀態 + oracle 在該狀態的動作標籤。
    重點是『模型停在觸及外』的接近狀態——oracle 在那會給移動到觸及內的目標。
    對手用強隊伍等級對手(長戰鬥多 surface 接近決策)。"""
    net.eval()
    OPP = ["battle_master", "champion", "evocation", "vengeance"]
    O, A, T = [], [], []
    for ep in range(n_eps):
        ident = rng.choice(IDENTS)
        _eq = _EQUIV.get(ident, 5)
        lvl = 8 if _eq == float("inf") else max(1, int(round(_eq)))
        if lvl > 12:
            lvl = 8
        n_opp = rng.choice([1, 1, 2])
        opps = [rng.choice(OPP) for _ in range(n_opp)]
        env = CombatEnvV2(seed=seed * 1_000_003 + ep, n_agents=1, n_opps=n_opp)
        try:
            obs, _ = env.reset(agent_archs=[ident], opp_archs=opps,
                               level=lvl, opp_level=lvl)
        except Exception:
            continue
        aid = env.agent_ids[0]
        script = make_archetype_policy(ident)
        done = False; t = 0
        while not done and t < 60:
            actor = env.current_agent_id
            ch = env.ws.characters[actor]
            if actor == aid:
                ob = blind_np_single(obs)
                # oracle 標籤
                dec = script.decide(actor, ch, env.ws, env.resources,
                                    env.ws.combat.round_number)
                if dec.action is None or getattr(dec, "fled", False):
                    enc, tt = [0, 0, 0], -1
                else:
                    try:
                        enc = list(encode_action(dec.action, env.ws, actor))
                        sks = available_skills(ch, env.ws)
                        tt = int(sks[enc[0]].features.target_type) if 0 <= enc[0] < len(sks) else -1
                    except Exception:
                        enc, tt = [0, 0, 0], -1
                O.append(ob); A.append(enc); T.append(tt)
                # roll 用模型自己的貪心動作(部署行為)
                ot = {k: torch.from_numpy(v).unsqueeze(0) for k, v in ob.items()}
                with torch.no_grad():
                    el, s, e, g = net(ot)
                s = apply_resource_mask(s, env.resources, env.ws, actor)
                e = apply_entity_mask(e, ot, env.ws, actor)
                act = list(pick_action(el[0], s[0], e[0], g[0], ws=env.ws, agent_id=actor))
                obs, _, term, trunc, _ = env.step(act)
            else:
                obs, _, term, trunc, _ = env.step([0, 0, 0])
            done = term or trunc; t += 1
    if not O:
        return None
    obs_np = {k: np.stack([o[k] for o in O]) for k in O[0]}
    return obs_np, np.array(A, np.int64), np.array(T, np.int64)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--warm", default="models/unified/uni_v7.pt")
    p.add_argument("--out_dir", default="models/unified_reach")
    p.add_argument("--rounds", type=int, default=6)
    p.add_argument("--roll_eps", type=int, default=120)
    p.add_argument("--steps_per_round", type=int, default=300)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--lr", type=float, default=3e-3)   # 只訓一列、可用較大 lr
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    rng = random.Random(args.seed)
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    net = load_student(args.warm)
    n_train = freeze_all_but_reach(net)
    optim = torch.optim.Adam([p for p in net.parameters() if p.requires_grad], lr=args.lr)
    print(f"warm={args.warm} 可訓練張量={n_train}(只 reach 列) idents={len(IDENTS)}", flush=True)
    buf = None
    for rd in range(1, args.rounds + 1):
        cur = collect(net, args.roll_eps, args.seed * 7919 + rd, rng)
        if cur is None:
            print(f"round {rd}: no states"); continue
        if buf is None:
            buf = cur
        else:
            buf = ({k: np.concatenate([buf[0][k], cur[0][k]], 0) for k in cur[0]},
                   np.concatenate([buf[1], cur[1]]), np.concatenate([buf[2], cur[2]]))
            if len(buf[1]) > 80000:
                sel = np.random.choice(len(buf[1]), 80000, replace=False)
                buf = ({k: v[sel] for k, v in buf[0].items()}, buf[1][sel], buf[2][sel])
        nB = len(buf[1])
        net.train()
        for step in range(args.steps_per_round):
            sel = np.random.randint(0, nB, size=args.batch)
            ob = {k: torch.from_numpy(v[sel]) for k, v in buf[0].items()}
            bc_loss_step(net, ob, torch.from_numpy(buf[1][sel]),
                         torch.from_numpy(buf[2][sel]), optim)
        net.eval()
        torch.save(net.state_dict(), out / f"reach_r{rd:02d}.pt")
        # 印出 reach 列範數(看它有沒有在學)
        rn = 0.0
        for name, pp in net.named_parameters():
            if name.endswith("grid_query_projs.0.weight"):
                rn = pp.data[-1].norm().item()
        print(f"round {rd}: states={nB} reach列norm={rn:.3f} saved reach_r{rd:02d}.pt",
              flush=True)


if __name__ == "__main__":
    main()
