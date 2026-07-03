"""根因修正(非遮罩)：教模型在貼臉/滿血時 NATIVELY 不做空動作。

診斷(數據)：uni_v7 是 DAgger 模型，但 oracle 為了接近敵人大量示範 move，BC 平均下
skill-head 養出「能動就動」強 move 先驗(貼臉時 move logit 仍 +2.3 > attack)→ move-to-self
空轉；second_wind 同理在滿血放。oracle 本身乾淨(0 空動作)，且在這些失敗狀態 oracle 給的是
attack/END(已驗證 91/94 矯正)。原 DAgger 沒修到是因 (a) 沒針對 move、(b) self-anchor 把非
focus 職錨在自己的 no-op 上。

本訓練 = 「只修壞決策、其餘自蒸餾保留」的針對性 DAgger，零遮罩：
  每輪 roll 盲化模型(貪心=部署行為)跨全部標準職+怪物，1v1/1v2/2v1：
    - 失敗狀態(空轉move / 滿血heal) → 目標 = oracle 動作(attack/END)         [F 桶, 矯正]
    - 其餘狀態                      → 目標 = 模型自己的貪心動作(自蒸餾)        [G 桶, 保留]
  每步 BC 各取 F、G 一批 → F 把 move 先驗在貼臉狀態壓下去、G 保住 std12/走位/選招邊際。
  失敗狀態被 oracle 標籤(不再自我強化)；G 排除失敗狀態(不反向鎖死 no-op)。

用法: python scripts/train_fix_null.py --warm models/unified/uni_v7.pt \
        --rounds 6 --roll_eps 60 --steps_per_round 300 --out_dir models/unified_fix
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
from trpg.rl.model import (CombatPolicyNet, apply_resource_mask,
                           apply_entity_mask, pick_action)
from trpg.rl.obs import N_GRID, GRID_CELL_SIZE_M
from trpg.rl.train_bc import bc_loss_step
from trpg.engine.skill import available_skills
from trpg.engine.combat_policy import make_archetype_policy
from train_population import blind_np_single
from synth_identity import STANDARD_IDS

MON = ["kobold", "goblin", "orc", "ogre", "wolf", "dire_wolf", "ghoul",
       "owlbear", "shadow", "wight", "skeleton", "zombie", "gargoyle",
       "bandit", "basilisk", "manticore"]
# 含 chimera(縫合怪)：殘留 no-op 集中在未訓練的 frost_troll/chimera_gish，
# 把它們納入訓練才能根治(make_archetype_policy 對 chimera fallback=通用EV乾淨oracle)。
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
IDENTS = list(STANDARD_IDS) + MON + CLS_CHIM + MON_CHIM
# 強對手(=審計用的調過專家)→長戰鬥→多 surface「貼臉回合」失敗狀態
OPP_PANEL = ["battle_master", "champion", "evocation", "vengeance"]


def load(path):
    # uni_v7 是 n_head_groups=1 單頭架構；必須用 load_student 載(不要 per-arch
    # 擴張成 12 頭=改架構)。fine-tune 後存回仍是單頭、與 deliverable/eval 相容。
    from distill_routed import load_student
    return load_student(path)


def _cur_cell(p):
    return (max(0, min(N_GRID - 1, int(p.x / GRID_CELL_SIZE_M))) * N_GRID
            + max(0, min(N_GRID - 1, int(p.y / GRID_CELL_SIZE_M))))


def _tt_of(ch, ws, skill_idx):
    sks = available_skills(ch, ws)
    return int(sks[skill_idx].features.target_type) if 0 <= skill_idx < len(sks) else -1


def _is_failure(ch, ws, resources, act):
    """失敗 = 空轉move(到自己格,還有移動力) 或 滿血純治療。"""
    sks = available_skills(ch, ws)
    if not (0 < act[0] < len(sks)):
        return False
    sk = sks[act[0]]
    if sk.skill_id == "move" and resources.get("movement", 0) >= 1.0 \
            and act[2] == _cur_cell(ch.position):
        return True
    f = sk.features
    if getattr(f, "expected_healing", 0) > 0 and getattr(f, "expected_damage", 0) <= 0:
        if ch.hp >= ch.max_hp * 0.99:
            return True
    return False


def collect(net, n_eps, seed, rng):
    """Roll 盲化模型；分流 F(失敗→oracle標籤) 與 G(其餘→自蒸餾)。"""
    net.eval()
    F_o, F_a, F_t, G_o, G_a, G_t = [], [], [], [], [], []
    for ep in range(n_eps):
        ident = rng.choice(IDENTS)
        _eq = _EQUIV.get(ident, 5)
        lvl = 8 if _eq == float("inf") else max(1, int(round(_eq)))
        if lvl > 12:
            lvl = 8
        n_opp = rng.choice([1, 1, 2])
        opps = [rng.choice(OPP_PANEL) for _ in range(n_opp)]
        # 等級對手(長戰鬥)為主，偶爾劣勢/優勢，貼合審計分佈
        olvl = {0: lvl, 1: lvl, 2: max(2, lvl - 3), 3: lvl}[rng.choice([0, 1, 2, 3])]
        env = CombatEnvV2(seed=seed * 1_000_003 + ep, n_agents=1, n_opps=n_opp)
        try:
            obs, _ = env.reset(agent_archs=[ident], opp_archs=opps,
                               level=lvl, opp_level=olvl)
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
                ot = {k: torch.from_numpy(v).unsqueeze(0) for k, v in ob.items()}
                with torch.no_grad():
                    el, s, e, g = net(ot)
                s = apply_resource_mask(s, env.resources, env.ws, actor)
                e = apply_entity_mask(e, ot, env.ws, actor)
                act = list(pick_action(el[0], s[0], e[0], g[0], ws=env.ws,
                                       agent_id=actor))
                if _is_failure(ch, env.ws, env.resources, act):
                    # oracle 矯正標籤
                    dec = script.decide(actor, ch, env.ws, env.resources,
                                        env.ws.combat.round_number)
                    if dec.action is None or getattr(dec, "fled", False):
                        enc, tt = [0, 0, 0], -1
                    else:
                        try:
                            enc = list(encode_action(dec.action, env.ws, actor))
                            tt = _tt_of(ch, env.ws, enc[0])
                        except Exception:
                            enc, tt = [0, 0, 0], -1
                    F_o.append(ob); F_a.append(enc); F_t.append(tt)
                else:
                    G_o.append(ob); G_a.append(act); G_t.append(_tt_of(ch, env.ws, act[0]))
                obs, _, term, trunc, _ = env.step(act)
            else:
                obs, _, term, trunc, _ = env.step([0, 0, 0])
            done = term or trunc; t += 1

    def pack(O, A, T):
        if not O:
            return None
        obs_np = {k: np.stack([o[k] for o in O]) for k in O[0]}
        return obs_np, np.array(A, np.int64), np.array(T, np.int64)
    return pack(F_o, F_a, F_t), pack(G_o, G_a, G_t)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--warm", default="models/unified/uni_v7.pt")
    p.add_argument("--out_dir", default="models/unified_fix")
    p.add_argument("--rounds", type=int, default=6)
    p.add_argument("--roll_eps", type=int, default=60)
    p.add_argument("--steps_per_round", type=int, default=300)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    rng = random.Random(args.seed)
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    net = load(args.warm);
    optim = torch.optim.Adam(net.parameters(), lr=args.lr)
    print(f"warm={args.warm} idents={len(IDENTS)}", flush=True)
    Fbuf = Gbuf = None
    for rd in range(1, args.rounds + 1):
        F, G = collect(net, args.roll_eps, args.seed * 7919 + rd, rng)
        # 聚合最近 2 輪
        def merge(prev, cur):
            if cur is None: return prev
            if prev is None: return cur
            o = {k: np.concatenate([prev[0][k], cur[0][k]], 0) for k in cur[0]}
            return o, np.concatenate([prev[1], cur[1]]), np.concatenate([prev[2], cur[2]])
        # F 累積(失敗矯正信號很稀少、必須保留)；G 滾動上限
        Fbuf = merge(Fbuf, F); Gbuf = merge(Gbuf, G)
        if Gbuf is not None and len(Gbuf[1]) > 60000:
            sel = np.random.choice(len(Gbuf[1]), 60000, replace=False)
            Gbuf = ({k: v[sel] for k, v in Gbuf[0].items()}, Gbuf[1][sel], Gbuf[2][sel])
        nF = 0 if Fbuf is None else len(Fbuf[1])
        nG = 0 if Gbuf is None else len(Gbuf[1])
        net.train()
        for step in range(args.steps_per_round):
            if nF > 0:
                sel = np.random.randint(0, nF, size=args.batch)
                ob = {k: torch.from_numpy(v[sel]) for k, v in Fbuf[0].items()}
                bc_loss_step(net, ob, torch.from_numpy(Fbuf[1][sel]),
                             torch.from_numpy(Fbuf[2][sel]), optim)
            if nG > 0:
                sel = np.random.randint(0, nG, size=args.batch)
                ob = {k: torch.from_numpy(v[sel]) for k, v in Gbuf[0].items()}
                bc_loss_step(net, ob, torch.from_numpy(Gbuf[1][sel]),
                             torch.from_numpy(Gbuf[2][sel]), optim)
        net.eval()
        torch.save(net.state_dict(), out / f"fix_r{rd:02d}.pt")
        print(f"round {rd}: F(失敗→oracle)={nF}  G(自蒸餾)={nG}  saved fix_r{rd:02d}.pt",
              flush=True)


if __name__ == "__main__":
    main()
