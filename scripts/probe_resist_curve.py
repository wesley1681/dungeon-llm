"""倍率-反應曲線探針：模型把 typed_resist 讀成「幅度」還是「旗標」？

背景（v9 dodge-collapse）：totem_bear L10 滿血 vs L1 shadow（物理全 0.5×），
攻擊合法卻 50 回合零攻擊；ablate typed_resist → 立刻改選攻擊。已證通道因果，
但「模型對抗性訊號的語義」未錘定。本探針掃倍率 m∈[0,2]，看 dodge→attack 在
哪一格翻轉：
  A 組：shadow 自然檔案、僅物理三型(斬/穿/鈍)=m —— 其他條目(黯蝕0/毒0/光耀2)不動
       → 若 A@m=1.0 仍 dodge ＝觸發源是「用不到的免疫條目」而非 0.5 抗性
  B 組：乾淨檔案、只有物理三型=m（無其他條目）
       → 翻轉點=模型對「自己主傷害型倍率」的語義曲線；m=1.0 編碼為全 0（=中性）
  對照：natural（原始重現）／empty（空檔案=通道 ablation 等價）

obs 編碼事實（obs.py I_DESC_RESIST）：resist_u[i] = multiplier − 1（0=中性）。

用法: python scripts/probe_resist_curve.py models/unified/uni_v9.pt
      [--ident totem_bear --opp shadow --lvl 10 --opp_lvl 1 --k 12]
"""
from __future__ import annotations
import sys, os, argparse, random
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout.reconfigure(encoding="utf-8")
import warnings; warnings.filterwarnings("ignore")
import torch

from trpg.scenarios.monsters import register_monsters
register_monsters()
try:
    from chimera_defs import register_chimeras
    register_chimeras()
except Exception:
    pass
try:
    from chimera_monsters import register_chimera_monsters
    register_chimera_monsters()
except Exception:
    pass
from trpg.rl.env_v2 import CombatEnvV2
from trpg.rl.model import apply_resource_mask, apply_entity_mask, pick_action
from trpg.engine.skill import available_skills, TargetType
from distill_routed import load_student
from eval_routed import stable_seed
from train_population import blind_np_single
from seed_switch_bc import damaging_options

PHYS = ["斬擊", "穿刺", "鈍擊"]
_OFFENSE_TT = (TargetType.SINGLE_ENEMY, TargetType.MULTI_ENEMY,
               TargetType.POINT, TargetType.LINE, TargetType.CONE)


def drive(net, ident, opp, lvl, opp_lvl, profile, key, k_dec=12, step_cap=400):
    """跑一場（同 key＝同種子成對可比），回 (決策序列, 攻擊數, 首決策oracle最佳EV, win)。
    profile=None＝不動自然檔案；dict＝整份覆蓋 damage_multipliers。"""
    random.seed(stable_seed(key))
    env = CombatEnvV2(seed=stable_seed(key) ^ 0x5A5A5A, n_agents=1, n_opps=1)
    obs, _ = env.reset(agent_archs=[ident], opp_archs=[opp],
                       level=lvl, opp_level=opp_lvl, layout="open")
    aid = env.agent_ids[0]
    oid = env.opp_ids[0]
    if profile is not None:
        env.ws.characters[oid].damage_multipliers = dict(profile)
    seq, n_atk, ev_first = [], 0, None
    done, steps = False, 0
    while not done and len(seq) < k_dec and steps < step_cap:
        steps += 1
        actor = env.current_agent_id
        if actor != aid:
            obs, _, term, trunc, _ = env.step([0, 0, 0])
            done = term or trunc
            continue
        ob = blind_np_single(obs)
        ot = {k: torch.from_numpy(v).unsqueeze(0) for k, v in ob.items()}
        with torch.no_grad():
            el, s, e, g = net(ot)
        s = apply_resource_mask(s, env.resources, env.ws, actor)
        e = apply_entity_mask(e, ot, env.ws, actor)
        act = list(pick_action(el[0], s[0], e[0], g[0], ws=env.ws, agent_id=actor))
        sks = available_skills(env.ws.characters[actor], env.ws)
        sk = sks[act[0]] if 0 < act[0] < len(sks) else None
        if ev_first is None:
            try:
                opts = damaging_options(env.ws, actor, oid)
                ev_first = max((t[4] for t in opts), default=0.0)
            except Exception:
                ev_first = float("nan")
        seq.append(sk.skill_id if sk is not None else "END")
        if (sk is not None
                and getattr(sk.features, "expected_damage", 0) > 0
                and sk.features.target_type in _OFFENSE_TT):
            n_atk += 1
        obs, _, term, trunc, _ = env.step(act)
        done = term or trunc
    win = not env.ws.characters[oid].is_alive()
    return seq, n_atk, ev_first, win


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt", nargs="?", default="models/unified/uni_v9.pt")
    ap.add_argument("--ident", default="totem_bear")
    ap.add_argument("--opp", default="shadow")
    ap.add_argument("--lvl", type=int, default=10)
    ap.add_argument("--opp_lvl", type=int, default=1)
    ap.add_argument("--k", type=int, default=12, help="每條件記錄前 k 個決策")
    ap.add_argument("--decomp", action="store_true",
                    help="外型條目逐欄分解：一次只放一種條目，定位哪些欄位觸發消極")
    args = ap.parse_args()

    net = load_student(args.ckpt)
    net.eval()
    key = f"resistcurve|{args.ident}|{args.opp}"   # 全條件同種子＝成對可比

    pe = CombatEnvV2(seed=1, n_agents=1, n_opps=1)
    pe.reset(agent_archs=[args.ident], opp_archs=[args.opp],
             level=args.lvl, opp_level=args.opp_lvl, layout="open")
    natural = dict(pe.ws.characters[pe.opp_ids[0]].damage_multipliers or {})
    print(f"ckpt={args.ckpt}  {args.ident} L{args.lvl} vs {args.opp} "
          f"L{args.opp_lvl} open  k={args.k}")
    print(f"敵自然 damage_multipliers = {natural}\n")
    print(f"{'條件':26s} {'首EV':>6s} {'攻擊數':>4s} {'win':>4s}  決策序列")

    if args.decomp:
        neg_only = {t: v for t, v in natural.items() if v < 1.0}
        conds = [
            ("empty(對照=會攻擊)", {}),
            ("僅 火=0.5", {"火": 0.5}),
            ("僅 火=0.0", {"火": 0.0}),
            ("僅 冰=0.5", {"冰": 0.5}),
            ("僅 閃電=0.5", {"閃電": 0.5}),
            ("五元素=0.5", {t: 0.5 for t in ["火", "冰", "閃電", "雷鳴", "強酸"]}),
            ("僅 黯蝕=0+毒=0", {"黯蝕": 0.0, "毒": 0.0}),
            ("僅 光耀=2.0", {"光耀": 2.0}),
            ("自然檔-光耀(純負)", neg_only),
            ("natural(原始)", None),
        ]
    else:
        ms = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0]
        conds = [("natural(原始重現)", None)]
        for m in ms:
            p = dict(natural)
            for t in PHYS:
                p[t] = m
            conds.append((f"A 自然檔+物理={m}", p))
        for m in ms:
            conds.append((f"B 僅物理={m}", {t: m for t in PHYS}))
        conds.append(("empty(空檔=ablation)", {}))

    for label, prof in conds:
        seq, n_atk, ev, win = drive(net, args.ident, args.opp, args.lvl,
                                    args.opp_lvl, prof, key, k_dec=args.k)
        print(f"{label:26s} {ev:6.1f} {n_atk:4d} {int(win):4d}  "
              f"{'/'.join(seq[:10])}")


if __name__ == "__main__":
    main()
