"""插樁滿血硬補：固定 1v1、bonus action 還在手，把 self HP 從 100% 掃到 10%，
dump 模型 skill-head 對「治療技」的 logit 與排名，看清是哪種根因：

  VALUE-FLAT  : 滿血時治療 logit 仍 ≥ 攻擊/其他 → 價值估計在 HP 頂端太平(可訓練)。
  THRESHOLDED : 治療 logit 隨 HP 上升而墜，滿血時已非 argmax → 模型其實會收手
                (=審計的滿血補是別的觸發點,需回戰鬥情境抓)。

對照腳本專家門檻(second_wind 0.4 / paladin 0.25 / cleric 0.25-0.5)：專家滿血絕不補。

用法: python scripts/probe_heal_full.py [模型路徑] [身分...]
"""
from __future__ import annotations
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding="utf-8")
import warnings; warnings.filterwarnings("ignore")
import numpy as np
import torch

from trpg.scenarios.monsters import register_monsters
register_monsters()
from trpg.rl.env_v2 import CombatEnvV2, MOVE_BUDGET_M
from trpg.rl.model import apply_resource_mask, apply_entity_mask
from trpg.rl.obs import build_obs, N_ARCHETYPES
from trpg.rl.neural_policy import _ENT_ARCH_START, _END_ARCH_START
from trpg.engine.skill import available_skills, TargetType


def load(path):
    from distill_routed import load_student
    return load_student(path)


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "models/unified/uni_v8.pt"
    idents = sys.argv[2:] or ["champion", "battle_master", "war", "devotion"]
    net = load(path)
    print(f"模型={os.path.basename(path)}\n")

    for ident in idents:
        env = CombatEnvV2(n_agents=1, n_opps=1)
        env.reset(level=8, opp_level=8, layout="open",
                  agent_archs=[ident], opp_archs=["battle_master"], seed=11)
        ch = env.ws.characters["agent_0"]
        foe = env.ws.characters["opp_0"]
        # 貼到攻擊距離內，確保攻擊合法、不被走位干擾。
        foe.position = ch.position.__class__(ch.position.x + 1.2, ch.position.y)
        max_hp = ch.max_hp
        skills = available_skills(ch, env.ws)
        heal_idx = [i for i, s in enumerate(skills)
                    if getattr(s.features, "expected_healing", 0) > 0]
        if not heal_idx:
            print(f"{ident:14s}: 無治療技,略過"); continue
        hnames = ",".join(skills[i].skill_id for i in heal_idx)
        print(f"== {ident}  治療技=[{hnames}]  max_hp={max_hp:.0f} ==")
        # bonus action 在手；模擬「主動作已用」(action=0) 與「未用」(action=1) 兩種。
        for act_left in (1, 0):
            print(f"   [action={act_left}, bonus=1, movement=0]")
            for frac in (1.0, 0.95, 0.8, 0.6, 0.4, 0.25, 0.1):
                ch.hp = max(1.0, max_hp * frac)
                res = {"action": act_left, "bonus_action": 1, "movement": 0.0}
                obs = build_obs(env.ws, "agent_0", res)
                obs["entities"][0, _ENT_ARCH_START:_ENT_ARCH_START + N_ARCHETYPES] = 0.0
                obs["end_features"][_END_ARCH_START:_END_ARCH_START + N_ARCHETYPES] = 0.0
                ot = {k: torch.from_numpy(v).unsqueeze(0) for k, v in obs.items()}
                with torch.no_grad():
                    el, s, e, g = net(ot)
                s = apply_resource_mask(s, res, env.ws, "agent_0")
                e = apply_entity_mask(e, ot, env.ws, "agent_0")
                row = s[0].clone()
                row[0] = -1e9  # 排除 end 槽
                legal = (row > -1e8)
                arg = int(row.argmax().item())
                arg_id = skills[arg].skill_id if arg < len(skills) else "?"
                # 治療技中 logit 最高者 + 其在合法技中的排名
                hl = [(i, row[i].item()) for i in heal_idx if legal[i]]
                if hl:
                    hi, hlogit = max(hl, key=lambda kv: kv[1])
                    rank = int((row[legal] > hlogit).sum())
                    nlegal = int(legal.sum())
                    flag = " ← 滿血補!" if (frac >= 0.95 and arg in heal_idx) else ""
                    print(f"      HP {frac*100:4.0f}%  argmax={arg_id:18s}"
                          f" heal_logit={hlogit:+.3f} rank {rank}/{nlegal}{flag}")
                else:
                    print(f"      HP {frac*100:4.0f}%  argmax={arg_id:18s} (治療技被resource_mask遮)")
        print()


if __name__ == "__main__":
    main()
