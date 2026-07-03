"""合成大量「卡死＋滿血」狀態 + 專家標籤=結束回合,灌給 end_head 訓練。

根因:滿血浪費治療的真觸發=搆不到敵(沒攻擊)＋滿血(補0)＋移動力耗光(走不到)＝零價值,
但 end-head 因「還有 action/bonus」不肯結束→硬塞 heal。這種狀態在自然 rollout 太稀有,
end-head 學不出尖銳條件。本檔直接造這種狀態(各種職業/距離/資源組合),標 end,讓 end-head 強學。

正例(label=end): 滿血 + 所有敵在我攻擊範圍外 + movement≈0 (走不到) → 此刻無價值動作 → 結束。
反例(label=don't-end, 用真實攻擊技): 滿血但敵在 reach 內 → 該攻擊不該結束(平衡,避免 end-head 變always-end)。

輸出 stuck_demos.npz,用 train_dagger.py --switch_demos 共訓(--freeze_casting 1 --keep_params end_head)。
"""
from __future__ import annotations
import sys, os, argparse, random
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding="utf-8")
import warnings; warnings.filterwarnings("ignore")
import numpy as np
import torch
from trpg.scenarios.monsters import register_monsters
register_monsters()
from trpg.rl.env_v2 import CombatEnvV2, MOVE_BUDGET_M
from trpg.rl.obs import build_obs
from trpg.engine.skill import available_skills, TargetType
from trpg.engine.vec2 import Vec2
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_population import blind_np_single
try:
    from chimera_defs import register_chimeras; register_chimeras()
except Exception:
    pass


def _reach(ch):
    try:
        return float(ch.get_weapon().range_normal) if ch.weapons else 1.5
    except Exception:
        return 1.5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="models/unified_heal2/stuck_demos.npz")
    ap.add_argument("--n", type=int, default=4000)
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    rng = random.Random(0)
    # 會卡死浪費治療的職業 + 大量近戰怪(讓「搆得到→攻擊」錨點覆蓋廣,抵抗 kite demo 外溢侵蝕)
    IDENTS = ["champion", "battle_master", "war", "devotion", "life", "vengeance",
              "totem_bear", "berserker", "assassin", "arcane_trickster",
              "ogre", "orc", "ghoul", "wolf", "dire_wolf", "owlbear", "wight",
              "skeleton", "zombie", "bandit", "gargoyle", "shadow", "kobold",
              "goblin", "manticore", "basilisk"]
    O, A, T = [], [], []
    ep = 0
    while len(A) < args.n:
        ident = rng.choice(IDENTS)
        env = CombatEnvV2(seed=ep * 2654435761 % (2**31), n_agents=1, n_opps=1)
        try:
            env.reset(agent_archs=[ident], opp_archs=["champion"],
                      level=rng.randint(3, 8), opp_level=5, layout="open")
        except Exception:
            ep += 1; continue
        ch = env.ws.characters["agent_0"]; foe = env.ws.characters["opp_0"]
        reach = _reach(ch)
        cx, cy = 15.0, 15.0
        ch.position = Vec2(cx, cy)
        # 三類:
        #  (a) 卡死+滿血+走不到 → 結束(別浪費治療)
        #  (b) 卡死+受傷+走不到 → 治療(別結束!治療此刻有價值)— 教 end_head 依 HP 條件化
        #  (c) 敵在 reach 內 → 攻擊(別結束)
        roll = rng.random()
        d = reach + rng.uniform(0.4, 2.5)
        mv = rng.uniform(0.0, 0.4)
        ang = rng.uniform(0, 6.2832)
        foe.position = Vec2(cx + d * np.cos(ang), cy + d * np.sin(ang))
        if roll < 0.25:                                   # (a) 卡死滿血 → 結束
            ch.hp = ch.max_hp
            label = [0, 0, 0]; tt = -1
        elif roll < 0.40:                                 # (b) 卡死受傷 → 治療(不結束)
            ch.hp = ch.max_hp * rng.uniform(0.2, 0.45)
            sks = available_skills(ch, env.ws)
            heal = next((i for i, s in enumerate(sks)
                         if getattr(s.features, "expected_healing", 0) > 0), None)
            if heal is None:                              # 無治療技→改成滿血結束例
                ch.hp = ch.max_hp; label = [0, 0, 0]; tt = -1
            else:
                label = [heal, 0, 0]
                tt = int(sks[heal].features.target_type)
        else:                                             # (c) 搆得到 → 攻擊
            ch.hp = ch.max_hp
            d = max(0.5, reach - rng.uniform(0.1, 0.6))    # 搆得到
            mv = rng.uniform(0.0, MOVE_BUDGET_M)
            foe.position = Vec2(cx + d * np.cos(ang), cy + d * np.sin(ang))
            sks = available_skills(ch, env.ws)
            atk = next((i for i, s in enumerate(sks)
                        if s.features.target_type in (TargetType.SINGLE_ENEMY,
                                                      TargetType.MULTI_ENEMY)
                        and s.skill_id != "move"), None)
            if atk is None:
                ep += 1; continue
            label = [atk, 0, 0]
            tt = int(sks[atk].features.target_type)
        res = {"action": rng.choice([1, 1, 2]), "bonus_action": rng.choice([0, 1, 1]),
               "movement": mv}
        ob = blind_np_single(build_obs(env.ws, "agent_0", res))
        O.append(ob); A.append(label); T.append(tt)
        ep += 1
    obs_np = {k: np.stack([o[k] for o in O]) for k in O[0]}
    save = {f"obs_{k}": v for k, v in obs_np.items()}
    save["actions"] = np.array(A, np.int64)
    save["target_types"] = np.array(T, np.int64)
    np.savez_compressed(args.out, **save)
    n_end = sum(1 for a in A if a[0] == 0)
    print(f"寫出 {len(A)} 筆 → {args.out}  (卡死→結束 {n_end}, 反例→攻擊 {len(A)-n_end})")


if __name__ == "__main__":
    main()
