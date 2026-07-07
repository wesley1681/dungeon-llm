"""確認診斷 + 校準獎勵壓力。

診斷:對站樁假人 + 寬鬆 50 步截止,選錯型(免疫)幾乎零代價(telescoping PBRS 路徑
無關,只剩 γ 折扣的微弱壓力)——所以「隨機各半打兩型」照樣靠對的那半殺死站樁敵。
用三個腳本策略量化證明,並掃 HP 找到「全對→贏、各半→超時」的敵人血量。

  always_correct : 每回合都打非免疫型(oracle 上限)
  always_wrong   : 每回合都打免疫型(0 傷下限)
  half_random    : 每回合隨機挑一型(≈模型現況)

若在低 HP 時 half_random 也高 WR ＝ 選錯零代價(診斷成立)。
掃 HP 找 always_correct WR≈100% 且 half_random WR≈0% 的點 = 有辨別壓力的敵人血量。
"""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout.reconfigure(encoding="utf-8")
import warnings; warnings.filterwarnings("ignore")
os.environ.setdefault("TRPG_WASTED_MOVE_COST", "0")
import random
import numpy as np

from exp_scratch_immune import register_kit, make_env, FIRE, COLD, W_FIRE, W_COLD
from trpg.rl.obs import ENEMY_SLOT_START
from trpg.engine.skill import available_skills

register_kit()


def _weapon_indices(env, aid):
    names = [s.skill_id for s in available_skills(env.ws.characters[aid], env.ws)]
    return names.index(f"weapon:{W_FIRE}"), names.index(f"weapon:{W_COLD}")


def run_policy(kind, hp, n=60, ac=13, max_steps=200,
               enemy_fights=True, enemy_regen=0, agent_hp=0):
    """kind: correct / wrong / half。回傳 (WR, 平均回報, 平均己末血, 平均步數)。"""
    wins = 0; rets = []; ahps = []; steps_l = []
    rng = random.Random(999)
    for gi in range(n):
        imm = FIRE if gi % 2 == 0 else COLD
        env = make_env(50_000 + gi, imm, hp, ac, 3, 3,
                       enemy_fights=enemy_fights, enemy_regen=enemy_regen,
                       agent_hp=agent_hp)
        aid = env.agent_ids[0]; oid = env.opp_ids[0]
        fi, ci = _weapon_indices(env, aid)
        correct_idx = ci if imm == FIRE else fi     # 非免疫型
        wrong_idx = fi if imm == FIRE else ci
        done = False; steps = 0; ep = 0.0
        while not done and steps < max_steps:
            steps += 1; actor = env.current_agent_id
            # 真實回合結構:有 action 才攻擊一次,攻擊完(action 用掉)就結束回合,
            # 讓敵人取得它的回合(回血/還手)。不再無視資源狂砸(那會一回合秒殺)。
            if actor == aid and env.resources.get("action", 0) > 0:
                if kind == "correct":
                    a = [correct_idx, ENEMY_SLOT_START, 0]
                elif kind == "wrong":
                    a = [wrong_idx, ENEMY_SLOT_START, 0]
                else:  # half
                    a = [rng.choice([fi, ci]), ENEMY_SLOT_START, 0]
            else:
                a = [0, 0, 0]     # 結束回合 / 敵人回合
            _, r, term, trunc, _ = env.step(a); done = term or trunc
            ep += float(r)
        A = env.ws.characters[aid]; W = env.ws.characters[oid]
        wins += (not W.is_alive() and A.is_alive())
        rets.append(ep); ahps.append(max(0., A.hp) / max(1, A.max_hp))
        steps_l.append(steps)
    return wins / n, float(np.mean(rets)), float(np.mean(ahps)), float(np.mean(steps_l))


print("站樁還手假人(StationaryAttacker,不移動但每回合反擊)——平滑梯度:對=進度,錯=挨打。")
print("找『全對→贏(己方存活)、各半→敗(耗死)』:")
print(f"{'敵HP/己HP/AC':>12} | {'correct WR/回報/己血':>24} | {'half WR/回報/己血':>22} | {'wrong WR/己血':>14}")
for ehp, ahp, ac in ((20, 40, 13), (24, 40, 13), (24, 48, 13),
                     (30, 48, 13), (30, 56, 13), (24, 60, 13)):
    cw, cr, cah, cs = run_policy("correct", ehp, ac=ac, agent_hp=ahp)
    hw, hr, hah, hs = run_policy("half", ehp, ac=ac, agent_hp=ahp)
    ww, wr, wah, ws = run_policy("wrong", ehp, ac=ac, agent_hp=ahp)
    ah_disp = ahp if ahp > 0 else 28
    print(f"{ehp:>3}/{ah_disp:>3}/{ac:<3} | {cw:4.0%} {cr:+6.2f} 己血{cah:3.0%} 步{cs:4.1f} | "
          f"{hw:4.0%} {hr:+6.2f} 己血{hah:3.0%} | {ww:4.0%} 己血{wah:3.0%}")
