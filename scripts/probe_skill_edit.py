"""技能彈性修改 → 模型能否讀出並改變行為？（資料驅動可讀性實驗）

問題：技能數值是否可彈性修改、修改是否進 obs、模型是否相應改變行為。
做法：拿 hill_giant（雙招：巨棒近戰 3d8 + 擲岩遠程 3d10）——冠軍模型 baseline
狂用擲岩 kite、幾乎不用巨棒。把「巨棒」傷害骰一行字 3d8→12d8（一個字串參數），
看 (a) SkillFeatures.expected_damage（=模型 obs 看到的招威力）是否變、(b) 模型行為
是否從擲岩轉向巨棒。轉了＝模型讀招的數值選招（資料驅動有效、kite 是 EV 計算非
死記）；沒轉＝模型死記用遠程。

附帶第二例（用戶問的「擲石更痛」）：擲岩 3d10→8d10。

Usage: python scripts/probe_skill_edit.py
"""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding="utf-8")
import warnings; warnings.filterwarnings("ignore")

from collections import Counter

from trpg.scenarios.monsters import MONSTER_DEFS, register_monsters
register_monsters()

from trpg.engine.items import WEAPON_DEFS
from trpg.rl.env_v2 import CombatEnvV2
from trpg.engine.skill import available_skills

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from distill_routed import load_student
from eval_monster_actor import run_episode, PARTY

CID = "hill_giant"
NAT = MONSTER_DEFS[CID].natural_level     # 5
PARTY_LVL = 3                              # party3 equiv floor


def show_skills(net=None):
    """印 hill_giant 兩把武器在 obs 裡的威力（SkillFeatures.expected_damage）。"""
    env = CombatEnvV2(seed=1, n_agents=1, n_opps=1)
    env.reset(agent_archs=[CID], opp_archs=["commoner"], level=NAT, opp_level=1)
    aid = env.agent_ids[0]
    rows = []
    for s in available_skills(env.ws.characters[aid], env.ws):
        if s.skill_id.startswith("weapon:"):
            rows.append((s.skill_id, s.features.expected_damage,
                         s.features.range_m))
    for sid, ev, rng in rows:
        print(f"    {sid:<16} obs威力(EV)={ev:5.1f}  射程={rng:.1f}m")


def boss_mix(net, games=20):
    usage = Counter(); turns = 0; wins = 0
    for k in range(games):
        key = f"skedit|{CID}|{k}"
        won, u, t, f = run_episode(CID, list(PARTY), NAT, PARTY_LVL, key, net)
        usage += u; turns += t; wins += int(won)
    melee = usage.get("weapon:巨人巨棒", 0) / max(1, turns)
    ranged = usage.get("weapon:擲岩", 0) / max(1, turns)
    return melee, ranged, wins / games


def main():
    net = load_student(f"models/mon_actor1/ma_u0032.pt"); net.eval()
    base_club = WEAPON_DEFS["巨人巨棒"].damage_dice
    base_rock = WEAPON_DEFS["擲岩"].damage_dice

    print(f"=== BASELINE（巨棒 {base_club} / 擲岩 {base_rock}）===")
    show_skills()
    m, r, w = boss_mix(net)
    print(f"  模型行為：巨棒×{m:.2f}/回合  擲岩×{r:.2f}/回合  勝率={w:.0%}\n")

    print(f"=== 實驗1：巨棒 {base_club}→12d8（把模型不用的近戰招改超強）===")
    WEAPON_DEFS["巨人巨棒"].damage_dice = "12d8"
    show_skills()
    m, r, w = boss_mix(net)
    print(f"  模型行為：巨棒×{m:.2f}/回合  擲岩×{r:.2f}/回合  勝率={w:.0%}")
    print(f"  → 巨棒使用率 {'上升＝模型讀數值轉招' if m > 0.15 else '沒上升'}\n")
    WEAPON_DEFS["巨人巨棒"].damage_dice = base_club

    print(f"=== 實驗2（你的「擲石更痛」）：擲岩 {base_rock}→8d10 ===")
    WEAPON_DEFS["擲岩"].damage_dice = "8d10"
    show_skills()
    m, r, w = boss_mix(net)
    print(f"  模型行為：巨棒×{m:.2f}/回合  擲岩×{r:.2f}/回合  勝率={w:.0%}")
    WEAPON_DEFS["擲岩"].damage_dice = base_rock


if __name__ == "__main__":
    main()
