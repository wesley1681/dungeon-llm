import numpy as np
import pytest
from trpg.rl.env_v2 import CombatEnvV2, ARCHETYPE_LIST
from trpg.rl.obs import OBS_KEYS, N_SKILL_SLOTS, N_ENTITY_SLOTS, ENTITY_DIM, N_GRID
from trpg.engine.skill import SKILL_FEATURE_DIM


def test_archetype_list_has_12_entries():
    assert len(ARCHETYPE_LIST) == 12


def test_reset_returns_dict_obs():
    env = CombatEnvV2(seed=0, n_agents=1, n_opps=1)
    obs, info = env.reset()
    for k in OBS_KEYS:
        assert k in obs


def test_reset_obs_shapes():
    env = CombatEnvV2(seed=0, n_agents=1, n_opps=1)
    obs, _ = env.reset()
    assert obs["skills"].shape == (N_SKILL_SLOTS, SKILL_FEATURE_DIM)
    assert obs["skill_mask"].shape == (N_SKILL_SLOTS,)
    assert obs["entities"].shape == (N_ENTITY_SLOTS, ENTITY_DIM)
    assert obs["resources"].shape == (4,)
    assert obs["terrain"].shape == (N_GRID, N_GRID)


def test_reset_with_fixed_archetype():
    env = CombatEnvV2(seed=0, n_agents=1, n_opps=1)
    env.reset(agent_archs=["battle_master"], opp_archs=["evocation"], level=5)
    assert env.ws.characters[env.agent_ids[0]].archetype_id == "battle_master"
    assert env.ws.characters[env.opp_ids[0]].archetype_id == "evocation"


def test_reset_opp_level_creates_asymmetric_levels():
    env = CombatEnvV2(seed=0, n_agents=2, n_opps=2)
    env.reset(agent_archs=["champion", "life"], opp_archs=["champion", "life"],
              level=5, opp_level=7)
    for aid in env.agent_ids:
        assert env.ws.characters[aid].level == 5
    for oid in env.opp_ids:
        assert env.ws.characters[oid].level == 7
    # higher level -> bigger HP pool (same archetype)
    assert (env.ws.characters[env.opp_ids[0]].max_hp
            > env.ws.characters[env.agent_ids[0]].max_hp)


def test_reset_opp_level_defaults_to_symmetric():
    env = CombatEnvV2(seed=0, n_agents=1, n_opps=1)
    env.reset(agent_archs=["champion"], opp_archs=["champion"], level=4)
    assert env.ws.characters[env.agent_ids[0]].level == 4
    assert env.ws.characters[env.opp_ids[0]].level == 4


def test_reset_seed_is_reproducible():
    env1 = CombatEnvV2(seed=42, n_agents=1, n_opps=1)
    env2 = CombatEnvV2(seed=42, n_agents=1, n_opps=1)
    obs1, _ = env1.reset()
    obs2, _ = env2.reset()
    np.testing.assert_array_equal(obs1["entities"], obs2["entities"])


def test_step_end_turn_returns_obs_reward_done():
    env = CombatEnvV2(seed=0, n_agents=1, n_opps=1)
    env.reset(agent_archs=["champion"], opp_archs=["champion"], level=5)
    obs, reward, terminated, truncated, info = env.step([0, 0, 0])
    assert isinstance(obs, dict)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)


def test_step_attack_reduces_enemy_hp_when_in_range():
    env = CombatEnvV2(seed=0, n_agents=1, n_opps=1)
    env.reset(agent_archs=["champion"], opp_archs=["champion"], level=5)
    agent_id = env.agent_ids[0]
    opp_id = env.opp_ids[0]
    # Force positions: agent and opponent adjacent
    from trpg.engine.vec2 import Vec2
    env.ws.characters[opp_id].position = Vec2(
        env.ws.characters[agent_id].position.x + 1.5,
        env.ws.characters[agent_id].position.y,
    )
    enemy_hp_before = env.ws.characters[opp_id].hp
    # Find the weapon-attack slot
    from trpg.engine.skill import available_skills
    skills = available_skills(env.ws.characters[agent_id], env.ws)
    weapon_idx = next(i for i, s in enumerate(skills) if s.skill_id.startswith("weapon:"))
    # Multiple steps: agent attacks (some may miss)
    for _ in range(20):
        obs, reward, term, trunc, info = env.step([weapon_idx, 3, 0])
        if env.ws.characters[opp_id].hp < enemy_hp_before:
            break
        if term or trunc:
            break
    # Either dealt damage or opponent killed us
    assert (env.ws.characters[opp_id].hp < enemy_hp_before
            or not env.ws.characters[agent_id].is_alive())


def test_step_terminates_when_enemy_dies():
    env = CombatEnvV2(seed=0, n_agents=1, n_opps=1)
    env.reset(agent_archs=["champion"], opp_archs=["champion"], level=5)
    agent_id = env.agent_ids[0]
    opp_id = env.opp_ids[0]
    env.ws.characters[opp_id].hp = 1   # one-shot kill
    from trpg.engine.vec2 import Vec2
    env.ws.characters[opp_id].position = Vec2(
        env.ws.characters[agent_id].position.x + 1.5,
        env.ws.characters[agent_id].position.y,
    )
    from trpg.engine.skill import available_skills
    skills = available_skills(env.ws.characters[agent_id], env.ws)
    weapon_idx = next(i for i, s in enumerate(skills) if s.skill_id.startswith("weapon:"))
    # May take a few tries to land a hit
    term = False
    for _ in range(20):
        obs, reward, term, trunc, info = env.step([weapon_idx, 3, 0])
        if term:
            break
    assert term, "enemy at 1 HP should die within 20 attacks"


def test_opp_turn_does_not_spin_on_engine_error():
    # BUG（play_gui seed 抽查）：對手席神經策略選了「遮罩放行但引擎必拒」的動作
    # （tarrasque_swallow 需要 restrained 目標，目標沒被束縛）→ execute_action 回
    # ERROR。舊 _run_opponent_turn 在 ERROR 時不扣資源、不推進 → 狀態沒變、決定性
    # 策略每次重選同一非法招 → 空轉到 _MAX_SUB_ACTIONS_PER_TURN(5)，整回合報廢。
    # agent 席 step() 早就對稱處理（ERROR→action=0），對手席漏了。此測試盯住「不
    # 空轉」：決定性非法策略被叫的次數必須有界（≤2），不是 5。
    from types import SimpleNamespace
    env = CombatEnvV2(seed=1, n_agents=1, n_opps=1)
    env.reset(agent_archs=["battle_master"], opp_archs=["battle_master"],
              level=5, opp_level=5, layout="open")
    opp_id = env.opp_ids[0]
    agent_id = env.agent_ids[0]
    assert not env.ws.characters[agent_id].has_status("restrained")

    class _AlwaysIllegalSwallow:
        def __init__(self, target_id):
            self.calls = 0
            self.target_id = target_id

        def decide(self, oid, opp, ws, resources, round_num):
            self.calls += 1
            action = {"type": "ATTACK", "attacker": oid, "target": self.target_id,
                      "weapon": "泰拉斯克巨顎", "n_attacks": 1, "skill_id": "tarrasque_swallow",
                      "requires_target_status": "restrained"}
            return SimpleNamespace(action=action, fled=False, ended=False)

    stub = _AlwaysIllegalSwallow(agent_id)
    env._opp_policies[opp_id] = stub
    env._run_opponent_turn(opp_id)
    assert stub.calls <= 2, (
        f"對手席在決定性非法動作上空轉了 {stub.calls} 次"
        f"（上限 5）——ERROR 沒有推進回合")


def test_step_incapacitated_agent_forced_end_no_error():
    # 規則：失能（paralyzed 等 INCAPACITATING_STATUSES）不能行動——env 必須強制
    # END，不執行策略給的動作（否則引擎彈 ERROR＝dead_action 噪音＋垃圾轉移）。
    # 對手席 _run_opponent_turn 早有這條跳過；agent 席的 _advance_to_next_agent
    # 護欄迴圈用盡兜底時會把控制權交給仍失能的 agent → step 端需要對稱守門。
    from trpg.engine.status import Paralyzed
    from trpg.engine.skill import available_skills
    env = CombatEnvV2(seed=3, n_agents=1, n_opps=1)
    env.reset(agent_archs=["champion"], opp_archs=["battle_master"],
              level=5, opp_level=5)
    aid = env.agent_ids[0]
    ch = env.ws.characters[aid]
    ch.add_status(Paralyzed())
    assert ch.is_incapacitated()
    # 攻擊動作三元組（skill=武器攻擊, entity=最近敵, grid=0）
    sks = available_skills(ch, env.ws)
    atk = next(i for i, s in enumerate(sks)
               if s.skill_id != "move" and s.features.expected_damage > 0)
    from trpg.rl.obs import ENEMY_SLOT_START
    obs, r, term, trunc, info = env.step([atk, ENEMY_SLOT_START, 0])
    res = info.get("action_result")
    assert not (isinstance(res, dict) and res.get("type") == "ERROR"), \
        f"失能 agent 的動作不該執行到引擎: {res}"
