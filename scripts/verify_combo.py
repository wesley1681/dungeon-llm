"""VERIFY (not assume) why the model loses on a class.

Hypothesis under test: "the model loses champion because it can't do the
action_surge combo, so it under-damages."

For a given archetype, play N games each driven by:
  - the MODEL (net, via env.step)
  - the EXPERT through env.step (same hampered path the model uses)
  - the EXPERT via the direct full-turn loop (its TRUE ceiling)
and measure, per game: win, damage DEALT to opp, damage TAKEN, rounds, and how
often the key combo skill (action_surge / the archetype's signature) fired.

If the model deals MUCH less damage AND fires the combo far less -> combo wall
confirmed. If the model deals SIMILAR damage but dies more -> it is NOT combo
(positioning / defense / luck), and the "can't learn combo" story is wrong.
"""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import torch
from trpg.rl.env_v2 import CombatEnvV2, ARCHETYPE_LIST, _MAX_SUB_ACTIONS_PER_TURN
from trpg.rl.model import (CombatPolicyNet, apply_resource_mask,
                           apply_entity_mask, pick_action)
from trpg.engine.combat_policy import make_archetype_policy
from trpg.rl.action import encode_action, decode_action
from trpg.engine.combat import (execute_action, consume_resources, MOVE_BUDGET_M,
                                 tick_terrain_damage, tick_aura_damage)
from trpg.engine.status import tick_status_effects

arch = sys.argv[1] if len(sys.argv) > 1 else "champion"
model_path = sys.argv[2] if len(sys.argv) > 2 else "models/ppo_v19_film/ppo_final.pt"
games = int(sys.argv[3]) if len(sys.argv) > 3 else 12
COMBO = {"champion": "action_surge", "battle_master": "action_surge",
         "vengeance": "vow_of_enmity", "devotion": "sacred_weapon"}.get(arch, "action_surge")

net = CombatPolicyNet()
sd = torch.load(model_path, map_location="cpu")
sd = CombatPolicyNet.adapt_state_dict_for_perarch(sd)
net.load_state_dict(sd, strict=False)
net.eval()


def play_envstep(driver, opp, seed):
    """driver: 'model' or 'expert'. Returns dict of metrics."""
    env = CombatEnvV2(seed=seed, n_agents=1, n_opps=1)
    obs, _ = env.reset(agent_archs=[arch], opp_archs=[opp], level=5)
    aid = env.agent_ids[0]; oid = env.opp_ids[0]
    opp_max = env.ws.characters[oid].max_hp
    ag_max = env.ws.characters[aid].max_hp
    expert = make_archetype_policy(arch) if driver == "expert" else None
    combo_fires = 0
    done = False
    noext = os.environ.get("NOEXT") == "1"
    while not done:
        actor = env.current_agent_id
        # Anti-over-extension: once the agent has spent its action this turn,
        # END instead of padding the turn with moves (tests whether the model's
        # under-ending / over-moving after its combo is what loses melee fights).
        if (driver == "model" and noext and actor == aid
                and env.resources.get("action", 0) <= 0):
            obs, _, term, trunc, _ = env.step([0, 0, 0])
            done = term or trunc
            continue
        if driver == "model":
            obs_t = {k: torch.from_numpy(v).unsqueeze(0) for k, v in obs.items()}
            with torch.no_grad():
                el, s, e, g = net(obs_t)
            s = apply_resource_mask(s, env.resources, env.ws, actor)
            e = apply_entity_mask(e, obs_t)
            act = list(pick_action(el[0], s[0], e[0], g[0], ws=env.ws, agent_id=actor))
            if actor == aid:
                from trpg.engine.skill import available_skills
                sk = available_skills(env.ws.characters[actor], env.ws)
                if 0 <= act[0] < len(sk) and sk[act[0]].skill_id == COMBO:
                    combo_fires += 1
        else:
            a = env.ws.characters[actor]
            dec = expert.decide(actor, a, env.ws, env.resources, env.ws.combat.round_number)
            if dec.action is None or dec.fled:
                act = [0, 0, 0]
            else:
                if dec.action.get("skill_id") == COMBO:
                    combo_fires += 1
                act = list(encode_action(dec.action, env.ws, actor))
        obs, _, term, trunc, _ = env.step(act)
        done = term or trunc
    opp_hp = max(0, env.ws.characters[oid].hp)
    ag_hp = max(0, env.ws.characters[aid].hp)
    won = (not env.ws.characters[oid].is_alive()) and env.ws.characters[aid].is_alive()
    return {"win": won, "dmg_dealt": opp_max - opp_hp, "dmg_dealt_frac": (opp_max - opp_hp)/opp_max,
            "dmg_taken_frac": (ag_max - ag_hp)/ag_max, "rounds": env.ws.combat.round_number,
            "combo": combo_fires}


def summarize(label, rows):
    n = len(rows)
    print(f"  {label:22s} win={np.mean([r['win'] for r in rows]):5.0%}  "
          f"dmg_dealt={np.mean([r['dmg_dealt_frac'] for r in rows]):5.0%}  "
          f"dmg_taken={np.mean([r['dmg_taken_frac'] for r in rows]):5.0%}  "
          f"rounds={np.mean([r['rounds'] for r in rows]):4.1f}  "
          f"{COMBO}/game={np.mean([r['combo'] for r in rows]):4.2f}")


archs = list(ARCHETYPE_LIST)
model_rows, expert_rows = [], []
for opp in archs:
    base = hash(f"{arch}_{opp}") & 0xFFFFFF
    for i in range(games):
        model_rows.append(play_envstep("model", opp, base + i))
        expert_rows.append(play_envstep("expert", opp, base + i))

print(f"=== {arch}: model vs expert (both via env.step), combo skill = {COMBO} ===")
print(f"    {len(model_rows)} games each, identical seeds/opponents")
summarize("MODEL", model_rows)
summarize("EXPERT (env.step)", expert_rows)
print()
md = np.mean([r['dmg_dealt_frac'] for r in model_rows])
ed = np.mean([r['dmg_dealt_frac'] for r in expert_rows])
mc = np.mean([r['combo'] for r in model_rows])
ec = np.mean([r['combo'] for r in expert_rows])
print(f"VERDICT inputs: model deals {md:.0%} vs expert {ed:.0%} of opp HP;  "
      f"combo fires model {mc:.2f} vs expert {ec:.2f}/game")
if md < ed - 0.12 and mc < ec - 0.3:
    print("=> consistent with COMBO/damage deficit")
elif abs(md - ed) < 0.12:
    print("=> model deals SIMILAR damage -> loss is NOT a damage/combo problem")
else:
    print("=> mixed; inspect dmg_taken / rounds")
