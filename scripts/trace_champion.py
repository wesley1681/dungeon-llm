"""Causal trace of champion's loss vs expert, on the real eval_v2 env.step path.

For each AGENT sub-action, record:
  - skill chosen (id)
  - whether action_surge was ATTEMPTED (slot selected) and whether it was a
    fresh fire vs a wasted re-select (surge already spent -> not in available)
  - in_reach of nearest enemy at decision time
  - weapon-attack landed damage
  - sub-actions per turn (over-extension)
Compares model vs expert over the same matchups/seeds.

Usage: python scripts/trace_champion.py <arch> <model.pt> [games]
"""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding="utf-8")
import warnings; warnings.filterwarnings("ignore")
import numpy as np
import torch
from trpg.rl.env_v2 import CombatEnvV2, ARCHETYPE_LIST
from trpg.rl.model import (CombatPolicyNet, apply_resource_mask,
                           apply_entity_mask, pick_action)
from trpg.engine.combat_policy import make_archetype_policy
from trpg.rl.action import encode_action
from trpg.engine.skill import available_skills

arch = sys.argv[1]
model_path = sys.argv[2]
games = int(sys.argv[3]) if len(sys.argv) > 3 else 12

net = CombatPolicyNet()
sd = torch.load(model_path, map_location="cpu")
sd = CombatPolicyNet.adapt_state_dict_for_perarch(sd)
_msd = net.state_dict()
sd = {k: v for k, v in sd.items() if not (k in _msd and v.shape != _msd[k].shape)}
net.load_state_dict(sd, strict=False); net.eval()


def reach_m(actor):
    weapon = actor.get_weapon() if actor.weapons else None
    return (weapon.range_normal if weapon else 1.5) or 1.5


def dist(a, b):
    return a.position.distance_to(b.position)


def play(driver, opp, seed, stats):
    env = CombatEnvV2(seed=seed, n_agents=1, n_opps=1)
    obs, _ = env.reset(agent_archs=[arch], opp_archs=[opp], level=5)
    aid = env.agent_ids[0]; oid = env.opp_ids[0]
    expert = make_archetype_policy(arch) if driver == "expert" else None
    surge_fired = 0
    surge_attempt_wasted = 0   # selected surge slot but it's an ERROR/no-op
    sub_actions = 0
    atk_dmg = 0.0
    out_of_reach_actions = 0   # non-move action chosen while not in reach
    done = False
    while not done:
        actor = env.current_agent_id
        ag = env.ws.characters[actor]
        opp_c = env.ws.characters[oid]
        skills = available_skills(ag, env.ws)
        if driver == "model":
            ot = {k: torch.from_numpy(v).unsqueeze(0) for k, v in obs.items()}
            with torch.no_grad():
                el, s, e, g = net(ot)
            s = apply_resource_mask(s, env.resources, env.ws, actor)
            e = apply_entity_mask(e, ot)
            act = list(pick_action(el[0], s[0], e[0], g[0], ws=env.ws, agent_id=actor))
        else:
            dec = expert.decide(actor, ag, env.ws, env.resources, env.ws.combat.round_number)
            act = ([0, 0, 0] if (dec.action is None or dec.fled)
                   else list(encode_action(dec.action, env.ws, actor)))

        is_agent = (actor == aid)
        sid = None
        if 0 <= act[0] < len(skills):
            sid = skills[act[0]].skill_id
        in_reach = dist(ag, opp_c) <= reach_m(ag) + 1e-6 if opp_c.is_alive() else False

        if is_agent and act[0] > 0:
            sub_actions += 1
            if sid == "action_surge":
                surge_fired += 1
                if env.resources.get("action", 0) > 0:
                    stats["surge_action_avail"] += 1
                if not in_reach:
                    stats["surge_oor"] += 1
            is_move = (sid == "move")
            if not is_move and not in_reach and sid not in ("second_wind",):
                out_of_reach_actions += 1
            # categorize
            if is_move:
                stats["c_move"] += 1
            elif sid and sid.startswith("weapon:"):
                if "弓" in sid or "bow" in sid.lower():
                    stats["c_ranged"] += 1
                else:
                    stats["c_melee"] += 1
            elif sid == "action_surge":
                stats["c_surge"] += 1
            elif sid == "second_wind":
                stats["c_heal"] += 1
            else:
                stats["c_other"] += 1

        hp_before = opp_c.hp
        ag_hp_before = env.ws.characters[aid].hp
        dist_before = dist(ag, opp_c) if opp_c.is_alive() else None
        if is_agent and opp_c.is_alive():
            stats["dist_sum"] += dist_before
            stats["dist_n"] += 1
        obs, _, term, trunc, _ = env.step(act)
        if is_agent and sid == "move" and dist_before is not None and opp_c.is_alive():
            dist_after = dist(ag, opp_c)
            if dist_after < dist_before - 0.3:
                stats["move_closing"] += 1
            elif dist_after > dist_before + 0.3:
                stats["move_away"] += 1
            else:
                stats["move_lateral"] += 1
        if is_agent and sid and sid.startswith("weapon:"):
            atk_dmg += max(0.0, hp_before - env.ws.characters[oid].hp)
        # damage taken by agent during this step (opp turn or terrain)
        stats["dmg_taken"] += max(0.0, ag_hp_before - env.ws.characters[aid].hp)
        done = term or trunc

    agent_alive = env.ws.characters[aid].is_alive()
    opp_alive = env.ws.characters[oid].is_alive()
    won = opp_alive is False and agent_alive
    if agent_alive and opp_alive:
        stats["timeout"] += 1          # both alive at end => truncated
        if (env.ws.characters[aid].hp / env.ws.characters[aid].max_hp >
                env.ws.characters[oid].hp / env.ws.characters[oid].max_hp):
            stats["timeout_ahead"] += 1   # we were winning on HP but ran out of rounds
    stats["rounds"] += env.ws.combat.round_number
    stats["games"] += 1
    stats["wins"] += int(won)
    stats["surge"] += surge_fired
    stats["subact"] += sub_actions
    stats["atk_dmg"] += atk_dmg
    stats["oor"] += out_of_reach_actions


for driver in ("model", "expert"):
    stats = dict(games=0, wins=0, surge=0, subact=0, atk_dmg=0.0, oor=0,
                 surge_action_avail=0, surge_oor=0, dmg_taken=0.0,
                 c_move=0, c_melee=0, c_ranged=0, c_surge=0, c_heal=0, c_other=0,
                 timeout=0, timeout_ahead=0, rounds=0,
                 move_closing=0, move_away=0, move_lateral=0, dist_sum=0.0, dist_n=0)
    for opp in ARCHETYPE_LIST:
        base = hash(f"{arch}_{opp}") & 0xFFFFFF
        for i in range(games):
            play(driver, opp, base + i, stats)
    n = stats["games"]
    print(f"{driver:7s}: WR={stats['wins']/n:4.0%}  "
          f"surge/game={stats['surge']/n:4.2f}  "
          f"subactions/game={stats['subact']/n:4.1f}  "
          f"atk_dmg/game={stats['atk_dmg']/n:5.1f}  "
          f"out-of-reach-actions/game={stats['oor']/n:4.2f}\n"
          f"         dmg_taken/game={stats['dmg_taken']/n:5.1f}  "
          f"per-game: move={stats['c_move']/n:4.1f} melee={stats['c_melee']/n:4.1f} "
          f"ranged={stats['c_ranged']/n:4.1f} surge={stats['c_surge']/n:4.2f} "
          f"heal={stats['c_heal']/n:4.2f} other={stats['c_other']/n:4.1f}\n"
          f"         avg_rounds={stats['rounds']/n:4.1f}  "
          f"timeouts={stats['timeout']/n:4.0%}  "
          f"timeout-while-HP-ahead={stats['timeout_ahead']/n:4.0%}\n"
          f"         moves: closing={stats['move_closing']/n:4.2f} away={stats['move_away']/n:4.2f} "
          f"lateral={stats['move_lateral']/n:4.2f}  avg_dist_at_decision={stats['dist_sum']/max(1,stats['dist_n']):4.1f}m")
