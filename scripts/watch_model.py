"""Watch the BC model play one combat episode step by step.

Usage:
    python scripts/watch_model.py [--model PATH] [--agent ARCH] [--opponent ARCH]
"""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding="utf-8")

import argparse
import torch
import numpy as np

from trpg.rl.env_v2 import CombatEnvV2, ARCHETYPE_LIST
from trpg.rl.model import CombatPolicyNet, apply_entity_mask
from trpg.rl.obs import N_SKILL_SLOTS
from trpg.engine.skill import available_skills


def get_skill_name(env, actor_id: str, skill_idx: int) -> str:
    agent = env.ws.characters[actor_id]
    skills = available_skills(agent, env.ws)
    if skill_idx < len(skills):
        s = skills[skill_idx]
        return f"{s.skill_id}({s.display_name})"
    return f"slot{skill_idx}[invalid]"


def get_entity_name(env, actor_id: str, entity_idx: int) -> str:
    from trpg.rl.obs import partition_entities
    allies, enemies = partition_entities(env.ws, actor_id)
    slots = {0: actor_id}
    for i, cid in enumerate(allies[:2]):  slots[1+i] = cid
    for i, cid in enumerate(enemies[:3]): slots[3+i] = cid
    cid = slots.get(entity_idx)
    if cid:
        c = env.ws.characters[cid]
        return f"{c.name}(HP {c.hp}/{c.max_hp})"
    return f"slot{entity_idx}[empty]"


def _hp_snapshot(env) -> str:
    """One-line summary of all combatants' HP — '[ally] cid HP/max'."""
    parts = []
    for cid in env.agent_ids:
        c = env.ws.characters[cid]
        parts.append(f"[A]{cid}:{c.hp}/{c.max_hp}")
    for cid in env.opp_ids:
        c = env.ws.characters[cid]
        parts.append(f"[O]{cid}:{c.hp}/{c.max_hp}")
    return "  ".join(parts)


def watch(net, agent_archs, opp_archs, level, seed, device,
          n_agents: int, n_opps: int):
    env = CombatEnvV2(seed=seed, n_agents=n_agents, n_opps=n_opps)
    obs, _ = env.reset(
        agent_archs=agent_archs, opp_archs=opp_archs, level=level,
    )

    print(f"\n隊伍：")
    for cid, arch in zip(env.agent_ids, env.agent_archs):
        c = env.ws.characters[cid]
        print(f"  [A] {c.name} (L{c.level} {arch}) HP={c.hp}")
    for cid, arch in zip(env.opp_ids, env.opp_archs):
        c = env.ws.characters[cid]
        print(f"  [O] {c.name} (L{c.level} {arch}) HP={c.hp}")
    print("=" * 60)

    step = 0
    done = False
    prev_round = 0

    while not done:
        step += 1
        round_num = env.ws.combat.round_number if env.ws.combat else 0
        if round_num != prev_round:
            print(f"\n--- Round {round_num} ---")
            prev_round = round_num

        actor_id = env.current_agent_id
        actor = env.ws.characters[actor_id]
        actor_arch = (env.agent_archs[env.agent_ids.index(actor_id)]
                      if actor_id in env.agent_ids
                      else env.opp_archs[env.opp_ids.index(actor_id)])

        # Get model's top-3 skill probabilities
        from trpg.rl.model import apply_resource_mask, pick_action
        obs_t = {k: torch.from_numpy(v).unsqueeze(0).to(device) for k, v in obs.items()}
        with torch.no_grad():
            end_l, skill_l, entity_l, grid_l = net(obs_t)
        skill_l  = apply_resource_mask(skill_l, env.resources, env.ws, actor_id)
        entity_l = apply_entity_mask(entity_l, obs_t)

        skill_probs = torch.softmax(skill_l[0], dim=-1).cpu().numpy()
        top3_skills = sorted(enumerate(skill_probs), key=lambda x: -x[1])[:3]

        action = list(pick_action(end_l[0], skill_l[0], entity_l[0], grid_l[0],
                                  ws=env.ws, agent_id=actor_id))
        chosen_skill = action[0]
        chosen_entity = action[1]

        skill_name  = get_skill_name(env, actor_id, chosen_skill)
        entity_name = get_entity_name(env, actor_id, chosen_entity)

        top3_str = ", ".join(
            f"{get_skill_name(env, actor_id, i).split('(')[0]}({p:.0%})"
            for i, p in top3_skills
        )
        end_prob = torch.sigmoid(end_l[0]).item()
        print(f"Step {step:2d} | {actor_id} {actor.name}({actor_arch}) "
              f"end_p={end_prob:.0%}")
        print(f"       | 考慮: [{top3_str}]")
        print(f"       | 選擇: {skill_name} -> {entity_name}")

        # HP-before snapshot for damage detection
        hp_before = {cid: env.ws.characters[cid].hp
                     for cid in env.agent_ids + env.opp_ids}
        obs, reward, term, trunc, info = env.step(action)
        done = term or trunc

        result = info.get("action_result")
        if result and result.get("type") not in (None, "ERROR"):
            # Find largest HP drop among any combatant
            dmg_msgs = []
            for cid, prev_hp in hp_before.items():
                drop = prev_hp - env.ws.characters[cid].hp
                if drop > 0:
                    dmg_msgs.append(f"{env.ws.characters[cid].name} -{drop}")
            if dmg_msgs:
                print(f"       | 結果: {result.get('type')}  傷害: {', '.join(dmg_msgs)}")
            elif result.get("type") == "MOVE":
                print(f"       | 結果: 移動")
            else:
                print(f"       | 結果: {result.get('type')}")
        elif result and result.get("type") == "ERROR":
            print(f"       | 結果: ERROR - {result.get('message','')}")

        print(f"       | HP: {_hp_snapshot(env)}")

    print("\n" + "=" * 60)
    opps_dead   = all(not env.ws.characters[c].is_alive() for c in env.opp_ids)
    agents_dead = all(not env.ws.characters[c].is_alive() for c in env.agent_ids)
    if opps_dead:
        winner = "我方勝"
    elif agents_dead:
        winner = "對方勝"
    else:
        winner = "平局（截斷）"
    print(f"結果: {winner}")
    print(f"最終: {_hp_snapshot(env)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",    type=str, default="models/bc_v1.pt")
    parser.add_argument("--agent",    type=str, default=None,
                        help=f"comma-separated archetypes (one per agent slot). "
                             f"Available: {ARCHETYPE_LIST}")
    parser.add_argument("--opponent", type=str, default=None,
                        help="comma-separated archetypes (one per opp slot)")
    parser.add_argument("--level",    type=int, default=5)
    parser.add_argument("--seed",     type=int, default=42)
    parser.add_argument("--n",        type=int, default=1, help="幾場")
    parser.add_argument("--n_agents", type=int, default=1)
    parser.add_argument("--n_opps",   type=int, default=1)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    net = CombatPolicyNet().to(device)
    net.load_state_dict(torch.load(args.model, map_location=device))
    net.eval()
    print(f"Model: {args.model}  Device: {device}")

    agent_archs = args.agent.split(",")    if args.agent    else None
    opp_archs   = args.opponent.split(",") if args.opponent else None

    for i in range(args.n):
        print(f"\n===== Episode {i+1}/{args.n} =====")
        watch(net, agent_archs, opp_archs, args.level, args.seed + i, device,
              n_agents=args.n_agents, n_opps=args.n_opps)


if __name__ == "__main__":
    main()
