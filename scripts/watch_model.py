"""Watch the BC model play one combat episode step by step.

Usage:
    python scripts/watch_model.py [--model PATH] [--agent ARCH] [--opponent ARCH]
"""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import torch
import numpy as np

from trpg.rl.env_v2 import CombatEnvV2, ARCHETYPE_LIST, _AGENT_ID, _OPPONENT_ID
from trpg.rl.model import CombatPolicyNet, apply_entity_mask
from trpg.rl.obs import N_SKILL_SLOTS
from trpg.engine.skill import available_skills


def get_skill_name(env, skill_idx: int) -> str:
    agent = env.ws.characters[_AGENT_ID]
    skills = available_skills(agent, env.ws)
    if skill_idx < len(skills):
        s = skills[skill_idx]
        return f"{s.skill_id}({s.display_name})"
    return f"slot{skill_idx}[invalid]"


def get_entity_name(env, entity_idx: int) -> str:
    from trpg.rl.obs import partition_entities
    allies, enemies = partition_entities(env.ws, _AGENT_ID)
    slots = {0: _AGENT_ID}
    for i, cid in enumerate(allies[:2]):  slots[1+i] = cid
    for i, cid in enumerate(enemies[:3]): slots[3+i] = cid
    cid = slots.get(entity_idx)
    if cid:
        c = env.ws.characters[cid]
        return f"{c.name}(HP {c.hp}/{c.max_hp})"
    return f"slot{entity_idx}[empty]"


def watch(net, agent_arch, opponent_arch, level, seed, device):
    env = CombatEnvV2(seed=seed)
    obs, _ = env.reset(agent_arch=agent_arch, opponent_arch=opponent_arch, level=level)

    agent = env.ws.characters[_AGENT_ID]
    opp   = env.ws.characters[_OPPONENT_ID]
    print(f"\nAgent:    {agent.name} (L{agent.level} {agent_arch}) HP={agent.hp}")
    print(f"Opponent: {opp.name}   (L{opp.level} {opponent_arch}) HP={opp.hp}")
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

        # Get model's top-3 skill probabilities
        from trpg.rl.model import apply_resource_mask
        obs_t = {k: torch.from_numpy(v).unsqueeze(0).to(device) for k, v in obs.items()}
        with torch.no_grad():
            skill_l, entity_l, grid_l = net(obs_t)
        skill_l  = apply_resource_mask(skill_l, env.resources, env.ws, _AGENT_ID)
        entity_l = apply_entity_mask(entity_l, obs_t)

        skill_probs = torch.softmax(skill_l[0], dim=-1).cpu().numpy()
        top3_skills = sorted(enumerate(skill_probs), key=lambda x: -x[1])[:3]

        action = [int(skill_l[0].argmax(-1)), int(entity_l[0].argmax(-1)), int(grid_l.argmax(-1))]
        chosen_skill = action[0]
        chosen_entity = action[1]

        skill_name  = get_skill_name(env, chosen_skill)
        entity_name = get_entity_name(env, chosen_entity)

        # Show top-3 candidates and chosen
        top3_str = ", ".join(
            f"{get_skill_name(env, i).split('(')[0]}({p:.0%})"
            for i, p in top3_skills
        )
        print(f"Step {step:2d} | 考慮: [{top3_str}]")
        print(f"       | 選擇: {skill_name} -> {entity_name}")

        # Execute and show result
        hp_before = opp.hp
        obs, reward, term, trunc, info = env.step(action)
        done = term or trunc

        result = info.get("action_result")
        if result and result.get("type") not in (None, "ERROR"):
            dmg = hp_before - opp.hp
            if dmg > 0:
                print(f"       | 結果: 造成 {dmg} 傷  (對手 HP {opp.hp}/{opp.max_hp})")
            elif result.get("type") == "MOVE":
                print(f"       | 結果: 移動")
            else:
                print(f"       | 結果: {result.get('type')} (未造成傷害)")
        elif result and result.get("type") == "ERROR":
            print(f"       | 結果: ERROR - {result.get('message','')}")

        print(f"       | 資源: action={env.resources.get('action',0)} "
              f"bonus={env.resources.get('bonus_action',0)} "
              f"move={env.resources.get('movement',0):.1f}m"
              f"  [agent HP {agent.hp}/{agent.max_hp}]")

    print("\n" + "=" * 60)
    if not opp.is_alive():
        winner = "Agent 勝"
    elif not agent.is_alive():
        winner = "Opponent 勝"
    else:
        winner = "平局（截斷）"
    print(f"結果: {winner}  (agent HP={agent.hp}, opp HP={opp.hp})")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",    type=str, default="models/bc_v1.pt")
    parser.add_argument("--agent",    type=str, default=None, help=f"archetype: {ARCHETYPE_LIST}")
    parser.add_argument("--opponent", type=str, default=None)
    parser.add_argument("--level",    type=int, default=5)
    parser.add_argument("--seed",     type=int, default=42)
    parser.add_argument("--n",        type=int, default=1, help="幾場")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    net = CombatPolicyNet().to(device)
    net.load_state_dict(torch.load(args.model, map_location=device))
    net.eval()
    print(f"Model: {args.model}  Device: {device}")

    import random
    rng = random.Random(args.seed)
    for i in range(args.n):
        agent_arch    = args.agent    or rng.choice(ARCHETYPE_LIST)
        opponent_arch = args.opponent or rng.choice(ARCHETYPE_LIST)
        print(f"\n===== Episode {i+1}/{args.n} =====")
        watch(net, agent_arch, opponent_arch, args.level, args.seed + i, device)


if __name__ == "__main__":
    main()
