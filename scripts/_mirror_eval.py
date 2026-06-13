"""Mirror-match eval: model-vs-expert same archetype.

50% means perfect imitation. <50% means model is worse than expert at the
mirror; >50% would mean model found edge cases vs expert.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding="utf-8")
import argparse
import torch
from trpg.engine.combat_policy import make_archetype_policy
from trpg.rl.env_v2 import CombatEnvV2, ARCHETYPE_LIST
from trpg.rl.model import CombatPolicyNet, apply_resource_mask, apply_entity_mask, pick_action


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--n", type=int, default=30)
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    net = CombatPolicyNet().to(device)
    net.load_state_dict(torch.load(args.model, map_location=device, weights_only=True))
    net.eval()
    print(f"Model: {args.model}")
    print(f"\n=== Mirror match: model vs expert (same archetype) ===")
    print(f"{'archetype':<20} {'W':>3} {'L':>3} {'T':>3}  win%")
    for arch in ARCHETYPE_LIST:
        w = l = t = 0
        for ep in range(args.n):
            env = CombatEnvV2(seed=ep, n_agents=1, n_opps=1)
            obs, _ = env.reset(agent_archs=[arch], opp_archs=[arch])
            agent_id = env.agent_ids[0]
            opp_id = env.opp_ids[0]
            done = False
            while not done:
                obs_t = {k: torch.from_numpy(v).unsqueeze(0).to(device) for k, v in obs.items()}
                with torch.no_grad():
                    end_l, s, e, g = net(obs_t)
                s = apply_resource_mask(s, env.resources, env.ws, agent_id)
                e = apply_entity_mask(e, obs_t)
                action = list(pick_action(end_l[0], s[0], e[0], g[0], ws=env.ws, agent_id=agent_id))
                obs, _, term, trunc, _ = env.step(action)
                done = term or trunc
            if not env.ws.characters[opp_id].is_alive():
                w += 1
            elif not env.ws.characters[agent_id].is_alive():
                l += 1
            else:
                t += 1
        wr = w / args.n * 100
        print(f"{arch:<20} {w:>3} {l:>3} {t:>3}  {wr:5.1f}%")


if __name__ == "__main__":
    main()
