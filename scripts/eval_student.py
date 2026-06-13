"""Official-style goal table for a DISTILLED student: one net plays ALL 12
archetypes vs every expert opponent; same-run expert baseline side by side.

Mirrors eval_goal.py methodology (same seeds/levels/2-sigma band) so numbers
are comparable with goal_1v1_* result files.

Usage: python scripts/eval_student.py <ckpt> [games_per_opp] [--blind]
"""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding="utf-8")
import warnings; warnings.filterwarnings("ignore")
import numpy as np
import torch

from trpg.rl.env_v2 import CombatEnvV2, ARCHETYPE_LIST
from trpg.rl.model import apply_resource_mask, apply_entity_mask, pick_action
from trpg.engine.combat_policy import make_archetype_policy
from trpg.rl.action import encode_action

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from distill_routed import load_student, blind_self_identity_t


def play(arch, net, expert, opp, seed, blind):
    env = CombatEnvV2(seed=seed, n_agents=1, n_opps=1)
    obs, _ = env.reset(agent_archs=[arch], opp_archs=[opp], level=5)
    aid = env.agent_ids[0]; oid = env.opp_ids[0]
    done = False
    while not done:
        actor = env.current_agent_id
        if net is not None:
            ot = {k: torch.from_numpy(v).unsqueeze(0) for k, v in obs.items()}
            if blind:
                ot = blind_self_identity_t(ot)
            with torch.no_grad():
                el, s, e, g = net(ot)
            s = apply_resource_mask(s, env.resources, env.ws, actor)
            e = apply_entity_mask(e, ot, env.ws, actor)
            act = list(pick_action(el[0], s[0], e[0], g[0],
                                   ws=env.ws, agent_id=actor))
        else:
            a = env.ws.characters[actor]
            dec = expert.decide(actor, a, env.ws, env.resources,
                                env.ws.combat.round_number)
            act = ([0, 0, 0] if (dec.action is None or dec.fled)
                   else list(encode_action(dec.action, env.ws, actor)))
        obs, _, term, trunc, _ = env.step(act)
        done = term or trunc
    return (not env.ws.characters[oid].is_alive()
            and env.ws.characters[aid].is_alive())


def main():
    ckpt = sys.argv[1]
    games = int(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[2].isdigit() else 10
    blind = "--blind" in sys.argv
    net = load_student(ckpt)
    print(f"student={ckpt} blind={blind} games/opp={games}")
    print(f"{'archetype':18s} {'model':>6s} {'expert':>7s} {'diff':>6s}  verdict")
    print("-" * 52)
    all_ok = True
    diffs = []
    n_per_arch = games * len(ARCHETYPE_LIST)
    band = 2 * 100 * (0.25 / n_per_arch) ** 0.5
    for arch in ARCHETYPE_LIST:
        expert = make_archetype_policy(arch)
        mw = ew = 0
        for opp in ARCHETYPE_LIST:
            base = hash(f"{arch}_{opp}") & 0xFFFFFF
            for i in range(games):
                if play(arch, net, None, opp, base + i, blind):
                    mw += 1
                if play(arch, None, expert, opp, base + i, False):
                    ew += 1
        m, e = mw / n_per_arch, ew / n_per_arch
        d = (m - e) * 100
        ok = d >= -band
        all_ok &= ok
        diffs.append(d)
        print(f"{arch:18s} {m:>6.0%} {e:>7.0%} {d:>+5.0f}%  "
              f"{'OK' if ok else 'SHORT'}")
    print("-" * 52)
    print(f"2-sigma tie band at n={n_per_arch}: +/-{band:.0f}%")
    print(f"mean diff {np.mean(diffs):+.1f}%")
    print("GOAL MET" if all_ok else "NOT MET")


if __name__ == "__main__":
    main()
