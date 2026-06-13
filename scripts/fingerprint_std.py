"""Deterministic fingerprint of standard expert-vs-expert fights.

Plays a fixed set of (arch, opp, seed) episodes with scripted experts on BOTH
sides through env_v2 (the official-gate engine path) and prints one line per
game: winner flag, rounds, final HPs — then a digest. Bit-identical output
across two source trees proves the standard-1v1 engine path (dice stream
included) is unchanged. No monsters, no models, no hash() salting.

Usage: python scripts/fingerprint_std.py > out.txt
"""
from __future__ import annotations
import sys, os, hashlib
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding="utf-8")
import warnings; warnings.filterwarnings("ignore")

from trpg.rl.env_v2 import CombatEnvV2
from trpg.rl.action import encode_action
from trpg.engine.combat_policy import make_archetype_policy

PAIRS = [
    ("battle_master", "champion"), ("battle_master", "evocation"),
    ("champion", "life"), ("evocation", "berserker"),
    ("assassin", "war"), ("devotion", "arcane_trickster"),
    ("totem_bear", "divination"), ("vengeance", "battle_master"),
]
SEEDS = range(25)

lines = []
for arch, opp in PAIRS:
    expert = make_archetype_policy(arch)
    for seed in SEEDS:
        env = CombatEnvV2(seed=seed, n_agents=1, n_opps=1)
        env.reset(agent_archs=[arch], opp_archs=[opp], level=6)
        aid, oid = env.agent_ids[0], env.opp_ids[0]
        done = False
        steps = 0
        while not done and steps < 300:
            actor = env.current_agent_id
            a = env.ws.characters[actor]
            dec = expert.decide(actor, a, env.ws, env.resources,
                                env.ws.combat.round_number)
            act = ([0, 0, 0] if (dec.action is None or dec.fled)
                   else list(encode_action(dec.action, env.ws, actor)))
            _, _, term, trunc, _ = env.step(act)
            done = term or trunc
            steps += 1
        ag, op = env.ws.characters[aid], env.ws.characters[oid]
        lines.append(f"{arch}|{opp}|{seed}|{int(ag.is_alive())}|"
                     f"{int(op.is_alive())}|{ag.hp}|{op.hp}|"
                     f"{env.ws.combat.round_number}|{steps}")

blob = "\n".join(lines)
print(blob)
print("DIGEST:", hashlib.sha256(blob.encode()).hexdigest())
