"""Zero-shot generalization probe: a distilled student plays the 3 chimeras.

The chimeras (scripts/chimera_defs.py) are skill/trait bundles no teacher ever
played — their archetype one-hot is all-zero (the monster condition). This
script measures BEHAVIOR, not just WR:

  - WR vs each of the 12 scripted expert opponents
  - opener distribution (first chosen skill per game)
  - per-skill usage per game
  - heal timing (self HP fraction when a heal is cast)
  - stance profile (mean distance to nearest living enemy at decision time)
  - wasted moves (move that doesn't change distance-to-nearest-enemy)
  - sample action traces (first N games verbatim)

Usage:
  python scripts/eval_chimera.py <ckpt> [games_per_opp] [--blind] [--out FILE]
"""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding="utf-8")
import warnings; warnings.filterwarnings("ignore")

from collections import Counter, defaultdict
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from chimera_defs import register_chimeras, CHIMERA_IDS

from trpg.rl.env_v2 import CombatEnvV2, ARCHETYPE_LIST
from trpg.rl.model import apply_resource_mask, apply_entity_mask, pick_action
from trpg.engine.skill import available_skills
from distill_routed import load_student, blind_self_identity_t
from eval_routed import stable_seed

HEAL_IDS = {"cure_wounds", "healing_word_life", "lay_on_hands_ability",
            "mass_cure_wounds_life"}


def _nearest_enemy_dist(ws, aid, enemy_ids):
    me = ws.characters[aid].position
    ds = [me.distance_to(ws.characters[e].position)
          for e in enemy_ids if ws.characters[e].is_alive()]
    return min(ds) if ds else None


def play_one(net, chim, opp, seed, blind, stats, trace=None):
    env = CombatEnvV2(seed=seed, n_agents=1, n_opps=1)
    obs, _ = env.reset(agent_archs=[chim], opp_archs=[opp], level=5)
    aid = env.agent_ids[0]; oid = env.opp_ids[0]
    first_skill = None
    done = False
    while not done:
        actor = env.current_agent_id
        ag = env.ws.characters[actor]
        d_before = _nearest_enemy_dist(env.ws, aid, [oid])
        ot = {k: torch.from_numpy(v).unsqueeze(0) for k, v in obs.items()}
        if blind:
            ot = blind_self_identity_t(ot)
        with torch.no_grad():
            el, s, e, g = net(ot)
        s = apply_resource_mask(s, env.resources, env.ws, actor)
        e = apply_entity_mask(e, ot, env.ws, actor)
        act = list(pick_action(el[0], s[0], e[0], g[0],
                               ws=env.ws, agent_id=actor))
        skills = available_skills(ag, env.ws)
        sid = None
        if act[0] > 0 and act[0] < len(skills):
            sid = skills[act[0]].skill_id
            stats["use"][sid] += 1
            if first_skill is None and sid != "move":
                first_skill = sid
            if sid in HEAL_IDS and ag.max_hp > 0:
                stats["heal_at_hp"].append(ag.hp / ag.max_hp)
            if d_before is not None:
                stats["dist_at_action"].append(d_before)
        if trace is not None:
            hpf = f"{ag.hp}/{ag.max_hp}"
            trace.append(f"r{env.ws.combat.round_number} hp={hpf} "
                         f"d={d_before if d_before is None else round(d_before,1)} "
                         f"-> {sid or ('END' if act[0]==0 else '?')}")
        obs, _, term, trunc, _ = env.step(act)
        if sid == "move":
            d_after = _nearest_enemy_dist(env.ws, aid, [oid])
            if (d_before is not None and d_after is not None
                    and abs(d_after - d_before) < 0.25):
                stats["wasted_moves"] += 1
        done = term or trunc
    stats["openers"][first_skill or "none"] += 1
    won = (not env.ws.characters[oid].is_alive()
           and env.ws.characters[aid].is_alive())
    stats["games"] += 1
    stats["wins"] += int(won)
    return won


def main():
    ckpt = sys.argv[1]
    games = int(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[2].isdigit() else 10
    blind = "--blind" in sys.argv
    out_path = None
    if "--out" in sys.argv:
        out_path = sys.argv[sys.argv.index("--out") + 1]

    register_chimeras()
    net = load_student(ckpt)
    lines = [f"student={ckpt} blind={blind} games/opp={games}", ""]

    for chim in CHIMERA_IDS:
        stats = {"use": Counter(), "openers": Counter(), "heal_at_hp": [],
                 "dist_at_action": [], "wasted_moves": 0, "games": 0,
                 "wins": 0}
        wr_by_opp = {}
        traces = []
        for opp in ARCHETYPE_LIST:
            w = 0
            for i in range(games):
                tr = [] if (i == 0 and len(traces) < 3) else None
                won = play_one(net, chim, opp, stable_seed(f"chim_{chim}_{opp}") + i,
                               blind, stats, trace=tr)
                if tr is not None:
                    traces.append((opp, won, tr))
                w += int(won)
            wr_by_opp[opp] = w / games
        n = stats["games"]
        lines.append(f"=== {chim} ===  overall WR {stats['wins']/n:.0%} "
                     f"({stats['wins']}/{n})")
        lines.append("  WR by opp: " + "  ".join(
            f"{o}:{wr_by_opp[o]:.0%}" for o in ARCHETYPE_LIST))
        lines.append("  openers: " + "  ".join(
            f"{k}x{v}" for k, v in stats["openers"].most_common()))
        lines.append("  skill use /game: " + "  ".join(
            f"{k}:{v/n:.2f}" for k, v in stats["use"].most_common()))
        if stats["heal_at_hp"]:
            arr = np.array(stats["heal_at_hp"])
            lines.append(f"  heals: {len(arr)} casts, self-HP at cast "
                         f"mean {arr.mean():.0%} (min {arr.min():.0%}, "
                         f"<=50%: {(arr <= 0.5).mean():.0%} of casts)")
        else:
            lines.append("  heals: NEVER cast")
        if stats["dist_at_action"]:
            lines.append(f"  stance: mean dist-to-enemy at action "
                         f"{np.mean(stats['dist_at_action']):.1f}m")
        lines.append(f"  wasted moves: {stats['wasted_moves']/n:.2f}/game")
        lines.append("  --- sample traces ---")
        for opp, won, tr in traces:
            lines.append(f"  vs {opp} ({'WIN' if won else 'LOSS'}): "
                         + " | ".join(tr[:28]))
        lines.append("")

    report = "\n".join(lines)
    print(report)
    if out_path:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(report)
        print(f"\nsaved -> {out_path}")


if __name__ == "__main__":
    main()
