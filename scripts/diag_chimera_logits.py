"""Where does B's probability mass sit on the UNUSED parts of a chimera kit?

The chimera eval showed degenerate behaviour (trickhealer: cleric kit never
cast; omni: heal starvation + stand-off stall). Before training anything,
measure WHY at the policy-distribution level: in representative states, is
the unused skill

  (a) masked out (resource/legality bug — PPO could never fix it),
  (b) crushed to ~0 probability (exploration would need a kickstart), or
  (c) merely out-ranked by the prototype action (plain PPO sampling will
      visit it and the reward signal can promote it)?

States probed per chimera: the round-1 opener, plus the same position with
self HP set to 35% (heal decisions live there), each at full resources and
at action-spent (bonus-only — the healing_word vs rage/cunning contest).

Usage: python scripts/diag_chimera_logits.py [ckpt] [--no-blind]
"""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding="utf-8")
import warnings; warnings.filterwarnings("ignore")

import torch
import torch.nn.functional as F

from trpg.rl.env_v2 import CombatEnvV2
from trpg.rl.model import apply_resource_mask
from trpg.rl.obs import build_obs
from trpg.engine.skill import available_skills

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from chimera_defs import register_chimeras, CHIMERA_IDS
from distill_routed import load_student, blind_self_identity_t
from eval_routed import stable_seed


def probe_state(net, env, aid, resources, blind, label):
    obs = build_obs(env.ws, aid, resources)
    ot = {k: torch.from_numpy(v).unsqueeze(0) for k, v in obs.items()}
    if blind:
        ot = blind_self_identity_t(ot)
    with torch.no_grad():
        el, s, e, g = net(ot)
    s = apply_resource_mask(s, resources, env.ws, aid)
    end_p = float(torch.sigmoid(el[0]))
    sl = s[0].clone()
    sl[0] = -1e9                       # slot 0 = end, decided by end head
    probs = F.softmax(sl, dim=0)
    skills = available_skills(env.ws.characters[aid], env.ws)
    print(f"  [{label}] end_p={end_p:.3f}")
    rows = []
    for i, sk in enumerate(skills):
        if i == 0:
            continue
        masked = bool(s[0, i] <= -1e8)
        rows.append((float(probs[i]), sk.skill_id, masked))
    for p, name, masked in sorted(rows, reverse=True):
        flag = " MASKED" if masked else ""
        print(f"      {p:7.4f}  {name}{flag}")


def main():
    ckpt = sys.argv[1] if len(sys.argv) > 1 else "models/distill_b/distill_e08.pt"
    blind = "--no-blind" not in sys.argv
    register_chimeras()
    net = load_student(ckpt)
    print(f"ckpt={ckpt} blind={blind}\n")

    for arch in CHIMERA_IDS:
        print(f"=== {arch} ===")
        env = CombatEnvV2(seed=stable_seed(f"diaglogit_{arch}"),
                          n_agents=1, n_opps=1)
        env.reset(agent_archs=[arch], opp_archs=["battle_master"], level=5)
        aid = env.agent_ids[0]
        ag = env.ws.characters[aid]

        full = {"action": 1, "bonus_action": 1, "movement": 9.0}
        bonus_only = {"action": 0, "bonus_action": 1, "movement": 0.0}

        probe_state(net, env, aid, full, blind, "opener, full HP, full res")

        ag.hp = max(1, int(ag.max_hp * 0.35))
        probe_state(net, env, aid, full, blind, "HP 35%, full res")
        probe_state(net, env, aid, bonus_only, blind, "HP 35%, bonus-only")
        print()


if __name__ == "__main__":
    main()
