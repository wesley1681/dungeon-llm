"""Turn-by-turn trace, MODEL vs SCRIPTED EXPERT on the SAME seed, to see WHY the
model removes ~76% of enemy HP but dies while the expert removes ~90% and lives.
probe_economy showed the model out-attacks the expert yet takes more damage ->
the gap is DEFENSE/survival, not offense. This logs, per agent decision:

  who acts, action type+skill, target, dmg dealt this step, agent HP, each
  enemy HP, and the agent's distance to each living enemy (how exposed it is).

General: no skill-name logic; just prints what happens. Run a few seeds.

Usage: python scripts/diag_trace.py --ident battle_master --seeds 0 1 2
"""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding="utf-8")
import warnings; warnings.filterwarnings("ignore")
import argparse, random
from zlib import crc32
import torch

from trpg.scenarios.monsters import register_monsters
register_monsters()
from trpg.rl.env_v2 import CombatEnvV2
from trpg.rl.action import decode_action, encode_action
from trpg.rl.model import apply_resource_mask, apply_entity_mask, pick_action
from trpg.engine.combat_policy import make_archetype_policy

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from distill_routed import load_student
from train_population import blind_np_single
from synth_identity import STANDARD_IDS


def _hps(env, oids):
    return [max(0, env.ws.characters[o].hp) for o in oids]


def _dists(env, aid, oids):
    a = env.ws.characters[aid].position
    out = []
    for o in oids:
        c = env.ws.characters[o]
        if c.is_alive():
            out.append((c.position - a).length())
    return out


def trace(driver, ident, gi, label):
    key = f"{ident}|2|{gi}"; k = crc32(key.encode()); random.seed(k)
    env = CombatEnvV2(seed=k ^ 0x5, n_agents=1, n_opps=2)
    opps = [STANDARD_IDS[(k + i) % len(STANDARD_IDS)] for i in range(2)]
    obs, _ = env.reset(agent_archs=[ident], opp_archs=opps, level=5,
                       opp_level=2, layout="open")
    aid = env.agent_ids[0]; oids = list(env.opp_ids)
    net = driver if driver != "expert" else None
    script = make_archetype_policy(ident) if net is None else None
    print(f"\n===== {label}  {ident} 1v2  seed={gi}  enemies={opps} =====")
    rnd = -1
    done = False
    while not done:
        actor = env.current_agent_id
        ch = env.ws.characters[actor]
        if net is not None:
            ob = blind_np_single(obs)
            ot = {kk: torch.from_numpy(v).unsqueeze(0) for kk, v in ob.items()}
            with torch.no_grad():
                el, s, e, g = net(ot)
            s = apply_resource_mask(s, env.resources, env.ws, actor)
            e = apply_entity_mask(e, ot, env.ws, actor)
            act = list(pick_action(el[0], s[0], e[0], g[0], ws=env.ws,
                                   agent_id=actor))
        else:
            dec = script.decide(actor, ch, env.ws, env.resources,
                                env.ws.combat.round_number)
            if dec.action is None or getattr(dec, "fled", False):
                act = [0, 0, 0]
            else:
                try:
                    act = list(encode_action(dec.action, env.ws, actor))
                except Exception:
                    act = [0, 0, 0]
        is_agent = (actor == aid)
        if is_agent:
            if env.ws.combat.round_number != rnd:
                rnd = env.ws.combat.round_number
                ds = _dists(env, aid, oids)
                print(f"  -- round {rnd}  agentHP={max(0,ch.hp)}/{ch.max_hp}  "
                      f"enemyHP={_hps(env, oids)}  dist={['%.1f'%d for d in ds]}")
            ad = decode_action(act, env.ws, aid)
            hp_before = _hps(env, oids)
        obs, _, term, trunc, _ = env.step(act)
        if is_agent:
            hp_after = _hps(env, oids)
            dealt = sum(max(0, b - a) for b, a in zip(hp_before, hp_after))
            if ad is None:
                desc = "END(turn)"
            else:
                typ = str(ad.get("type", "?"))
                sk = ad.get("skill_id") or ad.get("name") or ""
                tgt = ad.get("target_id") or ad.get("target") or ""
                desc = f"{typ} {sk} -> {tgt}"
            print(f"       {desc:<46} dmg={dealt:.0f}")
        done = term or trunc
    alive = env.ws.characters[aid].is_alive()
    ehp = _hps(env, oids)
    print(f"  RESULT: agent {'WON' if alive and not any(ehp) else 'LOST'}  "
          f"agentHP={max(0,env.ws.characters[aid].hp)}  enemyHP={ehp}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="models/seed_kit/kit_s1000.pt")
    p.add_argument("--ident", default="battle_master")
    p.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2])
    args = p.parse_args()
    net = load_student(args.ckpt); net.eval()
    for gi in args.seeds:
        trace(net, args.ident, gi, "MODEL")
        trace("expert", args.ident, gi, "EXPERT")


if __name__ == "__main__":
    main()
