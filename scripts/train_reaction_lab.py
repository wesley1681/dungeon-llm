"""CAN the model LEARN to read decision_context and react CONDITIONALLY from
reward (no teacher-forcing)? — Milestone 4 of the reaction/legendary wave.

Measure-first finding (probe_reaction_headroom.py): reactions are beneficial
(+7-8pp) but the current model uses Shield INDISCRIMINATELY (= "always", 100%
take-rate vs greedy's 36.5%) — it does NOT read the trigger. To make reading the
trigger MATTER (and thus learnable), we need a regime where blind always-shield
is WORSE than conditional shield.

THE DISCRIMINATION REGIME (the crux): an evocation wizard whose ONLY spell slots
are 3rd-level (its Fireball slots). Shield consumes the LOWEST available slot —
here that's a 3rd-level slot — so EVERY shield cast burns a Fireball. The d20
gives a natural spread of incoming-attack margins:
  - margin in [0,5): +5 AC FLIPS the hit → Shield saves real damage (worth a slot)
  - margin >= 5    : Shield can't flip it → casting Shield WASTES a Fireball
So "always-shield" throws away nukes on unflippable hits and loses the damage
race; "shield only when attack_margin flips" (reads decision_context[ATK_MARGIN])
preserves Fireballs and wins. The optimal policy is CONDITIONAL on a dctx feature
→ reading the channel is both necessary and rewarded. No oracle: reward is the
env return; the trigger→behaviour mapping must EMERGE.

Training = REINFORCE on the reaction decisions only (turn play left to the warm
net; an optional turn-BC-style anchor is omitted — we measure std12 regression
instead). Verify: (1) trained take-rate becomes CONDITIONAL on attack_margin
(shields flippable hits, declines unflippable) — vs the warm net's flat "always";
(2) CAUSAL — ablating decision_context collapses the conditional behaviour;
(3) BENEFICIAL — WR beats blind always-shield.

Usage:
  python scripts/train_reaction_lab.py --updates 40 --episodes 64 \
      --agent evocation --opp battle_master --level 5 --slots 3:2 \
      --warm models/pop_mon/pop_u0005.pt --out_dir models/reaction_lab
"""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding="utf-8")
import warnings; warnings.filterwarnings("ignore")

import argparse, random
from pathlib import Path
import numpy as np
import torch

from trpg.rl.env_v2 import CombatEnvV2
from trpg.rl.model import CombatPolicyNet, apply_resource_mask, apply_entity_mask, pick_action
from trpg.rl.obs import build_obs, reaction_decision_context, I_DCTX_ATK_MARGIN
from trpg.engine.combat import greedy_reaction_decider, effective_ac
from trpg.engine.skill import reaction_skills
from trpg.scenarios.monsters import register_monsters
register_monsters()


def _parse_slots(s):
    out = {}
    for part in s.split(","):
        lvl, n = part.split(":")
        out[int(lvl)] = int(n)
    return out


def _set_slots(env, agent_id, slots):
    env.ws.characters[agent_id].spell_slots = dict(slots)


class RecordingReactionDecider:
    """Samples the agent's reaction over [decline, shield] from the net's
    skill_head on the reaction obs, logging (obs, chosen_slot) for a REINFORCE
    update. Non-agent reactors keep the greedy default. ``ablate_dctx`` zeros the
    decision_context (causal control). ``mode``: 'sample' (train), 'argmax'
    (eval-trained), 'greedy'/'always'/'never' (baselines)."""

    def __init__(self, net, agent_ids, mode="sample", ablate_dctx=False,
                 device="cpu", log=None):
        self.net = net
        self.ids = set(agent_ids)
        self.mode = mode
        self.ablate = ablate_dctx
        self.device = device
        self.log = log              # list to append (obs_dict, slot) for training
        self.margins = []           # (attack_margin, took_reaction) for analysis

    def __call__(self, ctx):
        if ctx.reactor_id not in self.ids:
            return greedy_reaction_decider(ctx)
        if self.mode == "greedy":
            return greedy_reaction_decider(ctx)
        if self.mode == "never":
            return None
        if self.mode == "always":
            return ctx.options[0] if ctx.options else None

        cand = reaction_skills(ctx.reactor, ctx.options)      # [decline, shield..]
        margin = (float(ctx.attack_total) - effective_ac(ctx.reactor)
                  if ctx.trigger in ("attack", "uncanny") else 0.0)
        dctx = reaction_decision_context(ctx.trigger, attack_margin=margin,
                                         spell_level=ctx.spell_level)
        if self.ablate:
            dctx[:] = 0.0
        resources = {"action": 0, "bonus_action": 0, "movement": 0.0}
        obs = build_obs(ctx.world_state, ctx.reactor_id, resources,
                        decision_context=dctx, skills=cand)
        obs_t = {k: torch.from_numpy(v).unsqueeze(0).to(self.device)
                 for k, v in obs.items()}
        with torch.no_grad():
            _e, skill_l, _ent, _g = self.net(obs_t)
        valid = obs_t["skill_mask"][0] > 0.5
        logits = skill_l[0].masked_fill(~valid, float("-inf"))
        if self.mode == "argmax":
            slot = int(logits.argmax().item())
        else:  # sample
            slot = int(torch.distributions.Categorical(logits=logits).sample().item())
        if self.log is not None:
            self.log.append((obs, slot))
        took = 0 < slot < len(cand)
        self.margins.append((margin, took))
        if not took:
            return None
        return cand[slot].skill_id


def _agent_act(net, env, agent_id, device, sample=False):
    obs = build_obs(env.ws, agent_id, env.resources)
    obs_t = {k: torch.from_numpy(v).unsqueeze(0).to(device) for k, v in obs.items()}
    with torch.no_grad():
        end_l, skill_l, ent_l, grid_l = net(obs_t)
        skill_l = apply_resource_mask(skill_l, env.resources, env.ws, agent_id)
        ent_l = apply_entity_mask(ent_l, obs_t, env.ws, agent_id)
        return list(pick_action(end_l[0], skill_l[0], ent_l[0], grid_l[0],
                                ws=env.ws, agent_id=agent_id))


def play_episode(net, agent, opp, level, slots, seed, device, decider_factory,
                 opp_level=None):
    env = CombatEnvV2(seed=seed, n_agents=1, n_opps=1)
    env.reset(agent_archs=[agent], opp_archs=[opp], level=level,
              opp_level=opp_level, seed=seed)
    aid = env.agent_ids[0]
    if slots:
        _set_slots(env, aid, slots)
    dec = decider_factory(env)
    env.ws.reaction_decider = dec
    ret = 0.0
    for _ in range(env._max_steps):
        act = _agent_act(net, env, aid, device)
        _o, r, term, trunc, _info = env.step(act)
        ret += float(r)
        if term or trunc:
            break
    win = (all(not env.ws.characters[o].is_alive() for o in env.opp_ids)
           and any(env.ws.characters[a].is_alive() for a in env.agent_ids))
    return ret, win, dec


def train(args):
    device = args.device
    net = CombatPolicyNet()
    from trpg.sandbox.policy_loader import load_policy
    net = load_policy(args.warm).net
    net.to(device).train()
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)
    slots = _parse_slots(args.slots) if args.slots else None

    baseline = 0.0
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    for upd in range(1, args.updates + 1):
        batch_logs, batch_adv = [], []
        rets = []
        for ep in range(args.episodes):
            seed = upd * 100000 + ep
            random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
            log = []
            ret, win, _dec = play_episode(
                net, args.agent, args.opp, args.level, slots, seed, device,
                lambda env: RecordingReactionDecider(
                    net, env.agent_ids, mode="sample", device=device, log=log),
                opp_level=args.opp_level)
            rets.append(ret)
            adv = ret - baseline
            for (obs, slot) in log:
                batch_logs.append((obs, slot))
                batch_adv.append(adv)
        baseline = 0.9 * baseline + 0.1 * float(np.mean(rets))

        if not batch_logs:
            print(f"[u{upd:02d}] no reaction decisions sampled — check regime")
            continue
        # REINFORCE update over the reaction decisions (recompute logprob WITH grad)
        net.train()
        advs = torch.tensor(batch_adv, dtype=torch.float32, device=device)
        advs = (advs - advs.mean()) / (advs.std() + 1e-6)
        losses = []
        ents = []
        for (obs, slot), a in zip(batch_logs, advs):
            obs_t = {k: torch.from_numpy(v).unsqueeze(0).to(device)
                     for k, v in obs.items()}
            _e, skill_l, _ent, _g = net(obs_t)
            valid = obs_t["skill_mask"][0] > 0.5
            logits = skill_l[0].masked_fill(~valid, float("-inf"))
            dist = torch.distributions.Categorical(logits=logits)
            losses.append(-(a * dist.log_prob(torch.tensor(slot, device=device))))
            ents.append(dist.entropy())
        loss = torch.stack(losses).mean() - args.ent_coef * torch.stack(ents).mean()
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()

        if upd % args.eval_every == 0 or upd == args.updates:
            ev = evaluate(net, args, slots, device, games=args.eval_games)
            print(f"[u{upd:02d}] ret={np.mean(rets):+.2f} react_dec={len(batch_logs)} "
                  f"loss={loss.item():+.3f} | {ev}")
            torch.save(net.state_dict(), out_dir / f"rx_u{upd:04d}.pt")
    print(f"saved checkpoints to {out_dir}")


def _take_by_margin(margins):
    """take-rate split by whether the hit was in the +5 flip window (<5) or not."""
    flip = [t for m, t in margins if 0 <= m < 5]
    nonflip = [t for m, t in margins if m >= 5]
    f = (sum(flip) / len(flip)) if flip else float("nan")
    nf = (sum(nonflip) / len(nonflip)) if nonflip else float("nan")
    return f, nf, len(flip), len(nonflip)


def evaluate(net, args, slots, device, games=120):
    net.eval()
    out = {}
    for mode, abl in (("never", False), ("always", False), ("greedy", False),
                      ("argmax", False), ("argmax", True)):
        wins = 0
        all_margins = []
        for g in range(games):
            seed = 7_000_000 + g
            random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
            _ret, win, dec = play_episode(
                net, args.agent, args.opp, args.level, slots, seed, device,
                lambda env, m=mode, a=abl: RecordingReactionDecider(
                    net, env.agent_ids, mode=m, ablate_dctx=a, device=device),
                opp_level=args.opp_level)
            wins += int(win)
            all_margins += dec.margins
        wr = wins / games
        tag = mode + ("/abl" if abl else "")
        if mode == "argmax":
            f, nf, nf_n, nnf = _take_by_margin(all_margins)
            out[tag] = f"WR{wr*100:.0f}% take[flip{f*100:.0f}%/non{nf*100:.0f}%]"
        else:
            out[tag] = f"WR{wr*100:.0f}%"
    net.train()
    return " ".join(f"{k}={v}" for k, v in out.items())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--updates", type=int, default=40)
    ap.add_argument("--episodes", type=int, default=64)
    ap.add_argument("--agent", default="evocation")
    ap.add_argument("--opp", default="battle_master")
    ap.add_argument("--level", type=int, default=5)
    ap.add_argument("--opp_level", type=int, default=None)
    ap.add_argument("--eval_only", action="store_true")
    ap.add_argument("--slots", default="3:2")
    ap.add_argument("--warm", default="models/pop_mon/pop_u0005.pt")
    ap.add_argument("--out_dir", default="models/reaction_lab")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--ent_coef", type=float, default=0.01)
    ap.add_argument("--eval_every", type=int, default=5)
    ap.add_argument("--eval_games", type=int, default=120)
    ap.add_argument("--device", default="cpu")
    _args = ap.parse_args()
    if _args.eval_only:
        from trpg.sandbox.policy_loader import load_policy
        _net = load_policy(_args.warm).net
        _net.to(_args.device).eval()
        _slots = _parse_slots(_args.slots) if _args.slots else None
        print(f"eval-only regime: {_args.agent}@L{_args.level} slots={_args.slots} "
              f"vs {_args.opp}@L{_args.opp_level or _args.level}")
        print(evaluate(_net, _args, _slots, _args.device, games=_args.eval_games))
    else:
        train(_args)
