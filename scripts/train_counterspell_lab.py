"""M4: train the model to read decision_context.spell_level and counterspell
CONDITIONALLY (counter big spells, decline weak ones) — from REWARD, no oracle.

Headroom is real and clean (probe_counterspell_headroom.py, evocation mirror L7):
  smart(counter iff level>=3) 33% WR  >>  always 21% ≈ model 21% ≈ never 23%.
Indiscriminate counterspell is WORSE than never (it burns 3rd+ slots the wizard
needs for Fireball). The current model counters everything (=always) — it does
NOT read spell_level. The optimal policy is conditional on a decision_context
feature, so reading the channel is both necessary AND rewarded → learnable.

Training = REINFORCE on the agent's counterspell decisions only (sample over
[decline, counterspell] from skill_head on the reaction obs; reward = the
episode return). Turn play is left to the warm net; std12/turn regression is
measured, not anchored away (kept simple). NO oracle: the spell_level→decline/
counter mapping must EMERGE from the return.

Verify (eval each block):
  - argmax take-rate BY spell-level becomes conditional (counter L3, decline L1/L2)
    vs the warm net's flat "counter everything".
  - CAUSAL: argmax/abl (decision_context zeroed) collapses the conditional split.
  - BENEFICIAL: argmax WR climbs toward 'smart', beating 'always'/'never'.

Usage:
  python scripts/train_counterspell_lab.py --updates 40 --episodes 48 \
      --agent evocation --opp evocation --level 7 \
      --warm models/pop_mon/pop_u0005.pt --out_dir models/cspell_lab
"""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding="utf-8")
import warnings; warnings.filterwarnings("ignore")

import argparse, random
from collections import Counter
from pathlib import Path
import numpy as np
import torch

from trpg.rl.env_v2 import CombatEnvV2
from trpg.rl.model import apply_resource_mask, apply_entity_mask, pick_action
from trpg.rl.obs import build_obs, reaction_decision_context
from trpg.engine.combat import greedy_reaction_decider
from trpg.engine.skill import reaction_skills
from trpg.scenarios.monsters import register_monsters
register_monsters()


def _grant_counterspell(env, agent_id):
    c = env.ws.characters[agent_id]
    if "counterspell" not in (c.reactions or []):
        c.reactions = list(c.reactions or []) + ["counterspell"]


class CSDecider:
    """Agent counterspell decider. mode: sample(train) / argmax(eval) /
    never/always/smart(oracle baseline). Records (spell_level, took) and, in
    sample/argmax, logs (obs, slot) for REINFORCE. ablate_dctx zeros the
    decision_context (causal control)."""
    def __init__(self, net, ids, mode="sample", thresh=3, ablate_dctx=False,
                 device="cpu", log=None, step_ref=None):
        self.net, self.ids, self.mode = net, set(ids), mode
        self.thresh, self.ablate, self.device, self.log = thresh, ablate_dctx, device, log
        self.step_ref = step_ref     # [] whose len = #env rewards so far (for return-to-go)
        self.events = []

    def __call__(self, ctx):
        if ctx.reactor_id not in self.ids:
            return greedy_reaction_decider(ctx)
        if ctx.trigger != "spell" or not ctx.options:
            return greedy_reaction_decider(ctx)     # isolate counterspell
        if self.mode == "never":
            choice = None
        elif self.mode == "always":
            choice = "counterspell"
        elif self.mode == "smart":
            choice = "counterspell" if ctx.spell_level >= self.thresh else None
        else:
            cand = reaction_skills(ctx.reactor, ctx.options)   # [decline, counterspell]
            dctx = reaction_decision_context("spell", spell_level=ctx.spell_level)
            if self.ablate:
                dctx[:] = 0.0
            obs = build_obs(ctx.world_state, ctx.reactor_id,
                            {"action": 0, "bonus_action": 0, "movement": 0.0},
                            decision_context=dctx, skills=cand)
            obs_t = {k: torch.from_numpy(v).unsqueeze(0).to(self.device)
                     for k, v in obs.items()}
            with torch.no_grad():
                _e, skill_l, _en, _g = self.net(obs_t)
            valid = obs_t["skill_mask"][0] > 0.5
            logits = skill_l[0].masked_fill(~valid, float("-inf"))
            slot = (int(logits.argmax().item()) if self.mode == "argmax"
                    else int(torch.distributions.Categorical(logits=logits).sample().item()))
            if self.log is not None:
                step_idx = len(self.step_ref) if self.step_ref is not None else 0
                self.log.append((obs, slot, step_idx))
            choice = cand[slot].skill_id if 0 < slot < len(cand) else None
        self.events.append((ctx.spell_level, choice == "counterspell"))
        return choice


def _agent_act(net, env, aid, device):
    obs = build_obs(env.ws, aid, env.resources)
    obs_t = {k: torch.from_numpy(v).unsqueeze(0).to(device) for k, v in obs.items()}
    with torch.no_grad():
        e, sk, en, gr = net(obs_t)
        sk = apply_resource_mask(sk, env.resources, env.ws, aid)
        en = apply_entity_mask(en, obs_t, env.ws, aid)
        return list(pick_action(e[0], sk[0], en[0], gr[0], ws=env.ws, agent_id=aid))


def play(net, agent, opp, level, seed, device, dec_factory):
    env = CombatEnvV2(seed=seed, n_agents=1, n_opps=1)
    env.reset(agent_archs=[agent], opp_archs=[opp], level=level, seed=seed)
    aid = env.agent_ids[0]
    _grant_counterspell(env, aid)
    rewards = []                      # per-env-step reward stream (return-to-go)
    dec = dec_factory(env, aid, rewards)
    env.ws.reaction_decider = dec
    for _ in range(env._max_steps):
        _o, r, term, trunc, _i = env.step(_agent_act(net, env, aid, device))
        rewards.append(float(r))      # a reaction logged at step_idx i is credited
        if term or trunc:             # the return from rewards[i:] (incl. its own
            break                     # step, where the countered/allowed spell lands)
    win = (all(not env.ws.characters[o].is_alive() for o in env.opp_ids)
           and any(env.ws.characters[a].is_alive() for a in env.agent_ids))
    return sum(rewards), win, dec, rewards


def evaluate(net, args, device, games):
    net.eval()
    out = {}
    for mode, abl in (("never", False), ("always", False), ("smart", False),
                      ("argmax", False), ("argmax", True)):
        wins = 0; seen = Counter(); take = Counter()
        for g in range(games):
            s = 4000 + g                  # same winnable seed band as probe_counterspell
            random.seed(s); np.random.seed(s); torch.manual_seed(s)
            _r, win, dec, _rw = play(net, args.agent, args.opp, args.level, s, device,
                                     lambda env, aid, rw, m=mode, a=abl: CSDecider(
                                         net, {aid}, mode=m, thresh=args.thresh,
                                         ablate_dctx=a, device=device, step_ref=rw))
            wins += int(win)
            for lv, took in dec.events:
                seen[lv] += 1; take[lv] += int(took)
        wr = wins / games
        tag = mode + ("/abl" if abl else "")
        if mode == "argmax":
            lo = sum(take[lv] for lv in seen if lv < args.thresh)
            lo_n = sum(seen[lv] for lv in seen if lv < args.thresh)
            hi = sum(take[lv] for lv in seen if lv >= args.thresh)
            hi_n = sum(seen[lv] for lv in seen if lv >= args.thresh)
            out[tag] = (f"WR{wr*100:.0f}% take[<{args.thresh}:{lo}/{lo_n} "
                        f">={args.thresh}:{hi}/{hi_n}]")
        else:
            out[tag] = f"WR{wr*100:.0f}%"
    net.train()
    return " ".join(f"{k}={v}" for k, v in out.items())


def train(args):
    device = args.device
    from trpg.sandbox.policy_loader import load_policy
    net = load_policy(args.warm).net
    net.to(device).train()
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    print(f"baseline: {evaluate(net, args, device, args.eval_games)}")
    for upd in range(1, args.updates + 1):
        logs, advs, rets = [], [], []
        for ep in range(args.episodes):
            seed = upd * 100000 + ep
            random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
            ep_log = []
            ret, _w, _d, rewards = play(
                net, args.agent, args.opp, args.level, seed, device,
                lambda env, aid, rw: CSDecider(net, {aid}, mode="sample",
                                               device=device, log=ep_log, step_ref=rw))
            rets.append(ret)
            # Per-decision RETURN-TO-GO (discounted reward from the decision's env
            # step onward) — credits THIS counterspell's effect (Φ from the
            # prevented/allowed spell + the downstream slot economy), so the
            # gradient separates a good L3 counter from a wasted L1 counter WITHIN
            # an episode. A single per-episode advantage cannot (it would drift to
            # "counter more"). Baseline-subtracted to reduce variance.
            n = len(rewards)
            g = 0.0; rtg = [0.0] * n
            for t in range(n - 1, -1, -1):
                g = rewards[t] + args.gamma * g
                rtg[t] = g
            for (obs, slot, step_idx) in ep_log:
                i = min(step_idx, n - 1) if n else 0
                logs.append((obs, slot)); advs.append(rtg[i] if n else 0.0)
        if not logs:
            print(f"[u{upd:02d}] no counterspell decisions — check regime"); continue

        net.train()
        A = torch.tensor(advs, dtype=torch.float32, device=device)
        A = (A - A.mean()) / (A.std() + 1e-6)
        losses, ents = [], []
        for (obs, slot), a in zip(logs, A):
            obs_t = {k: torch.from_numpy(v).unsqueeze(0).to(device) for k, v in obs.items()}
            _e, skill_l, _en, _g = net(obs_t)
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
            print(f"[u{upd:02d}] ret={np.mean(rets):+.2f} dec={len(logs)} "
                  f"loss={loss.item():+.3f} | {evaluate(net, args, device, args.eval_games)}")
            torch.save(net.state_dict(), out_dir / f"cs_u{upd:04d}.pt")
    print(f"saved to {out_dir}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--updates", type=int, default=40)
    ap.add_argument("--episodes", type=int, default=48)
    ap.add_argument("--agent", default="evocation")
    ap.add_argument("--opp", default="evocation")
    ap.add_argument("--level", type=int, default=7)
    ap.add_argument("--thresh", type=int, default=3)
    ap.add_argument("--warm", default="models/pop_mon/pop_u0005.pt")
    ap.add_argument("--out_dir", default="models/cspell_lab")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--ent_coef", type=float, default=0.01)
    ap.add_argument("--eval_every", type=int, default=5)
    ap.add_argument("--eval_games", type=int, default=120)
    ap.add_argument("--device", default="cpu")
    train(ap.parse_args())
