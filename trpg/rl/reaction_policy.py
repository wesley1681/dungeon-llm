"""Model-controlled reaction / legendary-action deciders.

These plug into the engine decision seam (combat.ReactionContext /
combat_policy.LegendaryContext via ws.reaction_decider / ws.legendary_decider).
They route a MODEL-CONTROLLED creature's off-turn CHOICE through a
CombatPolicyNet, while any creature NOT in ``model_ids`` falls back to the engine
greedy default — so scripted opponents keep their historical behaviour and only
the model's own side learns to react.

The choice reuses the EXISTING policy machinery with no new action head: at a
decision point we build the reactor's obs with (a) the reaction/legendary
candidates placed in the skill slots (skill.reaction_skills) and (b) the
decision_context vector set (obs.reaction_decision_context / legendary_…), then
read skill_head's argmax over the candidate slots. Slot 0 is the DECLINE option
(skill.reaction_skills puts end_turn there) — picking it forgoes the reaction.
The legality mask is the engine's (ctx.options); the model only picks among legal
options or declines, exactly the contract the seam guarantees.
"""
from __future__ import annotations
import torch

from .obs import (build_obs, reaction_decision_context,
                  legendary_decision_context)


def _argmax_over_candidates(net, obs, device) -> int:
    """Run the policy on a decision obs and return the chosen skill slot index
    (0 = decline). Argmax over the VALID (present) candidate slots only — the
    decline slot competes as a normal skill logit, NOT via end_head (end_head's
    turn-ending semantics don't apply to an off-turn reaction)."""
    obs_t = {k: torch.from_numpy(v).unsqueeze(0).to(device) for k, v in obs.items()}
    with torch.no_grad():
        _end_l, skill_l, _ent_l, _grid_l = net(obs_t)
    logits = skill_l[0].clone()
    mask = obs_t["skill_mask"][0] > 0.5          # valid candidate slots
    logits = logits.masked_fill(~mask, float("-inf"))
    return int(logits.argmax().item())


class NeuralReactionDecider:
    """ws.reaction_decider that lets ``model_ids`` choose reactions via ``net``.

    ``greedy_fallback`` (combat.greedy_reaction_decider) handles every other
    reactor so the scripted side is untouched. Returns a chosen option skill_id
    (always one of ctx.options) or None to decline — decide_reaction re-checks
    membership, so an unexpected return can only ever degrade to a decline."""

    def __init__(self, net, model_ids, device: str = "cpu",
                 greedy_fallback=None):
        self.net = net
        self.model_ids = set(model_ids)
        self.device = device
        if greedy_fallback is None:
            from ..engine.combat import greedy_reaction_decider
            greedy_fallback = greedy_reaction_decider
        self._greedy = greedy_fallback

    def __call__(self, ctx):
        if ctx.reactor_id not in self.model_ids:
            return self._greedy(ctx)
        from ..engine.skill import reaction_skills
        from ..engine.combat import effective_ac
        cand = reaction_skills(ctx.reactor, ctx.options)   # [decline, opt...]
        margin = (float(ctx.attack_total) - effective_ac(ctx.reactor)
                  if ctx.trigger in ("attack", "uncanny") else 0.0)
        dctx = reaction_decision_context(
            ctx.trigger, attack_margin=margin, spell_level=ctx.spell_level)
        # Off-turn: the reactor has no action economy this step (the dctx is the
        # discriminator the policy reads, not the resource wallet).
        resources = {"action": 0, "bonus_action": 0, "movement": 0.0}
        obs = build_obs(ctx.world_state, ctx.reactor_id, resources,
                        decision_context=dctx, skills=cand)
        slot = _argmax_over_candidates(self.net, obs, self.device)
        if slot <= 0 or slot >= len(cand):
            return None                              # decline
        return cand[slot].skill_id


class NeuralLegendaryDecider:
    """ws.legendary_decider that lets ``model_ids`` choose legendary actions via
    ``net`` (falls back to greedy EV for any other actor). Returns a chosen
    skill_id from ctx.options, or None to forgo the legendary action."""

    def __init__(self, net, model_ids, device: str = "cpu",
                 greedy_fallback=None):
        self.net = net
        self.model_ids = set(model_ids)
        self.device = device
        if greedy_fallback is None:
            from ..engine.combat_policy import greedy_legendary_decider
            greedy_fallback = greedy_legendary_decider
        self._greedy = greedy_fallback

    def __call__(self, ctx):
        if ctx.actor_id not in self.model_ids:
            return self._greedy(ctx)
        # Candidate Skills come attached to each option (decide_legendary_action
        # populates "skill"); slot 0 = decline (end_turn).
        from ..engine.skill import end_turn_skill
        cand = [end_turn_skill()] + [o["skill"] for o in ctx.options
                                     if o.get("skill") is not None]
        if len(cand) <= 1:
            return None
        dctx = legendary_decision_context(ctx.actor.legendary_actions_remaining)
        resources = {"action": 0, "bonus_action": 0, "movement": 0.0}
        obs = build_obs(ctx.world_state, ctx.actor_id, resources,
                        decision_context=dctx, skills=cand)
        slot = _argmax_over_candidates(self.net, obs, self.device)
        if slot <= 0 or slot >= len(cand):
            return None
        return cand[slot].skill_id
