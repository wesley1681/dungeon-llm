"""Population PPO fine-tune of the BLIND distilled student.

Goal: make composed identities an IN-DISTRIBUTION event so the policy stops
collapsing chimeras onto the nearest teacher prototype (trickhealer playing
the assassin script with its cleric kit at literal zero use).

Recipe:
  - warm-start  models/distill_b/distill_e08.pt (1 head group, blind)
  - every episode samples a fresh identity: a standard class with prob
    (1 - p_synth), otherwise a freshly synthesized random legal ClassDef
    (synth_identity.py). Identity is fixed within the episode — like the
    --asym level-gap sampling, the POOL is what varies, never mid-game state.
  - opponents are scripted experts of random standard classes (the eval
    condition); rewards are the env defaults (HP-PBRS + wasted-move cost) —
    identity-agnostic by construction.
  - self-identity channels are zeroed at the env boundary (rollout AND
    update see blinded obs) — the capability-conditioning contract of B.
  - BC anchor: each update interleaves a few minibatches of the original
    distill dataset (blinded) — replay mixing against forgetting the 12
    standard kits (the life-class --replay_v1 precedent).
  - gradient-conflict monitoring: at each eval point, per-identity-group
    policy gradients on the last rollout are pairwise-cosined. This is the
    measurement that justified per-arch heads in the v16 era; if population
    PPO re-creates the pathology we see it in numbers, not in vibes.

Chimeras are NOT in the training pool — they are the held-out probe.

Usage:
  python scripts/train_population.py --updates 60 --out_dir models/pop_b
"""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding="utf-8")
import warnings; warnings.filterwarnings("ignore")

import argparse
import random
from pathlib import Path

import numpy as np
import torch

from trpg.rl.env_v2 import CombatEnvV2, _TEAM_CONFIGS, LAYOUTS
from trpg.rl.model import (CombatPolicyNet, apply_resource_mask,
                           apply_entity_mask, pick_action, _SPATIAL_C)
from trpg.rl.train_ppo import _sample_action, _compute_gae, ppo_update
from trpg.rl.train_bc import bc_loss_step
from trpg.engine.skill import available_skills
from trpg.engine.abilities import ABILITY_REGISTRY

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from distill_routed import (load_student, load_dataset, DATASET_PATH,
                            blind_self_identity_np, _ARCH_OH_START,
                            _END_ARCH_START)
from synth_identity import fresh_synth, STANDARD_IDS
from chimera_defs import register_chimeras, CHIMERA_IDS
from eval_routed import stable_seed

from trpg.rl.obs import (N_ARCHETYPES, ENT_DESC_START, N_V4_DESC,
                         migrate_entities_v3_to_v4)
from trpg.scenarios.monsters import (register_monsters, onev1_viable_monsters,
                                     EQUIV_LEVEL_1V1, MONSTER_DEFS)


# ── Episode identity+level sampling (fair 1v1 pairing) ───────────────────────

def sample_episode_spec(rng, mon_pool, p_synth, p_mon_agent, p_mon_opp, ep):
    """Sample (agent_id, agent_tag, agent_level, opp_id, opp_level) for one
    episode. Monsters fight at their natural_level; the class/synth facing a
    monster is set to the monster's MEASURED equiv_level so the matchup is a
    ~coin-flip (no monster→inf can reach here: mon_pool is pre-filtered).
    Standard-vs-standard keeps the historical symmetric random level."""
    r = rng.random()
    if mon_pool and r < p_mon_agent:
        agent, a_tag, a_mon = rng.choice(mon_pool), None, True
        a_tag = agent
    elif r < p_mon_agent + p_synth:
        agent, a_tag, a_mon = fresh_synth(rng, slot=ep), "synth", False
    else:
        agent = rng.choice(STANDARD_IDS); a_tag, a_mon = agent, False

    if mon_pool and rng.random() < p_mon_opp:
        opp, o_mon = rng.choice(mon_pool), True
    else:
        opp, o_mon = rng.choice(STANDARD_IDS), False

    if a_mon and not o_mon:
        a_lvl = MONSTER_DEFS[agent].natural_level
        o_lvl = max(1, round(EQUIV_LEVEL_1V1[agent]))
    elif o_mon and not a_mon:
        a_lvl = max(1, round(EQUIV_LEVEL_1V1[opp]))
        o_lvl = MONSTER_DEFS[opp].natural_level
    elif a_mon and o_mon:
        a_lvl = MONSTER_DEFS[agent].natural_level
        o_lvl = MONSTER_DEFS[opp].natural_level
    else:
        a_lvl = o_lvl = rng.randint(3, 8)
    return agent, a_tag, a_lvl, opp, o_lvl


def descriptor_weight_norm(net):
    """(desc_norm, rest_norm) of entity_mlp[0]. After obs-v4 migration the
    descriptor columns are exactly 0; a rising desc_norm PROVES the v4
    capability channel is being learned (the whole point of this wave)."""
    w = net.entity_mlp[0].weight.detach()
    desc = w[:, ENT_DESC_START:ENT_DESC_START + N_V4_DESC]
    rest = w[:, :ENT_DESC_START]
    return float(desc.norm()), float(rest.norm())


def blind_np_single(obs: dict) -> dict:
    """Blind one (unbatched) numpy obs dict, copy-on-write."""
    out = dict(obs)
    ent = obs["entities"].copy()
    ent[0, _ARCH_OH_START:_ARCH_OH_START + N_ARCHETYPES] = 0.0
    out["entities"] = ent
    ef = obs["end_features"].copy()
    ef[_END_ARCH_START:_END_ARCH_START + N_ARCHETYPES] = 0.0
    out["end_features"] = ef
    return out


# ── Rollout ──────────────────────────────────────────────────────────────────

def _sample_team_arch(rng, p_synth, slot):
    """One team-seat identity: a synth with prob p_synth, else a standard
    class. (Monsters stay on the 1v1 path — their fair level pairing is
    per-creature and doesn't generalise to mixed teams.)"""
    if rng.random() < p_synth:
        return fresh_synth(rng, slot=slot)
    return rng.choice(STANDARD_IDS)


# Layouts whose terrain actually BLOCKS line of sight (so a held-position enemy
# can sit out of sight and the agent must flank). difficult/lava slow movement
# but don't block LoS, so a hold-opponent there teaches nothing.
_LOS_BLOCKING_LAYOUTS = ("walls", "pillar")

# When set (list of class ids), the rollout runs ONLY lone-FOCUS-vs-2-standard
# 1v2/1v3 episodes — no synth/monster/other-class dilution. Set from --focus_only.
_FOCUS_ONLY = None


class _StationaryPolicy:
    """Opponent that holds position and never acts. Paired with a LoS-blocking
    layout it creates the ONLY regime that rewards flanking: an enemy that will
    NOT walk into view on its own, so the agent must reposition to regain line
    of sight before it can deal any damage. Approaching scripted experts hand
    LoS back for free (the enemy comes around the wall) → they reward WAITING,
    not flanking (measured: pillar blocked-runs resolve 90% by enemy-move), so
    the LoS row never gets a flank gradient without this."""
    def decide(self, opp_id, opp, ws, resources, round_number):
        from types import SimpleNamespace
        return SimpleNamespace(action=None, fled=False, ended=True)


def collect_population_rollout(net, n_steps: int, seed: int, p_synth: float,
                               rng: random.Random, gamma=0.99, lam=0.95,
                               mon_pool=(), p_mon_agent=0.0, p_mon_opp=0.0,
                               p_team=0.0, p_hold=0.0):
    """Sequential rollout; per-episode identity sampling; blinded obs.

    With prob ``p_team`` the episode is a MULTI-seat arrangement (1vN / Nv1 /
    NvN drawn from _TEAM_CONFIGS, classes+synths only) so the policy trains on
    the team sizes the 1v1-only history never covered; otherwise the existing
    1v1 path (with monster matchups) runs. The rollout loop is team-agnostic —
    it keys transitions on env.current_agent_id and the reward/PBRS are already
    team-level — so multi-seat needs only the env shape + arch lists here.

    Episodes always run to completion (the last one may overshoot n_steps a
    little) so GAE never bootstraps across an episode boundary mid-stream.
    Returns (batch, ident_tags) — ident_tags[i] is the identity GROUP of step i.
    """
    net.eval()
    obs_l, act_l, lp_l, rew_l, val_l, done_l = [], [], [], [], [], []
    skm_l, enm_l, grm_l, tag_l = [], [], [], []
    ep = 0
    # team configs with at least one side >1 (exclude the plain 1v1)
    team_cfgs = [c for c in _TEAM_CONFIGS if c[0] + c[1] > 2]
    team_wts = [c[2] for c in team_cfgs]
    while len(rew_l) < n_steps:
        if _FOCUS_ONLY:
            # Diagnostic/targeted regime: lone FOCUS-class agent vs 2 standard
            # experts (the exact 1v2 gap), NO other-class / synth / monster
            # dilution. Isolates whether PPO erosion is population-averaging or
            # the shared-head/reward itself.
            ident = rng.choice(_FOCUS_ONLY)
            no_ = rng.choice([2, 2, 3])
            o_archs = [rng.choice(STANDARD_IDS) for _ in range(no_)]
            tag = ident
            env = CombatEnvV2(seed=seed * 1_000_003 + ep, n_agents=1, n_opps=no_)
            obs, _ = env.reset(agent_archs=[ident], opp_archs=o_archs,
                               level=5, opp_level=rng.randint(2, 5))
        elif team_cfgs and rng.random() < p_team:
            na, no_, _ = rng.choices(team_cfgs, weights=team_wts, k=1)[0]
            a_archs = [_sample_team_arch(rng, p_synth, ep * 16 + i)
                       for i in range(na)]
            o_archs = [rng.choice(STANDARD_IDS) for _ in range(no_)]
            lvl = rng.randint(3, 8)
            tag = "team"
            env = CombatEnvV2(seed=seed * 1_000_003 + ep,
                              n_agents=na, n_opps=no_)
            obs, _ = env.reset(agent_archs=a_archs, opp_archs=o_archs,
                               level=lvl, opp_level=lvl)
        else:
            ident, tag, a_lvl, opp, o_lvl = sample_episode_spec(
                rng, mon_pool, p_synth, p_mon_agent, p_mon_opp, ep)
            env = CombatEnvV2(seed=seed * 1_000_003 + ep, n_agents=1, n_opps=1)
            # Sample the layout HERE (instead of letting reset pick) so we can
            # pair a held-position enemy with a LoS-blocking layout — the only
            # regime that gives the LoS row a flank gradient. The choice uses
            # the same LAYOUTS pool reset would, so the distribution is intact.
            layout = rng.choice(LAYOUTS)
            obs, _ = env.reset(agent_archs=[ident], opp_archs=[opp],
                               level=a_lvl, opp_level=o_lvl, layout=layout)
            if (p_hold > 0.0 and layout in _LOS_BLOCKING_LAYOUTS
                    and rng.random() < p_hold):
                for oid in env.opp_ids:
                    env._opp_policies[oid] = _StationaryPolicy()
                tag = "hold"
        obs = blind_np_single(obs)
        done = False
        while not done:
            aid = env.current_agent_id
            ot = {k: torch.from_numpy(v).unsqueeze(0) for k, v in obs.items()}
            with torch.no_grad():
                action, lp, val, skm, enm, grm = _sample_action(
                    net, ot, env.resources, env.ws, aid)
            a = action.numpy().tolist()
            obs2, r, term, trunc, _ = env.step(a)
            obs_l.append(obs); act_l.append(a); lp_l.append(lp.numpy())
            rew_l.append(float(r)); val_l.append(val)
            skm_l.append(skm.numpy()); enm_l.append(enm.numpy())
            grm_l.append(grm.numpy())
            done = term or trunc
            done_l.append(bool(done)); tag_l.append(tag)
            obs = blind_np_single(obs2) if not done else obs2
        ep += 1

    rewards = np.array(rew_l, dtype=np.float32)
    values = np.array(val_l, dtype=np.float32)
    dones = np.array(done_l, dtype=np.float32)
    adv, ret = _compute_gae(rewards, values, dones, next_value=0.0,
                            gamma=gamma, gae_lambda=lam)
    batch = {
        "obs": {k: np.stack([o[k] for o in obs_l]) for k in obs_l[0]},
        "actions": np.array(act_l, dtype=np.int64),
        "log_probs": np.array(lp_l, dtype=np.float32),
        "skill_masks": np.stack(skm_l).astype(np.bool_),
        "entity_masks": np.stack(enm_l).astype(np.bool_),
        "grid_masks": np.stack(grm_l).astype(np.bool_),
        "rewards": rewards, "values": values,
        "returns": ret, "advantages": adv, "dones": dones,
    }
    return batch, tag_l, ep


# ── Gradient-conflict measurement ────────────────────────────────────────────

def grad_cosines(net, batch, tags, max_per_group=192):
    """Pairwise cosine of per-identity-group policy gradients (plain PG loss
    -logp*adv on the group's own samples — the same quantity the v16
    diagnosis used). Returns list of (gi, gj, cos)."""
    groups: dict[str, list[int]] = {}
    for i, t in enumerate(tags):
        groups.setdefault(t, []).append(i)
    groups = {t: idx[:max_per_group] for t, idx in groups.items()
              if len(idx) >= 32}
    params = [p for p in net.parameters() if p.requires_grad]
    flats = {}
    adv_all = torch.from_numpy(batch["advantages"])
    adv_all = (adv_all - adv_all.mean()) / (adv_all.std() + 1e-8)
    for tag, idx in groups.items():
        sel = np.array(idx)
        obs_b = {k: torch.from_numpy(v[sel]) for k, v in batch["obs"].items()}
        acts = torch.from_numpy(batch["actions"][sel])
        skm = torch.from_numpy(batch["skill_masks"][sel])
        enm = torch.from_numpy(batch["entity_masks"][sel])
        adv = adv_all[sel]
        end_l, skill_l, entity_l, grid_l = net(obs_b)
        skill_l = skill_l.masked_fill(skm, -1e9)
        entity_l = entity_l.masked_fill(enm.unsqueeze(1), -1e9)
        skill_l_play = skill_l.clone(); skill_l_play[..., 0] = -1e9
        b = torch.arange(len(sel))
        ended = acts[:, 0] == 0
        end_lp = torch.distributions.Bernoulli(logits=end_l).log_prob(
            ended.float())
        sk_safe = acts[:, 0].clamp(min=1)
        sk_lp = torch.distributions.Categorical(
            logits=skill_l_play).log_prob(sk_safe).masked_fill(ended, 0.0)
        ent_lp = torch.distributions.Categorical(
            logits=entity_l[b, sk_safe]).log_prob(
            acts[:, 1]).masked_fill(ended, 0.0)
        gr_lp = torch.distributions.Categorical(
            logits=grid_l[b, sk_safe]).log_prob(
            acts[:, 2]).masked_fill(ended, 0.0)
        loss = -((end_lp + sk_lp + ent_lp + gr_lp) * adv).mean()
        net.zero_grad(set_to_none=True)
        loss.backward()
        flats[tag] = torch.cat([
            (p.grad if p.grad is not None else torch.zeros_like(p)).flatten()
            for p in params]).detach().clone()
    net.zero_grad(set_to_none=True)
    out = []
    keys = sorted(flats)
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            a, c = flats[keys[i]], flats[keys[j]]
            cos = float((a @ c) / (a.norm() * c.norm() + 1e-12))
            out.append((keys[i], keys[j], cos))
    return out


# ── Evals (fixed seeds — within-run comparable) ──────────────────────────────

_HEAL_IDS = {sid for sid, ab in ABILITY_REGISTRY.items()
             if ab.features.expected_healing > 0}


def play_probe(net, agent_arch, opp_arch, seed, level=5, opp_level=None):
    """One greedy blinded game; returns (win, metrics dict)."""
    env = CombatEnvV2(seed=seed, n_agents=1, n_opps=1)
    obs, _ = env.reset(agent_archs=[agent_arch], opp_archs=[opp_arch],
                       level=level, opp_level=opp_level)
    aid = env.agent_ids[0]; oid = env.opp_ids[0]
    heals = 0; heal_hp = []
    stall = max_stall = 0
    last_hp_sum = None
    done = False
    while not done:
        actor = env.current_agent_id
        obs_b = blind_np_single(obs)
        ot = {k: torch.from_numpy(v).unsqueeze(0) for k, v in obs_b.items()}
        with torch.no_grad():
            el, s, e, g = net(ot)
        s = apply_resource_mask(s, env.resources, env.ws, actor)
        e = apply_entity_mask(e, ot, env.ws, actor)
        act = list(pick_action(el[0], s[0], e[0], g[0], ws=env.ws,
                               agent_id=actor))
        ag = env.ws.characters[actor]
        if act[0] > 0:
            sks = available_skills(ag, env.ws)
            if act[0] < len(sks) and sks[act[0]].skill_id in _HEAL_IDS:
                heals += 1
                heal_hp.append(ag.hp / max(1, ag.max_hp))
        hp_sum = sum(c.hp for c in env.ws.characters.values())
        if last_hp_sum is not None and hp_sum == last_hp_sum:
            stall += 1
            max_stall = max(max_stall, stall)
        else:
            stall = 0
        last_hp_sum = hp_sum
        obs, _, term, trunc, _ = env.step(act)
        done = term or trunc
    win = (not env.ws.characters[oid].is_alive()
           and env.ws.characters[aid].is_alive())
    return win, {"heals": heals, "heal_hp": heal_hp, "max_stall": max_stall}


def chimera_probe(net, games=2):
    """Quick degeneracy panel: WR + heals/game + heal timing + stall."""
    out = {}
    for arch in CHIMERA_IDS:
        wins = n = heals = 0
        hp_at = []
        worst_stall = 0
        for opp in STANDARD_IDS:
            base = stable_seed(f"pop_chim_{arch}_{opp}")
            for k in range(games):
                w, m = play_probe(net, arch, opp, base + k)
                wins += int(w); n += 1
                heals += m["heals"]; hp_at += m["heal_hp"]
                worst_stall = max(worst_stall, m["max_stall"])
        out[arch] = {
            "wr": wins / n, "heals_pg": heals / n,
            "heal_hp": float(np.mean(hp_at)) if hp_at else float("nan"),
            "max_stall": worst_stall,
        }
    return out


def standard_probe(net, games=1):
    """Mean WR over the 12x12 standard matrix (greedy, blinded)."""
    wins = n = 0
    per = {}
    for a in STANDARD_IDS:
        w_a = n_a = 0
        for o in STANDARD_IDS:
            base = stable_seed(f"pop_std_{a}_{o}")
            for k in range(games):
                w, _ = play_probe(net, a, o, base + k)
                w_a += int(w); n_a += 1
        per[a] = w_a / n_a
        wins += w_a; n += n_a
    return wins / n, per


def monster_opp_probe(net, mon_pool, games=2):
    """Mean WR of the net (driving standard classes) vs each viable monster at
    the FAIR pairing level (class @ round(equiv_level), monster @ natural_level).
    This is the brand-new capability: the opponent's identity is visible ONLY
    through the v4 descriptor (monsters read all-zero in the one-hot)."""
    wins = n = 0
    per = {}
    for mon in mon_pool:
        lvl = max(1, round(EQUIV_LEVEL_1V1[mon]))
        m_lvl = MONSTER_DEFS[mon].natural_level
        w_m = n_m = 0
        for a in STANDARD_IDS:
            base = stable_seed(f"pop_mon_{a}_{mon}")
            for k in range(games):
                w, _ = play_probe(net, a, mon, base + k,
                                  level=lvl, opp_level=m_lvl)
                w_m += int(w); n_m += 1
        per[mon] = w_m / n_m
        wins += w_m; n += n_m
    return wins / n, per


def fmt_chim(c):
    return "  ".join(
        f"{a.replace('chimera_', '')}: wr={v['wr']:.0%} heal={v['heals_pg']:.2f}/g"
        f"@{v['heal_hp']*100:.0f}%hp stall={v['max_stall']}"
        for a, v in c.items())


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--warm", type=str, default="models/distill_b/distill_e08.pt")
    p.add_argument("--out_dir", type=str, default="models/pop_b")
    p.add_argument("--updates", type=int, default=60)
    p.add_argument("--steps", type=int, default=2048)
    p.add_argument("--p_synth", type=float, default=0.5)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--ent_coef", type=float, default=0.01)
    p.add_argument("--value_warmup", type=int, default=3)
    p.add_argument("--anchor_batches", type=int, default=4,
                   help="BC replay minibatches per update (0 = no anchor)")
    p.add_argument("--dataset", type=str, default=str(DATASET_PATH))
    p.add_argument("--focus_only", type=str, default="",
                   help="comma-sep class ids: rollout ONLY lone-vs-2-standard "
                        "1v2/1v3 for these (no dilution). Isolates PPO erosion "
                        "cause; also the targeted-improve regime.")
    p.add_argument("--kit_anchor", type=str, default="",
                   help="dir with seed_kit_bc's kit_demos.npz + self_anchor.npz "
                        "(current-obs, blinded). When set, the BC-replay anchor "
                        "uses THESE instead of the distill dataset — keeps the "
                        "freshly-seeded expert kit actions from PPO erosion.")
    p.add_argument("--eval_every", type=int, default=5)
    p.add_argument("--chim_games", type=int, default=2)
    p.add_argument("--std_games", type=int, default=1)
    p.add_argument("--mon_games", type=int, default=1)
    p.add_argument("--p_mon_opp", type=float, default=0.35,
                   help="prob the opponent is a 1v1-viable monster")
    p.add_argument("--p_mon_agent", type=float, default=0.15,
                   help="prob the agent identity is a 1v1-viable monster")
    p.add_argument("--mon_max_level", type=float, default=8.0,
                   help="equiv-level ceiling for the 1v1 monster pool")
    p.add_argument("--p_team", type=float, default=0.0,
                   help="fraction of episodes that are multi-seat 1vN/Nv1/NvN "
                        "(classes+synths) — the arrangement the 1v1 history "
                        "never trained")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--p_hold", type=float, default=0.0,
                   help="fraction of LoS-blocking-layout (walls/pillar) 1v1 "
                        "episodes whose enemy HOLDS POSITION (never acts) — the "
                        "regime that forces the agent to flank to regain line "
                        "of sight. Safe under --train_grid_los_only: only the "
                        "LoS row learns, so this cannot distort normal play.")
    p.add_argument("--los_lr", type=float, default=2e-3,
                   help="learning rate for the grid LoS row under "
                        "--train_grid_los_only (open-invariant -> safe to make "
                        "aggressive; it must grow from 0 to ~L2 2 to flank)")
    p.add_argument("--los_boost_lr", type=float, default=0.0,
                   help="COMBINED mode (full net trains, NOT frozen): put the "
                        "grid LoS row (grad-masked to that one row) in its OWN "
                        "high-lr group while the rest of the net trains at --lr. "
                        "The LoS row must grow fast (from 0) so flank MOVE cells "
                        "complete; the trunk/skill-head learn the open-safe-ish "
                        "'move when blocked' decision gently at --lr (+anchor). "
                        "This is the freeze fix the bit-exact LoS-only mode "
                        "cannot reach (56%% of blocked turns are non-move skill "
                        "choices). 0 = off.")
    p.add_argument("--train_grid_los_only", action="store_true",
                   help="freeze ALL params except the grid-head's LoS-channel "
                        "weight (grid_query_projs last output row) + the critic. "
                        "In open fields the LoS channel is all-ones -> a constant "
                        "shift to every cell logit -> argmax unchanged -> open "
                        "behaviour (and 2v2) is BIT-EXACT vs base regardless of "
                        "how this weight moves. Only walled-layout cell choice "
                        "can change -> the model LEARNS to flank without any "
                        "global drift. Pair with a wall-heavy TRPG_LAYOUTS.")
    args = p.parse_args()

    global _FOCUS_ONLY
    if args.focus_only:
        _FOCUS_ONLY = [s.strip() for s in args.focus_only.split(",") if s.strip()]
        print(f"FOCUS-ONLY rollout: {_FOCUS_ONLY} (lone vs 2-3 standard)", flush=True)

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "BLIND").write_text("population fine-tune of blind student\n")
    register_chimeras()
    register_monsters()
    mon_pool = onev1_viable_monsters(args.mon_max_level)
    rng = random.Random(args.seed)
    torch.manual_seed(args.seed); np.random.seed(args.seed)

    net = load_student(args.warm)
    if args.train_grid_los_only:
        # ── LoS-channel-only fine-tune ───────────────────────────────────
        # Freeze the whole net, then re-enable ONLY: (a) the critic (a
        # standalone value net — training it sharpens advantages but cannot
        # touch the policy), and (b) the grid-head's LoS-channel weight, the
        # LAST output row of every grid_query_projs[i] (grid_query[:,los] =
        # how strongly a cell's "can I see the enemy from here" feature scores
        # it). A grad hook zeroes every other row so the rest of the projection
        # (terrain/entity/distance queries) stays frozen. In open fields the
        # LoS channel is all-ones (los_grid fast path) -> grid_query[los]*1 is
        # the SAME additive constant on every cell logit -> argmax over cells
        # is invariant to this weight -> open-field play (hence symmetric 2v2)
        # is bit-exact vs base no matter what this weight learns. Only walled
        # layouts (LoS varies cell-to-cell) get a behaviour change: the policy
        # can learn to prefer cells that restore line of sight = flank a wall.
        los_row = _SPATIAL_C - 1
        for p_ in net.parameters():
            p_.requires_grad_(False)
        for p_ in net.critic.parameters():
            p_.requires_grad_(True)

        def _los_row_only(row):
            def hook(grad):
                g = torch.zeros_like(grad)
                g[row] = grad[row]
                return g
            return hook

        los_params = []
        for proj in net.grid_query_projs:
            proj.weight.requires_grad_(True)
            proj.bias.requires_grad_(True)
            proj.weight.register_hook(_los_row_only(los_row))
            proj.bias.register_hook(_los_row_only(los_row))
            los_params += [proj.weight, proj.bias]
        # ALSO train the skill-head's blocked-skill block (last 64 input cols =
        # blockedness × sk_emb). Like the LoS row it is EXACTLY ZERO in open
        # (blockedness=0) → open skill choice stays bit-exact for any weights →
        # zero 2v2 cost — but it lets the policy learn to CHOOSE move (vs
        # buff/end) when wall-blocked, the half the grid LoS row can't reach.
        # Grad-masked to the trailing 64 columns so the rest of the head (open
        # skill choice) never moves.
        def _last_cols_only(ncols):
            def hook(grad):
                g = torch.zeros_like(grad)
                g[:, -ncols:] = grad[:, -ncols:]
                return g
            return hook

        for hd in net.skill_heads:
            hd.weight.requires_grad_(True)
            hd.weight.register_hook(_last_cols_only(64))
            los_params.append(hd.weight)
        net._los_params = los_params  # picked up by the optimizer below
        n_train = sum(p_.numel() for p_ in net.parameters() if p_.requires_grad)
        print(f"[train_grid_los_only] frozen except critic + grid_query LoS "
              f"row {los_row}/{_SPATIAL_C} + skill_head blocked block (last 64 "
              f"cols); trainable params={n_train} (grad-masked, open bit-exact)",
              flush=True)
    elif args.los_boost_lr > 0.0:
        # ── COMBINED mode: full net trains at --lr, LoS row at --los_boost_lr ─
        # Nothing is frozen. The grid LoS row is grad-masked (only row los_row
        # of grid_query_projs moves) and put in a high-lr group so flank MOVE
        # cells actually complete; the rest of the net (trunk/skill-head) trains
        # gently at --lr so it can also learn to CHOOSE move when blocked — the
        # part the bit-exact LoS-only mode structurally cannot fix. Heavy anchor
        # replay (--anchor_batches) pins open-field skill choice against drift.
        los_row = _SPATIAL_C - 1

        def _los_row_only(row):
            def hook(grad):
                g = torch.zeros_like(grad)
                g[row] = grad[row]
                return g
            return hook

        los_params = []
        for proj in net.grid_query_projs:
            proj.weight.register_hook(_los_row_only(los_row))
            proj.bias.register_hook(_los_row_only(los_row))
            los_params += [proj.weight, proj.bias]
        net._los_params = los_params
        print(f"[los_boost] full net @lr={args.lr}; grid LoS row "
              f"{los_row}/{_SPATIAL_C} grad-masked @lr={args.los_boost_lr}",
              flush=True)
    print(f"warm-start={args.warm}  p_synth={args.p_synth} "
          f"steps={args.steps} updates={args.updates} "
          f"anchor={args.anchor_batches}x{args.batch}", flush=True)
    print(f"monster pool ({len(mon_pool)} 1v1-viable @equiv<={args.mon_max_level}): "
          f"p_mon_opp={args.p_mon_opp} p_mon_agent={args.p_mon_agent}\n  "
          f"{mon_pool}", flush=True)

    anchor = None
    if args.anchor_batches > 0 and args.kit_anchor:
        # Current-obs anchor (seed_kit_bc output): kit_demos.npz teaches/keeps the
        # expert's high-value actions; self_anchor.npz pins std12 behaviour. Both
        # were saved as blinded CURRENT obs (build_obs on the live net), so NO
        # migration and NO re-blinding — this is the anti-erosion anchor for the
        # PPO-exceed phase (warm from a seed_kit checkpoint).
        kd = Path(args.kit_anchor)
        parts = []
        for fn in ("kit_demos.npz", "self_anchor.npz"):
            z = np.load(kd / fn)
            o = {k[4:]: z[k] for k in z.files if k.startswith("obs_")}
            parts.append((o, z["actions"], z["target_types"]))
        keys = parts[0][0].keys()
        obs_np = {k: np.concatenate([p[0][k] for p in parts], axis=0)
                  for k in keys}
        act_np = np.concatenate([p[1] for p in parts], axis=0)
        tt_np = np.concatenate([p[2] for p in parts], axis=0)
        anchor = (obs_np, act_np, tt_np)
        print(f"kit anchor: {len(act_np)} pairs (current-obs, no migration)",
              flush=True)
    elif args.anchor_batches > 0:
        obs_np, act_np, tt_np = load_dataset(Path(args.dataset))
        obs_np = {k: v.copy() for k, v in obs_np.items()}
        # pre-v4 distill dataset stores v3-width entities; migrate to v4 so the
        # anchor replays through the migrated net (rescale + zero descriptors).
        obs_np["entities"] = migrate_entities_v3_to_v4(obs_np["entities"])
        # pre-12j distill dataset stores 53-wide skill features; the net's
        # skill_proj is now 66-wide (damage-type soft one-hot appended at the
        # tail, cols 53-65). Zero-pad those new columns — bit-exact migration,
        # same contract as the obs-v4 entity descriptor padding.
        want = net.skill_proj.in_features
        sk = obs_np["skills"]
        if sk.shape[-1] < want:
            pad = np.zeros(sk.shape[:-1] + (want - sk.shape[-1],),
                           dtype=sk.dtype)
            obs_np["skills"] = np.concatenate([sk, pad], axis=-1)
        # pre-reaction-wave dataset lacks the decision_context channel; normal
        # turns carry the all-zero context (no reaction/legendary trigger), so
        # zero-fill is the bit-exact migration.
        if "decision_context" not in obs_np:
            from trpg.rl.obs import N_DECISION_CTX
            n = obs_np["skills"].shape[0]
            obs_np["decision_context"] = np.zeros((n, N_DECISION_CTX),
                                                  dtype=np.float32)
        # pre-los-grid dataset: the per-cell LoS channel. The BC anchor is
        # open-field expert data (no walls) → LoS everywhere = all ones, the
        # same no-wall convention los_grid_obs itself returns.
        if "los_grid" not in obs_np:
            from trpg.rl.obs import N_LOS_GRID_CHANNELS, N_GRID
            n = obs_np["skills"].shape[0]
            obs_np["los_grid"] = np.ones(
                (n, N_LOS_GRID_CHANNELS, N_GRID, N_GRID), dtype=np.float32)
        blind_self_identity_np(obs_np)
        anchor = (obs_np, act_np, tt_np)
        print(f"anchor dataset: {len(act_np)} pairs "
              f"(blinded, entities->v4 {obs_np['entities'].shape[-1]}d)",
              flush=True)

    if args.train_grid_los_only:
        # The LoS row is grad-masked AND open-invariant (bit-exact for any
        # magnitude), so it carries zero risk to open play no matter how fast it
        # moves — but it starts at exactly 0 and must grow to ~the magnitude of
        # the frozen distance query (L2~2) to outweigh the "close distance"
        # pull that drives melee into the wall. At the base lr (1e-4) that took
        # ~150 updates (measured: L2 0.045 after 7). So give the LoS row its own
        # aggressive lr; the critic keeps the gentle lr (it affects value, not
        # policy, and high lr would destabilise it).
        los_ids = {id(p_) for p_ in net._los_params}
        critic_ps = [p_ for p_ in net.parameters()
                     if p_.requires_grad and id(p_) not in los_ids]
        optim = torch.optim.Adam([
            {"params": net._los_params, "lr": args.los_lr},
            {"params": critic_ps, "lr": args.lr},
        ])
        print(f"  optimizer: LoS-row lr={args.los_lr}  critic lr={args.lr}",
              flush=True)
    elif args.los_boost_lr > 0.0:
        los_ids = {id(p_) for p_ in net._los_params}
        rest_ps = [p_ for p_ in net.parameters() if id(p_) not in los_ids]
        optim = torch.optim.Adam([
            {"params": net._los_params, "lr": args.los_boost_lr},
            {"params": rest_ps, "lr": args.lr},
        ])
        print(f"  optimizer: LoS-row lr={args.los_boost_lr}  rest lr={args.lr}",
              flush=True)
    else:
        optim = torch.optim.Adam(
            [p_ for p_ in net.parameters() if p_.requires_grad], lr=args.lr)

    print("baseline probes (update 0):", flush=True)
    chim0 = chimera_probe(net, games=args.chim_games)
    std0, _ = standard_probe(net, games=args.std_games)
    mon0, mon0_per = monster_opp_probe(net, mon_pool, games=args.mon_games)
    d0, r0 = descriptor_weight_norm(net)
    print(f"  chim  {fmt_chim(chim0)}")
    print(f"  std12 mean WR = {std0:.1%}")
    print(f"  vs-monsters mean WR = {mon0:.1%}")
    print(f"  descriptor-weight norm = {d0:.4f}  (rest={r0:.2f})  "
          f"<- starts ~0 after v4 migration\n", flush=True)

    for update in range(1, args.updates + 1):
        batch, tags, n_eps = collect_population_rollout(
            net, args.steps, seed=args.seed * 7919 + update,
            p_synth=args.p_synth, rng=rng, mon_pool=mon_pool,
            p_mon_agent=args.p_mon_agent, p_mon_opp=args.p_mon_opp,
            p_team=args.p_team, p_hold=args.p_hold)
        info = ppo_update(net, batch, optim, n_epochs=args.epochs,
                          batch_size=args.batch, ent_coef=args.ent_coef,
                          device="cpu",
                          value_only=(update <= args.value_warmup))
        line = (f"Update {update:3d}/{args.updates}"
                f"{' [warmup]' if update <= args.value_warmup else ''}  "
                f"eps={n_eps}  policy={info['policy_loss']:+.4f}  "
                f"value={info['value_loss']:.4f}  "
                f"entropy={info['entropy']:.3f}")
        if anchor is not None and update > args.value_warmup:
            a_obs, a_act, a_tt = anchor
            accs = []
            for _ in range(args.anchor_batches):
                sel = np.random.randint(0, len(a_act), size=args.batch)
                ob = {k: torch.from_numpy(v[sel]) for k, v in a_obs.items()}
                _, acc = bc_loss_step(net, ob,
                                      torch.from_numpy(a_act[sel]),
                                      torch.from_numpy(a_tt[sel]), optim)
                accs.append(acc.get("skill", float("nan")))
            line += f"  anchor_acc={np.nanmean(accs):.2f}"
        print(line, flush=True)

        if update % args.eval_every == 0:
            net.eval()
            cos = grad_cosines(net, batch, tags)
            if cos:
                worst = min(cos, key=lambda x: x[2])
                print(f"  grad-cos: mean={np.mean([c for _, _, c in cos]):+.3f}"
                      f"  worst={worst[2]:+.3f} ({worst[0]}~{worst[1]})"
                      f"  pairs={len(cos)}", flush=True)
            chim = chimera_probe(net, games=args.chim_games)
            std, _ = standard_probe(net, games=args.std_games)
            mon, _ = monster_opp_probe(net, mon_pool, games=args.mon_games)
            dN, rN = descriptor_weight_norm(net)
            snap = out_dir / f"pop_u{update:04d}.pt"
            torch.save(net.state_dict(), snap)
            print(f"  => chim  {fmt_chim(chim)}")
            print(f"  => std12 {std:.1%} (base {std0:.1%})  "
                  f"vs-mon {mon:.1%} (base {mon0:.1%})  "
                  f"desc-norm {dN:.4f} (base {d0:.4f})  snap={snap.name}",
                  flush=True)
            net.train()

    print("done.", flush=True)


if __name__ == "__main__":
    main()
