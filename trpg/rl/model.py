"""PyTorch policy network for CombatEnvV2.

Hierarchical four-head action: end -> skill -> (entity | grid | nothing),
where the per-skill conditional structure mirrors the engine's action
semantics.

  - end head    scalar logit; sigmoid > 0.5 → end this sub-turn
  - skill head  over N_SKILL_SLOTS; trained only on act pairs;
                position 0 (end slot) is masked at inference
  - entity head [N_SKILL_SLOTS, N_ENTITY_SLOTS] — *per-skill* logits.
                During training only the (skill_idx, :) slice is supervised,
                and only when the skill's target_type needs an entity (ATTACK,
                HEAL, single/multi target spell). For POINT spells like
                fireball, this head's logits never receive a gradient.
  - grid head   [N_SKILL_SLOTS, N_GRID*N_GRID] — *per-skill* logits.
                Trained only when the skill's target_type is POINT/LINE/CONE.
                For melee attacks, this head is never trained.

The per-skill conditioning is the cure for the joint-vs-marginal problem:
the old shared heads were forced to fit one distribution across all
skills, so grid_head collapsed onto cell 0 (95% of training samples) and
entity_head onto "self" (every POINT spell's fallback). With per-skill
heads, each (skill, sub-head) pair sees only the samples relevant to it.

Encoder mixes a small Transformer for skills, an MLP for entities, a
3x3 Conv for terrain, and an MLP for resources. Outputs are concatenated
into a shared context and decoded into the four heads.
"""
from __future__ import annotations
import torch
import torch.nn as nn

from .obs import N_SKILL_SLOTS, N_ENTITY_SLOTS, ENTITY_DIM, N_GRID
from ..engine.skill import SKILL_FEATURE_DIM


class CombatPolicyNet(nn.Module):
    def __init__(self, hidden: int = 128):
        super().__init__()
        # Skills: TransformerEncoder over the 20 slots
        self.skill_proj = nn.Linear(SKILL_FEATURE_DIM, 64)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=64, nhead=4, dim_feedforward=128, batch_first=True,
        )
        self.skill_encoder = nn.TransformerEncoder(enc_layer, num_layers=2)

        # Entities: shared MLP per row
        self.entity_mlp = nn.Sequential(
            nn.Linear(ENTITY_DIM, 32), nn.ReLU(), nn.Linear(32, 32),
        )

        # Terrain: small CNN
        self.terrain_cnn = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(8, 8, kernel_size=3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(4),
        )
        self.terrain_proj = nn.Linear(8 * 4 * 4, 32)

        # Resources
        self.res_mlp = nn.Sequential(nn.Linear(4, 16), nn.ReLU())

        # Shared trunk
        # 64 (skills mean) + 32 (entities mean) + 32 (terrain) + 16 (res) = 144
        self.trunk = nn.Sequential(
            nn.Linear(144, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )

        # Heads — policy
        # end_head: scalar binary "end vs act"
        self.end_head = nn.Linear(hidden, 1)
        # skill_head: per-skill-slot logit (existing)
        self.skill_head = nn.Linear(64 + hidden, 1)
        # entity_head: per (skill, entity) — Linear over (sk_emb, h, ent_emb)
        # gives one scalar per (skill_slot, entity_slot) pair.
        self.entity_head = nn.Linear(64 + hidden + 32, 1)
        # grid_head: per skill — Linear over (sk_emb, h) gives N_GRID*N_GRID
        # logits for each skill_slot.
        self.grid_head = nn.Linear(64 + hidden, N_GRID * N_GRID)
        # Value head — critic (shared encoder, separate output)
        self.value_head = nn.Linear(hidden, 1)

    def _encode(self, obs: dict):
        """Shared encoder. Returns (sk_emb, ent_emb, h, key_padding)."""
        skills     = obs["skills"]
        skill_mask = obs["skill_mask"]
        entities   = obs["entities"]
        resources  = obs["resources"]
        terrain    = obs["terrain"]

        sk_proj     = self.skill_proj(skills)
        key_padding = skill_mask < 0.5
        sk_emb      = self.skill_encoder(sk_proj, src_key_padding_mask=key_padding)
        sk_mean     = (sk_emb * skill_mask.unsqueeze(-1)).sum(1) / skill_mask.sum(1, keepdim=True).clamp(min=1.0)

        ent_emb  = self.entity_mlp(entities)
        ent_mean = ent_emb.mean(1)

        terr     = self.terrain_cnn(terrain.unsqueeze(1)).flatten(1)
        terr_ctx = self.terrain_proj(terr)

        res_ctx = self.res_mlp(resources)

        ctx = torch.cat([sk_mean, ent_mean, terr_ctx, res_ctx], dim=-1)
        h   = self.trunk(ctx)
        return sk_emb, ent_emb, h, key_padding

    def forward(self, obs: dict) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns (end_logit, skill_logits, entity_logits, grid_logits).

        end_logit:    [B]
        skill_logits: [B, N_SKILL_SLOTS]
        entity_logits:[B, N_SKILL_SLOTS, N_ENTITY_SLOTS] — per-skill
        grid_logits:  [B, N_SKILL_SLOTS, N_GRID*N_GRID]  — per-skill
        """
        sk_emb, ent_emb, h, key_padding = self._encode(obs)
        B = h.shape[0]

        end_logit = self.end_head(h).squeeze(-1)

        # skill_head: per skill slot
        h_skill     = h.unsqueeze(1).expand(-1, N_SKILL_SLOTS, -1)
        skill_logits = self.skill_head(torch.cat([sk_emb, h_skill], dim=-1)).squeeze(-1)
        skill_logits = skill_logits.masked_fill(key_padding, -1e9)

        # entity_head per (skill, entity): combine (sk_emb[skill], h, ent_emb[entity])
        # Build [B, N_SKILL, N_ENTITY, 64+128+32] via broadcasting:
        sk_for_ent  = sk_emb.unsqueeze(2).expand(-1, -1, N_ENTITY_SLOTS, -1)   # [B, S, E, 64]
        h_for_ent   = h.unsqueeze(1).unsqueeze(2).expand(-1, N_SKILL_SLOTS, N_ENTITY_SLOTS, -1)  # [B, S, E, 128]
        ent_for_ent = ent_emb.unsqueeze(1).expand(-1, N_SKILL_SLOTS, -1, -1)   # [B, S, E, 32]
        ent_in      = torch.cat([sk_for_ent, h_for_ent, ent_for_ent], dim=-1)
        entity_logits = self.entity_head(ent_in).squeeze(-1)                   # [B, S, E]

        # grid_head per skill: combine (sk_emb[skill], h)
        h_for_grid = h.unsqueeze(1).expand(-1, N_SKILL_SLOTS, -1)              # [B, S, 128]
        grid_in    = torch.cat([sk_emb, h_for_grid], dim=-1)                   # [B, S, 64+128]
        grid_logits = self.grid_head(grid_in)                                  # [B, S, N_GRID*N_GRID]

        return end_logit, skill_logits, entity_logits, grid_logits

    def value(self, obs: dict) -> torch.Tensor:
        """Critic: estimate state value V(s). Returns [B]."""
        _, _, h, _ = self._encode(obs)
        return self.value_head(h).squeeze(-1)


def pick_action(end_logit: "torch.Tensor",
                skill_logits: "torch.Tensor",
                entity_logits: "torch.Tensor",
                grid_logits: "torch.Tensor",
                ws=None, agent_id: str = "") -> tuple[int, int, int]:
    """Hierarchical inference: return the action triplet (skill, entity, grid).

    Pipeline:
      1. End decision via end_head. Forced if every real skill is masked.
      2. Skill argmax (slot 0 excluded so end can't sneak in here).
      3. Look up the chosen skill's target_type:
            SELF             → entity=0, grid=0 (irrelevant; engine ignores)
            SINGLE_*/MULTI_* → argmax entity_logits[skill, :], grid=0
            POINT/LINE/CONE  → entity=0, argmax grid_logits[skill, :]

    ``skill_logits`` is expected to already be masked (key_padding +
    apply_resource_mask). ``entity_logits`` and ``grid_logits`` are the
    per-skill heads from ``forward()``.
    """
    import torch
    sl = skill_logits.clone()
    sl[..., 0] = -1e9  # never end via skill argmax — end_head decides
    if (sl <= -1e8).all(dim=-1).item():
        return (0, 0, 0)  # nothing feasible → forced end
    if torch.sigmoid(end_logit).item() > 0.5:
        return (0, 0, 0)

    skill_idx = int(sl.argmax(dim=-1).item())

    if ws is None or not agent_id:
        # Fallback when caller can't supply ws (degenerate; just argmax both)
        ent_idx = int(entity_logits[skill_idx].argmax().item())
        grid    = int(grid_logits[skill_idx].argmax().item())
        return (skill_idx, ent_idx, grid)

    from ..engine.skill import available_skills, TargetType
    agent = ws.characters[agent_id]
    skills = available_skills(agent, ws)
    if skill_idx >= len(skills):
        return (0, 0, 0)
    tt = skills[skill_idx].features.target_type

    ENTITY_TT = (TargetType.SINGLE_ENEMY, TargetType.SINGLE_ALLY,
                 TargetType.MULTI_ENEMY, TargetType.MULTI_ALLY)
    GRID_TT   = (TargetType.POINT, TargetType.LINE, TargetType.CONE)

    if tt in ENTITY_TT:
        ent_idx = int(entity_logits[skill_idx].argmax().item())
        return (skill_idx, ent_idx, 0)
    if tt in GRID_TT:
        grid = int(grid_logits[skill_idx].argmax().item())
        return (skill_idx, 0, grid)
    # SELF or unrecognised: both irrelevant
    return (skill_idx, 0, 0)


def apply_entity_mask(entity_logits: "torch.Tensor", obs: dict) -> "torch.Tensor":
    """Mask dead/empty entity slots and self.

    Handles both shapes:
      [B, N_ENTITY]              — legacy shared head
      [B, N_SKILL, N_ENTITY]     — per-skill head
    The entity feature layout (from obs.py _entity_row):
        index 5 = is_alive
        index 6 = is_self
    """
    import torch
    entities = obs["entities"]
    if not isinstance(entities, torch.Tensor):
        entities = torch.from_numpy(entities)
    entities = entities.to(entity_logits.device)
    is_alive = entities[..., 5]                       # [B, N_ENTITY]
    is_self  = entities[..., 6]                       # [B, N_ENTITY]
    # For [B, N_SKILL, N_ENTITY], broadcast the mask across the skill axis.
    if entity_logits.dim() == 3:
        is_alive = is_alive.unsqueeze(1)              # [B, 1, N_ENTITY]
        is_self  = is_self.unsqueeze(1)
    entity_logits = entity_logits.clone()
    entity_logits = entity_logits.masked_fill(is_alive < 0.5, -1e9)
    entity_logits = entity_logits.masked_fill(is_self  > 0.5, -1e9)
    return entity_logits


def apply_resource_mask(skill_logits: "torch.Tensor",
                        resources: dict,
                        ws=None, agent_id: str = "") -> "torch.Tensor":
    """Mask skill slots that the engine would reject.

    Iterates the current `available_skills()` list and masks any skill that:
      - is MOVE while the agent has no remaining movement budget
      - is a non-SELF skill whose range_m < distance to nearest enemy

    Slot positions are dynamic — they come from `available_skills`, so this
    function must look up each skill's features/skill_id, never assume a slot.
    """
    import torch
    skill_logits = skill_logits.clone()
    if ws is None or not agent_id:
        return skill_logits

    from .obs import partition_entities
    from ..engine.skill import available_skills, TargetType
    agent = ws.characters[agent_id]
    allies, enemies = partition_entities(ws, agent_id)
    enemy_dist: float | None = None
    if enemies:
        enemy_dist = agent.position.distance_to(ws.characters[enemies[0]].position)
    ally_dist: float | None = None
    if allies:
        ally_dist = agent.position.distance_to(ws.characters[allies[0]].position)

    # Fallback range for melee class abilities that leave range_m=0 in their
    # features (engine resolves the actual reach from the wielded weapon).
    fallback_reach = 1.5
    if agent.weapons:
        try:
            fallback_reach = float(agent.get_weapon().range_normal)
        except Exception:
            pass

    has_move   = resources.get("movement", 0.0) > 1e-6
    has_action = resources.get("action", 0) > 0
    has_bonus  = resources.get("bonus_action", 0) > 0
    skills = available_skills(agent, ws)
    for i, sk in enumerate(skills):
        if i >= skill_logits.shape[-1]:
            break
        if sk.skill_id == "move":
            if not has_move:
                skill_logits[..., i] = -1e9
            continue
        # Resource cost gates — engine accepts these without complaint when
        # consumes is empty, but the skill genuinely needs the slot to do
        # anything meaningful. Without this gate the policy can spam rage /
        # other bonus-action skills 3-4 times per turn while bonus_action is
        # already 0 (observed in watch_model with bc_v8 berserker).
        if sk.features.cost_action > 0 and not has_action:
            skill_logits[..., i] = -1e9
            continue
        if sk.features.cost_bonus > 0 and not has_bonus:
            skill_logits[..., i] = -1e9
            continue
        tt = sk.features.target_type
        if tt == TargetType.SELF:
            continue
        # Pick the relevant target distance for the skill's target type.
        if tt == TargetType.SINGLE_ALLY:
            target_dist = ally_dist if ally_dist is not None else (
                0.0 if agent.weapons else None  # self-heal via touch is OK
            )
            # Heal-touch skills target self when no ally → distance is 0
            if not allies:
                target_dist = 0.0
        else:
            target_dist = enemy_dist
        if target_dist is None:
            skill_logits[..., i] = -1e9
            continue
        rng = float(getattr(sk.features, "range_m", 0.0) or 0.0)
        if rng <= 0.0:
            rng = fallback_reach
        # Engine uses strict inequality (distance > range = out of range), so
        # distance == range gets rejected. Mask conservatively: anything at or
        # within 1 cm of the range boundary is treated as out-of-range.
        if target_dist >= rng - 0.01:
            skill_logits[..., i] = -1e9

    return skill_logits
