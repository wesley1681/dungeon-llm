"""PyTorch policy network for CombatEnvV2.

Four-head output. The end/act binary head decouples "should I end my
sub-turn?" from "which skill should I use if I act?". This prevents the
~50% of expert pairs that are turn-terminations from collapsing the
skill distribution onto skill_idx=0 (the end-turn slot).

  - end head    scalar logit; sigmoid > 0.5 → end this sub-turn
  - skill head  over N_SKILL_SLOTS; trained only on act pairs;
                position 0 (end slot) is masked at inference
  - entity head over N_ENTITY_SLOTS
  - grid head   over N_GRID*N_GRID

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
        self.end_head = nn.Linear(hidden, 1)
        self.skill_head = nn.Linear(64 + hidden, 1)
        self.entity_head = nn.Linear(32 + hidden, 1)
        self.grid_head = nn.Linear(hidden, N_GRID * N_GRID)
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

        end_logit:    [B] — sigmoid > 0.5 → end this sub-turn
        skill_logits: [B, N_SKILL_SLOTS] — slot 0 (end) is for PPO compatibility
                      and is masked at BC inference time
        """
        sk_emb, ent_emb, h, key_padding = self._encode(obs)

        end_logit = self.end_head(h).squeeze(-1)

        h_skill     = h.unsqueeze(1).expand(-1, N_SKILL_SLOTS, -1)
        skill_logits = self.skill_head(torch.cat([sk_emb, h_skill], dim=-1)).squeeze(-1)
        skill_logits = skill_logits.masked_fill(key_padding, -1e9)

        h_ent         = h.unsqueeze(1).expand(-1, N_ENTITY_SLOTS, -1)
        entity_logits = self.entity_head(torch.cat([ent_emb, h_ent], dim=-1)).squeeze(-1)

        grid_logits = self.grid_head(h)
        return end_logit, skill_logits, entity_logits, grid_logits

    def value(self, obs: dict) -> torch.Tensor:
        """Critic: estimate state value V(s). Returns [B]."""
        _, _, h, _ = self._encode(obs)
        return self.value_head(h).squeeze(-1)


def pick_skill_idx(end_logit: "torch.Tensor",
                   skill_logits: "torch.Tensor") -> int:
    """Two-head inference: return final skill_idx for the action triplet.

    Decision:
      1. If every real skill (positions 1..N-1) is masked to -inf,
         end is forced regardless of end_head.
      2. Else if sigmoid(end_logit) > 0.5, end.
      3. Else, argmax over masked real skills.

    ``skill_logits`` is expected to already have unavailable slots masked
    (via key_padding from forward and apply_resource_mask). Position 0
    (end slot) is masked here so it never wins the act-branch argmax.
    """
    import torch
    sl = skill_logits.clone()
    sl[..., 0] = -1e9  # never pick end via skill argmax — end_head decides
    if (sl <= -1e8).all(dim=-1).item():
        return 0  # nothing feasible → forced end
    if torch.sigmoid(end_logit).item() > 0.5:
        return 0
    return int(sl.argmax(dim=-1).item())


def apply_entity_mask(entity_logits: "torch.Tensor", obs: dict) -> "torch.Tensor":
    """Mask dead/empty entity slots and self.

    Entity feature layout (from obs.py _entity_row):
        index 5 = is_alive
        index 6 = is_self
    """
    import torch
    entities = obs["entities"]
    if not isinstance(entities, torch.Tensor):
        entities = torch.from_numpy(entities)
    entities = entities.to(entity_logits.device)
    entity_logits = entity_logits.clone()
    entity_logits = entity_logits.masked_fill(entities[..., 5] < 0.5, -1e9)  # dead/empty
    entity_logits = entity_logits.masked_fill(entities[..., 6] > 0.5, -1e9)  # self
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
