"""PyTorch policy network for CombatEnvV2.

Three-head output for the factored MultiDiscrete action:
  - skill head over N_SKILL_SLOTS, masked by skill_mask
  - entity head over N_ENTITY_SLOTS
  - grid head over N_GRID*N_GRID

Encoder mixes a small Transformer for skills, an MLP for entities, a
3x3 Conv for terrain, and an MLP for resources. Outputs are concatenated
into a shared context and decoded into the three heads.
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

    def forward(self, obs: dict) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        sk_emb, ent_emb, h, key_padding = self._encode(obs)

        h_skill     = h.unsqueeze(1).expand(-1, N_SKILL_SLOTS, -1)
        skill_logits = self.skill_head(torch.cat([sk_emb, h_skill], dim=-1)).squeeze(-1)
        skill_logits = skill_logits.masked_fill(key_padding, -1e9)

        h_ent         = h.unsqueeze(1).expand(-1, N_ENTITY_SLOTS, -1)
        entity_logits = self.entity_head(torch.cat([ent_emb, h_ent], dim=-1)).squeeze(-1)

        grid_logits = self.grid_head(h)
        return skill_logits, entity_logits, grid_logits

    def value(self, obs: dict) -> torch.Tensor:
        """Critic: estimate state value V(s). Returns [B]."""
        _, _, h, _ = self._encode(obs)
        return self.value_head(h).squeeze(-1)


def apply_resource_mask(skill_logits: "torch.Tensor",
                        resources: dict,
                        ws=None, agent_id: str = "") -> "torch.Tensor":
    """Mask invalid skill slots so the model never picks an unexecutable action.

    Slot layout (from available_skills()):
        0 = end
        1 = move
        2+ = weapons, spells, class abilities, dodge, hide

    Masks applied:
    - slot 0 (END):  blocked while action > 0
    - slot 1 (MOVE): blocked when movement == 0
    - weapon slots:  blocked when nearest enemy is outside weapon range
                     (requires ws + agent_id to be passed)
    """
    import torch
    has_action = resources.get("action", 0) > 0
    has_move   = resources.get("movement", 0.0) > 1e-6

    skill_logits = skill_logits.clone()
    if has_action:
        skill_logits[..., 0] = -1e9
    if not has_move:
        skill_logits[..., 1] = -1e9

    # Mask weapon-attack slots when nearest enemy is out of reach
    if ws is not None and agent_id and has_action:
        try:
            from .obs import partition_entities
            from ..engine.skill import available_skills
            agent = ws.characters[agent_id]
            _, enemies = partition_entities(ws, agent_id)
            if enemies:
                nearest = ws.characters[enemies[0]]
                dist = agent.position.distance_to(nearest.position)
                skills = available_skills(agent, ws)
                for i, sk in enumerate(skills):
                    if i >= skill_logits.shape[-1]:
                        break
                    if sk.skill_id.startswith("weapon:"):
                        # Get weapon range from the skill features
                        weapon_range = getattr(sk.features, "range_m", 1.5) or 1.5
                        if dist > weapon_range + 1e-6:
                            skill_logits[..., i] = -1e9
        except Exception:
            pass   # never crash inference due to mask logic

    return skill_logits
