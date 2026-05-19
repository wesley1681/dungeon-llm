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
                        resources: dict) -> "torch.Tensor":
    """Mask skill slot 0 (END) when the agent still has resources to spend.

    Called at inference time to prevent the model from ending its turn when
    it has actions/bonus-actions remaining — a common BC failure mode when
    the resources signal is under-learned.

    Slot 0 is always END (see available_skills() — 'end' is always first).
    """
    import torch
    has_action = resources.get("action", 0) > 0
    has_move   = resources.get("movement", 0.0) > 1e-6

    skill_logits = skill_logits.clone()
    if has_action:
        # Prevent END while the agent still has their main action to spend.
        skill_logits[..., 0] = -1e9
    if not has_move:
        # Prevent MOVE when movement budget is exhausted (slot 1 is always MOVE).
        skill_logits[..., 1] = -1e9
    return skill_logits
