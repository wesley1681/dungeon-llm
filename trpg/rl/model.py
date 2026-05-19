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

        # Heads
        # Skill head reuses per-slot embeddings (point-wise scoring)
        self.skill_head = nn.Linear(64 + hidden, 1)
        self.entity_head = nn.Linear(32 + hidden, 1)
        self.grid_head = nn.Linear(hidden, N_GRID * N_GRID)

    def forward(self, obs: dict) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        skills = obs["skills"]            # [B, S, 53]
        skill_mask = obs["skill_mask"]    # [B, S]
        entities = obs["entities"]        # [B, E, 8]
        resources = obs["resources"]      # [B, 4]
        terrain = obs["terrain"]          # [B, G, G]
        B = skills.shape[0]

        # Skill encoder (mask: True = pad, so invert)
        sk_proj = self.skill_proj(skills)
        key_padding = skill_mask < 0.5    # [B, S], True = pad
        sk_emb = self.skill_encoder(sk_proj, src_key_padding_mask=key_padding)
        sk_mean = (sk_emb * skill_mask.unsqueeze(-1)).sum(1) / skill_mask.sum(1, keepdim=True).clamp(min=1.0)

        # Entities
        ent_emb = self.entity_mlp(entities)              # [B, E, 32]
        ent_mean = ent_emb.mean(1)                       # [B, 32]

        # Terrain
        terr = terrain.unsqueeze(1)                      # [B, 1, G, G]
        terr = self.terrain_cnn(terr).flatten(1)         # [B, 128]
        terr_ctx = self.terrain_proj(terr)               # [B, 32]

        # Resources
        res_ctx = self.res_mlp(resources)                # [B, 16]

        # Shared
        ctx = torch.cat([sk_mean, ent_mean, terr_ctx, res_ctx], dim=-1)  # [B, 144]
        h = self.trunk(ctx)                              # [B, hidden]

        # Heads
        h_skill = h.unsqueeze(1).expand(-1, N_SKILL_SLOTS, -1)
        skill_logits = self.skill_head(torch.cat([sk_emb, h_skill], dim=-1)).squeeze(-1)
        # Apply mask: pad slots -> -inf
        skill_logits = skill_logits.masked_fill(key_padding, -1e9)

        h_ent = h.unsqueeze(1).expand(-1, N_ENTITY_SLOTS, -1)
        entity_logits = self.entity_head(torch.cat([ent_emb, h_ent], dim=-1)).squeeze(-1)

        grid_logits = self.grid_head(h)
        return skill_logits, entity_logits, grid_logits
