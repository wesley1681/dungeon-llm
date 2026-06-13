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

from .obs import (
    N_SKILL_SLOTS, N_ENTITY_SLOTS, ENTITY_DIM, N_GRID,
    N_ENTITY_GRID_CHANNELS, N_DISTANCE_GRID_CHANNELS, END_FEATURES_DIM,
    N_ARCHETYPES, N_V3_EXTRA, N_V4_DESC, N_V5_TRAIT, ENEMY_SLOT_START,
    I_ENT_LEVEL, I_ENT_MAXHP, I_DESC_RESIST, I_ENT_ENEMY,
    LEVEL_NORM, MAXHP_NORM, V3_LEVEL_NORM, V3_MAXHP_NORM,
    N_DECISION_CTX,
)
from ..engine.skill import SKILL_FEATURE_DIM, SKILL_DTYPE_START
from ..engine.damage import N_DAMAGE_TYPES


# entities[:, 0] is the agent itself. Indices 7..7+N_ARCHETYPES-1 store the
# self archetype one-hot — see obs._entity_row's layout comment.
_ARCH_OH_START = 7
_ARCH_OH_END   = 7 + N_ARCHETYPES

# Frozen obs-era entity widths, used by the checkpoint migration chain
# (adapt_state_dict_for_obs_v3 / _v4 / _v5). The LIVE width is obs.ENTITY_DIM;
# these describe dead file formats and must never track future obs growth.
# Each is derived by peeling the LAST-appended tail off the live width, so they
# stay correct as new tails are added (peel v5 trait → v4, then v4 desc → v3…).
_ENTITY_DIM_V4 = ENTITY_DIM - N_V5_TRAIT    # pre-v5: v4 descriptor, no trait tail
_ENTITY_DIM_V3 = _ENTITY_DIM_V4 - N_V4_DESC  # v3: base + one-hots + v3 tail
_ENTITY_DIM_V2 = _ENTITY_DIM_V3 - N_V3_EXTRA  # pre-v3: no tail
# Frozen pre-dtype skill-feature width (2026-06-12i schema surgery added the
# damage-type tail + the matchup-join head input). File-format constant for
# adapt_state_dict_for_skill_dtype — never track live SKILL_FEATURE_DIM.
_SKILL_DIM_V1 = 53


# CNN removed. spatial_feat is now the raw stack of pre-computed per-cell
# channels — terrain + entity overlay + distance grids. CNN was bottlenecking
# grid_head because Euclidean distance to entities is a global quantity that
# a local 3x3 kernel can't compute. With distance pre-computed as channels,
# grid_head only needs a linear scoring (grid_query · cell_features), which
# is what bmm in forward() does.
_SPATIAL_C = 1 + N_ENTITY_GRID_CHANNELS + N_DISTANCE_GRID_CHANNELS  # terrain + entity + distance

# Pre-v3 slot layout, frozen as a file-format constant for checkpoint
# migration: [self, 2 allies, 3 enemies]. In the v3 slot layout those six
# entities live at rows [0, 1, 2, ENEMY_SLOT_START..+2] — the rows a migrated
# pre-v3 checkpoint's attention is restricted to (see attn_legacy buffer).
_LEGACY_N_ALLY  = 2
_LEGACY_N_SLOTS = 6
_LEGACY_ATTN_ROWS = (list(range(1 + _LEGACY_N_ALLY))
                     + list(range(ENEMY_SLOT_START,
                                  ENEMY_SLOT_START + (_LEGACY_N_SLOTS - 1 - _LEGACY_N_ALLY))))


class _CriticNet(nn.Module):
    """Standalone value network, decoupled from the policy encoder.

    Reads the scalars that actually determine V(s) — the entity rows (HP
    fractions, positions, distance-to-self, alive/enemy flags, per-entity
    status), resources, and end_features (hp, action economy, enemies/allies
    alive) — through a small MLP. Keeping it separate means value MSE never
    perturbs the policy representation, and V(s) is fit directly instead of off
    frozen policy features (which under-fit and gave PPO noisy advantages).
    """

    def __init__(self, hidden: int = 128):
        super().__init__()
        # + SKILL_FEATURE_DIM: a masked mean over the currently-available skill
        # features. Without it the critic could not tell "champion WITH
        # action_surge / a maneuver die available" from "without" — so V(s) was
        # blind to combo-resource state and PPO got biased advantages that
        # steered the melee combo classes AWAY from attacking. Pooling the
        # available-skill features lets the critic value loaded combo potential.
        # + N_DECISION_CTX at the END: the value of a reaction / legendary
        # decision state differs from a normal turn (e.g. "I'm about to be hit
        # for X" vs "my turn"). Zero on normal turns → migrated checkpoints stay
        # bit-exact (the appended columns are zero-padded; zero input → 0).
        in_dim = (N_ENTITY_SLOTS * ENTITY_DIM + 4 + END_FEATURES_DIM
                  + SKILL_FEATURE_DIM + N_DECISION_CTX)
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, obs: dict) -> torch.Tensor:
        ent = obs["entities"].flatten(1)              # [B, N_ENTITY*ENTITY_DIM]
        m = obs["skill_mask"].unsqueeze(-1)           # [B, N_SLOT, 1] 1=valid
        skill_pool = (obs["skills"] * m).sum(1) / m.sum(1).clamp(min=1.0)  # [B, F]
        x = torch.cat([ent, obs["resources"], obs["end_features"], skill_pool,
                       obs["decision_context"]], dim=-1)
        return self.net(x).squeeze(-1)


class CombatPolicyNet(nn.Module):
    def __init__(self, hidden: int = 128, n_head_groups: int | None = None):
        """``n_head_groups``: how many sibling copies of the skill/entity/grid
        heads to build. Default N_ARCHETYPES = the per-archetype-head design
        (identity-routed). 1 = a single shared head group — used by the
        distilled student nets, where supervised targets are fixed so the
        cross-class gradient conflict that motivated per-arch heads does not
        exist. Checkpoints are only shape-compatible with the same group
        count they were saved with."""
        super().__init__()
        # Skills: TransformerEncoder over the 20 slots
        self.skill_proj = nn.Linear(SKILL_FEATURE_DIM, 64)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=64, nhead=4, dim_feedforward=128, batch_first=True,
        )
        self.skill_encoder = nn.TransformerEncoder(enc_layer, num_layers=2)

        # Entities: shared MLP per row. First layer is widened past ENTITY_DIM
        # so the per-entity status multi-hot (sparse, ~27 bits) isn't crushed
        # against the small projection. Output stays at 32 — downstream sizing
        # unchanged.
        self.entity_mlp = nn.Sequential(
            nn.Linear(ENTITY_DIM, 64), nn.ReLU(), nn.Linear(64, 32),
        )

        # No CNN. spatial_feat is the raw stack of pre-computed per-cell
        # channels (terrain + entity overlay + distance grids). Local pattern
        # detection (e.g. "wall adjacent") can be added later by precomputing
        # those features too — bypasses CNN's local-only inductive bias.
        self._spatial_total_c = _SPATIAL_C

        # Resources
        self.res_mlp = nn.Sequential(nn.Linear(4, 16), nn.ReLU())

        # Shared trunk — NO global mean pools. Position info reaches `h`
        # through self-indexed reads instead of dilution-by-averaging:
        #   - self_emb     = ent_emb[:, 0]                  (32)  ← self row, not entity mean
        #   - self_cell_feat = spatial_feat at self's cell  (_spatial_total_c)
        #                                                          ← single cell, not 30x30 pool
        # Old design pooled both (`ent_emb.mean(1)` and `spatial_feat.mean((2,3))`),
        # which destroyed positional info before it ever reached `h`, causing
        # the encoder to output near-constant representations across all obs.
        # + N_DECISION_CTX at the END of the trunk concat: the decision-context
        # vector (reaction/legendary trigger info) reaches every head through h.
        # Zero on normal turns → the appended trunk columns (zero-padded on
        # migrated checkpoints) contribute exactly 0, so normal-turn behaviour is
        # bit-exact forever, however these columns later train (a zero INPUT kills
        # the product regardless of the weight — that is the invariance).
        _trunk_in = 64 + 32 + self._spatial_total_c + 16 + N_DECISION_CTX
        self.trunk = nn.Sequential(
            nn.Linear(_trunk_in, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )

        # FiLM modulation on trunk output. Two linear projections from the
        # self archetype one-hot produce per-archetype (gamma, beta) vectors
        # the same width as ``h``, then we apply h' = gamma*h + beta. Reason:
        # under shared-encoder PPO every archetype's gradient overwrites the
        # same trunk weights, so a strong-signal archetype (e.g. vengeance)
        # drifts the encoder in directions that destroy a weak-signal one
        # (battle_master collapsed from 49% to 4% in v16). FiLM gives each
        # archetype a private scale+shift on h so per-archetype optimisation
        # can't sideswipe the others. Initialised so gamma=1, beta=0 — the
        # module is identity at start, so a BC checkpoint loaded with
        # strict=False is unchanged until FiLM weights actually train.
        self.film_gamma = nn.Linear(N_ARCHETYPES, hidden)
        self.film_beta  = nn.Linear(N_ARCHETYPES, hidden)
        nn.init.zeros_(self.film_gamma.weight)
        nn.init.ones_(self.film_gamma.bias)
        nn.init.zeros_(self.film_beta.weight)
        nn.init.zeros_(self.film_beta.bias)

        # Heads — policy
        # end_head: scalar binary "end vs act". DECOUPLED from the shared
        # encoder — reads from a small hand-picked end_features vector instead
        # of the shared trunk output `h`. Reason: when an archetype loses
        # every game (e.g. wizards vs much stronger experts), all skill/entity
        # gradients push the shared encoder to slide `h` along end_head's
        # weight direction (the "lazy gradient path" — one dim of motion
        # reduces every skill's joint probability via the (1-σ(end)) factor).
        # That drift forces end_p → 1 even though end_head's own weights
        # barely change. Routing end_head through end_features means the
        # shared encoder's drift has zero pathway to end_p.
        self.end_head = nn.Sequential(
            nn.Linear(END_FEATURES_DIM, 32), nn.ReLU(),
            nn.Linear(32, 1),
        )
        # skill_head: per-skill-slot logit (existing)
        # Projects skill embeddings (64-d) to entity space (32-d) so each
        # skill can attend over entity embeddings in forward(), giving
        # skill_head per-entity resolution without mean-pooling dilution.
        self.skill_ent_attn_proj = nn.Linear(64, 32)
        # Per-archetype heads. Each of skill / entity / grid has 12 sibling
        # copies — at forward time we look up the agent's archetype id (from
        # the self entity's one-hot) and read THAT copy's output. Reason:
        # under fully-shared heads, the cross-class gradient cosine was -0.55
        # on average and reached -0.99 for some archetype pairs, so any
        # gradient step that improves vengeance literally undoes battle_master
        # on the head weights — measured directly. FiLM only fixed the encoder
        # path; the heads themselves remained shared and limited the ceiling
        # to ~41%. Per-archetype heads cut the conflict at the heads, at the
        # cost of 12× the per-head parameter count (heads are tiny — total
        # adds ~25k params — so the wall-clock hit is negligible).
        #
        # Loading: when a checkpoint pre-dates per-arch heads, ``load_perarch``
        # below tiles the shared weights into all 12 copies so warm-starts
        # don't lose the shared-head behaviour the BC step produced.
        from .obs import N_ARCHETYPES as _N_ARCH
        n_groups = _N_ARCH if n_head_groups is None else n_head_groups
        # +1: the typed-matchup join — skill row's damage-type soft one-hot
        # dotted with an entity's typed-resist descriptor slice, computed in
        # forward() from RAW obs columns with NO per-type weights. Because the
        # dot product is shared across all 13 damage-type columns, whatever
        # the head learns about "join < 0 ⇒ this skill's damage is resisted"
        # transfers zero-shot to types never seen in training — per-type
        # one-hot weights alone cannot do that (orthogonal columns get no
        # gradient until their type appears). skill_head sees the best join
        # over present enemies; entity_head sees the per-(skill, entity) join.
        self.skill_heads = nn.ModuleList(
            [nn.Linear(64 + hidden + 32 + 1, 1) for _ in range(n_groups)])
        self.entity_heads = nn.ModuleList(
            [nn.Linear(64 + hidden + 32 + 1, 1) for _ in range(n_groups)])
        self.grid_query_projs = nn.ModuleList(
            [nn.Linear(64 + hidden, self._spatial_total_c) for _ in range(n_groups)])
        # Pre-v3 compatibility flag, carried INSIDE the checkpoint. 0 = native
        # v3 net (presence-masked attention). 1 = migrated pre-v3 checkpoint:
        # restrict the skill→entity attention softmax to the LEGACY slot rows,
        # which reproduces the old attention bit-exactly (each legacy row's
        # embedding is unchanged — the new feature columns are zero-padded —
        # and the softmax sees the same 6 logits, pad rows included). Without
        # this, old checkpoints shift behaviour either way: widening to 10
        # unmasked rows broke berserker (-8pp), presence-masking broke
        # devotion (-13pp) — both measured, diag_obsv3_shift.py. Wave retrains
        # clear the flag (train scripts) and adopt the presence mask.
        self.register_buffer("attn_legacy", torch.zeros(()))
        # Critic — a STANDALONE value network (own weights, own simple encoder),
        # decoupled from the policy. See value() for the rationale: a value head
        # bolted onto the (detached) policy encoder under-fit badly and stalled
        # PPO. The critic reads the value-relevant scalars directly.
        self.critic = _CriticNet(hidden)

    # Pre-v3 entity slot layout (module-level constants above). Live code
    # must use the obs.py slot constants — these describe a dead format.
    _LEGACY_N_ALLY  = _LEGACY_N_ALLY
    _LEGACY_N_SLOTS = _LEGACY_N_SLOTS

    @staticmethod
    def adapt_state_dict_for_obs_v3(state_dict: dict) -> dict:
        """Migrate a pre-v3-obs checkpoint to the V3 obs schema (2026-06-11).

        v3 appended N_V3_EXTRA per-entity features (level / max_hp / ac /
        is_dying / death saves) and widened the slot layout 6 → N_ENTITY_SLOTS.
        Pre-v3 columns are a strict prefix of each v3 entity row, so old
        behaviour is preserved EXACTLY by:
          - ``entity_mlp.0.weight``: zero-pad the new feature columns — the
            migrated net ignores the new features until trained on them;
          - ``critic.net.0.weight``: re-scatter entity-block columns by the
            slot map (self/ally slots keep their index, enemy slots shift to
            ENEMY_SLOT_START..) and zero-fill new columns; the tail
            (resources / end_features / skill-pool) moves to its new offset.
        New slots hold all-zero rows in old-era episodes, and zero rows
        through zero-padded columns contribute nothing — outputs are bitwise
        identical on the same game state. A checkpoint already at v3 dims
        (or too old to recognise) is returned untouched.

        Targets the FROZEN v3 width (_ENTITY_DIM_V3), not the live
        ENTITY_DIM — adapt_state_dict_for_obs_v4 finishes the chain.
        """
        new_sd = dict(state_dict)
        w = new_sd.get("entity_mlp.0.weight")
        if w is None or w.shape[1] != _ENTITY_DIM_V2:
            return new_sd   # not a pre-v3 ckpt — nothing to do here
        old_dim = w.shape[1]
        pad = torch.zeros(w.shape[0], N_V3_EXTRA, dtype=w.dtype)
        new_sd["entity_mlp.0.weight"] = torch.cat([w, pad], dim=1)
        # Mark the checkpoint as pre-v3 so forward() restricts the
        # skill→entity attention to the legacy slot rows (bit-exact old
        # behaviour). Wave retrains clear this and adopt the presence mask.
        new_sd["attn_legacy"] = torch.ones(())

        cw = new_sd.get("critic.net.0.weight")
        # Pre-v3 checkpoints also predate the dtype tail — frozen 53-wide
        # skill pool (adapt_state_dict_for_skill_dtype finishes the chain).
        tail = 4 + END_FEATURES_DIM + _SKILL_DIM_V1
        legacy_in = CombatPolicyNet._LEGACY_N_SLOTS * old_dim + tail
        if cw is not None and cw.shape[1] == legacy_in:
            new_cw = torch.zeros(cw.shape[0],
                                 N_ENTITY_SLOTS * _ENTITY_DIM_V3 + tail,
                                 dtype=cw.dtype)
            for s_old in range(CombatPolicyNet._LEGACY_N_SLOTS):
                if s_old <= CombatPolicyNet._LEGACY_N_ALLY:
                    s_new = s_old
                else:
                    s_new = ENEMY_SLOT_START + (s_old - CombatPolicyNet._LEGACY_N_ALLY - 1)
                new_cw[:, s_new * _ENTITY_DIM_V3 : s_new * _ENTITY_DIM_V3 + old_dim] = \
                    cw[:, s_old * old_dim : (s_old + 1) * old_dim]
            new_cw[:, N_ENTITY_SLOTS * _ENTITY_DIM_V3:] = \
                cw[:, CombatPolicyNet._LEGACY_N_SLOTS * old_dim:]
            new_sd["critic.net.0.weight"] = new_cw
        return new_sd

    @staticmethod
    def adapt_state_dict_for_obs_v4(state_dict: dict) -> dict:
        """Migrate a v3-obs checkpoint to the v4 obs schema (2026-06-12,
        MONSTER_CATALOG §4). Two coupled changes, both EXACTLY compensated:

        1. Capability descriptor: N_V4_DESC new per-entity columns appended
           after the v3 tail (strict-prefix trick again) — zero-padded input
           weights, so the migrated net ignores the descriptor until trained.
        2. NORM rescale: level / max_hp features shrank by ×2 / ×8
           (LEVEL_NORM 20→40, MAXHP_NORM 100→800). The matching input-weight
           columns are scaled UP by the same factors. Both are powers of two,
           so feature-shrink and weight-grow are pure fp32 exponent shifts and
           every product round-trips bit-exactly — outputs on the same game
           state are bitwise identical to the v3 net on v3 obs.

        Touches ``entity_mlp.0.weight`` and ``critic.net.0.weight`` (per-slot
        blocks). Already-v4 or unrecognised checkpoints pass through.
        """
        _LVL_SCALE = LEVEL_NORM / V3_LEVEL_NORM    # 2.0  (power of 2)
        _HP_SCALE  = MAXHP_NORM / V3_MAXHP_NORM    # 8.0  (power of 2)

        new_sd = dict(state_dict)
        w = new_sd.get("entity_mlp.0.weight")
        if w is None or w.shape[1] != _ENTITY_DIM_V3:
            return new_sd   # already v4 / unrecognised era
        pad = torch.zeros(w.shape[0], N_V4_DESC, dtype=w.dtype)
        w2 = torch.cat([w.clone(), pad], dim=1)
        w2[:, I_ENT_LEVEL] *= _LVL_SCALE
        w2[:, I_ENT_MAXHP] *= _HP_SCALE
        new_sd["entity_mlp.0.weight"] = w2

        cw = new_sd.get("critic.net.0.weight")
        # v3-era checkpoints predate the dtype tail — frozen 53-wide skill pool.
        # v4 produces its OWN frozen entity stride (_ENTITY_DIM_V4), NOT the live
        # ENTITY_DIM — the v5 step widens it further afterwards (this used to read
        # ENTITY_DIM back when v4 was the latest schema).
        tail = 4 + END_FEATURES_DIM + _SKILL_DIM_V1
        v3_in = N_ENTITY_SLOTS * _ENTITY_DIM_V3 + tail
        if cw is not None and cw.shape[1] == v3_in:
            new_cw = torch.zeros(cw.shape[0], N_ENTITY_SLOTS * _ENTITY_DIM_V4 + tail,
                                 dtype=cw.dtype)
            for s in range(N_ENTITY_SLOTS):
                blk = cw[:, s * _ENTITY_DIM_V3 : (s + 1) * _ENTITY_DIM_V3].clone()
                blk[:, I_ENT_LEVEL] *= _LVL_SCALE
                blk[:, I_ENT_MAXHP] *= _HP_SCALE
                new_cw[:, s * _ENTITY_DIM_V4 : s * _ENTITY_DIM_V4 + _ENTITY_DIM_V3] = blk
            new_cw[:, N_ENTITY_SLOTS * _ENTITY_DIM_V4:] = \
                cw[:, N_ENTITY_SLOTS * _ENTITY_DIM_V3:]
            new_sd["critic.net.0.weight"] = new_cw
        return new_sd

    @staticmethod
    def adapt_state_dict_for_skill_dtype(state_dict: dict) -> dict:
        """Migrate a pre-dtype checkpoint to the 66-wide skill schema +
        join-augmented heads (2026-06-12i). Strict-append again, so old
        behaviour is preserved EXACTLY by zero-padding the new input columns:

          - ``skill_proj.weight``        [64, 53] → [64, 66]
          - ``critic.net.0.weight``      skill-pool block sits at the END of
                                          the input concat → +13 zero columns
          - skill/entity head weights    (single pre-perarch or per-group
                                          ModuleList form) +1 zero column for
                                          the matchup-join input

        New dtype features and the join meet only zero weights until trained;
        on identical game states the migrated net's outputs are bitwise
        identical. Era detection keys on skill_proj's input width.
        """
        w = state_dict.get("skill_proj.weight")
        if w is None or w.shape[1] != _SKILL_DIM_V1:
            return state_dict   # already dtype-era / unrecognised
        new_sd = dict(state_dict)
        pad = torch.zeros(w.shape[0], SKILL_FEATURE_DIM - _SKILL_DIM_V1,
                          dtype=w.dtype)
        new_sd["skill_proj.weight"] = torch.cat([w, pad], dim=1)

        cw = new_sd.get("critic.net.0.weight")
        # skill-dtype runs BEFORE the v5 entity-widen step, so the entity blocks
        # are still at the v4 width here (NOT live ENTITY_DIM, which since v5 is
        # wider). The skill pool sits at the end → just +13 columns.
        old_in = N_ENTITY_SLOTS * _ENTITY_DIM_V4 + 4 + END_FEATURES_DIM + _SKILL_DIM_V1
        if cw is not None and cw.shape[1] == old_in:
            cpad = torch.zeros(cw.shape[0], SKILL_FEATURE_DIM - _SKILL_DIM_V1,
                               dtype=cw.dtype)
            new_sd["critic.net.0.weight"] = torch.cat([cw, cpad], dim=1)

        import re
        head_re = re.compile(r"^(skill_head|entity_head)(s\.\d+)?\.weight$")
        for k in list(new_sd):
            if head_re.match(k):
                hw = new_sd[k]
                hpad = torch.zeros(hw.shape[0], 1, dtype=hw.dtype)
                new_sd[k] = torch.cat([hw, hpad], dim=1)
        return new_sd

    @staticmethod
    def adapt_state_dict_for_obs_v5(state_dict: dict) -> dict:
        """Migrate a v4-obs checkpoint to the v5 obs schema (2026-06-13): the
        N_V5_TRAIT per-entity passive-trait columns appended after the v4
        descriptor (a creature's own pack tactics / regen — see obs.py v5 block).
        PURE strict append — no NORM rescale this time — so old behaviour is
        preserved EXACTLY by zero-padding the new input columns (entity_mlp) and
        zero-filling them in each critic slot block; the migrated net ignores the
        trait tail until trained on it. Runs LAST in the obs chain (after
        skill-dtype), so the critic's skill-pool tail is the live 66-wide pool.
        Touches ``entity_mlp.0.weight`` and ``critic.net.0.weight`` (per-slot
        blocks). Already-v5 or unrecognised checkpoints pass through.
        """
        new_sd = dict(state_dict)
        w = new_sd.get("entity_mlp.0.weight")
        if w is None or w.shape[1] != _ENTITY_DIM_V4:
            return new_sd   # already v5 / unrecognised era
        pad = torch.zeros(w.shape[0], N_V5_TRAIT, dtype=w.dtype)
        new_sd["entity_mlp.0.weight"] = torch.cat([w.clone(), pad], dim=1)

        cw = new_sd.get("critic.net.0.weight")
        # v5 is last in the chain → skill pool already at the live width.
        tail = 4 + END_FEATURES_DIM + SKILL_FEATURE_DIM
        v4_in = N_ENTITY_SLOTS * _ENTITY_DIM_V4 + tail
        if cw is not None and cw.shape[1] == v4_in:
            new_cw = torch.zeros(cw.shape[0], N_ENTITY_SLOTS * ENTITY_DIM + tail,
                                 dtype=cw.dtype)
            for s in range(N_ENTITY_SLOTS):
                blk = cw[:, s * _ENTITY_DIM_V4 : (s + 1) * _ENTITY_DIM_V4]
                new_cw[:, s * ENTITY_DIM : s * ENTITY_DIM + _ENTITY_DIM_V4] = blk
            new_cw[:, N_ENTITY_SLOTS * ENTITY_DIM:] = \
                cw[:, N_ENTITY_SLOTS * _ENTITY_DIM_V4:]
            new_sd["critic.net.0.weight"] = new_cw
        return new_sd

    @staticmethod
    def adapt_state_dict_for_decision_ctx(state_dict: dict) -> dict:
        """Migrate a pre-decision-context checkpoint to the reaction/legendary
        schema (2026-06-13). N_DECISION_CTX columns were appended to the END of
        BOTH the trunk's first-layer input (decision_context after res_ctx) and
        the critic's first-layer input (decision_context after the skill pool).
        Strict append → zero-pad the new columns so the migrated net is bit-exact
        on normal turns (decision_context is all-zero there; a zero input nulls
        the product regardless of the column weights). Runs LAST in the chain,
        after the v5 entity-widen has brought both layers to the live width.
        Touches ``trunk.0.weight`` and ``critic.net.0.weight``; already-migrated
        or unrecognised checkpoints pass through (per-weight width detection).
        """
        new_sd = dict(state_dict)
        tw = new_sd.get("trunk.0.weight")
        # Pre-dctx trunk input width = sk_mean(64)+self_emb(32)+spatial+res(16).
        old_trunk_in = 64 + 32 + _SPATIAL_C + 16
        if tw is not None and tw.shape[1] == old_trunk_in:
            pad = torch.zeros(tw.shape[0], N_DECISION_CTX, dtype=tw.dtype)
            new_sd["trunk.0.weight"] = torch.cat([tw, pad], dim=1)

        cw = new_sd.get("critic.net.0.weight")
        # Pre-dctx critic width (entity blocks already at live ENTITY_DIM after
        # the v5 step): entities + resources(4) + end_features + skill pool(66).
        old_critic_in = (N_ENTITY_SLOTS * ENTITY_DIM + 4 + END_FEATURES_DIM
                         + SKILL_FEATURE_DIM)
        if cw is not None and cw.shape[1] == old_critic_in:
            cpad = torch.zeros(cw.shape[0], N_DECISION_CTX, dtype=cw.dtype)
            new_sd["critic.net.0.weight"] = torch.cat([cw, cpad], dim=1)
        return new_sd

    @staticmethod
    def adapt_state_dict_for_obs(state_dict: dict) -> dict:
        """Full obs-era chain: pre-v3 → v3 → v4 → skill-dtype → v5 →
        decision-ctx. Loaders that don't need the per-arch head tiling (e.g.
        1-head-group distilled students) call this."""
        return CombatPolicyNet.adapt_state_dict_for_decision_ctx(
            CombatPolicyNet.adapt_state_dict_for_obs_v5(
                CombatPolicyNet.adapt_state_dict_for_skill_dtype(
                    CombatPolicyNet.adapt_state_dict_for_obs_v4(
                        CombatPolicyNet.adapt_state_dict_for_obs_v3(state_dict)))))

    @staticmethod
    def adapt_state_dict_for_perarch(state_dict: dict) -> dict:
        """Universal load-time checkpoint adapter (every loader calls this).

        Chain: obs migrations first (pre-v3 → v3 → v4), then tile shared-head
        weights into per-arch copies for warm-start.

        Old BC checkpoints have ``skill_head.weight`` etc. The current model
        replaces those with ModuleLists of 12 per-arch copies. To preserve
        BC behaviour on warm-start, copy the shared weights into every per-arch
        copy — then PPO is free to diverge them per archetype.
        """
        new_sd = CombatPolicyNet.adapt_state_dict_for_obs(state_dict)
        mappings = {
            "skill_head":      "skill_heads",
            "entity_head":     "entity_heads",
            "grid_query_proj": "grid_query_projs",
        }
        for old_base, new_base in mappings.items():
            for suffix in ("weight", "bias"):
                old_key = f"{old_base}.{suffix}"
                if old_key in new_sd:
                    value = new_sd.pop(old_key)
                    for i in range(N_ARCHETYPES):
                        new_sd[f"{new_base}.{i}.{suffix}"] = value.clone()
        return new_sd

    def _encode(self, obs: dict):
        """Shared encoder. Returns (sk_emb, ent_emb, h, key_padding, spatial_feat).

        ``spatial_feat`` is [B, _SPATIAL_C, N_GRID, N_GRID] — per-cell features
        that the spatial grid_head reads to produce per-(skill, cell) logits.
        """
        skills      = obs["skills"]
        # ``_dtype_era_v1`` reproduces the pre-dtype computation (53-wide
        # skill rows, no matchup join) — diagnosis-only escape hatch for
        # diag_skill_dtype_shift.py, same pattern as legacy_unmasked_attn.
        if getattr(self, "_dtype_era_v1", False):
            skills = skills[:, :, :_SKILL_DIM_V1]
        skill_mask  = obs["skill_mask"]
        entities    = obs["entities"]
        resources   = obs["resources"]
        terrain     = obs["terrain"]
        entity_grid = obs["entity_grid"]   # [B, N_ENTITY_GRID_CHANNELS, N_GRID, N_GRID]

        sk_proj     = self.skill_proj(skills)
        key_padding = skill_mask < 0.5
        sk_emb      = self.skill_encoder(sk_proj, src_key_padding_mask=key_padding)
        sk_mean     = (sk_emb * skill_mask.unsqueeze(-1)).sum(1) / skill_mask.sum(1, keepdim=True).clamp(min=1.0)

        ent_emb  = self.entity_mlp(entities)
        # Self-indexed read instead of mean: entities[:, 0] is the agent itself.
        # Averaging across 6 slots dilutes self's features to 1/6 strength;
        # taking slot 0 directly preserves them.
        self_emb = ent_emb[:, 0]                          # [B, 32]

        # spatial_feat = stack of pre-computed per-cell features. No CNN —
        # all "global structure" features (distances) are precomputed in
        # obs.distance_grid_obs, and "local" features (entity presence) are
        # already discrete in entity_grid. grid_head's bmm dot-product
        # against this gives per-cell linear scoring, no learned spatial
        # reasoning needed.
        terrain_ch    = terrain.unsqueeze(1)              # [B, 1, N_GRID, N_GRID]
        distance_grid = obs["distance_grid"]              # [B, 2, N_GRID, N_GRID]
        spatial_feat  = torch.cat(                        # [B, _SPATIAL_C, N_GRID, N_GRID]
            [terrain_ch, entity_grid, distance_grid], dim=1)

        # Self-indexed read instead of global mean pool. Entities obs row 0
        # stores self.position normalised to [0, 1] in columns 1 (x) and 2 (y);
        # multiplying by N_GRID gives the cell index. Reading spatial_feat
        # at exactly that cell preserves position info — global mean-pool
        # would average it across 900 cells (1/900 strength).
        self_x_norm = entities[:, 0, 1].clamp(0.0, 1.0 - 1e-6)
        self_y_norm = entities[:, 0, 2].clamp(0.0, 1.0 - 1e-6)
        self_ix = (self_x_norm * N_GRID).long()           # [B]
        self_iy = (self_y_norm * N_GRID).long()           # [B]
        self_cell_idx = self_ix * N_GRID + self_iy        # [B]
        spatial_flat = spatial_feat.flatten(2)            # [B, C, N_GRID*N_GRID]
        gather_idx = self_cell_idx.view(-1, 1, 1).expand(-1, spatial_flat.shape[1], 1)
        self_cell_feat = spatial_flat.gather(2, gather_idx).squeeze(-1)  # [B, C]

        res_ctx = self.res_mlp(resources)

        # decision_context (zeros on normal turns) appended at the END so the
        # trunk's matching columns are the ones the migration zero-pads.
        dctx = obs["decision_context"]
        ctx = torch.cat([sk_mean, self_emb, self_cell_feat, res_ctx, dctx], dim=-1)
        h   = self.trunk(ctx)

        # FiLM: per-archetype gain & bias on h. arch_oh is taken from the
        # self entity row (slot 0) — see _ARCH_OH_* and obs._entity_row.
        arch_oh = entities[:, 0, _ARCH_OH_START:_ARCH_OH_END]  # [B, N_ARCHETYPES]
        gamma   = self.film_gamma(arch_oh)                    # [B, hidden]
        beta    = self.film_beta(arch_oh)                     # [B, hidden]
        h = gamma * h + beta

        return sk_emb, ent_emb, h, key_padding, spatial_feat

    def forward(self, obs: dict) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns (end_logit, skill_logits, entity_logits, grid_logits).

        end_logit:    [B]
        skill_logits: [B, N_SKILL_SLOTS]
        entity_logits:[B, N_SKILL_SLOTS, N_ENTITY_SLOTS] — per-skill
        grid_logits:  [B, N_SKILL_SLOTS, N_GRID*N_GRID]  — per-skill
        """
        sk_emb, ent_emb, h, key_padding, spatial_feat = self._encode(obs)
        B = h.shape[0]

        # Per-sample archetype id (used to gather the right per-arch head).
        # Clamped to the head-group count: a single-group (distilled) net
        # routes every identity — including all-zero one-hots (monsters,
        # chimeras) — to its one shared head. On 12-group nets the clamp is
        # a no-op (argmax of 12 bits is already < 12).
        entities = obs["entities"]
        arch_ids = entities[:, 0, _ARCH_OH_START:_ARCH_OH_END].argmax(dim=1)  # [B]
        arch_ids = arch_ids.clamp(max=len(self.skill_heads) - 1)

        # end_head reads from end_features (NOT from the shared trunk h) —
        # see __init__ docstring for the encoder-drift rationale.
        end_logit = self.end_head(obs["end_features"]).squeeze(-1)

        # Skill-to-entity attention: each skill queries entity embeddings so
        # skill_head can condition on per-entity status (e.g. vow_target).
        # ABSENT slots (all-zero rows) are masked out of the softmax: a zero
        # row still embeds to entity_mlp's bias vector, so unmasked padding
        # rows leak into the context — and worse, the leak SCALES with the
        # slot count (widening 6→10 slots shifted sk_ent_ctx enough to flip
        # berserker's move/reckless_attack decision, diag_obsv3_shift.py).
        # Masking makes the policy invariant to padding-slot count. Slot 0
        # (self) is always present, so the row can never be fully masked.
        sk_query    = self.skill_ent_attn_proj(sk_emb)                              # [B, S, 32]
        attn_scores = torch.bmm(sk_query, ent_emb.transpose(1, 2)) * (32 ** -0.5)  # [B, S, E]
        # ``legacy_unmasked_attn`` reproduces the pre-mask softmax (padding
        # rows included, whatever rows the obs carries) — diagnosis-only
        # escape hatch, see diag_obsv3_shift.py.
        if getattr(self, "legacy_unmasked_attn", False):
            pass
        elif self.attn_legacy.item() > 0:
            # Migrated pre-v3 checkpoint: softmax over the legacy slot rows
            # only — bit-exact pre-migration attention (see buffer comment).
            E = ent_emb.shape[1]
            if E > len(_LEGACY_ATTN_ROWS):
                legacy = torch.zeros(E, dtype=torch.bool, device=ent_emb.device)
                legacy[_LEGACY_ATTN_ROWS] = True
                attn_scores = attn_scores.masked_fill(~legacy.view(1, 1, E), -1e9)
        else:
            # Native v3: ABSENT slots (all-zero rows) are masked out — a zero
            # row still embeds to entity_mlp's bias vector, so unmasked
            # padding leaks into the context and the leak SCALES with slot
            # count. Masking makes the policy invariant to padding-slot
            # count. Slot 0 (self) is always present.
            ent_present = entities.abs().sum(dim=-1) > 1e-6                         # [B, E]
            attn_scores = attn_scores.masked_fill(~ent_present.unsqueeze(1), -1e9)
        sk_ent_ctx  = torch.bmm(torch.softmax(attn_scores, dim=-1), ent_emb)        # [B, S, 32]

        # Typed-matchup join (raw obs, no learned per-type weights): each
        # skill row's damage-type soft one-hot ⋅ each entity's typed-resist
        # slice (multiplier − 1 per type). join ≈ expected-damage multiplier
        # − 1 of using that skill on that entity: 0 = neutral / no info /
        # untyped skill, −1 = fully immune, +1 = doubly vulnerable. Zeroed
        # descriptors (desc-off arms, pre-v4 episodes) give exactly 0.
        dtype_v1 = getattr(self, "_dtype_era_v1", False)   # diagnosis-only
        sk_dt   = obs["skills"][:, :, SKILL_DTYPE_START:
                                SKILL_DTYPE_START + N_DAMAGE_TYPES]            # [B, S, 13]
        ent_rs  = entities[:, :, I_DESC_RESIST:I_DESC_RESIST + N_DAMAGE_TYPES] # [B, E, 13]
        # mul+sum over the trailing axis instead of bmm: each (s, e) pair
        # reduces its own contiguous 13 values, so the result is bitwise
        # independent of the slot count E (bmm's K-reduction order varies
        # with output tiling — broke padding-count invariance by 1 ulp).
        join_se = (sk_dt.unsqueeze(2) * ent_rs.unsqueeze(1)).sum(-1)           # [B, S, E]
        # skill_head summary: best matchup over PRESENT ENEMY rows — "the
        # damage this skill keeps against the most favourable target".
        # Enemy-ness comes from the row's own is_enemy bit, not its slot
        # index, so the join is invariant to slot layout / padding count.
        # No enemy rows present (shouldn't happen mid-combat) → 0.
        ent_present = entities.abs().sum(dim=-1) > 1e-6                        # [B, E]
        enemy_rows  = entities[:, :, I_ENT_ENEMY] > 0.5
        valid_enemy = (ent_present & enemy_rows).unsqueeze(1)                  # [B, 1, E]
        join_skill  = join_se.masked_fill(~valid_enemy, float("-inf")).amax(-1)
        join_skill  = torch.where(valid_enemy.any(-1), join_skill,
                                  torch.zeros_like(join_skill))                # [B, S]

        # skill_head: per skill slot. Stack all N_ARCH copies' outputs, then
        # gather the one matching each sample's archetype.
        h_skill      = h.unsqueeze(1).expand(-1, N_SKILL_SLOTS, -1)
        if dtype_v1:   # pre-dtype heads have no join column
            skill_in = torch.cat([sk_emb, h_skill, sk_ent_ctx], dim=-1)
        else:
            skill_in = torch.cat([sk_emb, h_skill, sk_ent_ctx,
                                  join_skill.unsqueeze(-1)], dim=-1)           # [B, S, 64+h+32+1]
        skill_all    = torch.stack(
            [hd(skill_in).squeeze(-1) for hd in self.skill_heads], dim=1)     # [B, N_ARCH, S]
        sk_gather_idx = arch_ids.view(B, 1, 1).expand(-1, 1, N_SKILL_SLOTS)
        skill_logits = skill_all.gather(1, sk_gather_idx).squeeze(1)          # [B, S]
        skill_logits = skill_logits.masked_fill(key_padding, -1e9)

        # entity_head per (skill, entity) — same per-arch gather pattern.
        # Gets the raw per-pair join so target choice can route a typed skill
        # AWAY from the resistant enemy and onto the vulnerable one.
        sk_for_ent  = sk_emb.unsqueeze(2).expand(-1, -1, N_ENTITY_SLOTS, -1)   # [B, S, E, 64]
        h_for_ent   = h.unsqueeze(1).unsqueeze(2).expand(-1, N_SKILL_SLOTS, N_ENTITY_SLOTS, -1)  # [B, S, E, h]
        ent_for_ent = ent_emb.unsqueeze(1).expand(-1, N_SKILL_SLOTS, -1, -1)   # [B, S, E, 32]
        if dtype_v1:
            ent_in = torch.cat([sk_for_ent, h_for_ent, ent_for_ent], dim=-1)
        else:
            ent_in = torch.cat([sk_for_ent, h_for_ent, ent_for_ent,
                                join_se.unsqueeze(-1)], dim=-1)   # [B, S, E, 64+h+32+1]
        ent_all     = torch.stack(
            [hd(ent_in).squeeze(-1) for hd in self.entity_heads], dim=1)      # [B, N_ARCH, S, E]
        ent_gather_idx = arch_ids.view(B, 1, 1, 1).expand(-1, 1, N_SKILL_SLOTS, N_ENTITY_SLOTS)
        entity_logits = ent_all.gather(1, ent_gather_idx).squeeze(1)          # [B, S, E]

        # Spatial grid_head — per-arch gather on the spatial query projection.
        h_for_grid = h.unsqueeze(1).expand(-1, N_SKILL_SLOTS, -1)              # [B, S, hidden]
        grid_in    = torch.cat([sk_emb, h_for_grid], dim=-1)                   # [B, S, 64+hidden]
        gq_all     = torch.stack(
            [proj(grid_in) for proj in self.grid_query_projs], dim=1)         # [B, N_ARCH, S, _SPATIAL_C]
        gq_gather_idx = arch_ids.view(B, 1, 1, 1).expand(
            -1, 1, N_SKILL_SLOTS, self._spatial_total_c)
        grid_query = gq_all.gather(1, gq_gather_idx).squeeze(1)               # [B, S, _SPATIAL_C]
        spatial_flat = spatial_feat.flatten(2)                                 # [B, _SPATIAL_C, N_GRID*N_GRID]
        grid_logits = torch.bmm(grid_query, spatial_flat)                      # [B, S, N_GRID*N_GRID]

        return end_logit, skill_logits, entity_logits, grid_logits

    def value(self, obs: dict, detach_encoder: bool = False) -> torch.Tensor:
        """Critic: estimate state value V(s). Returns [B].

        Uses a SEPARATE critic network (``self.critic``) with its own weights,
        NOT the shared policy encoder. History: the value head used to read the
        policy encoder output ``h``; to stop value MSE (targets ±10) from
        dominating the tiny policy gradient (~0.001) we detached the encoder —
        but then the value head could only fit V(s) on FROZEN policy features
        and stayed under-fit (MSE 10-15, RMSE ~3-4), giving noisy advantages
        that stalled PPO at ~40%. A standalone critic fits V(s) from the
        value-relevant scalars (HP fractions, positions, distances, resources)
        directly and is fully decoupled from the policy, so the policy encoder
        gets pure policy-gradient signal AND advantages get a well-fit value.
        ``detach_encoder`` is accepted for call-site compatibility but ignored
        (the critic never touches the policy encoder).
        """
        return self.critic(obs)


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
    # NOTE: lowering this threshold (ending turns earlier, closer to the expert's
    # 51-66% end rate) was measured to HURT win rate (v19: 43%→38% at thr 0.2),
    # so the policy's longer move-padded turns are net-positive. Keep 0.5.
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
        # Legality mask for the aim point — the grid-head counterpart of
        # apply_entity_mask. Cells decode_action would reject (blocked / no
        # LoS) are excluded so the policy can't no-op its turn into a wall.
        from .action import point_validity_mask
        row = grid_logits[skill_idx]
        inv = point_validity_mask(ws, agent_id, skills[skill_idx])
        if inv.any() and not inv.all():
            row = row.clone().masked_fill(
                torch.from_numpy(inv).to(row.device), -1e9)
        grid = int(row.argmax().item())
        return (skill_idx, 0, grid)
    # SELF or unrecognised: both irrelevant
    return (skill_idx, 0, 0)


def apply_entity_mask(entity_logits: "torch.Tensor", obs: dict,
                      ws=None, agent_id: str = "") -> "torch.Tensor":
    """Mask empty (padding) entity slots and self.

    Handles both shapes:
      [B, N_ENTITY]              — legacy shared head
      [B, N_SKILL, N_ENTITY]     — per-skill head
    The entity feature layout (from obs.py _entity_row):
        index 5 = is_alive (0 for dying — still targetable)
        index 6 = is_self

    When ``ws``/``agent_id`` are provided (single live state, B=1) and the
    logits are per-skill, additionally mask — for each skill whose target_type
    is SINGLE_ALLY/MULTI_ALLY — every slot that is not an ALLY within the
    skill's range: enemy slots would be bounced to self by decode_action
    anyway, and an out-of-range ally would make the engine reject the cast,
    wasting the whole action. A fully-masked row argmaxes/samples to a
    non-ally slot which decode_action resolves to a legal SELF target, so
    1v1 behaviour (no allies) is unchanged.
    """
    import torch
    entities = obs["entities"]
    if not isinstance(entities, torch.Tensor):
        entities = torch.from_numpy(entities)
    entities = entities.to(entity_logits.device)
    # Empty/padding slots are all-zero rows. DYING characters (hp 0, alive
    # bit 0 but row populated — archetype bits etc.) stay SELECTABLE: they
    # can be healed back up (revive) or attacked (finish off). The old
    # `is_alive < 0.5` mask treated them as padding and made revive
    # untargetable at the model layer.
    is_present = entities.abs().sum(dim=-1) > 1e-6    # [B, N_ENTITY]
    is_self    = entities[..., 6]                     # [B, N_ENTITY]
    # For [B, N_SKILL, N_ENTITY], broadcast the mask across the skill axis.
    if entity_logits.dim() == 3:
        is_present = is_present.unsqueeze(1)          # [B, 1, N_ENTITY]
        is_self    = is_self.unsqueeze(1)
    entity_logits = entity_logits.clone()
    entity_logits = entity_logits.masked_fill(~is_present, -1e9)
    entity_logits = entity_logits.masked_fill(is_self > 0.5, -1e9)

    if (ws is not None and agent_id
            and entity_logits.dim() == 3 and entity_logits.shape[0] == 1):
        from ..engine.skill import available_skills, TargetType
        from .obs import partition_entities, N_ALLY_SLOTS
        agent = ws.characters[agent_id]
        allies, _ = partition_entities(ws, agent_id)
        # Slot layout mirrors entities_obs: [self, allies.., enemies..].
        ally_of_slot = {1 + i: cid for i, cid in enumerate(allies[:N_ALLY_SLOTS])}
        ALLY_TT = (TargetType.SINGLE_ALLY, TargetType.MULTI_ALLY)
        skills = available_skills(agent, ws)
        for i, sk in enumerate(skills):
            if i >= entity_logits.shape[1]:
                break
            if sk.features.target_type not in ALLY_TT:
                continue
            rng = float(getattr(sk.features, "range_m", 0.0) or 0.0)
            for slot in range(entity_logits.shape[2]):
                cid = ally_of_slot.get(slot)
                in_range = (
                    cid is not None
                    and agent.position.distance_to(
                        ws.characters[cid].position) < rng - 0.01)
                if not in_range:
                    entity_logits[0, i, slot] = -1e9

        # Enemy-target skills: mask enemies WITHOUT line of sight — exactly
        # the condition under which decode_action silently converts the
        # action into an end-turn no-op. Applied only when at least one enemy
        # remains visible, so a fully-blocked board keeps today's semantics
        # (row stays unmasked; decode no-ops; turn ends).
        bf = ws.combat.battlefield if ws.combat else None
        if bf is not None:
            from .obs import ENEMY_SLOT_START
            ENEMY_TT = (TargetType.SINGLE_ENEMY, TargetType.MULTI_ENEMY)
            _, enemies = partition_entities(ws, agent_id)
            vis = {}
            for j, cid in enumerate(enemies):
                slot = ENEMY_SLOT_START + j
                if slot >= entity_logits.shape[2]:
                    break
                vis[slot] = bf.has_line_of_sight(
                    agent.position, ws.characters[cid].position)
            if vis and any(vis.values()) and not all(vis.values()):
                for i, sk in enumerate(skills):
                    if i >= entity_logits.shape[1]:
                        break
                    if sk.features.target_type not in ENEMY_TT:
                        continue
                    for slot, ok in vis.items():
                        if not ok:
                            entity_logits[0, i, slot] = -1e9
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
    # LoS to ANY enemy: when every enemy is wall-blocked, decode_action turns
    # each enemy-target action into a silent end-turn no-op — mask the whole
    # skill instead so the policy spends the turn on something executable
    # (move / self / point skills). Same legality contract as the range gate.
    bf = ws.combat.battlefield if ws.combat else None
    any_enemy_los = True
    if bf is not None and enemies:
        any_enemy_los = any(
            bf.has_line_of_sight(agent.position, ws.characters[e].position)
            for e in enemies)

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
    # 5e one-leveled-spell-per-turn: once the actor has cast a leveled spell
    # this turn, mask out all other leveled-spell skills (cantrips still OK).
    leveled_locked = getattr(agent, "leveled_spell_cast_this_turn", False)
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
        if leveled_locked and getattr(sk.features, "cost_slot_level", 0) > 0:
            skill_logits[..., i] = -1e9
            continue
        tt = sk.features.target_type
        if tt == TargetType.SELF:
            continue
        if tt in (TargetType.SINGLE_ALLY, TargetType.MULTI_ALLY):
            # Ally-target skills always have a legal in-range target: SELF
            # (decode_action resolves non-ally entity picks to self, and the
            # engine accepts self at distance 0). Range-masking these by some
            # ally's distance both blocked legitimate self-heals and used an
            # arbitrary (unsorted) ally — per-ally range gating lives in
            # apply_entity_mask instead.
            continue
        target_dist = enemy_dist
        if target_dist is None:
            skill_logits[..., i] = -1e9
            continue
        if (not any_enemy_los
                and tt in (TargetType.SINGLE_ENEMY, TargetType.MULTI_ENEMY)):
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
