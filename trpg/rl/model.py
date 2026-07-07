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
    N_ENTITY_GRID_CHANNELS, N_DISTANCE_GRID_CHANNELS, N_LOS_GRID_CHANNELS,
    N_REACH_GRID_CHANNELS, N_THREAT_GRID_CHANNELS,
    END_FEATURES_DIM,
    N_ARCHETYPES, N_V3_EXTRA, N_V4_DESC, N_V5_TRAIT, N_V6_CIMMUN, N_V7_ABILITY,
    ENEMY_SLOT_START,
    I_ENT_LEVEL, I_ENT_MAXHP, I_DESC_RESIST, I_DESC_CIMMUN, I_ENT_ENEMY,
    LEVEL_NORM, MAXHP_NORM, V3_LEVEL_NORM, V3_MAXHP_NORM,
    N_DECISION_CTX,
)
from ..engine.skill import SKILL_FEATURE_DIM, SKILL_DTYPE_START, N_STATUS_SLOTS
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
_ENTITY_DIM_V6 = ENTITY_DIM - N_V7_ABILITY    # pre-v7: v6 cimmun tail, no ability
_ENTITY_DIM_V5 = _ENTITY_DIM_V6 - N_V6_CIMMUN  # pre-v6: v5 trait tail, no cimmun
_ENTITY_DIM_V4 = _ENTITY_DIM_V5 - N_V5_TRAIT  # pre-v5: v4 descriptor, no trait tail
_ENTITY_DIM_V3 = _ENTITY_DIM_V4 - N_V4_DESC  # v3: base + one-hots + v3 tail
_ENTITY_DIM_V2 = _ENTITY_DIM_V3 - N_V3_EXTRA  # pre-v3: no tail
# Frozen pre-dtype skill-feature width (2026-06-12i schema surgery added the
# damage-type tail + the matchup-join head input). File-format constant for
# adapt_state_dict_for_skill_dtype — never track live SKILL_FEATURE_DIM.
_SKILL_DIM_V1 = 53
# applies_status multi-hot offset inside the skill feature vector (N_STATUS_SLOTS
# bits, at [SKILL_DTYPE_START-N_STATUS_SLOTS : SKILL_DTYPE_START]) — the skill
# side of the v6 condition-immunity matchup join, mirror of the damage-type
# join's SKILL_DTYPE_START slice.
SKILL_STATUS_START = SKILL_DTYPE_START - N_STATUS_SLOTS


# CNN removed. spatial_feat is now the raw stack of pre-computed per-cell
# channels — terrain + entity overlay + distance grids. CNN was bottlenecking
# grid_head because Euclidean distance to entities is a global quantity that
# a local 3x3 kernel can't compute. With distance pre-computed as channels,
# grid_head only needs a linear scoring (grid_query · cell_features), which
# is what bmm in forward() does.
_SPATIAL_C = (1 + N_ENTITY_GRID_CHANNELS + N_DISTANCE_GRID_CHANNELS
              + N_LOS_GRID_CHANNELS
              + N_REACH_GRID_CHANNELS
              + N_THREAT_GRID_CHANNELS)  # terrain+entity+distance+los+reach+threat

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
    def __init__(self, hidden: int = 128, n_head_groups: int | None = None,
                 skill_combo_dim: int = 0, ablate_immunity_joins: bool = False,
                 drop_noop_h: bool = False, ablate_archetype: bool = False,
                 encode_entity_skills: bool = False):
        """``n_head_groups``: how many sibling copies of the skill/entity/grid
        heads to build. Default N_ARCHETYPES = the per-archetype-head design
        (identity-routed). 1 = a single shared head group — used by the
        distilled student nets, where supervised targets are fixed so the
        cross-class gradient conflict that motivated per-arch heads does not
        exist. Checkpoints are only shape-compatible with the same group
        count they were saved with."""
        super().__init__()
        # EXPERIMENT-ONLY structural ablation. Default False ⇒ production /
        # uni_v10 nets are BIT-EXACT and load unchanged (the historical
        # architecture is preserved). When True, the two hand-wired immunity
        # joins (typed-resist + condition-immunity) are STRUCTURALLY ABSENT:
        # the skill/entity heads are built narrower and the joins are never
        # concatenated in forward(), so immunity can reach the policy ONLY via
        # the learned sk_ent_ctx attention. Set on fresh experiment nets only;
        # a net built with this flag is NOT checkpoint-compatible with a net
        # built without it (different head input width — by design).
        self.ablate_immunity_joins = ablate_immunity_joins
        # EXPERIMENT-ONLY. When True, the trunk vector h is NOT fed to skill_head
        # or entity_head. Rationale: h is broadcast identically to every skill
        # slot (and every entity slot per skill), so W_h·h is a per-softmax
        # CONSTANT that cancels in the skill / entity choice — dead weights that
        # also receive ~zero gradient (softmax is shift-invariant). Verified:
        # zeroing h's head columns moves the skill/entity softmax by <1e-7.
        # grid_head KEEPS h (its query is dotted with per-cell features, so h's
        # contribution varies per cell — not a constant). Default False ⇒ v10 /
        # production keep h in all heads and load bit-exact.
        self.drop_noop_h = drop_noop_h
        # EXPERIMENT-ONLY structural removal of the ARCHETYPE (class) identity.
        # NOT masking (NoArch zeroes the arch one-hot but keeps the weights that
        # consume it) — here the arch pathway does not EXIST: the arch columns are
        # sliced out of every entity row before entity_mlp (narrower first layer),
        # FiLM (film_gamma/beta, the class-conditioner on h) is never built nor
        # applied, end_head drops the end_features arch tail, and per-arch routing
        # is fixed to head 0. So the net can infer identity ONLY from the skill
        # pool (codeword) / non-class features. A net built with this flag has
        # different entity_mlp/end_head widths and NO film params ⇒ NOT checkpoint-
        # compatible with a normal net (by design). Default False ⇒ uni_v10 bit-exact.
        self.ablate_archetype = ablate_archetype
        # EXPERIMENT (obs v8): read every entity's STATIC kit (obs "entity_skills")
        # through the SAME skill encoder the self uses, summarising each creature's
        # skill set into a vector added to its entity embedding. Closes the gap
        # where distinct kits with an identical capability_descriptor (raging
        # berserker vs champion; two different L6 wizards) read identical. The
        # add is via a ZERO-INIT projection ⇒ a net warm-started from a pre-v8
        # checkpoint is bit-exact until entity_kit_proj trains. Default False ⇒
        # the "entity_skills" obs key is simply ignored (older archs bit-exact).
        self.encode_entity_skills = encode_entity_skills
        # Source-of-truth record of the exact constructor args that define THIS
        # architecture. A checkpoint's .pt stores only weights (no arch); pairing
        # it with these kwargs is what lets a loader rebuild the right net instead
        # of guessing the default and silently dropping mismatched weights. See
        # trpg/rl/arch.py (ARCH_PRESETS / save_net / load_net).
        self.arch_kwargs = dict(
            hidden=hidden, n_head_groups=n_head_groups,
            skill_combo_dim=skill_combo_dim,
            ablate_immunity_joins=ablate_immunity_joins,
            drop_noop_h=drop_noop_h, ablate_archetype=ablate_archetype,
            encode_entity_skills=encode_entity_skills)
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
        # ablate_archetype: the arch one-hot columns are sliced out of every
        # entity row before this MLP, so its first layer is that much narrower —
        # the weights that would read the class simply do not exist.
        _ent_in = ENTITY_DIM - (N_ARCHETYPES if ablate_archetype else 0)
        self.entity_mlp = nn.Sequential(
            nn.Linear(_ent_in, 64), nn.ReLU(), nn.Linear(64, 32),
        )
        # encode_entity_skills: projects each entity's kit summary (64-d, from the
        # SHARED skill encoder) into entity-embedding space (32-d) and ADDS it to
        # ent_emb. Zero-init ⇒ inert at start (bit-exact warm-start from pre-v8).
        if encode_entity_skills:
            self.entity_kit_proj = nn.Linear(64, 32)
            nn.init.zeros_(self.entity_kit_proj.weight)
            nn.init.zeros_(self.entity_kit_proj.bias)
        else:
            self.entity_kit_proj = None

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
        # ablate_archetype: FiLM is the sole class-conditioning module (its input
        # IS the arch one-hot). Structurally absent in ablate mode — not built,
        # not applied. None so attribute access stays valid.
        if ablate_archetype:
            self.film_gamma = None
            self.film_beta = None
        else:
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
        #
        # +N_THREAT+N_REACH: the self-cell THREAT and REACH flags (threat_grid /
        # reach_grid at the agent's OWN cell) are concatenated to end_features so
        # the decoupled end_head can condition the stop-vs-continue decision on
        # VALUE, not just resource availability. Unified root cause of two WR
        # bugs (measured): end_head saw only resources/archetype, so it kept
        # acting whenever a resource remained, regardless of whether acting was
        # worth anything —
        #   • threat flag → fixes over-kiting (stop repositioning once SAFE);
        #   • reach flag  → fixes full-HP heal waste (when I CANNOT attack from
        #     here (reach=0) and I'm at full HP, no action has value → end the
        #     turn and KEEP the once-per-rest heal, instead of burning it).
        # These are the only threat/reach pathways into end_head (the spatial
        # grid otherwise feeds the move/skill heads, not end_head).
        # ablate_archetype: end_features carries the agent's arch one-hot as its
        # trailing N_ARCHETYPES block — sliced off before end_head, so its input
        # is that much narrower (no weights consume the class).
        _end_feat = END_FEATURES_DIM - (N_ARCHETYPES if ablate_archetype else 0)
        self.end_head = nn.Sequential(
            nn.Linear(_end_feat + N_THREAT_GRID_CHANNELS
                      + N_REACH_GRID_CHANNELS, 32), nn.ReLU(),
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
        # +64 = the "blocked → which skill" feature: blockedness × sk_emb, where
        # blockedness = 1 − self-cell LoS proximity (0 in open layouts, where the
        # LoS channel is all-ones). It is EXACTLY ZERO whenever the agent has line
        # of sight, so training its weights changes ONLY blocked-state skill
        # choice and leaves open play (and symmetric team combat) bit-exact — the
        # one open-safe lever for "choose to MOVE when wall-blocked instead of
        # buffing/ending" (56% of blocked turns were non-move; a shared skill
        # head couldn't learn this without the open anchor fighting it). The head
        # LEARNS which skill embedding to boost when blocked (it converges on
        # move) — no hardcoded "moveness". entity_head is unchanged (target
        # choice, not the move-vs-buff decision).
        # skill_heads input: sk_emb(64) + h + sk_ent_ctx(32) + typed-join(1) +
        # blocked_feat(64) + cimmun-join(1, v6). entity_heads: sk(64)+h+ent(32) +
        # typed-join(1) + cimmun-join(1). The v6 condition-immunity join is the
        # LAST column of each (appended after blocked_feat / after the typed
        # join) so its checkpoint migration is a single trailing zero-pad — see
        # adapt_state_dict_for_cimmun_heads.
        # _j = per-head immunity-join column count: 0 when ablated (structurally
        # absent), else 1 typed + 1 cimmun. Default preserves production width
        # (skill 290 / entity 226) ⇒ uni_v10 bit-exact; ablated = skill 288 /
        # entity 224 (no immunity wiring at all).
        _j = 0 if self.ablate_immunity_joins else 1
        # _h = hidden unless drop_noop_h drops the constant-across-slots h from
        # skill_head / entity_head (grid_head always keeps h — see grid section).
        _h = 0 if self.drop_noop_h else hidden
        self.skill_heads = nn.ModuleList(
            [nn.Linear(64 + _h + 32 + _j + 64 + _j, 1) for _ in range(n_groups)])
        self.entity_heads = nn.ModuleList(
            [nn.Linear(64 + _h + 32 + _j + _j, 1) for _ in range(n_groups)])
        # Named input-column index of the typed-matchup join in BOTH head kinds.
        # Surgery/probe scripts MUST use this instead of positional math like
        # ``ncol-1``: blocked_feat(+64) and the cimmun join(+1) were appended
        # AFTER the typed join, which silently broke every ``ncol-1`` consumer
        # (the v7 rsw surgery trained the cimmun column — feature ≡ 0 vs the
        # lab's status-immunity-less enemies — while the real typed join stayed
        # frozen; the pooled resist columns became the only carrier = shortcut).
        self.tjoin_col = 64 + hidden + 32
        self.grid_query_projs = nn.ModuleList(
            [nn.Linear(64 + hidden, self._spatial_total_c) for _ in range(n_groups)])
        # Skill-combination identity codeword (Deep-Sets over the held skills).
        # A per-skill projection is masked-MEAN-pooled into a k-d codeword that
        # summarises WHICH skills the actor holds — order-invariant (mean is
        # symmetric) and fixed-length (pool collapses the variable-size set).
        # The codeword → a 64-d preference direction, dotted with EACH skill's
        # embedding: a BILINEAR term that varies per skill. (Contrast the h
        # branch, which is broadcast identically to every slot = a per-state
        # constant that cancels in the skill softmax/argmax and cannot move the
        # choice — measured: masking it flips 0/183 argmaxes.) This lets an
        # archetype-blind net infer identity from its kit and let that identity
        # shift skill selection. 0 = disabled: no modules built, forward path
        # skipped ⇒ bit-exact with the base net (old checkpoints load clean).
        self.skill_combo_dim = skill_combo_dim
        if skill_combo_dim > 0:
            self.skill_combo_proj = nn.Sequential(
                nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, skill_combo_dim))
            self.skill_combo_pref = nn.Linear(skill_combo_dim, 64)
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
        # v5 produces its OWN frozen entity stride (_ENTITY_DIM_V5), NOT the live
        # ENTITY_DIM — since obs v6 the cimmun step (adapt_state_dict_for_cimmun_
        # entity) widens the critic further afterwards. The skill-dtype step
        # already ran, so the skill pool tail is the live 66-wide pool.
        tail = 4 + END_FEATURES_DIM + SKILL_FEATURE_DIM
        v4_in = N_ENTITY_SLOTS * _ENTITY_DIM_V4 + tail
        if cw is not None and cw.shape[1] == v4_in:
            new_cw = torch.zeros(cw.shape[0], N_ENTITY_SLOTS * _ENTITY_DIM_V5 + tail,
                                 dtype=cw.dtype)
            for s in range(N_ENTITY_SLOTS):
                blk = cw[:, s * _ENTITY_DIM_V4 : (s + 1) * _ENTITY_DIM_V4]
                new_cw[:, s * _ENTITY_DIM_V5 : s * _ENTITY_DIM_V5 + _ENTITY_DIM_V4] = blk
            new_cw[:, N_ENTITY_SLOTS * _ENTITY_DIM_V5:] = \
                cw[:, N_ENTITY_SLOTS * _ENTITY_DIM_V4:]
            new_sd["critic.net.0.weight"] = new_cw
        return new_sd

    @staticmethod
    def adapt_state_dict_for_cimmun_entity(state_dict: dict) -> dict:
        """Widen a pre-v6 checkpoint's ENTITY-side weights for the condition-
        immunity descriptor (2026-07-01): N_V6_CIMMUN columns appended at the END
        of every entity row (after the v5 trait tail). Touches entity_mlp.0.weight
        (zero-pad the trailing cols → the immunity descriptor is ignored until
        trained ⇒ bit-exact) and critic.net.0.weight (per-slot entity blocks widen
        _ENTITY_DIM_V5 → _ENTITY_DIM_V6, the frozen pre-v7 stride; the ability step
        finishes to live ENTITY_DIM, new cols zero). The head-side cimmun join
        column is added separately by adapt_state_dict_for_cimmun_heads (which must
        run after blocked_skill). Runs RIGHT AFTER obs_v5 and BEFORE decision_ctx,
        so a pre-dctx checkpoint's critic is already at the live entity stride when
        decision_ctx (which keys on N_ENTITY_SLOTS*ENTITY_DIM) runs. Already-v6 /
        unrecognised checkpoints pass through (entity_mlp width detection)."""
        new_sd = dict(state_dict)
        w = new_sd.get("entity_mlp.0.weight")
        if w is None or w.shape[1] != _ENTITY_DIM_V5:
            return new_sd   # already v6 / unrecognised era
        pad = torch.zeros(w.shape[0], N_V6_CIMMUN, dtype=w.dtype)
        new_sd["entity_mlp.0.weight"] = torch.cat([w.clone(), pad], dim=1)

        # critic: per-slot entity blocks widen _ENTITY_DIM_V5 → _ENTITY_DIM_V6
        # (the FROZEN v6 stride, NOT live ENTITY_DIM — since obs v7 the ability
        # step, adapt_state_dict_for_ability_entity, widens it the rest of the way
        # to live). The trailing tail (resources+end+skill-pool, optionally +dctx
        # depending on era) is preserved verbatim — try both tail widths so pre-
        # and post-dctx checkpoints both re-block cleanly.
        cw = new_sd.get("critic.net.0.weight")
        if cw is not None:
            base_tail = 4 + END_FEATURES_DIM + SKILL_FEATURE_DIM
            for tail in (base_tail, base_tail + N_DECISION_CTX):
                if cw.shape[1] == N_ENTITY_SLOTS * _ENTITY_DIM_V5 + tail:
                    new_cw = torch.zeros(cw.shape[0],
                                         N_ENTITY_SLOTS * _ENTITY_DIM_V6 + tail,
                                         dtype=cw.dtype)
                    for s in range(N_ENTITY_SLOTS):
                        blk = cw[:, s * _ENTITY_DIM_V5 : (s + 1) * _ENTITY_DIM_V5]
                        new_cw[:, s * _ENTITY_DIM_V6 : s * _ENTITY_DIM_V6 + _ENTITY_DIM_V5] = blk
                    new_cw[:, N_ENTITY_SLOTS * _ENTITY_DIM_V6:] = \
                        cw[:, N_ENTITY_SLOTS * _ENTITY_DIM_V5:]
                    new_sd["critic.net.0.weight"] = new_cw
                    break
        return new_sd

    @staticmethod
    def adapt_state_dict_for_ability_entity(state_dict: dict) -> dict:
        """Widen a pre-v7 checkpoint's ENTITY-side weights for the six ability-
        modifier columns (obs v7, 2026-07-06): N_V7_ABILITY columns appended at the
        END of every entity row (after the v6 cimmun tail). Touches
        entity_mlp.0.weight (zero-pad the trailing cols → the ability descriptor is
        ignored until trained ⇒ bit-exact) and critic.net.0.weight (per-slot entity
        blocks widen the FROZEN _ENTITY_DIM_V6 → live ENTITY_DIM, new cols zero).
        There is NO head-side join for this tail (unlike cimmun) — the ability mods
        reach the policy through the entity embedding only. Runs RIGHT AFTER
        cimmun_entity and BEFORE decision_ctx so the critic is already at the live
        entity stride when decision_ctx (which keys on N_ENTITY_SLOTS*ENTITY_DIM)
        runs. Already-v7 / unrecognised checkpoints pass through (entity_mlp width
        detection)."""
        new_sd = dict(state_dict)
        w = new_sd.get("entity_mlp.0.weight")
        if w is None or w.shape[1] != _ENTITY_DIM_V6:
            return new_sd   # already v7 / unrecognised era
        pad = torch.zeros(w.shape[0], N_V7_ABILITY, dtype=w.dtype)
        new_sd["entity_mlp.0.weight"] = torch.cat([w.clone(), pad], dim=1)

        cw = new_sd.get("critic.net.0.weight")
        if cw is not None:
            base_tail = 4 + END_FEATURES_DIM + SKILL_FEATURE_DIM
            for tail in (base_tail, base_tail + N_DECISION_CTX):
                if cw.shape[1] == N_ENTITY_SLOTS * _ENTITY_DIM_V6 + tail:
                    new_cw = torch.zeros(cw.shape[0],
                                         N_ENTITY_SLOTS * ENTITY_DIM + tail,
                                         dtype=cw.dtype)
                    for s in range(N_ENTITY_SLOTS):
                        blk = cw[:, s * _ENTITY_DIM_V6 : (s + 1) * _ENTITY_DIM_V6]
                        new_cw[:, s * ENTITY_DIM : s * ENTITY_DIM + _ENTITY_DIM_V6] = blk
                    new_cw[:, N_ENTITY_SLOTS * ENTITY_DIM:] = \
                        cw[:, N_ENTITY_SLOTS * _ENTITY_DIM_V6:]
                    new_sd["critic.net.0.weight"] = new_cw
                    break
        return new_sd

    @staticmethod
    def adapt_state_dict_for_cimmun_heads(state_dict: dict) -> dict:
        """Add the v6 condition-immunity JOIN column to the skill/entity heads
        (2026-07-01). One trailing input column each:
          - skill_heads.*.weight: appended AFTER blocked_feat → detect width
            (64+hidden)+32+1+64 and pad +1.
          - entity_heads.*.weight: appended AFTER the typed join → detect width
            (64+hidden)+32+1 and pad +1.
        Zero-pad ⇒ bit-exact (the cimmun join meets a zero weight until trained).
        The pre-feature widths are inferred from grid_query_projs' input (=64+
        hidden), so already-migrated / unrecognised checkpoints pass through. Runs
        AFTER blocked_skill (so skill_head already carries its +64 blocked block)."""
        new_sd = dict(state_dict)
        gq = None
        for k, v in new_sd.items():
            if k.startswith("grid_query_projs.") and k.endswith(".weight") and v.dim() == 2:
                gq = v.shape[1]                       # = 64 + hidden
                break
        if gq is None:
            return new_sd
        old_skill_in = gq + 32 + 1 + 64   # sk_emb in gq; +ctx +typed-join +blocked
        old_entity_in = gq + 32 + 1       # sk_emb in gq; +ctx +typed-join
        for key in list(new_sd.keys()):
            base = key.rsplit(".", 1)[0]
            if not key.endswith(".weight"):
                continue
            w = new_sd[key]
            if w.dim() != 2:
                continue
            is_skill = base == "skill_head" or base.startswith("skill_heads.")
            is_entity = base == "entity_head" or base.startswith("entity_heads.")
            if is_skill and w.shape[1] == old_skill_in:
                new_sd[key] = torch.cat(
                    [w, torch.zeros(w.shape[0], 1, dtype=w.dtype)], dim=1)
            elif is_entity and w.shape[1] == old_entity_in:
                new_sd[key] = torch.cat(
                    [w, torch.zeros(w.shape[0], 1, dtype=w.dtype)], dim=1)
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
        # The los-grid AND reach-grid channels are appended to spatial LATER in
        # the chain, so at the dctx step the spatial block is still the pre-los/
        # pre-reach width — subtract both (else a bumped _SPATIAL_C makes this
        # width test miss pre-dctx ckpts).
        old_trunk_in = (64 + 32
                        + (_SPATIAL_C - N_LOS_GRID_CHANNELS - N_REACH_GRID_CHANNELS
                           - N_THREAT_GRID_CHANNELS)
                        + 16)
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
    def adapt_state_dict_for_los_grid(state_dict: dict) -> dict:
        """Migrate a pre-LoS-channel checkpoint (2026-06-13d). One spatial
        channel (per-cell line-of-sight to nearest enemy) was appended to the
        END of spatial_feat. That widens TWO things by 1:
          - every grid_query projection's OUTPUT (it projects to _SPATIAL_C):
            append a zero row → the new channel gets a zero query → ignored.
          - the trunk's first-layer INPUT, where self_cell_feat (_SPATIAL_C
            wide) sits at columns [96 : 96+_SPATIAL_C] inside
            [sk_mean(64), self_emb(32), self_cell_feat, res(16), dctx]: insert
            a zero column at the END of that slice (index 96+_SPATIAL_C-1).
        Zero-pad ⇒ bit-exact (an unused channel contributes nothing). Per-weight
        width detection ⇒ already-migrated / unrecognised checkpoints pass
        through. Runs LAST in the chain (after all earlier widens reach live).
        """
        new_sd = dict(state_dict)
        # Width BEFORE los was added = _SPATIAL_C minus the channels appended
        # AFTER it too (los, then reach). Subtracting reach as well keeps this
        # migration matching ONLY genuinely pre-los checkpoints (a post-los/
        # pre-reach ckpt like uni_v7 has output = _SPATIAL_C - N_REACH, which is
        # N_LOS wider than this old_spatial, so it correctly passes through here
        # and is handled by the reach migration instead).
        old_spatial = (_SPATIAL_C - N_LOS_GRID_CHANNELS - N_REACH_GRID_CHANNELS
                       - N_THREAT_GRID_CHANNELS)

        # grid_query projections: both the pre-tiling single `grid_query_proj.*`
        # and the per-group/per-arch `grid_query_projs.N.*`. Pad any whose first
        # (output) dim == old spatial width.
        for key in list(new_sd.keys()):
            base = key.rsplit(".", 1)[0]
            if not (base == "grid_query_proj" or base.startswith("grid_query_projs.")):
                continue
            w = new_sd[key]
            if key.endswith(".weight") and w.dim() == 2 and w.shape[0] == old_spatial:
                pad = torch.zeros(N_LOS_GRID_CHANNELS, w.shape[1], dtype=w.dtype)
                new_sd[key] = torch.cat([w, pad], dim=0)
            elif key.endswith(".bias") and w.shape[0] == old_spatial:
                pad = torch.zeros(N_LOS_GRID_CHANNELS, dtype=w.dtype)
                new_sd[key] = torch.cat([w, pad], dim=0)

        # trunk first layer: insert zero column(s) at the end of self_cell_feat.
        tw = new_sd.get("trunk.0.weight")
        old_trunk_in = 64 + 32 + old_spatial + 16 + N_DECISION_CTX
        if tw is not None and tw.shape[1] == old_trunk_in:
            ins = 64 + 32 + old_spatial               # end of self_cell_feat slice
            pad = torch.zeros(tw.shape[0], N_LOS_GRID_CHANNELS, dtype=tw.dtype)
            new_sd["trunk.0.weight"] = torch.cat(
                [tw[:, :ins], pad, tw[:, ins:]], dim=1)
        return new_sd

    @staticmethod
    def adapt_state_dict_for_reach_grid(state_dict: dict) -> dict:
        """Migrate a pre-REACH-channel checkpoint (2026-06-19). One spatial
        channel (per-cell weapon-reach proximity to nearest enemy) was appended
        to the END of spatial_feat, AFTER los. Same zero-pad surgery as
        adapt_state_dict_for_los_grid, one column later:
          - grid_query projection OUTPUT: append N_REACH zero rows.
          - trunk first-layer INPUT: insert N_REACH zero cols at the end of the
            self_cell_feat slice (now _SPATIAL_C wide, reach is its last col).
        Zero-pad ⇒ bit-exact. Width detection keys on the post-los/pre-reach
        width so already-migrated checkpoints pass through. Runs after los."""
        new_sd = dict(state_dict)
        # post-los width = _SPATIAL_C minus the channels appended AFTER reach
        # (reach itself + threat), so this matches ONLY post-los/pre-reach ckpts.
        old_spatial = (_SPATIAL_C - N_REACH_GRID_CHANNELS
                       - N_THREAT_GRID_CHANNELS)
        for key in list(new_sd.keys()):
            base = key.rsplit(".", 1)[0]
            if not (base == "grid_query_proj" or base.startswith("grid_query_projs.")):
                continue
            w = new_sd[key]
            if key.endswith(".weight") and w.dim() == 2 and w.shape[0] == old_spatial:
                pad = torch.zeros(N_REACH_GRID_CHANNELS, w.shape[1], dtype=w.dtype)
                new_sd[key] = torch.cat([w, pad], dim=0)
            elif key.endswith(".bias") and w.shape[0] == old_spatial:
                pad = torch.zeros(N_REACH_GRID_CHANNELS, dtype=w.dtype)
                new_sd[key] = torch.cat([w, pad], dim=0)
        tw = new_sd.get("trunk.0.weight")
        old_trunk_in = 64 + 32 + old_spatial + 16 + N_DECISION_CTX
        if tw is not None and tw.shape[1] == old_trunk_in:
            ins = 64 + 32 + old_spatial
            pad = torch.zeros(tw.shape[0], N_REACH_GRID_CHANNELS, dtype=tw.dtype)
            new_sd["trunk.0.weight"] = torch.cat(
                [tw[:, :ins], pad, tw[:, ins:]], dim=1)
        return new_sd

    @staticmethod
    def adapt_state_dict_for_threat_grid(state_dict: dict) -> dict:
        """Migrate a pre-THREAT-channel checkpoint (2026-06-19). One spatial
        channel (per-cell melee-threat flag) was appended to the END of
        spatial_feat, AFTER reach. Same zero-pad surgery as the los/reach
        migrations, one column later:
          - grid_query projection OUTPUT: append N_THREAT zero rows.
          - trunk first-layer INPUT: insert N_THREAT zero cols at the end of the
            self_cell_feat slice (now _SPATIAL_C wide, threat is its last col).
        Zero-pad ⇒ bit-exact. Width detection keys on the post-reach/pre-threat
        width so already-migrated checkpoints pass through. Runs after reach."""
        new_sd = dict(state_dict)
        old_spatial = _SPATIAL_C - N_THREAT_GRID_CHANNELS   # = post-reach width
        for key in list(new_sd.keys()):
            base = key.rsplit(".", 1)[0]
            if not (base == "grid_query_proj" or base.startswith("grid_query_projs.")):
                continue
            w = new_sd[key]
            if key.endswith(".weight") and w.dim() == 2 and w.shape[0] == old_spatial:
                pad = torch.zeros(N_THREAT_GRID_CHANNELS, w.shape[1], dtype=w.dtype)
                new_sd[key] = torch.cat([w, pad], dim=0)
            elif key.endswith(".bias") and w.shape[0] == old_spatial:
                pad = torch.zeros(N_THREAT_GRID_CHANNELS, dtype=w.dtype)
                new_sd[key] = torch.cat([w, pad], dim=0)
        tw = new_sd.get("trunk.0.weight")
        old_trunk_in = 64 + 32 + old_spatial + 16 + N_DECISION_CTX
        if tw is not None and tw.shape[1] == old_trunk_in:
            ins = 64 + 32 + old_spatial
            pad = torch.zeros(tw.shape[0], N_THREAT_GRID_CHANNELS, dtype=tw.dtype)
            new_sd["trunk.0.weight"] = torch.cat(
                [tw[:, :ins], pad, tw[:, ins:]], dim=1)
        return new_sd

    @staticmethod
    def adapt_state_dict_for_threat_end(state_dict: dict) -> dict:
        """Migrate a checkpoint whose end_head input is narrower than the live
        [end_features, self_threat, self_reach] width (2026-06-19). The self-cell
        threat then reach flags were appended to end_head's input, so
        end_head.0.weight gains trailing INPUT columns. Zero-pad ⇒ bit-exact
        (new columns null their product until trained). Handles BOTH a pre-threat
        ckpt (width END_FEATURES_DIM → +N_THREAT+N_REACH) and a threat-era ckpt
        (width END_FEATURES_DIM+N_THREAT → +N_REACH): pad whatever is short up to
        the live width, appending zeros at the END (matching the concat order).
        critic is untouched — these scalars feed end_head only."""
        new_sd = dict(state_dict)
        target = (END_FEATURES_DIM + N_THREAT_GRID_CHANNELS
                  + N_REACH_GRID_CHANNELS)
        w = new_sd.get("end_head.0.weight")
        if (w is not None and w.dim() == 2
                and END_FEATURES_DIM <= w.shape[1] < target):
            pad = torch.zeros(w.shape[0], target - w.shape[1], dtype=w.dtype)
            new_sd["end_head.0.weight"] = torch.cat([w, pad], dim=1)
        return new_sd

    @staticmethod
    def adapt_state_dict_for_blocked_skill(state_dict: dict) -> dict:
        """Migrate a pre-blocked-skill-feature checkpoint (2026-06-14). The
        skill_head input gained a trailing 64-wide block (blockedness × sk_emb)
        so the policy can learn to choose MOVE when wall-blocked. Append 64 zero
        columns to the END of every skill_heads.*.weight (and the legacy single
        skill_head.weight) → the new feature gets zero weight → bit-exact (it is
        also exactly 0 in open layouts by construction, so a trained head stays
        bit-exact there too). The pre-feature input width is inferred from
        grid_query_projs (its input = 64+hidden) + sk_ent_ctx(32) + join(1), so
        already-migrated / unrecognised checkpoints pass through untouched.
        Runs LAST in the chain. entity_heads are NOT touched (unchanged width)."""
        new_sd = dict(state_dict)
        gq = None
        for k, v in new_sd.items():
            if k.startswith("grid_query_projs.") and k.endswith(".weight") and v.dim() == 2:
                gq = v.shape[1]                       # = 64 + hidden
                break
        if gq is None:
            return new_sd
        old_skill_in = gq + 32 + 1                     # sk_emb in gq; +ctx +join
        for key in list(new_sd.keys()):
            base = key.rsplit(".", 1)[0]
            if not (base == "skill_head" or base.startswith("skill_heads.")):
                continue
            if not key.endswith(".weight"):
                continue
            w = new_sd[key]
            if w.dim() == 2 and w.shape[1] == old_skill_in:
                pad = torch.zeros(w.shape[0], 64, dtype=w.dtype)
                new_sd[key] = torch.cat([w, pad], dim=1)
        return new_sd

    @staticmethod
    def adapt_state_dict_for_obs(state_dict: dict) -> dict:
        """Full obs-era chain: pre-v3 → v3 → v4 → skill-dtype → v5 →
        cimmun-entity → decision-ctx → los → reach → threat → blocked-skill →
        threat-end → cimmun-heads. Loaders that don't need the per-arch head
        tiling (e.g. 1-head-group distilled students) call this."""
        return CombatPolicyNet.adapt_state_dict_for_cimmun_heads(
          CombatPolicyNet.adapt_state_dict_for_threat_end(
           CombatPolicyNet.adapt_state_dict_for_blocked_skill(
            CombatPolicyNet.adapt_state_dict_for_threat_grid(
             CombatPolicyNet.adapt_state_dict_for_reach_grid(
              CombatPolicyNet.adapt_state_dict_for_los_grid(
                CombatPolicyNet.adapt_state_dict_for_decision_ctx(
                 CombatPolicyNet.adapt_state_dict_for_ability_entity(
                  CombatPolicyNet.adapt_state_dict_for_cimmun_entity(
                    CombatPolicyNet.adapt_state_dict_for_obs_v5(
                        CombatPolicyNet.adapt_state_dict_for_skill_dtype(
                            CombatPolicyNet.adapt_state_dict_for_obs_v4(
                                CombatPolicyNet.adapt_state_dict_for_obs_v3(
                                    state_dict)))))))))))))

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

        # ablate_archetype: drop the arch one-hot columns [_ARCH_OH_START:_END]
        # from EVERY entity row before the MLP — the class never reaches a weight.
        # (entities is kept intact for the position reads / joins / presence below.)
        if self.ablate_archetype:
            ent_in = torch.cat([entities[..., :_ARCH_OH_START],
                                entities[..., _ARCH_OH_END:]], dim=-1)
        else:
            ent_in = entities
        ent_emb  = self.entity_mlp(ent_in)

        # encode_entity_skills (obs v8): summarise EACH entity's static kit through
        # the SAME skill encoder the self uses, and add it into that entity's
        # embedding — so distinct kits with identical capability_descriptors stop
        # reading identical. Zero-init entity_kit_proj ⇒ inert until trained.
        if self.encode_entity_skills:
            es  = obs["entity_skills"]          # [B, E, S, F]
            esm = obs["entity_skill_mask"]      # [B, E, S]
            B_, E_, S_, F_ = es.shape
            kpm = esm.reshape(B_ * E_, S_) < 0.5                 # True = padding slot
            # A fully-empty entity row (every slot padded) would make the encoder
            # softmax over an all-masked sequence → NaN; un-mask those rows for the
            # forward, then the masked mean below zeros them out anyway.
            all_pad = kpm.all(dim=1, keepdim=True)
            enc = self.skill_encoder(
                self.skill_proj(es.reshape(B_ * E_, S_, F_)),
                src_key_padding_mask=kpm & ~all_pad)            # [B*E, S, 64]
            w = esm.reshape(B_ * E_, S_, 1)
            kit = (enc * w).sum(1) / w.sum(1).clamp(min=1.0)    # [B*E, 64]; empty→0
            ent_emb = ent_emb + self.entity_kit_proj(kit.reshape(B_, E_, 64))

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
        los_grid      = obs["los_grid"]                   # [B, 1, N_GRID, N_GRID]
        reach_grid    = obs["reach_grid"]                  # [B, 1, N_GRID, N_GRID]
        threat_grid   = obs["threat_grid"]                 # [B, 1, N_GRID, N_GRID]
        # los, reach, threat appended LAST (in that order) so each migration zero-
        # pads the matching final spatial column / grid_query row, bit-exact for
        # old checkpoints. threat is the newest → outermost (last) column.
        spatial_feat  = torch.cat(                        # [B, _SPATIAL_C, N_GRID, N_GRID]
            [terrain_ch, entity_grid, distance_grid, los_grid, reach_grid,
             threat_grid], dim=1)

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
        # ablate_archetype: the class-conditioner is absent entirely.
        if not self.ablate_archetype:
            arch_oh = entities[:, 0, _ARCH_OH_START:_ARCH_OH_END]  # [B, N_ARCHETYPES]
            gamma   = self.film_gamma(arch_oh)                    # [B, hidden]
            beta    = self.film_beta(arch_oh)                     # [B, hidden]
            h = gamma * h + beta

        return sk_emb, ent_emb, h, key_padding, spatial_feat, self_cell_feat

    def forward(self, obs: dict) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns (end_logit, skill_logits, entity_logits, grid_logits).

        end_logit:    [B]
        skill_logits: [B, N_SKILL_SLOTS]
        entity_logits:[B, N_SKILL_SLOTS, N_ENTITY_SLOTS] — per-skill
        grid_logits:  [B, N_SKILL_SLOTS, N_GRID*N_GRID]  — per-skill
        """
        sk_emb, ent_emb, h, key_padding, spatial_feat, self_cell_feat = self._encode(obs)
        B = h.shape[0]

        # Per-sample archetype id (used to gather the right per-arch head).
        # Clamped to the head-group count: a single-group (distilled) net
        # routes every identity — including all-zero one-hots (monsters,
        # chimeras) — to its one shared head. On 12-group nets the clamp is
        # a no-op (argmax of 12 bits is already < 12).
        entities = obs["entities"]
        # ablate_archetype: don't read the arch columns at all — every sample
        # routes to the single shared head (this flag implies n_head_groups=1).
        if self.ablate_archetype:
            arch_ids = torch.zeros(B, dtype=torch.long, device=h.device)
        else:
            arch_ids = entities[:, 0, _ARCH_OH_START:_ARCH_OH_END].argmax(dim=1)  # [B]
            arch_ids = arch_ids.clamp(max=len(self.skill_heads) - 1)

        # end_head reads from end_features (NOT from the shared trunk h) — see
        # __init__ docstring. The self-cell THREAT and REACH flags are appended
        # so the stop decision sees "am I in danger" (threat → kite-stop) and
        # "can I attack from here" (reach → end+keep-heal when I can't). Spatial
        # channel order is [...los, reach, threat], threat last:
        #   threat = self_cell_feat[:, C-N_THREAT : C]
        #   reach  = self_cell_feat[:, C-N_THREAT-N_REACH : C-N_THREAT]
        # Concat order [end_features, threat, reach] → reach is the newest (last)
        # column; zero-padded weight columns for old checkpoints ⇒ bit-exact.
        _thr0 = _SPATIAL_C - N_THREAT_GRID_CHANNELS
        self_threat = self_cell_feat[:, _thr0:_SPATIAL_C]
        self_reach  = self_cell_feat[:, _thr0 - N_REACH_GRID_CHANNELS:_thr0]
        # ablate_archetype: drop end_features' trailing arch one-hot block.
        ef = obs["end_features"]
        if self.ablate_archetype:
            ef = ef[..., :END_FEATURES_DIM - N_ARCHETYPES]
        end_in = torch.cat([ef, self_threat, self_reach], dim=-1)
        end_logit = self.end_head(end_in).squeeze(-1)

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

        # v6 condition-immunity join (mirror of the typed-matchup join): each
        # skill row's applies_status bits ⋅ each entity's condition-immunity
        # slice = how many of the conditions this skill inflicts simply BOUNCE
        # off that target. NEGATED so the sign matches the typed join (≤0 = bad):
        # 0 = lands / no condition / no info, −k = k inflicted conditions wasted
        # on an immune target. Zeroed cimmun descriptors (pre-v6 episodes) → 0,
        # and a migrated pre-v6 head has a zero cimmun weight column ⇒ bit-exact.
        sk_st   = obs["skills"][:, :, SKILL_STATUS_START:
                                SKILL_STATUS_START + N_STATUS_SLOTS]           # [B, S, 16]
        ent_ci  = entities[:, :, I_DESC_CIMMUN:I_DESC_CIMMUN + N_V6_CIMMUN]    # [B, E, 16]
        cimmun_se = -(sk_st.unsqueeze(2) * ent_ci.unsqueeze(1)).sum(-1)        # [B, S, E]
        # skill_head summary: LEAST-wasted target over present enemy rows (amax
        # of the ≤0 values = the target fewest of this skill's conditions bounce
        # off). Same enemy-row masking / no-enemy→0 convention as the typed join.
        cimmun_skill = cimmun_se.masked_fill(~valid_enemy, float("-inf")).amax(-1)
        cimmun_skill = torch.where(valid_enemy.any(-1), cimmun_skill,
                                   torch.zeros_like(cimmun_skill))            # [B, S]

        # skill_head: per skill slot. Stack all N_ARCH copies' outputs, then
        # gather the one matching each sample's archetype.
        h_skill      = h.unsqueeze(1).expand(-1, N_SKILL_SLOTS, -1)
        # blocked → which-skill feature: blockedness × sk_emb. blockedness =
        # 1 − self-cell LoS proximity. The LoS channel is the second-to-last
        # spatial channel now that the reach channel is appended AFTER it, so
        # index explicitly past reach (NOT _SPATIAL_C-1, which is reach). 0
        # whenever the agent can see an enemy (open layouts: LoS all-ones → 0),
        # so this term vanishes in open → bit-exact there.
        _LOS_IDX = _SPATIAL_C - 1 - N_REACH_GRID_CHANNELS - N_THREAT_GRID_CHANNELS
        blockedness = (1.0 - self_cell_feat[:, _LOS_IDX]).clamp(min=0.0)        # [B]
        blocked_feat = blockedness.view(-1, 1, 1) * sk_emb                     # [B, S, 64]
        if dtype_v1:   # pre-dtype heads have no join column
            skill_in = torch.cat([sk_emb, h_skill, sk_ent_ctx], dim=-1)
        else:
            # order must match the skill_head input width built in __init__:
            # sk_emb | (h) | sk_ent_ctx | (typed-join) | blocked | (cimmun-join).
            # h dropped when drop_noop_h (constant across slots ⇒ cancels);
            # joins dropped when ablate_immunity_joins.
            parts = [sk_emb]
            if not self.drop_noop_h:
                parts.append(h_skill)
            parts.append(sk_ent_ctx)
            if not self.ablate_immunity_joins:
                parts.append(join_skill.unsqueeze(-1))
            parts.append(blocked_feat)
            if not self.ablate_immunity_joins:
                parts.append(cimmun_skill.unsqueeze(-1))
            skill_in = torch.cat(parts, dim=-1)
        skill_all    = torch.stack(
            [hd(skill_in).squeeze(-1) for hd in self.skill_heads], dim=1)     # [B, N_ARCH, S]
        sk_gather_idx = arch_ids.view(B, 1, 1).expand(-1, 1, N_SKILL_SLOTS)
        skill_logits = skill_all.gather(1, sk_gather_idx).squeeze(1)          # [B, S]
        if self.skill_combo_dim > 0:
            # Skill-combination identity → per-skill preference (bilinear term).
            proj = self.skill_combo_proj(sk_emb)                     # [B, S, k]
            m    = (~key_padding).to(proj.dtype).unsqueeze(-1)       # [B, S, 1] held skills
            c    = (proj * m).sum(1) / m.sum(1).clamp(min=1.0)       # [B, k] order-invariant
            pref = self.skill_combo_pref(c)                          # [B, 64]
            skill_logits = skill_logits + torch.einsum("bsd,bd->bs", sk_emb, pref)
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
            # order matches entity_head width: sk_emb | (h) | ent_emb | (joins).
            # h dropped when drop_noop_h (constant across the per-skill entity
            # softmax ⇒ cancels); joins dropped when ablate_immunity_joins.
            parts = [sk_for_ent]
            if not self.drop_noop_h:
                parts.append(h_for_ent)
            parts.append(ent_for_ent)
            if not self.ablate_immunity_joins:
                parts.append(join_se.unsqueeze(-1))
                parts.append(cimmun_se.unsqueeze(-1))
            ent_in = torch.cat(parts, dim=-1)
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
        # Move own-cell DEADLOCK-BREAKER (null-effect family, NARROW scope;
        # miner mine|666|0136: two teammates stacked on one cell spamming
        # "move to my own cell" for 50+ turns, enemies 2.6m away). The first
        # version masked the own cell unconditionally — measured regression:
        # standing ground is a legitimate stance (let melee walk into your
        # reach), and forcing a reselect turned it into aimless wandering
        # (seed0 no_engage 0→60, affected-game loss rate 58%, single-variable
        # A/B on mine|0|0000: 0dmg→34dmg with the guard off). v2 mirrors the
        # pure-turtle guard's activation exactly: fire ONLY when the agent has
        # attempted zero attacks all episode AND a living enemy sits within
        # weapon reach + one move budget, and reselect ONLY among cells
        # strictly closer to the nearest living enemy — one attack attempt
        # unlocks stand-still forever; kiting-with-output never triggers.
        if skills[skill_idx].skill_id == "move":
            from .obs import (N_GRID as _NGm, GRID_CELL_SIZE_M as _CSm,
                              partition_entities as _pe)
            _gx = max(0, min(_NGm - 1, int(agent.position.x / _CSm)))
            _gy = max(0, min(_NGm - 1, int(agent.position.y / _CSm)))
            _own = _gx * _NGm + _gy
            if grid == _own and getattr(agent, "_outgoing_attempts", 0) == 0:
                from ..engine.combat import MOVE_BUDGET_M as _MVm
                try:
                    _reach = float(agent.get_weapon().range_normal) or 1.5
                except Exception:
                    _reach = 1.5
                _, _foes = _pe(ws, agent_id)
                _live = [ws.characters[x].position for x in _foes
                         if ws.characters[x].is_alive()]
                _nd = min((agent.position.distance_to(p) for p in _live),
                          default=None)
                if _nd is not None and _nd <= _reach + _MVm + 0.05:
                    import numpy as _np2
                    _tgt = min(_live,
                               key=lambda p: agent.position.distance_to(p))
                    _cs = (_np2.arange(_NGm, dtype=_np2.float32) + 0.5) * _CSm
                    _cx = _np2.repeat(_cs, _NGm); _cy = _np2.tile(_cs, _NGm)
                    _dist = _np2.sqrt((_cx - _tgt.x) ** 2 + (_cy - _tgt.y) ** 2)
                    _closer = torch.from_numpy(_dist < _nd - 1e-3).to(row.device)
                    _ok = (row > -1e8) & _closer
                    _ok[_own] = False        # 自格中心可能比站位更近敵——顯式排除
                    if bool(_ok.any()):
                        grid = int(row.clone().masked_fill(~_ok, -1e9)
                                   .argmax().item())
        # AoE dominated-cell reselect (geometric, archetype-agnostic): if the
        # chosen damaging-AoE cell catches allies (incl. self) while some
        # clean cell covers AT LEAST as many living enemies, the choice is
        # strictly dominated — reselect by the model's own logits WITHIN the
        # safe set {zero allies, enemy coverage ≥ original}. Coverage can
        # never drop and no behaviour is invented (model preference decides
        # among safe cells); a melee scrum (empty safe set) is untouched, so
        # nuking an engaged cluster stays legal. Pure masking was rejected:
        # argmax over the unstructured remainder can whiff the cast entirely
        # (worse than friendly fire). Same guard family as disengage-null /
        # heal-at-full / entity-range masks.
        f = skills[skill_idx].features
        rad = float(getattr(f, "aoe_radius_m", 0.0) or 0.0)
        if rad > 0 and getattr(f, "expected_damage", 0) > 0:
            import numpy as _np
            from .obs import (partition_entities, N_GRID as _NG,
                              GRID_CELL_SIZE_M as _CS)
            allies, enemies = partition_entities(ws, agent_id)
            apos = [ws.characters[a].position for a in allies
                    if a in ws.characters and ws.characters[a].is_alive()]
            apos.append(agent.position)
            epos = [ws.characters[e].position for e in enemies
                    if ws.characters[e].is_alive()]
            if apos and epos:
                cs = (_np.arange(_NG, dtype=_np.float32) + 0.5) * _CS
                cx = _np.repeat(cs, _NG); cy = _np.tile(cs, _NG)
                ne = _np.zeros(_NG * _NG, dtype=_np.int32)
                na = _np.zeros(_NG * _NG, dtype=_np.int32)
                for p in epos:
                    ne += (((cx - p.x) ** 2 + (cy - p.y) ** 2) <= rad * rad)
                for p in apos:
                    na += (((cx - p.x) ** 2 + (cy - p.y) ** 2) <= rad * rad)
                if na[grid] > 0:
                    legal = (row > -1e8).cpu().numpy()
                    safe = (na == 0) & (ne >= max(1, int(ne[grid]))) & legal
                    if safe.any():
                        r2 = row.clone().masked_fill(
                            torch.from_numpy(~safe), -1e9)
                        grid = int(r2.argmax().item())
        return (skill_idx, 0, grid)
    # SELF or unrecognised: both irrelevant
    return (skill_idx, 0, 0)


def apply_entity_mask(entity_logits: "torch.Tensor", obs: dict,
                      ws=None, agent_id: str = "",
                      mask_immune_null: bool = True) -> "torch.Tensor":
    """Mask empty (padding) entity slots and self.

    ``mask_immune_null`` (default True) governs ONLY the entity-layer immune
    whiff gate (masking an immune target for a pure-damage skill). Set False
    for the 2026-07-06 sk_ent_ctx ablation. All production callers keep the
    default ⇒ bit-exact.

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

        # Enemy-target skills — per-(skill, slot) ENGINE-legality gate:
        #   • LoS：decode_action 會把無視線目標靜默轉 end-turn no-op（原有條件）
        #   • range／目標狀態（bug_miner 2026-07-02 挖出）：技能層遮罩只保證
        #     「最近敵在射程內」，實體層原本不看距離 → 模型可鎖 reach 外的較遠敵
        #     → 引擎 ERROR 白費回合（v5 一場 75 回合 0 傷平手）。鏡射引擎原語：
        #     ATTACK 用 attack_range_check（近戰 reach／遠程 range_long＋LoS）＋
        #     requires/blocked_by_target_status；帶 range_m 的單體法術比距離。
        # 每列守門：某技能列全被遮就還原該列＝遮罩永遠不製造「無合法目標」新狀態
        # （維持舊語義：decode no-op／引擎 ERROR 仍是後盾）。
        from ..engine.combat import attack_range_check
        from .obs import ENEMY_SLOT_START
        ENEMY_TT = (TargetType.SINGLE_ENEMY, TargetType.MULTI_ENEMY)
        bf = ws.combat.battlefield if ws.combat else None
        _, enemies = partition_entities(ws, agent_id)
        slot_cid = {}
        for j, cid in enumerate(enemies):
            slot = ENEMY_SLOT_START + j
            if slot >= entity_logits.shape[2]:
                break
            slot_cid[slot] = cid
        for i, sk in enumerate(skills):
            if i >= entity_logits.shape[1]:
                break
            if sk.features.target_type not in ENEMY_TT or not slot_cid:
                continue
            legal = {}
            for slot, cid in slot_cid.items():
                tch = ws.characters[cid]
                ok = bf is None or bf.has_line_of_sight(
                    agent.position, tch.position)
                if ok and sk.features.target_type == TargetType.SINGLE_ENEMY:
                    # ATTACK range/目標狀態/魅惑門＋單體法術 range_m＝共用真源
                    ok = _enemy_target_legal(agent, agent_id, sk, cid, ws, bf)
                    # 實體層免疫空砸（skill 層 null_dmg_idx 的目標層 sibling；
                    # miner mine|666|0065：shadow 反覆摸黯蝕免疫的敵 shadow、
                    # 旁邊站著能打的 orc）：純傷害技（無狀態 rider、無治療）
                    # 對佔比加權倍率≈0 的目標＝引擎可證 null——遮掉該目標。
                    # 「全遮則還原」後盾（下方既有）保證單免疫敵時不剝奪揮擊。
                    f_ = sk.features
                    if (ok and mask_immune_null and f_.expected_damage > 0
                            and not any(f_.applies_status)
                            and f_.expected_healing <= 0):
                        pairs_ = list(f_.iter_damage_types())
                        if pairs_:
                            mult_ = sum(
                                sh * float((tch.damage_multipliers or {})
                                           .get(tok, 1.0))
                                for tok, sh in pairs_)
                            if mult_ <= 0.05:
                                ok = False
                legal[slot] = ok
            if any(legal.values()) and not all(legal.values()):
                for slot, ok in legal.items():
                    if not ok:
                        entity_logits[0, i, slot] = -1e9
    return entity_logits


def _enemy_target_legal(agent, agent_id, sk, cid, ws, bf):
    """引擎同源的 per-(skill, enemy) 當下合法性：LoS＋ATTACK（range/目標狀態
    前置/魅惑門）＋單體法術 range_m。apply_entity_mask 逐槽門與
    apply_resource_mask 的 legal-now 支配預掃共用的唯一真源。
    語義保守：builder 回 None/丟例外＝不判非法（維持 decode/引擎後盾）。"""
    from ..engine.combat import attack_range_check
    tch = ws.characters[cid]
    if bf is not None and not bf.has_line_of_sight(agent.position, tch.position):
        return False
    try:
        act = sk.build_action(agent_id, cid, (tch.position.x, tch.position.y))
    except Exception:
        return True
    if not isinstance(act, dict):
        return True
    if act.get("type") == "ATTACK":
        w = agent.get_weapon(act.get("weapon", ""))
        ok, _, _ = attack_range_check(agent, tch, w, bf)
        if not ok:
            return False
        rts = act.get("requires_target_status")
        if rts and not tch.has_status(rts):
            return False
        bts = act.get("blocked_by_target_status")
        if bts and tch.has_status(bts):
            return False
        # 魅惑鏡射（0230 家族的受害席）：引擎 ATTACK 門拒絕受魅者攻擊施魅者
        if any(getattr(fx, "name", "") == "charmed"
               and getattr(fx, "source_id", "") == cid
               for fx in agent.status_effects):
            return False
    elif "range_m" in act:
        # 與引擎逐位元同條件（同一 Vec2 距離、同容差）
        if (agent.position.distance_to(tch.position)
                > float(act["range_m"]) + 1e-6):
            return False
    return True


def apply_resource_mask(skill_logits: "torch.Tensor",
                        resources: dict,
                        ws=None, agent_id: str = "",
                        mask_immune_null: bool = True) -> "torch.Tensor":
    """Mask skill slots that the engine would reject.

    ``mask_immune_null`` (default True) governs ONLY the immune-null attack
    pre-pass (masking a pure-damage skill whose type every enemy resists to 0).
    Set False to let the policy freely pick an immune attack — used by the
    2026-07-06 sk_ent_ctx ablation experiment, where this engine-truth gate
    would otherwise solve immunity for free (a fresh net already scores 100%).
    Every other legality gate (resource cost, range, LoS, heal-at-full, …) is
    unaffected. All production callers keep the default ⇒ bit-exact.

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
    from ..engine.abilities import ABILITY_REGISTRY
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
    # Immune-null attack pre-pass: a PURE-damage skill (no status rider, no
    # heal) whose damage-share-weighted multiplier is ~0 against EVERY living
    # enemy deals literally nothing — engine-provable null. Mask it ONLY when
    # some other damaging skill in the current kit has a non-null multiplier
    # (strict dominance; a monotype kit facing full immunity keeps its swing —
    # never create a no-legal-offense state). Shares come from
    # features.iter_damage_types(), the same engine data damaging_options /
    # the obs dtype tail use, so a divine_smite row keeps its 光耀 viability
    # when the weapon type is immune. Same guard family as disengage-when-far
    # / heal-at-full / repeat-dodge / AoE dominated-cell reselect.
    null_dmg_idx: set[int] = set()
    live_pairs = [(e, ws.characters[e]) for e in enemies
                  if ws.characters[e].is_alive()]
    live_enemies = [oc for _, oc in live_pairs]
    if live_enemies and mask_immune_null:
        _mults = {}
        _mults_now = {}
        for i, sk in enumerate(skills):
            f = sk.features
            if (i == 0 or f.expected_damage <= 0 or any(f.applies_status)
                    or f.expected_healing > 0):
                continue
            pairs = list(f.iter_damage_types())
            if not pairs:
                continue
            def _m(oc, _pairs=pairs):
                return sum(sh * float((oc.damage_multipliers or {})
                                      .get(tok, 1.0)) for tok, sh in _pairs)
            _mults[i] = max(_m(oc) for oc in live_enemies)
            # legal-now 視角（0150 家族）：只對「當下引擎合法可執行」的目標
            # 取最佳倍率——遠處的高 EV 目標不是本回合的替代選項
            if f.target_type in (TargetType.SINGLE_ENEMY,
                                 TargetType.MULTI_ENEMY):
                now = [_m(oc) for eid, oc in live_pairs
                       if _enemy_target_legal(agent, agent_id, sk, eid, ws, bf)]
                if now:
                    _mults_now[i] = max(now)
        if _mults and max(_mults.values()) > 0.05:
            null_dmg_idx = {i for i, m in _mults.items() if m <= 0.05}
        # legal-now 嚴格支配（miner mine|666|0150：雙武器身體、唯一搆得到的
        # 敵免疫黯蝕——「生命吸取打免疫者=0」被「鬼爪打同一人=正傷」支配）：
        # 存在當下合法且非 null 的純傷害選項時，遮掉當下全 null 的純傷害技。
        # 單型 kit／敵全免疫時 _mults_now 全 ≤0.05＝門不開＝揮擊保留。
        if _mults_now and max(_mults_now.values()) > 0.05:
            null_dmg_idx |= {i for i, m in _mults_now.items() if m <= 0.05}
    for i, sk in enumerate(skills):
        if i >= skill_logits.shape[-1]:
            break
        if sk.skill_id == "move":
            if not has_move:
                skill_logits[..., i] = -1e9
            continue
        if i in null_dmg_idx:
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
        # Pure heals (heal > 0, no damage, no status) are NULL-EFFECT when every
        # legal target is effectively full — same masking contract as the
        # disengage-when-far gate below. Threshold mirrors the degen-audit
        # heal_full detector (hp ≥ 95% max); dying targets (hp 0) always count
        # as hurt, so revive-capable heals stay available.
        _f = sk.features
        if (_f.expected_healing > 0 and _f.expected_damage <= 0
                and not any(_f.applies_status)):
            if tt == TargetType.SELF:
                pool = [agent]
            elif tt in (TargetType.SINGLE_ALLY, TargetType.MULTI_ALLY):
                pool = [agent] + [ws.characters[x] for x in allies]
            else:
                pool = []
            if pool and all(c.hp >= c.max_hp * 0.95 for c in pool):
                skill_logits[..., i] = -1e9
                continue
        if tt == TargetType.SELF:
            # APPLY_MOD-already-active recast guard (null-effect family:
            # heal-at-full / disengage-when-far / repeat-dodge / pure-turtle).
            # add_status is idempotent on the modifier name (character.py), so
            # re-casting a self-buff whose modifier is already on the agent
            # does NOTHING to the target — engine-identical null (and the
            # re-concentrate path can even DROP an active concentration buff).
            # Worse, a zero-cost permanent buff (a passive trait modelled as a
            # free active: consumes=[], max_uses=0) is an infinite argmax
            # sink — miner 0012: grafted evasion recast 3-4×/turn, adjacent
            # kill legal, 22 rounds, 0 attack attempts, death. First cast
            # stays legal (1 action for a permanent buff = real value); the
            # mask only fires while the modifier is already active, so no
            # engine-honoured behaviour is ever removed.
            _bd = sk.build_action(agent_id, agent_id,
                                  (agent.position.x, agent.position.y))
            if (isinstance(_bd, dict) and _bd.get("type") == "APPLY_MOD"
                    and agent.has_status(_bd.get("modifier", ""))):
                skill_logits[..., i] = -1e9
                continue
            # Disengage only avoids opportunity attacks when you LEAVE an
            # enemy's reach. With no enemy adjacent it is a NULL-EFFECT action —
            # offering it lets an OOD / near-tie policy sink the turn into it
            # (measured: a caster whose primary damage is immune vs a melee
            # enemy at range disengage-loops, 0 dmg — combat_log 2026-07-01).
            # Mask when the nearest enemy is beyond any plausible melee reach
            # (3m > 5-ft normal AND 10-ft reach weapons); within that keep it
            # (a real disengage-then-reposition). Only disengage among the SELF
            # actions is null-when-far — dodge helps vs ranged, hide/buffs act.
            # Also null with NO remaining movement: disengage's only effect is
            # that the movement you make AFTERWARDS provokes no opportunity
            # attacks — with the budget already spent there is nothing for it
            # to protect (miner 0456: circle-move burns 9m, then disengage at
            # mv=0 + END, 50 rounds, 0 attacks, weapon legal at 1.4m).
            if sk.skill_id == "disengage" and (enemy_dist is None
                                               or enemy_dist > 3.0
                                               or resources.get("movement",
                                                                0.0) <= 0.05):
                skill_logits[..., i] = -1e9
            # Repeat-dodge with ZERO incoming pressure = empirically-null
            # repeat (dodge only matters if someone attacks before my next
            # turn; last round nobody even TRIED). First dodge is always
            # legal; any attack attempt (hit or miss — resolve_attack bumps
            # _incoming_attempts) unlocks it again. Kills the documented
            # defensive-inertia pocket (berserker dodge-brace loops vs
            # passive enemies, adjacent, 75-turn 0-damage draws) without
            # touching legit tanking-under-fire. Same null-effect family as
            # disengage-when-far / heal-at-full.
            if sk.skill_id == "dodge":
                ldr = getattr(agent, "_last_dodge_round", None)
                rnd = ws.combat.round_number if ws.combat else None
                if (ldr is not None and rnd is not None
                        and rnd <= ldr + 1
                        and getattr(agent, "_incoming_attempts", 0)
                        == getattr(agent, "_dodge_snapshot", -1)):
                    skill_logits[..., i] = -1e9
            # Pure-turtle guard (engage-commitment family): SOLO (no living
            # ally to carry the fight), enemy reachable THIS TURN (within
            # reach + remaining movement), action in hand, and ZERO offensive
            # attempts ALL GAME (execute_action bumps _outgoing_attempts,
            # save-spells included) — a 1v1 zero-output turtle can never win
            # (draw at best), so dodge/hide there is strictly non-winning;
            # ONE attack attempt unlocks them forever (tanking-with-output
            # stays legal). Trace-proven shapes: berserker adjacent (0.9m)
            # chip-dodging 50 rounds, and the doorstep stand (2.0m, movement
            # unspent, first dodge then death). Berserker is the documented
            # retrain-fragile class (BC surgery rejected repeatedly) — this
            # is the decision-layer fix. Disengage is the same defensive-turn
            # sink (miner 0456: disengage+END loop, zero attacks, weapon
            # legal); a zero-output kite can't win 1v1 either, and one attack
            # attempt likewise unlocks it (real kiting keeps output).
            if sk.skill_id in ("dodge", "hide", "disengage"):
                if (getattr(agent, "_outgoing_attempts", 0) == 0
                        and enemy_dist is not None
                        and enemy_dist <= fallback_reach
                        + resources.get("movement", 0.0) + 0.05
                        and not any(ws.characters[x].is_alive()
                                    for x in allies if x in ws.characters)):
                    skill_logits[..., i] = -1e9
            continue
        if tt in (TargetType.SINGLE_ALLY, TargetType.MULTI_ALLY):
            # Ally-target skills always have a legal in-range target: SELF
            # (decode_action resolves non-ally entity picks to self, and the
            # engine accepts self at distance 0). Range-masking these by some
            # ally's distance both blocked legitimate self-heals and used an
            # arbitrary (unsorted) ally — per-ally range gating lives in
            # apply_entity_mask instead.
            continue
        # Target-state precondition gate (engine-identical legality, NOT
        # strategy): execute_action's ATTACK path returns ERROR when
        # requires_target_status is absent / blocked_by_target_status is present
        # (combat.py rts/bts gate), and the scripted expert skips the same
        # (combat_policy). available_skills LISTS these abilities (the actor can
        # use them in principle) but is target-agnostic, so a swallow (needs a
        # `restrained` target) stays selectable against a lone non-restrained
        # enemy → engine ERROR → wasted/looped turn (play_gui seed audit). This
        # is the ONE absolutely-illegal enemy-target source available_skills
        # can't filter. Mask ONLY when NO living enemy satisfies the gate; in
        # 1vN a qualifying enemy keeps the skill and apply_entity_mask picks it.
        ab = ABILITY_REGISTRY.get(sk.skill_id)
        if ab is not None and (ab.requires_target_status
                               or ab.blocked_by_target_status):
            _rts, _bts = ab.requires_target_status, ab.blocked_by_target_status
            if not any((not _rts or e.has_status(_rts))
                       and (not _bts or not e.has_status(_bts))
                       for e in live_enemies):
                skill_logits[..., i] = -1e9
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
        # Engine-identical range gate (attack_range_check: `d > reach` rejects,
        # so d == reach is LEGAL; spell executors use `> range_m + 1e-6`).
        # The old conservative form (>= rng - 0.01) wrongly masked the legal
        # boundary window [reach-0.01, reach]: a champion standing at exactly
        # 1.5m from a permanently-restrained enemy had its attack masked
        # forever while move no-ops zeroed and dodge/hide hit the turtle guard
        # → 50-round zero-attempt stall (miner 0477, opp seat). Same distance
        # math (Vec2) on both sides ⇒ bit-safe to mirror the engine exactly.
        if target_dist > rng + 1e-6:
            skill_logits[..., i] = -1e9

    return skill_logits


class CombatPolicyNetNoArch(CombatPolicyNet):
    """CombatPolicyNet with the archetype (class) encoding fully removed.

    Structurally the SAME network as the parent (skill transformer, entity MLP,
    spatial grid, resource MLP, trunk, four hierarchical heads, standalone
    critic — plus every zero-weight carrier: typed-join, cimmun-join,
    blocked_feat, threat/reach/los channels). The only difference is that the
    net is given NO class label anywhere and so must infer class-appropriate
    behaviour from the skill pool, HP / AC / level, resources and positions:

      1. the per-entity archetype one-hot (self, allies AND enemies) is zeroed
         before it reaches the entity MLP / end_head / critic;
      2. the self archetype one-hot in end_features is zeroed;
      3. FiLM — the per-archetype gain/bias on the trunk — is pinned to identity
         (gamma=1, beta=0) and frozen, so there is no per-class OR global affine.

    Single shared head group (n_head_groups=1): the per-arch head machinery is
    inert because an all-zero one-hot argmaxes to head 0 and is clamped there.
    Fresh-init only (an experiment net); no checkpoint migration is involved.
    """

    def __init__(self, hidden: int = 128, skill_combo_dim: int = 0,
                 ablate_immunity_joins: bool = False, drop_noop_h: bool = False):
        # ablate_immunity_joins / drop_noop_h are handled structurally by the
        # base __init__ (narrower heads + omitted inputs in forward). Passed
        # through here.
        super().__init__(hidden=hidden, n_head_groups=1,
                         skill_combo_dim=skill_combo_dim,
                         ablate_immunity_joins=ablate_immunity_joins,
                         drop_noop_h=drop_noop_h)
        # FiLM → identity and frozen. arch is zeroed anyway (so gamma/beta would
        # otherwise read only their learnable bias = a global affine); pinning
        # to identity makes "no archetype modulation" exact and gradient-free.
        with torch.no_grad():
            self.film_gamma.weight.zero_(); self.film_gamma.bias.fill_(1.0)
            self.film_beta.weight.zero_();  self.film_beta.bias.zero_()
        for _p in (*self.film_gamma.parameters(), *self.film_beta.parameters()):
            _p.requires_grad_(False)

    @staticmethod
    def _strip_arch(obs: dict) -> dict:
        """Return a shallow copy of obs with every archetype one-hot zeroed.

        Feature-based slicing (no hardcoded class list): entity arch lives at
        [_ARCH_OH_START:_ARCH_OH_END] on every slot; the self-arch one-hot is
        the trailing N_ARCHETYPES block of end_features. Only the two tensors
        that carry archetype are cloned; all other obs keys pass through.
        """
        obs = dict(obs)
        ent = obs["entities"].clone()
        ent[..., _ARCH_OH_START:_ARCH_OH_END] = 0.0
        obs["entities"] = ent
        ef = obs["end_features"].clone()
        ef[..., -N_ARCHETYPES:] = 0.0
        obs["end_features"] = ef
        return obs

    def forward(self, obs: dict, *args, **kwargs):
        return super().forward(self._strip_arch(obs), *args, **kwargs)

    def value(self, obs: dict, *args, **kwargs):
        return super().value(self._strip_arch(obs), *args, **kwargs)
