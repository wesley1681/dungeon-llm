import numpy as np
import torch
from trpg.rl.model import CombatPolicyNet
from trpg.rl.obs import N_SKILL_SLOTS, N_ENTITY_SLOTS, ENTITY_DIM, N_GRID
from trpg.rl.obs import (N_ENTITY_GRID_CHANNELS, N_DISTANCE_GRID_CHANNELS,
                          END_FEATURES_DIM, N_DECISION_CTX)
from trpg.engine.skill import SKILL_FEATURE_DIM


def _fake_obs_batch(batch=2):
    return {
        "skills":           torch.zeros(batch, N_SKILL_SLOTS, SKILL_FEATURE_DIM),
        "skill_mask":       torch.ones(batch, N_SKILL_SLOTS),
        "entities":         torch.zeros(batch, N_ENTITY_SLOTS, ENTITY_DIM),
        "resources":        torch.zeros(batch, 4),
        "terrain":          torch.zeros(batch, N_GRID, N_GRID),
        "entity_grid":      torch.zeros(batch, N_ENTITY_GRID_CHANNELS, N_GRID, N_GRID),
        "distance_grid":    torch.zeros(batch, N_DISTANCE_GRID_CHANNELS, N_GRID, N_GRID),
        "end_features":     torch.zeros(batch, END_FEATURES_DIM),
        "decision_context": torch.zeros(batch, N_DECISION_CTX),
    }


def test_model_forward_output_shapes():
    net = CombatPolicyNet()
    obs = _fake_obs_batch(batch=4)
    end_logit, skill_logits, entity_logits, grid_logits = net(obs)
    assert end_logit.shape == (4,)
    assert skill_logits.shape == (4, N_SKILL_SLOTS)
    assert entity_logits.shape == (4, N_SKILL_SLOTS, N_ENTITY_SLOTS)
    assert grid_logits.shape == (4, N_SKILL_SLOTS, N_GRID * N_GRID)


def test_model_skill_mask_zeros_padding():
    net = CombatPolicyNet()
    obs = _fake_obs_batch(batch=2)
    obs["skill_mask"][:, 5:] = 0   # only first 5 valid
    _, skill_logits, _, _ = net(obs)
    # Padded slots should have very negative logits
    assert (skill_logits[:, 5:] < -1e8).all()


def test_skill_ent_attn_proj_exists():
    net = CombatPolicyNet()
    assert hasattr(net, "skill_ent_attn_proj"), "skill_ent_attn_proj layer missing"
    assert net.skill_ent_attn_proj.in_features == 64
    assert net.skill_ent_attn_proj.out_features == 32


def test_skill_head_input_size():
    """skill_heads[i] must accept sk_emb(64) + h(hidden) + sk_ent_ctx(32)
    + the typed-matchup join scalar."""
    net = CombatPolicyNet()
    hidden = net.trunk[0].out_features
    assert net.skill_heads[0].in_features == 64 + hidden + 32 + 1, (
        f"Expected {64+hidden+32+1}, got {net.skill_heads[0].in_features}"
    )
    # Per-arch sanity: should have N_ARCHETYPES copies
    from trpg.rl.obs import N_ARCHETYPES
    assert len(net.skill_heads) == N_ARCHETYPES


def _shrink_decision_ctx_sd(sd: dict) -> dict:
    """Strip the decision-context era (inverse of adapt_state_dict_for_decision_ctx).

    decision_ctx is appended LAST in the load chain (trunk.0.weight and
    critic.net.0.weight each grew N_DECISION_CTX columns at the END), so building
    an OLDER-era fixture peels it off FIRST. After this the rest of the shrink
    chain sees the pre-dctx widths it was written against."""
    from trpg.rl.obs import N_DECISION_CTX
    out = dict(sd)
    for key in ("trunk.0.weight", "critic.net.0.weight"):
        if key in out:
            out[key] = out[key][:, :-N_DECISION_CTX].clone()
    return out


def _shrink_skill_dtype_sd(sd: dict) -> dict:
    """Strip the skill-dtype era from a current state dict (test fixture).

    Inverse of adapt_state_dict_for_skill_dtype — run FIRST when building
    older-era fixtures, since that adapter runs LAST in the load chain:
    skill_proj back to the frozen 53 input columns, heads lose the matchup-
    join column, the critic's skill-pool tail block narrows 66 → 53.
    (decision_ctx is peeled even earlier — see _shrink_decision_ctx_sd.)
    """
    import re
    from trpg.rl.obs import END_FEATURES_DIM
    from trpg.rl.model import _SKILL_DIM_V1
    out = _shrink_decision_ctx_sd(sd)
    out["skill_proj.weight"] = sd["skill_proj.weight"][:, :_SKILL_DIM_V1].clone()
    head_re = re.compile(r"^(skill_head|entity_head)(s\.\d+)?\.weight$")
    for k in list(out):
        if head_re.match(k):
            out[k] = out[k][:, :-1].clone()
    cw = out["critic.net.0.weight"]
    full_in = (N_ENTITY_SLOTS * ENTITY_DIM + 4 + END_FEATURES_DIM
               + SKILL_FEATURE_DIM)
    assert cw.shape[1] == full_in
    out["critic.net.0.weight"] = \
        cw[:, :full_in - (SKILL_FEATURE_DIM - _SKILL_DIM_V1)].clone()
    return out


def _shrink_to_legacy_sd(sd: dict) -> dict:
    """Build a pre-v3-shaped state dict from a current one (test fixture).

    Inverse of adapt_state_dict_for_obs_v3's mapping: slice the entity_mlp
    input columns down to the legacy ENTITY_DIM and re-pack the critic's
    first-layer entity-block columns into the legacy 6-slot layout.
    """
    from trpg.rl.obs import ENEMY_SLOT_START, END_FEATURES_DIM
    from trpg.rl.model import _ENTITY_DIM_V2, _SKILL_DIM_V1
    sd = _shrink_skill_dtype_sd(sd)
    legacy_sd = dict(sd)
    old_dim = _ENTITY_DIM_V2
    legacy_sd["entity_mlp.0.weight"] = sd["entity_mlp.0.weight"][:, :old_dim].clone()

    n_ally_legacy, n_slots_legacy = (CombatPolicyNet._LEGACY_N_ALLY,
                                      CombatPolicyNet._LEGACY_N_SLOTS)
    cw = sd["critic.net.0.weight"]
    tail = 4 + END_FEATURES_DIM + _SKILL_DIM_V1
    legacy_cw = torch.zeros(cw.shape[0], n_slots_legacy * old_dim + tail)
    for s_old in range(n_slots_legacy):
        s_new = s_old if s_old <= n_ally_legacy else ENEMY_SLOT_START + (s_old - n_ally_legacy - 1)
        legacy_cw[:, s_old * old_dim:(s_old + 1) * old_dim] = \
            cw[:, s_new * ENTITY_DIM: s_new * ENTITY_DIM + old_dim]
    legacy_cw[:, n_slots_legacy * old_dim:] = cw[:, N_ENTITY_SLOTS * ENTITY_DIM:]
    legacy_sd["critic.net.0.weight"] = legacy_cw
    return legacy_sd


def test_obs_v3_state_dict_migration_preserves_behaviour():
    """A migrated pre-v3 checkpoint must (a) load strict through the FULL
    obs chain (v3 then v4), (b) ignore every post-v2 feature column —
    outputs identical whether the v3 tail + v4 descriptor are zero or
    arbitrary garbage on the same legacy-visible state."""
    from trpg.rl.model import _ENTITY_DIM_V2
    torch.manual_seed(0)
    donor = CombatPolicyNet()
    legacy_sd = _shrink_to_legacy_sd(donor.state_dict())

    adapted = CombatPolicyNet.adapt_state_dict_for_obs(legacy_sd)
    net = CombatPolicyNet()
    net.load_state_dict(adapted)   # strict — every key/shape must match
    net.eval()

    from trpg.rl.obs import ENEMY_SLOT_START
    obs_a = _fake_obs_batch(batch=2)
    # legacy-visible state: self row with position + archetype one-hot,
    # plus one present enemy row
    obs_a["entities"][:, 0, 1] = 0.4
    obs_a["entities"][:, 0, 2] = 0.6
    obs_a["entities"][:, 0, 7] = 1.0
    obs_a["entities"][:, ENEMY_SLOT_START, 0] = 1.0
    obs_a["entities"][:, ENEMY_SLOT_START, 4] = 1.0
    obs_a["entities"][:, ENEMY_SLOT_START, 5] = 1.0
    obs_a["skills"] = torch.randn_like(obs_a["skills"])
    obs_b = {k: v.clone() for k, v in obs_a.items()}
    # garbage in the v3 tail + v4 descriptor of PRESENT rows — a migrated
    # pre-v3 net must not react. (Absent rows stay all-zero: a row whose only
    # non-zero content is post-v2 columns cannot occur in real obs, and making
    # one up would legitimately flip the attention presence mask.)
    for row in (0, ENEMY_SLOT_START):
        obs_b["entities"][:, row, _ENTITY_DIM_V2:] = \
            torch.randn(2, ENTITY_DIM - _ENTITY_DIM_V2)

    with torch.no_grad():
        out_a = net(obs_a)
        out_b = net(obs_b)
        v_a = net.value(obs_a)
        v_b = net.value(obs_b)
    for ta, tb in zip(out_a, out_b):
        assert torch.equal(ta, tb)
    assert torch.equal(v_a, v_b)


def test_policy_invariant_to_padding_slot_count():
    """Policy outputs must not depend on how many EMPTY slots the obs carries.

    This is the invariance that broke during the 6→10 slot widening: unmasked
    skill→entity attention let padding rows (bias embeddings) leak into the
    context, scaling with slot count. Forward the same present entities under
    the full slot layout and under a monkeypatched 6-slot layout — end/skill
    logits and the shared entity-logit rows must be bitwise identical.
    """
    import trpg.rl.model as M
    from trpg.rl.obs import ENEMY_SLOT_START
    torch.manual_seed(1)
    net = CombatPolicyNet()
    net.eval()

    obs = _fake_obs_batch(batch=2)
    obs["skills"] = torch.randn_like(obs["skills"])
    # present: self + one enemy; everything else padding
    obs["entities"][:, 0, :] = torch.randn(2, ENTITY_DIM)
    obs["entities"][:, 0, 1:3] = 0.5
    obs["entities"][:, 0, 6] = 1.0
    obs["entities"][:, 0, 7] = 1.0
    obs["entities"][:, ENEMY_SLOT_START, :] = torch.randn(2, ENTITY_DIM)

    legacy_rows = [0, 1, 2] + list(range(ENEMY_SLOT_START, ENEMY_SLOT_START + 3))
    obs6 = {k: v.clone() for k, v in obs.items()}
    obs6["entities"] = obs["entities"][:, legacy_rows, :]

    with torch.no_grad():
        end10, skill10, ent10, grid10 = net(obs)
        saved = M.N_ENTITY_SLOTS
        M.N_ENTITY_SLOTS = len(legacy_rows)
        try:
            end6, skill6, ent6, grid6 = net(obs6)
        finally:
            M.N_ENTITY_SLOTS = saved

    assert torch.equal(end10, end6)
    assert torch.equal(skill10, skill6)
    assert torch.equal(grid10, grid6)
    assert torch.equal(ent10[:, :, legacy_rows], ent6)


def test_migrated_checkpoint_attention_bitexact_vs_legacy():
    """Migration must set attn_legacy=1 and reproduce the PRE-v3 attention
    bit-exactly: forward under the 10-slot layout (legacy-row restricted
    softmax) must equal the genuine pre-migration computation — a 6-slot
    view with the original unmasked softmax."""
    import trpg.rl.model as M
    torch.manual_seed(2)
    donor = CombatPolicyNet()
    legacy_sd = _shrink_to_legacy_sd(donor.state_dict())
    legacy_sd.pop("attn_legacy", None)     # real old checkpoints predate the buffer

    adapted = CombatPolicyNet.adapt_state_dict_for_obs(legacy_sd)
    assert float(adapted["attn_legacy"]) == 1.0
    net = CombatPolicyNet()
    net.load_state_dict(adapted)
    net.eval()

    from trpg.rl.obs import ENEMY_SLOT_START
    obs = _fake_obs_batch(batch=2)
    obs["skills"] = torch.randn_like(obs["skills"])
    obs["entities"][:, 0, :] = torch.randn(2, ENTITY_DIM)
    obs["entities"][:, 0, 1:3] = 0.5
    obs["entities"][:, 0, 6] = 1.0
    obs["entities"][:, 0, 7] = 1.0
    obs["entities"][:, ENEMY_SLOT_START, :] = torch.randn(2, ENTITY_DIM)

    legacy_rows = M._LEGACY_ATTN_ROWS
    obs6 = {k: v.clone() for k, v in obs.items()}
    obs6["entities"] = obs["entities"][:, legacy_rows, :]

    with torch.no_grad():
        out10 = net(obs)
        net.legacy_unmasked_attn = True
        saved = M.N_ENTITY_SLOTS
        M.N_ENTITY_SLOTS = len(legacy_rows)
        try:
            out6 = net(obs6)
        finally:
            M.N_ENTITY_SLOTS = saved
            net.legacy_unmasked_attn = False

    end10, skill10, ent10, grid10 = out10
    end6, skill6, ent6, grid6 = out6
    assert torch.equal(end10, end6)
    assert torch.equal(skill10, skill6)
    assert torch.equal(grid10, grid6)
    assert torch.equal(ent10[:, :, legacy_rows], ent6)


def test_obs_v3_migration_noop_on_current_checkpoints():
    """A state dict already at current dims passes through both adapters."""
    net = CombatPolicyNet()
    sd = net.state_dict()
    adapted = CombatPolicyNet.adapt_state_dict_for_obs(dict(sd))
    assert torch.equal(adapted["entity_mlp.0.weight"], sd["entity_mlp.0.weight"])
    assert torch.equal(adapted["critic.net.0.weight"], sd["critic.net.0.weight"])


def _shrink_to_v3_sd(sd: dict) -> dict:
    """Build a v3-era state dict from a current (v4) one (test fixture).

    Inverse of adapt_state_dict_for_obs_v4: drop the descriptor columns and
    UN-scale the level/max_hp columns (the adapter scales them back up).
    """
    from trpg.rl.model import _ENTITY_DIM_V3, _SKILL_DIM_V1
    from trpg.rl.obs import (I_ENT_LEVEL, I_ENT_MAXHP, N_ENTITY_SLOTS,
                             END_FEATURES_DIM, LEVEL_NORM, MAXHP_NORM,
                             V3_LEVEL_NORM, V3_MAXHP_NORM)
    lvl_s = LEVEL_NORM / V3_LEVEL_NORM
    hp_s = MAXHP_NORM / V3_MAXHP_NORM
    sd = _shrink_skill_dtype_sd(sd)
    v3_sd = dict(sd)
    w = sd["entity_mlp.0.weight"][:, :_ENTITY_DIM_V3].clone()
    w[:, I_ENT_LEVEL] /= lvl_s
    w[:, I_ENT_MAXHP] /= hp_s
    v3_sd["entity_mlp.0.weight"] = w
    cw = sd["critic.net.0.weight"]
    tail = 4 + END_FEATURES_DIM + _SKILL_DIM_V1
    v3_cw = torch.zeros(cw.shape[0], N_ENTITY_SLOTS * _ENTITY_DIM_V3 + tail)
    for s in range(N_ENTITY_SLOTS):
        blk = cw[:, s * ENTITY_DIM: s * ENTITY_DIM + _ENTITY_DIM_V3].clone()
        blk[:, I_ENT_LEVEL] /= lvl_s
        blk[:, I_ENT_MAXHP] /= hp_s
        v3_cw[:, s * _ENTITY_DIM_V3:(s + 1) * _ENTITY_DIM_V3] = blk
    v3_cw[:, N_ENTITY_SLOTS * _ENTITY_DIM_V3:] = cw[:, N_ENTITY_SLOTS * ENTITY_DIM:]
    v3_sd["critic.net.0.weight"] = v3_cw
    return v3_sd


def test_obs_v4_migration_ignores_descriptor_garbage():
    """A v4-migrated v3-era checkpoint must load strict and be invariant to
    the descriptor columns (its descriptor input weights are zero)."""
    from trpg.rl.obs import ENT_DESC_START, N_V4_DESC, ENEMY_SLOT_START
    torch.manual_seed(4)
    donor = CombatPolicyNet()
    v3_sd = _shrink_to_v3_sd(donor.state_dict())

    adapted = CombatPolicyNet.adapt_state_dict_for_obs(v3_sd)
    net = CombatPolicyNet()
    net.load_state_dict(adapted)   # strict
    net.eval()

    obs_a = _fake_obs_batch(batch=2)
    obs_a["skills"] = torch.randn_like(obs_a["skills"])
    obs_a["entities"][:, 0, :] = torch.randn(2, ENTITY_DIM)
    obs_a["entities"][:, ENEMY_SLOT_START, :] = torch.randn(2, ENTITY_DIM)
    obs_b = {k: v.clone() for k, v in obs_a.items()}
    # scribble garbage across the whole descriptor tail (v4 capability descriptor
    # + v5 passive-trait columns) — a v3-era checkpoint migrated through the full
    # chain has zero input weights on ALL of them, so it must stay invariant.
    for row in (0, ENEMY_SLOT_START):
        obs_b["entities"][:, row, ENT_DESC_START:] = \
            torch.randn(2, ENTITY_DIM - ENT_DESC_START)

    with torch.no_grad():
        out_a = net(obs_a)
        out_b = net(obs_b)
        v_a = net.value(obs_a)
        v_b = net.value(obs_b)
    for ta, tb in zip(out_a, out_b):
        assert torch.equal(ta, tb)
    assert torch.equal(v_a, v_b)


def test_decision_ctx_migration_ignores_context():
    """A checkpoint migrated through the decision-context step must (a) load
    strict and (b) be EXACTLY invariant to the decision_context vector — its
    trunk / critic dctx columns are zero-padded, so an arbitrary context vector
    has no effect until those columns train. This is the guarantee that off-turn
    reaction/legendary obs never perturbs a pre-wave checkpoint's turn play."""
    from trpg.rl.obs import N_DECISION_CTX, ENEMY_SLOT_START
    torch.manual_seed(7)
    donor = CombatPolicyNet()
    pre_sd = _shrink_decision_ctx_sd(donor.state_dict())   # peel dctx columns
    adapted = CombatPolicyNet.adapt_state_dict_for_decision_ctx(pre_sd)  # re-pad 0
    net = CombatPolicyNet()
    net.load_state_dict(adapted)   # strict — shapes round-trip
    net.eval()

    obs_a = _fake_obs_batch(batch=2)
    obs_a["skills"] = torch.randn_like(obs_a["skills"])
    obs_a["entities"][:, 0, :] = torch.randn(2, ENTITY_DIM)
    obs_a["entities"][:, ENEMY_SLOT_START, :] = torch.randn(2, ENTITY_DIM)
    obs_b = {k: v.clone() for k, v in obs_a.items()}
    # arbitrary reaction/legendary context — must not move any output
    obs_b["decision_context"] = torch.randn(2, N_DECISION_CTX)

    with torch.no_grad():
        out_a, out_b = net(obs_a), net(obs_b)
        v_a, v_b = net.value(obs_a), net.value(obs_b)
    for ta, tb in zip(out_a, out_b):
        assert torch.equal(ta, tb)
    assert torch.equal(v_a, v_b)


def test_obs_v4_rescale_compensation_exact():
    """The NORM rescale (LEVEL ×2, MAXHP ×8 — powers of two) must be EXACTLY
    compensated by the scaled weight columns. Exactness holds at the
    per-product level (bitwise: power-of-2 scaling is an fp exponent shift,
    rounding commutes); the matmul OUTPUT differs only by BLAS summation
    order across the different row widths (measured ≤1e-6 — behaviour gate
    is diag_obsv4_shift's action-agreement run)."""
    from trpg.rl.model import _ENTITY_DIM_V3, _ENTITY_DIM_V4
    from trpg.rl.obs import (I_ENT_LEVEL, I_ENT_MAXHP, LEVEL_NORM, MAXHP_NORM,
                             V3_LEVEL_NORM, V3_MAXHP_NORM)
    torch.manual_seed(5)
    w3 = torch.randn(64, _ENTITY_DIM_V3)
    # v4-only migration produces the v4 width (descriptor appended); the v5
    # trait tail is a separate later step, so test the v4 step at its own width.
    w4 = CombatPolicyNet.adapt_state_dict_for_obs_v4(
        {"entity_mlp.0.weight": w3.clone()})["entity_mlp.0.weight"]
    assert w4.shape == (64, _ENTITY_DIM_V4)

    raw_lvl = torch.tensor([1.0, 3.0, 5.0, 12.0, 30.0])
    raw_hp = torch.tensor([7.0, 59.0, 105.0, 546.0, 676.0])
    x4 = torch.randn(5, _ENTITY_DIM_V4)
    x4[:, I_ENT_LEVEL] = raw_lvl / LEVEL_NORM
    x4[:, I_ENT_MAXHP] = raw_hp / MAXHP_NORM
    x3 = x4[:, :_ENTITY_DIM_V3].clone()
    x3[:, I_ENT_LEVEL] = raw_lvl / V3_LEVEL_NORM
    x3[:, I_ENT_MAXHP] = raw_hp / V3_MAXHP_NORM

    # 1. per-element products bitwise identical over the shared columns
    prod4 = (x4.unsqueeze(1) * w4.unsqueeze(0))
    prod3 = (x3.unsqueeze(1) * w3.unsqueeze(0))
    assert torch.equal(prod4[:, :, :_ENTITY_DIM_V3], prod3)
    # 2. descriptor columns contribute exactly zero
    assert float(prod4[:, :, _ENTITY_DIM_V3:].abs().max()) == 0.0
    # 3. matmul outputs equal up to summation-order drift
    assert torch.allclose(x4 @ w4.T, x3 @ w3.T, atol=2e-6, rtol=0)
