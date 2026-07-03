import numpy as np
import torch
from trpg.rl.model import CombatPolicyNet
from trpg.rl.obs import N_SKILL_SLOTS, N_ENTITY_SLOTS, ENTITY_DIM, N_GRID
from trpg.rl.obs import (N_ENTITY_GRID_CHANNELS, N_DISTANCE_GRID_CHANNELS,
                          N_LOS_GRID_CHANNELS, N_REACH_GRID_CHANNELS,
                          N_THREAT_GRID_CHANNELS,
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
        "los_grid":         torch.zeros(batch, N_LOS_GRID_CHANNELS, N_GRID, N_GRID),
        "reach_grid":       torch.zeros(batch, N_REACH_GRID_CHANNELS, N_GRID, N_GRID),
        "threat_grid":      torch.zeros(batch, N_THREAT_GRID_CHANNELS, N_GRID, N_GRID),
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
    + the typed-matchup join scalar(1) + the blocked-skill feature
    (blockedness × sk_emb, 64) + the v6 condition-immunity join scalar(1).
    entity_heads[i]: sk(64)+h+ent(32) + typed-join(1) + cimmun-join(1)."""
    net = CombatPolicyNet()
    hidden = net.trunk[0].out_features
    assert net.skill_heads[0].in_features == 64 + hidden + 32 + 1 + 64 + 1, (
        f"Expected {64+hidden+32+1+64+1}, got {net.skill_heads[0].in_features}"
    )
    assert net.entity_heads[0].in_features == 64 + hidden + 32 + 1 + 1, (
        f"Expected {64+hidden+32+1+1}, got {net.entity_heads[0].in_features}"
    )
    # Per-arch sanity: should have N_ARCHETYPES copies
    from trpg.rl.obs import N_ARCHETYPES
    assert len(net.skill_heads) == N_ARCHETYPES


def _shrink_cimmun_heads_sd(sd: dict) -> dict:
    """Strip the condition-immunity head-join era (inverse of
    adapt_state_dict_for_cimmun_heads). It is the OUTERMOST load step, so a
    pre-cimmun fixture peels it FIRST: one trailing column off every
    skill_heads.*.weight (appended after blocked_feat) and every
    entity_heads.*.weight (appended after the typed join)."""
    import re
    out = dict(sd)
    gq = None
    for k, v in out.items():
        if k.startswith("grid_query_projs.") and k.endswith(".weight") and v.dim() == 2:
            gq = v.shape[1]
            break
    if gq is None:
        return out
    skill_in = gq + 32 + 1 + 64 + 1     # sk_emb in gq; +ctx +typed +blocked +cimmun
    entity_in = gq + 32 + 1 + 1         # sk_emb in gq; +ctx +typed +cimmun
    sp = re.compile(r"^skill_head(s\.\d+)?\.weight$")
    ep = re.compile(r"^entity_head(s\.\d+)?\.weight$")
    for k in list(out):
        if sp.match(k) and out[k].dim() == 2 and out[k].shape[1] == skill_in:
            out[k] = out[k][:, :-1].clone()
        elif ep.match(k) and out[k].dim() == 2 and out[k].shape[1] == entity_in:
            out[k] = out[k][:, :-1].clone()
    return out


def _shrink_blocked_skill_sd(sd: dict) -> dict:
    """Strip the blocked-skill era (inverse of adapt_state_dict_for_blocked_skill).

    blocked-skill is the second-to-last step of the load chain (after los-grid,
    before cimmun-heads). Peel cimmun-heads FIRST, then this. It appended 64
    columns (blockedness × sk_emb) to the END of every skill_heads.*.weight
    (before the cimmun column, so with cimmun already peeled it is the tail)."""
    import re
    out = _shrink_cimmun_heads_sd(sd)   # peel the outermost cimmun head col FIRST
    gq = None
    for k, v in out.items():
        if k.startswith("grid_query_projs.") and k.endswith(".weight") and v.dim() == 2:
            gq = v.shape[1]
            break
    if gq is None:
        return out
    new_skill_in = gq + 32 + 1 + 64
    pat = re.compile(r"^skill_head(s\.\d+)?\.weight$")
    for k in list(out):
        if pat.match(k) and out[k].dim() == 2 and out[k].shape[1] == new_skill_in:
            out[k] = out[k][:, :-64].clone()
    return out


def _shrink_threat_grid_sd(sd: dict) -> dict:
    """Strip the threat-grid era (inverse of adapt_state_dict_for_threat_grid).

    threat-grid is appended AFTER reach (load chain: ...reach ← threat ←
    blocked). Peel blocked-skill FIRST, then this. grid_query OUTPUT −N_THREAT
    rows; trunk self_cell_feat −N_THREAT cols at index 64+32+(post-reach
    width)."""
    import re
    from trpg.rl.obs import N_THREAT_GRID_CHANNELS
    from trpg.rl.model import _SPATIAL_C
    out = _shrink_blocked_skill_sd(sd)
    old_spatial = _SPATIAL_C - N_THREAT_GRID_CHANNELS    # post-reach width
    ins = 64 + 32 + old_spatial
    tw = out.get("trunk.0.weight")
    if tw is not None and tw.shape[1] >= ins + N_THREAT_GRID_CHANNELS:
        out["trunk.0.weight"] = torch.cat(
            [tw[:, :ins], tw[:, ins + N_THREAT_GRID_CHANNELS:]], dim=1).clone()
    gq = re.compile(r"^grid_query_proj(s\.\d+)?\.(weight|bias)$")
    for k in list(out):
        if gq.match(k):
            out[k] = out[k][:-N_THREAT_GRID_CHANNELS].clone()
    return out


def _shrink_reach_grid_sd(sd: dict) -> dict:
    """Strip the reach-grid era (inverse of adapt_state_dict_for_reach_grid).

    reach-grid is appended AFTER los (load chain: ...los ← reach ← threat ←
    blocked). Peel threat-grid FIRST (which peels blocked-skill), then this.
    old spatial width subtracts BOTH reach and threat (channels added after the
    pre-reach era)."""
    import re
    from trpg.rl.obs import N_REACH_GRID_CHANNELS, N_THREAT_GRID_CHANNELS
    from trpg.rl.model import _SPATIAL_C
    out = _shrink_threat_grid_sd(sd)
    old_spatial = _SPATIAL_C - N_REACH_GRID_CHANNELS - N_THREAT_GRID_CHANNELS
    ins = 64 + 32 + old_spatial
    tw = out.get("trunk.0.weight")
    if tw is not None and tw.shape[1] >= ins + N_REACH_GRID_CHANNELS:
        out["trunk.0.weight"] = torch.cat(
            [tw[:, :ins], tw[:, ins + N_REACH_GRID_CHANNELS:]], dim=1).clone()
    gq = re.compile(r"^grid_query_proj(s\.\d+)?\.(weight|bias)$")
    for k in list(out):
        if gq.match(k):
            out[k] = out[k][:-N_REACH_GRID_CHANNELS].clone()
    return out


def _shrink_los_grid_sd(sd: dict) -> dict:
    """Strip the los-grid era (inverse of adapt_state_dict_for_los_grid).

    Peels reach-grid FIRST (which peels threat then blocked-skill), then strips
    los. old spatial width subtracts los, reach AND threat (the channels added
    after the pre-los era)."""
    import re
    from trpg.rl.obs import (N_LOS_GRID_CHANNELS, N_REACH_GRID_CHANNELS,
                             N_THREAT_GRID_CHANNELS)
    from trpg.rl.model import _SPATIAL_C
    out = _shrink_reach_grid_sd(sd)
    old_spatial = (_SPATIAL_C - N_LOS_GRID_CHANNELS - N_REACH_GRID_CHANNELS
                   - N_THREAT_GRID_CHANNELS)
    ins = 64 + 32 + old_spatial
    tw = out.get("trunk.0.weight")
    if tw is not None and tw.shape[1] >= ins + N_LOS_GRID_CHANNELS:
        out["trunk.0.weight"] = torch.cat(
            [tw[:, :ins], tw[:, ins + N_LOS_GRID_CHANNELS:]], dim=1).clone()
    gq = re.compile(r"^grid_query_proj(s\.\d+)?\.(weight|bias)$")
    for k in list(out):
        if gq.match(k):
            out[k] = out[k][:-N_LOS_GRID_CHANNELS].clone()
    return out


def _shrink_decision_ctx_sd(sd: dict) -> dict:
    """Strip the decision-context era (inverse of adapt_state_dict_for_decision_ctx).

    decision_ctx grew trunk.0.weight and critic.net.0.weight by N_DECISION_CTX
    columns at the END. It sits just before los-grid in the load chain, so the
    shrink peels los-grid FIRST (here) then the dctx columns. After this the
    rest of the shrink chain sees the pre-dctx widths it was written against."""
    from trpg.rl.obs import N_DECISION_CTX
    out = _shrink_los_grid_sd(sd)
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
    pre_sd = _shrink_decision_ctx_sd(donor.state_dict())   # peel los + dctx
    adapted = CombatPolicyNet.adapt_state_dict_for_obs(pre_sd)  # re-pad both 0
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


# ── 滿血遮純治療技（null-effect 遮罩統一原則；heal_full 家族） ────────────────────

def _heal_world():
    from trpg.engine.character import Character, Stats
    from trpg.engine.world_state import WorldState, CombatState
    from trpg.engine.combat import setup_combat_positions
    from trpg.engine.items import WEAPON_DEFS
    a = Character(name="a", race="人類", class_="戰士", level=3,
                  stats=Stats(STR=14, DEX=12, CON=14),
                  hp=24, max_hp=24, ac=14,
                  weapons=[WEAPON_DEFS["長劍"]], is_npc=False)
    a.known_abilities = ["second_wind"]
    b = Character(name="b", race="哥布林", class_="戰士", level=1,
                  stats=Stats(STR=10, DEX=12),
                  hp=10, max_hp=10, ac=12,
                  weapons=[WEAPON_DEFS["短劍"]], is_npc=True, attitude=0)
    ws = WorldState(characters={"a": a, "b": b}, scene="test",
                    pc_ids=["a"], party_ids=["a"])
    ws.combat = CombatState(active=True, initiative_order=["a", "b"],
                            round_number=1)
    setup_combat_positions(ws, ws.combat)
    return ws


def _pure_heal_idx(ws, cid):
    from trpg.engine.skill import available_skills
    sks = available_skills(ws.characters[cid], ws)
    for i, s in enumerate(sks):
        f = s.features
        if (f.expected_healing > 0 and f.expected_damage <= 0
                and not any(f.applies_status)):
            return i
    raise AssertionError("no pure heal in kit")


def test_resource_mask_blocks_pure_heal_at_full_hp():
    from trpg.rl.model import apply_resource_mask
    from trpg.rl.action import ACTION_DIMS
    ws = _heal_world()
    i = _pure_heal_idx(ws, "a")
    res = {"movement": 6.0, "action": 1, "bonus_action": 1}
    logits = torch.zeros(1, ACTION_DIMS[0])
    out = apply_resource_mask(logits, res, ws, "a")
    assert out[0, i].item() <= -1e8          # 滿血 → 純治療＝null-effect，遮
    ws.characters["a"].hp = 10               # 受傷 → 開
    out2 = apply_resource_mask(logits, res, ws, "a")
    assert out2[0, i].item() > -1e8


# ── AoE 落點支配性剪枝（teamff/aoe_ally 家族；幾何真值、零技能名） ────────────────

def _aoe_world():
    from trpg.engine.character import Character, Stats
    from trpg.engine.world_state import WorldState, CombatState
    from trpg.engine.combat import setup_combat_positions
    from trpg.engine.vec2 import Vec2
    mage = Character(name="m", race="人類", class_="法師", level=5,
                     stats=Stats(STR=8, DEX=14, CON=12, INT=16),
                     hp=20, max_hp=20, ac=12, is_npc=False,
                     spell_slots={3: 2}, spellcasting_ability="INT")
    mage.known_abilities = ["fireball_ev"]
    ally = Character(name="al", race="人類", class_="戰士", level=5,
                     stats=Stats(STR=16, DEX=12, CON=14),
                     hp=40, max_hp=40, ac=16, is_npc=False)
    e1 = Character(name="e1", race="獸人", class_="—", level=3,
                   stats=Stats(STR=14, DEX=10), hp=15, max_hp=15, ac=12,
                   is_npc=True, attitude=0)
    e2 = Character(name="e2", race="獸人", class_="—", level=3,
                   stats=Stats(STR=14, DEX=10), hp=15, max_hp=15, ac=12,
                   is_npc=True, attitude=0)
    ws = WorldState(characters={"m": mage, "al": ally, "e1": e1, "e2": e2},
                    scene="", pc_ids=["m", "al"], party_ids=["m", "al"])
    ws.combat = CombatState(active=True,
                            initiative_order=["m", "al", "e1", "e2"],
                            round_number=1)
    setup_combat_positions(ws, ws.combat)
    # 幾何：ally 貼著 e1（炸 e1 必炸 ally）；e2 單獨在遠處（乾淨落點、同敵覆蓋 1）
    mage.position = Vec2(2.0, 2.0)
    ally.position = Vec2(10.0, 10.0)
    e1.position = Vec2(11.0, 10.0)
    e2.position = Vec2(22.0, 22.0)
    return ws


def test_pick_action_aoe_relocates_off_ally_when_clean_cell_exists():
    import torch
    from trpg.rl.model import pick_action
    from trpg.rl.action import ACTION_DIMS
    from trpg.engine.skill import available_skills, TargetType
    from trpg.rl.obs import N_GRID, GRID_CELL_SIZE_M
    ws = _aoe_world()
    sks = available_skills(ws.characters["m"], ws)
    fb = next(i for i, s in enumerate(sks)
              if (getattr(s.features, "aoe_radius_m", 0) or 0) > 0
              and s.features.target_type == TargetType.POINT)
    rad = float(sks[fb].features.aoe_radius_m)
    # 造 logits：偏好 e1 的格（炸 ally），乾淨替代=e2 的格存在且敵覆蓋相同(1)
    end = torch.tensor(-50.0)
    sl = torch.full((ACTION_DIMS[0],), -20.0); sl[fb] = 10.0
    el = torch.zeros(ACTION_DIMS[0], ACTION_DIMS[1])
    gx1, gy1 = int(11.0 / GRID_CELL_SIZE_M), int(10.0 / GRID_CELL_SIZE_M)
    gl = torch.zeros(ACTION_DIMS[0], N_GRID * N_GRID)
    gl[fb, gx1 * N_GRID + gy1] = 10.0            # 最愛=炸 e1+ally 的格
    act = pick_action(end, sl, el, gl, ws=ws, agent_id="m")
    assert act[0] == fb
    tx = (act[2] // N_GRID + 0.5) * GRID_CELL_SIZE_M
    ty = (act[2] % N_GRID + 0.5) * GRID_CELL_SIZE_M
    al = ws.characters["al"].position
    assert ((al.x - tx) ** 2 + (al.y - ty) ** 2) ** 0.5 > rad, \
        "存在同敵覆蓋的乾淨落點時不該選炸隊友的格"
    # 至少覆蓋一個敵人（不是逃到空角落）
    hit_enemy = any(((c.position.x - tx) ** 2 + (c.position.y - ty) ** 2) ** 0.5 <= rad
                    for c in (ws.characters["e1"], ws.characters["e2"]))
    assert hit_enemy


def test_pick_action_aoe_scrum_not_blocked():
    # 肉搏團（唯一敵人貼著隊友、無乾淨格）→ 不遮，照炸
    import torch
    from trpg.rl.model import pick_action
    from trpg.rl.action import ACTION_DIMS
    from trpg.engine.skill import available_skills, TargetType
    from trpg.rl.obs import N_GRID, GRID_CELL_SIZE_M
    ws = _aoe_world()
    ws.characters["e2"].hp = 0   # 只剩 e1（貼著 ally）
    sks = available_skills(ws.characters["m"], ws)
    fb = next(i for i, s in enumerate(sks)
              if (getattr(s.features, "aoe_radius_m", 0) or 0) > 0
              and s.features.target_type == TargetType.POINT)
    rad = float(sks[fb].features.aoe_radius_m)
    end = torch.tensor(-50.0)
    sl = torch.full((ACTION_DIMS[0],), -20.0); sl[fb] = 10.0
    el = torch.zeros(ACTION_DIMS[0], ACTION_DIMS[1])
    gx1, gy1 = int(11.0 / GRID_CELL_SIZE_M), int(10.0 / GRID_CELL_SIZE_M)
    gl = torch.zeros(ACTION_DIMS[0], N_GRID * N_GRID)
    gl[fb, gx1 * N_GRID + gy1] = 10.0
    act = pick_action(end, sl, el, gl, ws=ws, agent_id="m")
    tx = (act[2] // N_GRID + 0.5) * GRID_CELL_SIZE_M
    ty = (act[2] % N_GRID + 0.5) * GRID_CELL_SIZE_M
    e1 = ws.characters["e1"].position
    assert ((e1.x - tx) ** 2 + (e1.y - ty) ** 2) ** 0.5 <= rad, \
        "無乾淨替代時（肉搏團）不該阻止對敵團開火"


# ── 重複 dodge 零壓力遮罩（防禦惰性家族；引擎實證 null-effect） ──────────────────

def test_resource_mask_blocks_repeat_dodge_without_pressure():
    import torch
    from trpg.rl.model import apply_resource_mask
    from trpg.rl.action import ACTION_DIMS
    from trpg.engine.skill import available_skills
    from trpg.engine.combat import execute_action, resolve_attack
    from trpg.engine.vec2 import Vec2
    ws = _heal_world()
    # 敵放 reach 外：隔離「零壓力重複」語義（reach 內的零出手連續 dodge
    # 屬純龜縮守門，另測）
    ws.characters["a"].position = Vec2(10.0, 10.0)
    ws.characters["b"].position = Vec2(20.0, 10.0)   # reach+mv(6m) 之外
    a = ws.characters["a"]
    sks = available_skills(a, ws)
    di = next(i for i, s in enumerate(sks) if s.skill_id == "dodge")
    res = {"movement": 6.0, "action": 1, "bonus_action": 1}
    logits = torch.zeros(1, ACTION_DIMS[0])
    # 第 1 輪 dodge；第 2 輪期間零來襲 → 重複 dodge＝實證無效 → 遮
    ws.combat.round_number = 1
    execute_action({"type": "DODGE", "character": "a", "skill_id": "dodge"}, ws)
    ws.combat.round_number = 2
    out = apply_resource_mask(logits, res, ws, "a")
    assert out[0, di].item() <= -1e8, "連續 dodge 且零來襲該遮"
    # 期間有攻擊嘗試（含 miss）＝dodge 有效（壓力存在）→ 開
    resolve_attack(ws.characters["b"], a, ws.characters["b"].get_weapon())
    out2 = apply_resource_mask(logits, res, ws, "a")
    assert out2[0, di].item() > -1e8, "有來襲壓力時 dodge 是合法戰術"
    # 非連續輪（隔了一輪沒 dodge）→ 開（首次 dodge 永遠合法）
    ws.combat.round_number = 5
    out3 = apply_resource_mask(logits, res, ws, "a")
    assert out3[0, di].item() > -1e8


# ── 免疫空砸技能遮罩（純傷害技 vs 全免疫目標＝null；有 EV>0 替代才遮） ─────────────

def _smite_world(grant_smite=True):
    from trpg.engine.character import Character, Stats
    from trpg.engine.world_state import WorldState, CombatState
    from trpg.engine.combat import setup_combat_positions
    from trpg.engine.items import WEAPON_DEFS
    from trpg.engine.vec2 import Vec2
    a = Character(name="a", race="人類", class_="聖騎士", level=6,
                  stats=Stats(STR=16, DEX=10, CON=14, CHA=14),
                  hp=40, max_hp=40, ac=16, is_npc=False,
                  weapons=[WEAPON_DEFS["長劍"]],
                  spell_slots={1: 2}, spellcasting_ability="CHA")
    if grant_smite:
        a.known_abilities = ["divine_smite_ven"]
    b = Character(name="b", race="獸人", class_="—", level=3,
                  stats=Stats(STR=14, DEX=10), hp=20, max_hp=20, ac=12,
                  is_npc=True, attitude=0,
                  damage_multipliers={"斬擊": 0.0})
    ws = WorldState(characters={"a": a, "b": b}, scene="",
                    pc_ids=["a"], party_ids=["a"])
    ws.combat = CombatState(active=True, initiative_order=["a", "b"],
                            round_number=1)
    setup_combat_positions(ws, ws.combat)
    a.position = Vec2(10.0, 10.0)
    b.position = Vec2(11.0, 10.0)
    return ws


def test_resource_mask_blocks_null_attack_into_immunity_with_alternative():
    import torch
    from trpg.rl.model import apply_resource_mask
    from trpg.rl.action import ACTION_DIMS
    from trpg.engine.skill import available_skills
    ws = _smite_world(grant_smite=True)
    sks = available_skills(ws.characters["a"], ws)
    sw = next(i for i, s in enumerate(sks) if s.skill_id.startswith("weapon:"))
    sm = next(i for i, s in enumerate(sks) if s.skill_id == "divine_smite_ven")
    res = {"movement": 6.0, "action": 1, "bonus_action": 1}
    logits = torch.zeros(1, ACTION_DIMS[0])
    out = apply_resource_mask(logits, res, ws, "a")
    assert out[0, sw].item() <= -1e8, "斬擊全免疫且有 smite 替代 → 劍該遮"
    assert out[0, sm].item() > -1e8, "smite（光耀）是正解，不該遮"


def test_resource_mask_keeps_null_attack_when_no_alternative():
    import torch
    from trpg.rl.model import apply_resource_mask
    from trpg.rl.action import ACTION_DIMS
    from trpg.engine.skill import available_skills
    ws = _smite_world(grant_smite=False)     # 單型：無替代 → 不遮（不可贏局）
    sks = available_skills(ws.characters["a"], ws)
    sw = next(i for i, s in enumerate(sks) if s.skill_id.startswith("weapon:"))
    res = {"movement": 6.0, "action": 1, "bonus_action": 1}
    logits = torch.zeros(1, ACTION_DIMS[0])
    out = apply_resource_mask(logits, res, ws, "a")
    assert out[0, sw].item() > -1e8, "無替代時不遮（避免製造零合法動作）"


# ── 目標狀態前置門控（swallow 需 restrained 目標；無合法目標＝絕對非法該遮） ────────

def _swallow_world(restrained=False):
    from trpg.engine.character import Character, Stats
    from trpg.engine.world_state import WorldState, CombatState
    from trpg.engine.combat import setup_combat_positions
    from trpg.engine.items import WEAPON_DEFS
    from trpg.engine.vec2 import Vec2
    from trpg.engine.status import Restrained
    a = Character(name="a", race="人類", class_="戰士", level=5,
                  stats=Stats(STR=16, DEX=12, CON=14),
                  hp=50, max_hp=50, ac=16, is_npc=False,
                  weapons=[WEAPON_DEFS["長劍"]])
    a.known_abilities = ["tarrasque_swallow"]     # GUI 嫁接式：職業帶怪物能力
    b = Character(name="b", race="獸人", class_="—", level=5,
                  stats=Stats(STR=14, DEX=10), hp=40, max_hp=40, ac=14,
                  is_npc=True, attitude=0)
    ws = WorldState(characters={"a": a, "b": b}, scene="",
                    pc_ids=["a"], party_ids=["a"])
    ws.combat = CombatState(active=True, initiative_order=["a", "b"],
                            round_number=1)
    setup_combat_positions(ws, ws.combat)
    a.position = Vec2(10.0, 10.0)
    b.position = Vec2(11.0, 10.0)     # 1.0m：射程不成問題（swallow reach 3.0/劍 1.5）
    if restrained:
        b.add_status(Restrained())
    return ws


def test_resource_mask_blocks_swallow_when_no_valid_target():
    # 絕對非法（引擎 execute_action 必回 ERROR，combat.py rts 閘）：tarrasque_swallow
    # 需要 restrained 目標；唯一敵人沒被束縛→本回合對任何目標都不可能合法。
    # available_skills 列出它（角色原則上會這招）但看不到「目標狀態」→ 遮罩層必須
    # 補這條前置門控，否則神經策略選它、引擎 ERROR、對手席空轉（play_gui 抽查）。
    import torch
    from trpg.rl.model import apply_resource_mask
    from trpg.rl.action import ACTION_DIMS
    from trpg.engine.skill import available_skills
    ws = _swallow_world(restrained=False)
    sks = available_skills(ws.characters["a"], ws)
    sid = next(i for i, s in enumerate(sks) if s.skill_id == "tarrasque_swallow")
    res = {"movement": 6.0, "action": 1, "bonus_action": 1}
    out = apply_resource_mask(torch.zeros(1, ACTION_DIMS[0]), res, ws, "a")
    assert out[0, sid].item() <= -1e8, "無 restrained 目標→吞噬絕對非法→該遮"
    sw = next(i for i, s in enumerate(sks) if s.skill_id.startswith("weapon:"))
    assert out[0, sw].item() > -1e8, "長劍仍合法（不製造零合法攻擊）"


def test_resource_mask_keeps_swallow_when_target_restrained():
    # 目標已 restrained → 吞噬對牠合法 → 不該遮（1vN 由 entity 遮罩挑正確那隻）。
    import torch
    from trpg.rl.model import apply_resource_mask
    from trpg.rl.action import ACTION_DIMS
    from trpg.engine.skill import available_skills
    ws = _swallow_world(restrained=True)
    sks = available_skills(ws.characters["a"], ws)
    sid = next(i for i, s in enumerate(sks) if s.skill_id == "tarrasque_swallow")
    res = {"movement": 6.0, "action": 1, "bonus_action": 1}
    out = apply_resource_mask(torch.zeros(1, ACTION_DIMS[0]), res, ws, "a")
    assert out[0, sid].item() > -1e8, "目標已 restrained→吞噬合法→不該遮"


# ── 純龜縮守門（1v1 零出手 + 連續 dodge + 在 reach 內＝恆不可勝的重複） ─────────────

def test_resource_mask_blocks_pure_turtle_dodge():
    import torch
    from trpg.rl.model import apply_resource_mask
    from trpg.rl.action import ACTION_DIMS
    from trpg.engine.skill import available_skills
    from trpg.engine.combat import execute_action
    from trpg.engine.vec2 import Vec2
    ws = _heal_world()
    a = ws.characters["a"]; b = ws.characters["b"]
    a.position = Vec2(10.0, 10.0); b.position = Vec2(10.8, 10.0)  # 在 reach 內
    sks = available_skills(a, ws)
    di = next(i for i, s in enumerate(sks) if s.skill_id == "dodge")
    res = {"movement": 6.0, "action": 1, "bonus_action": 1}
    logits = torch.zeros(1, ACTION_DIMS[0])
    # 第 1 輪 dodge＋敵有攻（壓力存在→零壓力遮罩不觸）；第 2 輪零出手＋連續
    # ＋在 reach＋無隊友 → 純龜縮遮罩觸發
    ws.combat.round_number = 1
    execute_action({"type": "DODGE", "character": "a", "skill_id": "dodge"}, ws)
    from trpg.engine.combat import resolve_attack
    resolve_attack(b, a, b.get_weapon())          # 來襲壓力（零壓力遮罩不觸）
    ws.combat.round_number = 2
    out = apply_resource_mask(logits, res, ws, "a")
    assert out[0, di].item() <= -1e8, "1v1 零出手連續 dodge 在 reach 內該遮"
    # 出手一次（武器攻擊嘗試）→ 永久解鎖（坦克行為保留）
    sw = next(i for i, s in enumerate(sks) if s.skill_id.startswith("weapon:"))
    act = sks[sw].build_action("a", "b", (b.position.x, b.position.y))
    execute_action(act, ws)
    execute_action({"type": "DODGE", "character": "a", "skill_id": "dodge"}, ws)
    resolve_attack(b, a, b.get_weapon())      # 敵持續施壓（零壓力遮罩不觸）
    ws.combat.round_number = 3
    out2 = apply_resource_mask(logits, res, ws, "a")
    assert out2[0, di].item() > -1e8, "出手過後 dodge 是合法戰術"


def test_resource_mask_allows_attack_at_exact_reach_boundary():
    # 引擎語義（attack_range_check）：d > reach 才拒 ⇒ d == reach 合法。
    # 舊遮罩 >= reach-0.01 在 [1.49,1.5] 錯殺合法攻擊——0477 對手席永久卡死
    # 真因（定身敵不動＝距離恆 1.5、move 空轉、dodge/hide 被龜縮守門遮、
    # 只剩 END＝50 回合零嘗試）。遮罩必須引擎同源：> reach + 1e-6。
    import torch
    from trpg.rl.model import apply_resource_mask
    from trpg.rl.action import ACTION_DIMS
    from trpg.engine.skill import available_skills
    from trpg.engine.vec2 import Vec2
    ws = _heal_world()
    a = ws.characters["a"]; b = ws.characters["b"]
    a.position = Vec2(10.0, 10.0)
    b.position = Vec2(11.5, 10.0)                  # 恰好 reach=1.5
    sks = available_skills(a, ws)
    sw = next(i for i, s in enumerate(sks) if s.skill_id.startswith("weapon:"))
    res = {"movement": 0.0, "action": 1, "bonus_action": 1}
    logits = torch.zeros(1, ACTION_DIMS[0])
    out = apply_resource_mask(logits, res, ws, "a")
    assert out[0, sw].item() > -1e8, "d==reach 引擎合法，不該遮"
    b.position = Vec2(11.51, 10.0)                 # 超出 1cm → 引擎拒 → 遮
    out2 = apply_resource_mask(logits, res, ws, "a")
    assert out2[0, sw].item() <= -1e8


# ── disengage 零移動/零輸出遮罩（null-effect＋engage-commitment 家族） ────────────

def test_resource_mask_blocks_disengage_without_movement():
    # miner mine|666|0456：繞圈 move 吃光預算 → mv=0 放 disengage → END 迴圈。
    # disengage 唯一效果＝讓「之後的移動」不引發藉機攻擊；movement=0 ⇒ 嚴格 null。
    import torch
    from trpg.rl.model import apply_resource_mask
    from trpg.rl.action import ACTION_DIMS
    from trpg.engine.skill import available_skills
    from trpg.engine.vec2 import Vec2
    ws = _heal_world()
    a = ws.characters["a"]; b = ws.characters["b"]
    a.position = Vec2(10.0, 10.0); b.position = Vec2(11.4, 10.0)  # 3m 內（far 遮不觸）
    a._outgoing_attempts = 1                 # 隔離：龜縮守門不觸
    sks = available_skills(a, ws)
    di = next(i for i, s in enumerate(sks) if s.skill_id == "disengage")
    logits = torch.zeros(1, ACTION_DIMS[0])
    res0 = {"movement": 0.0, "action": 1, "bonus_action": 1}
    out = apply_resource_mask(logits, res0, ws, "a")
    assert out[0, di].item() <= -1e8, "movement=0 的 disengage＝null，該遮"
    res6 = {"movement": 6.0, "action": 1, "bonus_action": 1}
    out2 = apply_resource_mask(logits, res6, ws, "a")
    assert out2[0, di].item() > -1e8, "有移動預算＝真 disengage-然後-脫離，合法"


def test_resource_mask_blocks_pure_turtle_disengage():
    # 純龜縮守門的 sibling 補位：零輸出+單挑+可及+有行動時 dodge/hide 已遮，
    # disengage 同為防禦沉沒點（零輸出 1v1 永不能贏）卻漏列——mine|666|0456。
    import torch
    from trpg.rl.model import apply_resource_mask
    from trpg.rl.action import ACTION_DIMS
    from trpg.engine.skill import available_skills
    from trpg.engine.combat import execute_action
    from trpg.engine.vec2 import Vec2
    ws = _heal_world()
    a = ws.characters["a"]; b = ws.characters["b"]
    a.position = Vec2(10.0, 10.0); b.position = Vec2(10.8, 10.0)  # reach 內
    sks = available_skills(a, ws)
    di = next(i for i, s in enumerate(sks) if s.skill_id == "disengage")
    res = {"movement": 6.0, "action": 1, "bonus_action": 1}   # 有移動（(a) 不觸）
    logits = torch.zeros(1, ACTION_DIMS[0])
    out = apply_resource_mask(logits, res, ws, "a")
    assert out[0, di].item() <= -1e8, "1v1 零輸出、敵可及、有行動 → disengage 該遮"
    sw = next(i for i, s in enumerate(sks) if s.skill_id.startswith("weapon:"))
    act = sks[sw].build_action("a", "b", (b.position.x, b.position.y))
    execute_action(act, ws)                  # 出手一次 → 永久解鎖
    out2 = apply_resource_mask(logits, res, ws, "a")
    assert out2[0, di].item() > -1e8, "出手過後 disengage 是合法戰術（風箏/重定位）"


# ── APPLY_MOD 已生效重放遮罩（null-effect 家族；add_status 冪等＝引擎同源） ─────────

def _buff_world():
    from trpg.engine.character import Character, Stats
    from trpg.engine.world_state import WorldState, CombatState
    from trpg.engine.combat import setup_combat_positions
    from trpg.engine.items import WEAPON_DEFS
    from trpg.engine.vec2 import Vec2
    a = Character(name="a", race="人類", class_="盜賊", level=8,
                  stats=Stats(STR=14, DEX=16, CON=12),
                  hp=40, max_hp=40, ac=14,
                  weapons=[WEAPON_DEFS["短劍"]], is_npc=False)
    a.known_abilities = ["evasion_rogue"]
    b = Character(name="b", race="哥布林", class_="戰士", level=1,
                  stats=Stats(STR=10, DEX=12),
                  hp=10, max_hp=10, ac=12,
                  weapons=[WEAPON_DEFS["短劍"]], is_npc=True, attitude=0)
    ws = WorldState(characters={"a": a, "b": b}, scene="test",
                    pc_ids=["a"], party_ids=["a"])
    ws.combat = CombatState(active=True, initiative_order=["a", "b"],
                            round_number=1)
    setup_combat_positions(ws, ws.combat)
    a.position = Vec2(10.0, 10.0)
    b.position = Vec2(10.8, 10.0)
    return ws


def test_resource_mask_blocks_recast_of_active_self_mod():
    # 根因 B 迴歸（miner mine|0|0012）：零消耗永久 self-buff（嫁接 evasion）
    # 已生效仍可重放 → argmax 空轉誘餌（貼臉 22 回合 0 攻擊嘗試站著死）。
    # 引擎同源：add_status 對同名冪等 ⇒ 重放對目標什麼都不做 ⇒ null-effect 遮；
    # 首放合法（1 行動換永久閃避＝真價值），不發明行為。
    import torch
    from trpg.rl.model import apply_resource_mask
    from trpg.rl.action import ACTION_DIMS
    from trpg.engine.skill import available_skills
    from trpg.engine.combat import execute_action
    ws = _buff_world()
    a = ws.characters["a"]
    sks = available_skills(a, ws)
    ei = next(i for i, s in enumerate(sks) if s.skill_id == "evasion_rogue")
    res = {"movement": 6.0, "action": 1, "bonus_action": 1}
    logits = torch.zeros(1, ACTION_DIMS[0])
    out = apply_resource_mask(logits, res, ws, "a")
    assert out[0, ei].item() > -1e8, "尚未生效 → 首放合法"
    act = sks[ei].build_action("a", "a", (a.position.x, a.position.y))
    r = execute_action(act, ws)
    assert r.get("type") != "ERROR"
    assert a.has_status("evasion")
    out2 = apply_resource_mask(logits, res, ws, "a")
    assert out2[0, ei].item() <= -1e8, "已生效 → 重放＝引擎 no-op，該遮"
    # 攻擊列不受影響（argmax 自然落到攻擊＝空轉迴圈被打斷）
    sw = next(i for i, s in enumerate(sks) if s.skill_id.startswith("weapon:"))
    assert out2[0, sw].item() > -1e8


def test_recast_guard_generic_over_self_apply_mod_registry():
    # sibling audit as test：登錄表裡所有 SELF+APPLY_MOD 能力，modifier 已在
    # 身上時一律遮（通用機制、零技能名硬編碼——迴圈掃全表）。
    import torch
    from trpg.rl.model import apply_resource_mask
    from trpg.rl.action import ACTION_DIMS
    from trpg.engine.abilities import ABILITY_REGISTRY
    from trpg.engine.skill import available_skills, TargetType
    from trpg.engine.status import MODIFIER_CLASSES
    covered = []
    for aid, ab in sorted(ABILITY_REGISTRY.items()):
        if ab.features.target_type != TargetType.SELF or ab.builder is None:
            continue
        try:
            d = ab.build_action("a", "a", (0.0, 0.0))
        except Exception:
            continue                     # builder 需要 char 上下文 → 掃描略過
        if not (isinstance(d, dict) and d.get("type") == "APPLY_MOD"):
            continue
        mod = d.get("modifier", "")
        assert mod in MODIFIER_CLASSES, f"{aid}: 未知 modifier {mod!r}"
        ws = _buff_world()
        a = ws.characters["a"]
        a.level = 20
        a.known_abilities = [aid]
        a.spell_slots = {1: 4, 2: 4, 3: 4, 4: 4, 5: 4}
        sks = available_skills(a, ws)
        idx = next((i for i, s in enumerate(sks) if s.skill_id == aid), None)
        if idx is None:
            continue                     # 此身份列不出（等級/職業門檻）
        cls = MODIFIER_CLASSES[mod]
        try:
            fx = cls(applied_round=1, source_id="a")
        except TypeError:
            fx = cls(applied_round=1)
        a.add_status(fx)
        res = {"movement": 6.0, "action": 1, "bonus_action": 1}
        logits = torch.zeros(1, ACTION_DIMS[0])
        out = apply_resource_mask(logits, res, ws, "a")
        assert out[0, idx].item() <= -1e8, \
            f"{aid}: modifier {mod!r} 已生效仍未遮"
        covered.append(aid)
    assert "evasion_rogue" in covered
