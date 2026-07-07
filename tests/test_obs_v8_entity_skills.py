"""obs v8 + encode_entity_skills:模型讀「每個實體的靜態 kit」以區分聚合相同的敵人。

動機(數據):capability_descriptor 把 raging berserker 和 champion(L2 1.29,暴怒抗性隱形)、
兩個 L6 法師(evocation vs divination,L2 **0.00**)糊成一團。obs v8 新增 per-entity 技能
矩陣;encode_entity_skills 用**共享的** skill encoder 把它總結進 ent_emb。

驗收:
  (1) zero-init entity_kit_proj ⇒ **惰性**:擾動 entity_skills 不改輸出(warm-start bit-exact)。
  (2) 投影訓練後 ⇒ **真的讀** entity_skills(擾動→輸出變)。
  (3) 全空實體列不 NaN(全 padding 序列的守門)。
  (4) 預設網(flag off)忽略新 key(有無 entity_skills 輸出一致)。
"""
import torch
import pytest

import trpg.rl.obs as O
from trpg.engine.skill import SKILL_FEATURE_DIM
from trpg.rl.model import CombatPolicyNet


def _obs(B=2, seed=0, with_es=True):
    g = torch.Generator().manual_seed(seed)
    d = {
        "skills": torch.randn(B, O.N_SKILL_SLOTS, SKILL_FEATURE_DIM, generator=g),
        "skill_mask": (lambda m: (m.__setitem__((slice(None), slice(0, 6)), 1.0), m)[1])(
            torch.zeros(B, O.N_SKILL_SLOTS)),
        "entities": torch.rand(B, O.N_ENTITY_SLOTS, O.ENTITY_DIM, generator=g),
        "resources": torch.rand(B, 4, generator=g),
        "terrain": torch.zeros(B, O.N_GRID, O.N_GRID),
        "entity_grid": torch.zeros(B, O.N_ENTITY_GRID_CHANNELS, O.N_GRID, O.N_GRID),
        "distance_grid": torch.rand(B, O.N_DISTANCE_GRID_CHANNELS, O.N_GRID, O.N_GRID, generator=g),
        "los_grid": torch.ones(B, O.N_LOS_GRID_CHANNELS, O.N_GRID, O.N_GRID),
        "reach_grid": torch.ones(B, O.N_REACH_GRID_CHANNELS, O.N_GRID, O.N_GRID),
        "threat_grid": torch.zeros(B, O.N_THREAT_GRID_CHANNELS, O.N_GRID, O.N_GRID),
        "end_features": torch.rand(B, O.END_FEATURES_DIM, generator=g),
        "decision_context": torch.zeros(B, O.N_DECISION_CTX),
    }
    if with_es:
        d["entity_skills"] = torch.rand(B, O.N_ENTITY_SLOTS, O.N_SKILL_SLOTS,
                                        SKILL_FEATURE_DIM, generator=g)
        esm = torch.zeros(B, O.N_ENTITY_SLOTS, O.N_SKILL_SLOTS)
        esm[:, :3, :5] = 1.0                    # 前 3 個實體各 5 招,其餘空
        d["entity_skill_mask"] = esm
    return d


def _net(**kw):
    return CombatPolicyNet(n_head_groups=1, skill_combo_dim=8, ablate_archetype=True,
                           encode_entity_skills=True, **kw).eval()


def test_zero_init_projection_is_inert():
    net = _net()
    assert net.entity_kit_proj is not None
    assert torch.all(net.entity_kit_proj.weight == 0) and torch.all(net.entity_kit_proj.bias == 0)
    obs = _obs(seed=1)
    obs2 = dict(obs); obs2["entity_skills"] = torch.rand_like(obs["entity_skills"])
    with torch.no_grad():
        a, b = net(obs), net(obs2)
    for x, y in zip(a, b):
        assert torch.equal(x, y), "zero-init ⇒ entity_skills 該惰性(warm-start bit-exact)"


def test_trained_projection_reads_entity_skills():
    net = _net()
    with torch.no_grad():                        # 「訓練」投影
        net.entity_kit_proj.weight.normal_(0.0, 0.5)
        net.entity_kit_proj.bias.normal_(0.0, 0.5)
    obs = _obs(seed=2)
    obs2 = dict(obs); obs2["entity_skills"] = torch.rand_like(obs["entity_skills"])
    with torch.no_grad():
        a, b = net(obs), net(obs2)
    assert any(not torch.equal(x, y) for x, y in zip(a, b)), "訓練後該真的讀 entity_skills"


def test_all_empty_entities_no_nan():
    net = _net()
    with torch.no_grad():
        net.entity_kit_proj.weight.normal_(0.0, 0.5)   # 非零,確保真的走了 encoder
    obs = _obs(seed=3)
    obs["entity_skill_mask"] = torch.zeros_like(obs["entity_skill_mask"])  # 全實體全空
    with torch.no_grad():
        out = net(obs)
    for x in out:
        assert torch.isfinite(x).all(), "全 padding 序列不該 NaN"


def test_default_net_ignores_entity_skills():
    net = CombatPolicyNet(n_head_groups=1, skill_combo_dim=8).eval()  # flag off
    assert net.entity_kit_proj is None
    obs_with = _obs(seed=4, with_es=True)
    obs_without = {k: v for k, v in obs_with.items() if not k.startswith("entity_skill")}
    with torch.no_grad():
        a, b = net(obs_with), net(obs_without)
    for x, y in zip(a, b):
        assert torch.equal(x, y), "flag off ⇒ 新 key 該被忽略"


def test_forward_shapes():
    net = _net()
    with torch.no_grad():
        el, sl, ent, grid = net(_obs(B=3, seed=5))
    assert el.shape[0] == 3 and sl.shape == (3, O.N_SKILL_SLOTS)
