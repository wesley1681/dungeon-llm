"""obs v8 + encode_entity_skills:模型讀「每個實體的靜態 kit」以區分聚合相同的敵人。

動機(數據):capability_descriptor 把 raging berserker 和 champion(L2 1.29,暴怒抗性隱形)、
兩個 L6 法師(evocation vs divination,L2 **0.00**)糊成一團。obs v8 新增 per-entity 技能
矩陣;encode_entity_skills 用**共享的** skill encoder 把它總結成 64-d,再**拼接**進 ent_emb
(不是相加、不壓縮)——ent_emb 從 32 → 96 維,下游一律以 self._ent_emb_dim 取寬。

驗收:
  (1) 拼接使 ent_emb 變寬(_ent_emb_dim==96)、沒有投影層(entity_kit_proj 已移除)。
  (2) 從零初始就**真的讀** entity_skills(擾動→輸出變;無 warm-start/惰性設計)。
  (3) 全空實體列不 NaN(全 padding 序列的守門)。
  (4) 預設網(flag off)ent_emb 維持 32、忽略新 key(有無 entity_skills 輸出一致)。
"""
import torch

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


def test_concat_widens_entity_embedding():
    """拼接(非相加):ent_emb 32→96,無 64→32 投影層。"""
    net = _net()
    assert net.encode_entity_skills is True
    assert net._ent_emb_dim == 96                     # 32 base + 64 kit(不壓縮)
    assert not hasattr(net, "entity_kit_proj") or net.entity_kit_proj is None
    # 下游各層真的照 96 建:skill→entity 注意力的 query 投影對齊 ent_emb 寬度。
    assert net.skill_ent_attn_proj.out_features == 96


def test_reads_entity_skills_at_fresh_init():
    """從零初始就讀 entity_skills(共享 encoder 非零)——沒有惰性/warm-start 設計。"""
    net = _net()
    obs = _obs(seed=2)
    obs2 = dict(obs); obs2["entity_skills"] = torch.rand_like(obs["entity_skills"])
    with torch.no_grad():
        a, b = net(obs), net(obs2)
    assert any(not torch.equal(x, y) for x, y in zip(a, b)), \
        "拼接下 entity_skills 從第 0 步就進 ent_emb,擾動它該改變輸出"


def test_all_empty_entities_no_nan():
    """全實體全 padding(空 kit 序列)不該 NaN——all_pad 守門。"""
    net = _net()
    obs = _obs(seed=3)
    obs["entity_skill_mask"] = torch.zeros_like(obs["entity_skill_mask"])  # 全實體全空
    with torch.no_grad():
        out = net(obs)
    for x in out:
        assert torch.isfinite(x).all(), "全 padding 序列不該 NaN"


def test_default_net_ignores_entity_skills():
    """flag off ⇒ ent_emb 維持 32、忽略新 key。"""
    net = CombatPolicyNet(n_head_groups=1, skill_combo_dim=8).eval()  # flag off
    assert net.encode_entity_skills is False and net._ent_emb_dim == 32
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
