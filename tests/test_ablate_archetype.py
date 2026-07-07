"""ablate_archetype (2026-07-06): 實驗模式下把「職業」從架構中**結構性移除**。

不是遮罩(NoArch 把職業 one-hot 欄餵零) —— 而是那些欄位/模組在架構裡**不存在**:
  1. entity_mlp 進第一層前切掉職業欄 → 入寬 = ENTITY_DIM - N_ARCHETYPES(權重不存在)
  2. FiLM(film_gamma/beta,職業條件化 h=γh+β)整條不建、forward 不套用
  3. end_head 切掉 end_features 職業尾 → 入寬 -N_ARCHETYPES
  4. per-arch 路由 arch_ids 不讀職業欄(固定 0,單一共享頭)

驗收核心(behavioural):ablate 網對「任何職業欄的擾動」**輸出變化 exactly 0**,證明架構
裡真的沒有職業通路;對照組=普通網擾動職業欄輸出**會變**(證明這個不變性測試抓得到洩漏);
預設(ablate=False)網結構不變(film 還在、entity_mlp 入寬仍 ENTITY_DIM)。
"""
import copy
import torch
import pytest

import trpg.rl.obs as O
from trpg.rl.obs import N_ARCHETYPES, ENTITY_DIM, END_FEATURES_DIM
from trpg.rl.model import CombatPolicyNet, _ARCH_OH_START, _ARCH_OH_END
from trpg.engine.skill import SKILL_FEATURE_DIM


def _obs(B=2, seed=0):
    g = torch.Generator().manual_seed(seed)
    ent = torch.rand(B, O.N_ENTITY_SLOTS, ENTITY_DIM, generator=g)  # 全列非零=全部 present
    m = torch.zeros(B, O.N_SKILL_SLOTS); m[:, :6] = 1.0
    return {
        "skills": torch.randn(B, O.N_SKILL_SLOTS, SKILL_FEATURE_DIM, generator=g),
        "skill_mask": m,
        "entities": ent,
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


def _perturb_all_arch(obs):
    """把每個實體列的職業欄 + end_features 職業尾 全部改成不同值(仍保持列非零)。"""
    o = copy.deepcopy(obs)
    o["entities"][:, :, _ARCH_OH_START:_ARCH_OH_END] = torch.rand(
        obs["entities"].shape[0], O.N_ENTITY_SLOTS, N_ARCHETYPES)
    o["end_features"][:, END_FEATURES_DIM - N_ARCHETYPES:] = torch.rand(
        obs["end_features"].shape[0], N_ARCHETYPES)
    return o


def _kw(**extra):
    return dict(n_head_groups=1, skill_combo_dim=8, **extra)


def test_ablate_removes_film_and_narrows_inputs():
    net = CombatPolicyNet(**_kw(ablate_archetype=True))
    assert net.ablate_archetype is True
    assert getattr(net, "film_gamma", None) is None
    assert getattr(net, "film_beta", None) is None
    assert net.entity_mlp[0].in_features == ENTITY_DIM - N_ARCHETYPES
    assert net.end_head[0].in_features == (END_FEATURES_DIM - N_ARCHETYPES
                                           + O.N_REACH_GRID_CHANNELS
                                           + O.N_THREAT_GRID_CHANNELS)


def test_default_net_keeps_arch_structure():
    net = CombatPolicyNet(**_kw())
    assert net.ablate_archetype is False
    assert net.film_gamma is not None and net.film_beta is not None
    assert net.entity_mlp[0].in_features == ENTITY_DIM
    assert net.end_head[0].in_features == (END_FEATURES_DIM
                                           + O.N_REACH_GRID_CHANNELS
                                           + O.N_THREAT_GRID_CHANNELS)


def test_ablate_output_invariant_to_every_arch_column():
    """核心驗收:ablate 網對任何職業欄的擾動,輸出變化 exactly 0。"""
    net = CombatPolicyNet(**_kw(ablate_archetype=True)).eval()
    obs = _obs(seed=1)
    obs2 = _perturb_all_arch(obs)
    with torch.no_grad():
        a = net(obs); b = net(obs2)
    for i, (x, y) in enumerate(zip(a, b)):
        assert torch.equal(x, y), f"head {i}:ablate 網竟對職業欄敏感=有殘留通路"


def test_default_net_IS_sensitive_to_arch():
    """對照:普通網擾動職業欄輸出會變 → 證明上面的不變性測試抓得到洩漏(非假陽)。"""
    net = CombatPolicyNet(**_kw()).eval()
    obs = _obs(seed=2)
    obs2 = _perturb_all_arch(obs)
    with torch.no_grad():
        a = net(obs); b = net(obs2)
    assert any(not torch.equal(x, y) for x, y in zip(a, b)), \
        "普通網該對職業欄敏感(entity_mlp/end_head 隨機權重吃職業欄)"


def test_ablate_forward_shapes():
    net = CombatPolicyNet(**_kw(ablate_archetype=True)).eval()
    with torch.no_grad():
        el, sl, ent, grid = net(_obs(B=3, seed=3))
    assert el.shape[0] == 3
    assert sl.shape == (3, O.N_SKILL_SLOTS)
    assert ent.shape[:2] == (3, O.N_SKILL_SLOTS)
