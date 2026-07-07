"""architectures/ 時間軸登記表:一檔一版本、代號綁 checkpoint、載錯 raise、可回退。

核心性質:
  (1) 時間軸有序、每版有代號/日期/前身(發展軌跡)。
  (2) save_net→load_net round-trip:載回同版本、輸出位元一致。
  (3) **用錯代號載入會 raise**(核心權重形狀不合)——防「靜默載壞」的關鍵保證。
  (4) 無 sidecar 的舊 bare 檔 → 警告並當最舊的 'uni'。
"""
import json
import torch
import pytest

import trpg.rl.obs as O
from trpg.engine.skill import SKILL_FEATURE_DIM
from trpg.rl import architectures as A


def _obs(B=2, seed=0):
    g = torch.Generator().manual_seed(seed)
    ent = torch.rand(B, O.N_ENTITY_SLOTS, O.ENTITY_DIM, generator=g)
    m = torch.zeros(B, O.N_SKILL_SLOTS); m[:, :6] = 1.0
    return {
        "skills": torch.randn(B, O.N_SKILL_SLOTS, SKILL_FEATURE_DIM, generator=g),
        "skill_mask": m, "entities": ent, "resources": torch.rand(B, 4, generator=g),
        "terrain": torch.zeros(B, O.N_GRID, O.N_GRID),
        "entity_grid": torch.zeros(B, O.N_ENTITY_GRID_CHANNELS, O.N_GRID, O.N_GRID),
        "distance_grid": torch.rand(B, O.N_DISTANCE_GRID_CHANNELS, O.N_GRID, O.N_GRID, generator=g),
        "los_grid": torch.ones(B, O.N_LOS_GRID_CHANNELS, O.N_GRID, O.N_GRID),
        "reach_grid": torch.ones(B, O.N_REACH_GRID_CHANNELS, O.N_GRID, O.N_GRID),
        "threat_grid": torch.zeros(B, O.N_THREAT_GRID_CHANNELS, O.N_GRID, O.N_GRID),
        "end_features": torch.rand(B, O.END_FEATURES_DIM, generator=g),
        "decision_context": torch.zeros(B, O.N_DECISION_CTX),
    }


def test_timeline_is_ordered_named_and_linked():
    names = [m.CODENAME for m in A.TIMELINE]
    assert names == ["uni", "codeword", "codeword-noimmune", "codeword-noarch",
                     "codeword-noarch-enemyskill"]
    for m in A.TIMELINE:
        assert m.DATE                                   # 每版有日期
        assert m.PARENT is None or m.PARENT in A.REGISTRY   # 前身鏈有效
    assert A.TIMELINE[0].PARENT is None                 # uni = 根


def test_current_arch_codename_builds_noarch():
    net = A.build("codeword-noarch")
    assert net.ablate_archetype is True and net.film_gamma is None
    assert net.arch_kwargs["skill_combo_dim"] == 8


def test_save_load_roundtrip_bit_exact(tmp_path):
    net = A.build("codeword-noarch").eval()
    p = str(tmp_path / "m.pt")
    A.save_net(net, p, "codeword-noarch")
    meta = json.load(open(p + ".arch.json", encoding="utf-8"))
    assert meta["codename"] == "codeword-noarch"
    assert meta["arch_kwargs"]["ablate_archetype"] is True
    net2 = A.load_net(p)                                 # 無參數:靠 sidecar 認代號
    assert net2.ablate_archetype is True and net2.film_gamma is None
    obs = _obs(seed=3)
    with torch.no_grad():
        a, b = net(obs), net2(obs)
    for x, y in zip(a, b):
        assert torch.equal(x, y)


def test_wrong_codename_raises_not_silent(tmp_path):
    """關鍵:codeword-noarch 的 checkpoint 用 'uni' 載 → 必須 raise。"""
    net = A.build("codeword-noarch").eval()
    p = str(tmp_path / "m.pt")
    A.save_net(net, p, "codeword-noarch")
    with pytest.raises(ValueError, match="架構不符"):
        A.load_net(p, "uni")


def test_missing_sidecar_warns_and_assumes_uni(tmp_path):
    net = A.build("uni").eval()
    p = str(tmp_path / "bare.pt")
    torch.save(net.state_dict(), p)                     # bare,無 sidecar(舊檔)
    with pytest.warns(UserWarning, match="無 sidecar"):
        net2 = A.load_net(p)
    assert net2.arch_kwargs == A.build("uni").arch_kwargs


def test_unknown_codename_raises():
    with pytest.raises(KeyError):
        A.build("does-not-exist")


def test_save_records_architecture_fingerprint(tmp_path):
    """存檔＝把當時的架構存下來:sidecar 要有各層形狀 + obs 版面。"""
    net = A.build("codeword-noarch").eval()
    p = str(tmp_path / "m.pt")
    A.save_net(net, p, "codeword-noarch")
    fp = json.load(open(p + ".arch.json", encoding="utf-8"))["fingerprint"]
    assert "entity_mlp.0.weight" in fp["params"]                 # 有層形狀
    assert fp["obs"]["ENTITY_DIM"] == O.ENTITY_DIM               # 有 obs 版面
    assert fp["obs"]["N_V4_DESC"] == O.N_V4_DESC


def test_obs_layout_drift_fails_loud(tmp_path):
    """模擬「大改 obs 版面」:sidecar 記的 obs 與現在不同 → 載入必須大聲報錯,不靜默載壞。"""
    net = A.build("codeword-noarch").eval()
    p = str(tmp_path / "m.pt")
    A.save_net(net, p, "codeword-noarch")
    # 竄改 sidecar 的 obs 指紋(相當於「存檔後有人把 obs 大改了一遍」)
    sc = p + ".arch.json"
    meta = json.load(open(sc, encoding="utf-8"))
    meta["fingerprint"]["obs"]["ENTITY_DIM"] += 7               # 假裝 obs 版面變了
    json.dump(meta, open(sc, "w", encoding="utf-8"))
    with pytest.raises(ValueError, match="obs 版面已改變"):
        A.load_net(p)


def test_no_auto_migration_by_default(tmp_path):
    """預設不自動遷移:形狀對不上就報錯,不會偷偷 pad/丟鍵把它硬載進去。"""
    net = A.build("codeword-noarch").eval()                     # 有 entity_kit_proj? 沒有
    p = str(tmp_path / "m.pt")
    A.save_net(net, p, "codeword-noarch")
    # 用 enemyskill 版(多了 entity_kit_proj)去載 → 缺鍵,預設該 raise(而非靜默補零)
    with pytest.raises(ValueError, match="架構不符"):
        A.load_net(p, "codeword-noarch-enemyskill")
