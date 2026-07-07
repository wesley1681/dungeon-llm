"""模型架構的**時間軸登記表**:一檔一版本、按時間排序、各有代號、可回退。

政策(為什麼這樣分):
  - 共用的穩定合約(obs schema+遷移鏈、決策層遮罩 apply_*/pick_action、checkpoint
    adapter、env/訓練介面)**永不複製**——它是基礎設施,不是「架構」。
  - 每個架構版本 = 這資料夾裡的一個檔 `vNN_代號.py`,定義自己的 `Net`,記
    CODENAME / DATE / PARENT / 改動說明。小變體 = 繼承 CombatPolicyNet 只設 delta
    kwargs(改動一眼可見、勝過複製 1800 行把 delta 埋進相同的行裡);大改動 = 該檔
    直接定義獨立 class(不繼承)。
  - checkpoint 靠**代號**綁定(sidecar `<path>.arch.json`);載入時核心權重形狀不合
    就 raise,不靜默載壞。
  - 回退 = 直接用舊版本檔的 `Net`(舊檔凍結、不因新版而動)= 你要的「改壞了回上一版」。

新增架構:開 `vNN_代號.py`(下一個編號)→ 定義 CODENAME/PARENT/DATE/Net → 加進下面
TIMELINE 尾端。就這樣。
"""
from __future__ import annotations
import json
import os
import warnings

import torch

from ..model import CombatPolicyNet
from . import (v01_uni, v02_codeword, v03_codeword_noimmune, v04_codeword_noarch,
               v05_codeword_noarch_enemyskill)

# 時間順序(舊 → 新)。這份 list 就是「發展軌跡」。
TIMELINE = [v01_uni, v02_codeword, v03_codeword_noimmune, v04_codeword_noarch,
            v05_codeword_noarch_enemyskill]
REGISTRY = {m.CODENAME: m for m in TIMELINE}


def build(codename: str, **kw) -> CombatPolicyNet:
    """照代號建一個對的網。"""
    if codename not in REGISTRY:
        raise KeyError(f"未知架構代號 {codename!r};已登記(時間序):"
                       f"{[m.CODENAME for m in TIMELINE]}")
    return REGISTRY[codename].Net(**kw)


def timeline() -> str:
    """印出時間軸(給人看發展軌跡)。"""
    rows = []
    for i, m in enumerate(TIMELINE, 1):
        head = ((m.__doc__ or "").strip().splitlines() or [""])[0]
        rows.append(f"{i:>2}. {m.CODENAME:<18} {m.DATE:<10} 前身:{str(m.PARENT or '—'):<16} {head}")
    return "\n".join(rows)


# ── 自我描述的存/載(sidecar 記代號) ──────────────────────────────────────────
def _sidecar(path: str) -> str:
    return path + ".arch.json"


# ── 架構指紋:「把當時的架構存下來」的具體形式 ────────────────────────────────
# 存檔時記下網路的結構快照(各層形狀 + obs 版面);載入時由 codename 重建網、比對指紋,
# 不合就大聲報錯並印 diff。**沒有自動遷移**(那是 append-only 假設、綁手綁腳的來源)——
# 你可以隨便重構/重寫整個檔案,舊 checkpoint 只會乾淨地載入失敗、告訴你差在哪,不靜默載壞。
def _obs_signature() -> dict:
    """obs 版面的結構簽章 —— 抓 obs 重排/改寬/加減欄(連「同形狀但語意變」torch 抓不到的)。"""
    import trpg.rl.obs as _O
    return {
        "OBS_KEYS": list(_O.OBS_KEYS),
        "ENTITY_DIM": int(_O.ENTITY_DIM),
        "N_V3_EXTRA": int(_O.N_V3_EXTRA), "N_V4_DESC": int(_O.N_V4_DESC),
        "N_V5_TRAIT": int(_O.N_V5_TRAIT), "N_V6_CIMMUN": int(_O.N_V6_CIMMUN),
        "N_V7_ABILITY": int(_O.N_V7_ABILITY),
        "I_DESC_RESIST": int(_O.I_DESC_RESIST), "I_DESC_CIMMUN": int(_O.I_DESC_CIMMUN),
        "N_SKILL_SLOTS": int(_O.N_SKILL_SLOTS),
        "SKILL_FEATURE_DIM": int(_O.SKILL_FEATURE_DIM),
        "N_ENTITY_SLOTS": int(_O.N_ENTITY_SLOTS),
    }


def _fingerprint(net: CombatPolicyNet) -> dict:
    """checkpoint 的架構快照:各層形狀 + obs 版面。存這個 = 把當時的架構存下來。"""
    return {"params": {k: list(v.shape) for k, v in net.state_dict().items()},
            "obs": _obs_signature()}


def _dict_diff(a: dict, b: dict) -> dict:
    """逐鍵 diff(給人看差在哪):{鍵: [存檔時, 現在]}。"""
    return {k: [a.get(k), b.get(k)] for k in (set(a) | set(b)) if a.get(k) != b.get(k)}


def _meta(codename: str, net: CombatPolicyNet) -> dict:
    return {"codename": codename, "arch_kwargs": net.arch_kwargs,
            "fingerprint": _fingerprint(net)}


def save_net(net: CombatPolicyNet, path: str, codename: str) -> None:
    """存 state_dict(bare)+ sidecar(代號 + arch_kwargs + **架構指紋**)。"""
    if codename not in REGISTRY:
        raise KeyError(f"未知架構代號 {codename!r}")
    torch.save(net.state_dict(), path)
    with open(_sidecar(path), "w", encoding="utf-8") as f:
        json.dump(_meta(codename, net), f, ensure_ascii=False, indent=2)


def write_sidecar(path: str, codename: str) -> None:
    """給既有 .pt 補一張 sidecar(記**當前 build 的**指紋;僅對相容的檔有意義)。"""
    if codename not in REGISTRY:
        raise KeyError(f"未知架構代號 {codename!r}")
    with open(_sidecar(path), "w", encoding="utf-8") as f:
        json.dump(_meta(codename, build(codename)), f, ensure_ascii=False, indent=2)


def _read_meta(path: str) -> dict | None:
    sc = _sidecar(path)
    if not os.path.exists(sc):
        return None
    with open(sc, encoding="utf-8") as f:
        return json.load(f)


def read_codename(path: str) -> str | None:
    m = _read_meta(path)
    return m.get("codename") if m else None


def load_net(path: str, codename: str | None = None, *, eval_mode: bool = True,
             migrate: bool = False) -> CombatPolicyNet:
    """由 codename 重建網 → **嚴格載入**;結構對不上就 raise 並印清楚 diff。

    機制=「存架構、重建、比對」,不做自動遷移 → 你重構/重寫整個檔案時舊 checkpoint 只會
    乾淨報錯(不靜默載壞)、也不逼你 append-only。`migrate=True` 才會套用 legacy 遷移橋接
    (append 時代的舊檔)。critic(value head,推論不用)的形狀差異被容許。
    """
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    meta = _read_meta(path)
    if codename is None:
        codename = (meta or {}).get("codename")
        if codename is None:
            warnings.warn(f"{path} 無 sidecar → 假設最舊的 'uni';實驗網請傳 codename=… "
                          f"或用 write_sidecar 補上。")
            codename = "uni"
    net = build(codename)
    sd = torch.load(path, map_location="cpu", weights_only=True)
    if migrate:   # 明確 opt-in 的 legacy 橋接;預設完全不走(不做 append-only 假設)
        sd = CombatPolicyNet.adapt_state_dict_for_perarch(sd)

    # (1) obs 版面漂移:比對 sidecar 記的 vs 現在的(抓同形狀但重排/改語意)
    rec = (meta or {}).get("fingerprint") or {}
    if rec.get("obs") is not None and rec["obs"] != _obs_signature():
        raise ValueError(
            f"架構不符(obs 版面已改變):{path}\n  差異(存檔時→現在):"
            f"{_dict_diff(rec['obs'], _obs_signature())}\n  → 此 checkpoint 在不同 obs 版面下"
            f"訓練。重訓、用當時的碼,或 load_net(migrate=True) 若有 legacy 橋接。")

    # (2) 參數形狀:對不上就 raise 並印 diff(critic 可差異、推論不用)
    msd = net.state_dict()
    bad = {k: [list(sd[k].shape), list(msd[k].shape)]
           for k in msd if k in sd and tuple(sd[k].shape) != tuple(msd[k].shape)
           and not k.startswith("critic")}
    missing = [k for k in msd if k not in sd and not k.startswith("critic")]
    extra = [k for k in sd if k not in msd and not k.startswith("critic")]
    if bad or missing or extra:
        raise ValueError(
            f"架構不符:{path} 的權重對不上 codename={codename!r} 建出的網。\n"
            f"  形狀不同:{dict(list(bad.items())[:5])}\n"
            f"  缺少(新網有、檔沒有):{missing[:5]}\n"
            f"  多餘(檔有、新網沒有):{extra[:5]}\n"
            f"  → codename 的碼在存檔後結構漂移了。重訓、用當時的碼,或 migrate=True。")

    net.load_state_dict({k: v for k, v in sd.items()
                         if k in msd and tuple(sd[k].shape) == tuple(msd[k].shape)},
                        strict=False)
    if eval_mode:
        net.eval()
    return net
