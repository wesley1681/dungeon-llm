"""exp_scratch 實驗線的共用平行 rollout 原語（把「你自己的 collect()」平行化）。

為什麼需要它，而不是直接用 train_ppo.collect_ppo_rollout：
  collect_ppo_rollout 是**統一訓練管線**用的，env 設置寫死（隨機抽職業／自我對弈／
  固定隊形）。exp_scratch 每支實驗要**自訂 env**（釘 level、站樁靶、免疫注入、鏡像
  腳本專家、wmc、mask_immune_null…），所以各自有一份 bespoke `collect()`。這個原語
  只負責把那份 collect() 拆到多進程跑，env 邏輯留在實驗裡。

瓶頸依據（實測 2026-07-06，RTX 5060 Ti + 32 核）：
  scratch PPO 是 **rollout-bound**（rollout 佔一次 update 的 72–81%），而 rollout 是
  batch=1 小網路（~31 萬參數）的逐步 forward——對它灑 torch 執行緒**反而更慢**
  （16 threads: 15s/update；1 thread: 12s）。**GPU 無用**（per-op 啟動開銷 > CPU 運算）。
  正解＝**進程級平行**：K 個 worker 各 threads=1、各收 n_steps/K 步。實測 K=16 讓
  rollout 8.6s→0.9s（9.4x 近線性），整個 update 15s→4s（~3.7x）。

實驗端契約（模組頂層需有這兩個函式）：
  build_net(**net_kwargs) -> CombatPolicyNet
  collect(net, n_steps, seed, *rest) -> (batch_dict, n_episodes)             # 或 3-tuple
      batch_dict 需含 "obs"(dict of arrays) 與其餘 1D/2D array 欄位（與 ppo_update 相容）。
      **可選**：回 (batch_dict, n_episodes, aux_list) 第三個＝逐局附帶資料的 list（如
      線上強度貨幣要收回主程序 update 的 (隊A,隊B,勝負) 三元組）。用 run_aux 取回。
  *rest 必須可 pickle（spawn）——closure/lambda 不行，改傳參數在 worker 內重建（見
   exp_matchup_train.collect：傳 model.to_dict()+draw 參數，worker 內 from_dict/重建 draw）。

用法（在實驗 main 內）：
  from trpg.rl.exp_parallel import ExpParallel
  pc = ExpParallel("exp_scratch_battlemaster", workers=16,
                   scripts_dir=os.path.dirname(os.path.abspath(__file__)),
                   net_kwargs=dict(skill_combo_dim=8))
  batch, neps = pc.run(net, steps, base_seed, level, opp_level, wmc)   # -> 合併後 batch
  batch, neps, aux = pc.run_aux(net, steps, base_seed, *rest)          # 需逐局附帶資料時
  ...
  pc.close()

主程序執行緒：rollout 靠 workers 平行，主程序只 ppo_update 用得到執行緒，設 8 左右即可
（torch.set_num_threads(8)）——不是設越高越好，見上。
"""
from __future__ import annotations
import sys
import importlib
import multiprocessing as mp
import numpy as np

# 每個 worker 進程的快取：實驗模組 + 該進程重用的 net。
_W: dict = {}


def _init(scripts_dir, module_name, net_kwargs):
    """Pool initializer：worker 一律單執行緒、把實驗模組匯入一次並快取。"""
    import torch
    import warnings
    torch.set_num_threads(1)            # 關鍵：batch=1 forward 灑多執行緒反而更慢
    warnings.filterwarnings("ignore")
    if scripts_dir and scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    _W["mod"] = importlib.import_module(module_name)
    _W["net"] = None
    _W["net_kwargs"] = net_kwargs or {}


def _task(args):
    """單一 worker chunk：載入最新權重、跑實驗自己的 collect()。"""
    sd, per, seed, rest = args
    mod = _W["mod"]
    if _W["net"] is None:
        _W["net"] = mod.build_net(**_W["net_kwargs"])
    net = _W["net"]
    net.load_state_dict(sd)
    net.eval()
    return mod.collect(net, per, seed, *rest)


def _merge(batches):
    """把各 worker 的 batch dict 串接成一個平坦 PPO batch（GAE 已在各 worker 內算好）。"""
    out = {}
    for k in batches[0]:
        if k == "obs":
            out["obs"] = {ok: np.concatenate([b["obs"][ok] for b in batches], 0)
                          for ok in batches[0]["obs"]}
        else:
            out[k] = np.concatenate([b[k] for b in batches], 0)
    return out


class ExpParallel:
    """持久化 spawn Pool，跨 update 重用（沿用 train_ppo._get_pool 同樣的 spawn 慣例）。"""

    def __init__(self, module_name, workers=16, scripts_dir=None, net_kwargs=None):
        self.workers = workers
        ctx = mp.get_context("spawn")   # Windows 唯一支援；顯式指定跨平台一致
        self.pool = ctx.Pool(
            processes=workers,
            initializer=_init,
            initargs=(scripts_dir, module_name, net_kwargs or {}),
        )

    def _run(self, net, n_steps, base_seed, rest):
        """核心：拆 n_steps 給 workers、合併。回 (batch, neps, aux)。
        aux＝各 worker collect() 第三個回傳（若有）攤平串接的 list——給需要把 rollout
        逐局附帶資料（如線上強度貨幣的 (隊A,隊B,勝負) 三元組）收回主程序的實驗用；
        collect() 只回 (batch, neps) 時 aux＝[]（向後相容）。"""
        sd = {k: v.cpu() for k, v in net.state_dict().items()}
        per = max(1, n_steps // self.workers)
        tasks = [(sd, per, base_seed * 131 + i * 100_003, rest)
                 for i in range(self.workers)]
        results = self.pool.map(_task, tasks)
        batches = [r[0] for r in results]
        neps = sum(r[1] for r in results)
        aux = []
        for r in results:
            if len(r) > 2 and r[2]:
                aux.extend(r[2])
        return _merge(batches), neps, aux

    def run(self, net, n_steps, base_seed, *rest):
        """把 n_steps 拆給 workers 個 worker 同時收集。rest 原封傳給 collect()（level 等）。
        回 (batch, neps)——collect() 的第三個回傳（若有）在此被丟棄，要收就用 run_aux。"""
        batch, neps, _aux = self._run(net, n_steps, base_seed, rest)
        return batch, neps

    def run_aux(self, net, n_steps, base_seed, *rest):
        """同 run，但**保留** collect() 第三個回傳（攤平串接）→ 回 (batch, neps, aux)。"""
        return self._run(net, n_steps, base_seed, rest)

    def close(self):
        self.pool.close()
        self.pool.join()
