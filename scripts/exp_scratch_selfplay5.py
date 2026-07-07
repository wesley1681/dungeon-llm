"""實驗:5 職自我對弈 — 對手席從「純腳本」換成「腳本 + 凍結快照池」的混合。

動機:接下來要讓模型扮演更多角色(尤其怪物,腳本寫不動也寫不好),不能再只靠腳本當
對手。這支用 5 個**有專家腳本**的職業當試驗場——因為它們有腳本,我們才**保有一把量尺**
(5×5 vs-專家矩陣),能量出自我對弈到底是侵蝕、持平、還是超越專家。專案史有實測地雷:
「有腳本的身份卻讓對手席由自我對弈快照驅動 → agent 學打神經對手而非腳本 → vs-腳本
−12pp(梯度衝突 −0.97)」。所以這裡用**混合**(50% 腳本當強錨防漂移 + 50% 快照給多樣性),
並把「−12pp 定律在職業上重不重演」變成可測結果。

架構 = **v05 `codeword-noarch-enemyskill`**(使用者指定「最新架構不妥協」):
  n_head_groups=1, skill_combo_dim=8, drop_noop_h=True, ablate_archetype=True(noarch,
  沒有職業標籤、身分全靠技能 codeword + entity_skills 推), encode_entity_skills=True
  (讀每個實體的靜態 kit,拼接進 ent_emb)。noarch ⇒ 對手席 blind 旗標無意義(網不讀
  職業 one-hot),史上「blind 網餵非-blind obs → 龜縮」那類退化在此不會發生。

兩階段(唯一跨階段變數 = 對手席):
  base :純專家腳本。產出 = v05 專家-only 5×5 矩陣(錨)+ 自我對弈起點。也是 v05 首訓體檢。
  mixed:warm-start base,對手席每局擲骰 = p_script 機率腳本 / 其餘從凍結快照池均勻抽。
         快照池:每 k_refresh updates 存一份當前權重、留最近 pool_size 份;第 1 份 =
         warm-start 來源(所以池從不空/冷)。

判讀:mixed 的 5×5 vs-專家 ≥ base(零侵蝕)= 成功;> = 自我對弈真的磨得更好;
明顯 < = −12pp 重演,停下查根因別硬訓。n≥48/格,greedy。vs-快照勝率只當監看(≈50%)。

用法:
  python scripts/exp_scratch_selfplay5.py --smoke
  # Phase 0 基線(純專家):
  python scripts/exp_scratch_selfplay5.py --phase base --updates 60 \
      --out_dir models/exp_selfplay5_base
  # Phase 1 混合自我對弈(warm-start base 末檔):
  python scripts/exp_scratch_selfplay5.py --phase mixed --updates 60 \
      --init_from models/exp_selfplay5_base/sp5_u0060.pt \
      --out_dir models/exp_selfplay5_mixed
  # 只評測:
  python scripts/exp_scratch_selfplay5.py --eval_only <ckpt>
"""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout.reconfigure(encoding="utf-8")
import warnings; warnings.filterwarnings("ignore")
os.environ.setdefault("TRPG_WASTED_MOVE_COST", "0")   # fresh PPO:別讓 wmc 擋探索

import argparse, random
from pathlib import Path
import numpy as np
import torch

import trpg.rl.obs  # noqa: F401  freeze N_ARCHETYPES before registry use
from trpg.rl.env_v2 import CombatEnvV2
from trpg.rl.model import (CombatPolicyNet, apply_resource_mask,
                           apply_entity_mask, pick_action)
from trpg.rl.train_ppo import _sample_action, _compute_gae, ppo_update
from trpg.rl.obs import build_obs
from trpg.engine.skill import available_skills
from trpg.engine.combat_policy import make_archetype_policy

# 5 職業:archetype_id → 顯示名(專家 policy 由 make_archetype_policy 自動對應)
ARCHS = ("war", "battle_master", "assassin", "evocation", "vengeance")
ARCH_NAME = {"war": "戰爭牧", "battle_master": "戰技師", "assassin": "刺客",
             "evocation": "塑能法", "vengeance": "復仇騎"}
CODENAME = "codeword-noarch-enemyskill"   # v05,checkpoint sidecar 用


# ── 架構契約(ExpParallel 要求模組頂層有 build_net);預設 = v05 ──────────────────
def build_net(hidden=128, skill_combo_dim=8, ablate_immunity_joins=False,
              drop_noop_h=True, ablate_archetype=True, encode_entity_skills=True):
    return CombatPolicyNet(hidden=hidden, n_head_groups=1,
                           skill_combo_dim=skill_combo_dim,
                           ablate_immunity_joins=ablate_immunity_joins,
                           drop_noop_h=drop_noop_h,
                           ablate_archetype=ablate_archetype,
                           encode_entity_skills=encode_entity_skills)


def make_env(seed, agent_arch, opp_arch, level, opp_net=None):
    """1v1 open。opp_net=None → 對手 = 專家腳本(env.reset 自動指派);
    opp_net 給網 → 對手 = 該(凍結快照)神經網。noarch ⇒ blind 無意義,傳 True 無害。"""
    env = CombatEnvV2(seed=seed, n_agents=1, n_opps=1)
    if opp_net is not None:
        env.use_self_play_opponent(opp_net, blind=True)
    env.reset(agent_archs=[agent_arch], opp_archs=[opp_arch],
              level=level, opp_level=level, layout="open")
    return env


# ── rollout(ExpParallel 契約:collect(net, n_steps, seed, *rest) -> (batch, neps)) ──
# rest = (level, pool, p_script, net_kwargs)。pool = list[state_dict](凍結快照,可 pickle);
# p_script = 腳本對手機率(base 階段 = 1.0 → 永遠腳本);net_kwargs 供 build 快照對手殼。
_BUF_KEYS = ("obs", "act", "lp", "rew", "val", "done", "skm", "enm", "grm")


def collect(net, n_steps, seed, level, pool, p_script, net_kwargs,
            gamma=0.99, lam=0.95):
    net.eval()
    rng = random.Random(seed)
    # 快照對手殼:整個 collect 只 build 一次,每局把抽中的快照 state_dict 載進去(load
    # 便宜、build 貴)。pool 空(base 階段)就不需要。
    opp_shell = build_net(**net_kwargs) if pool else None
    if opp_shell is not None:
        opp_shell.eval()
    buf = {k: [] for k in _BUF_KEYS}
    ep = 0
    while len(buf["rew"]) < n_steps:
        a_arch = rng.choice(ARCHS)          # 每局獨立均勻抽:模型職 + 對手職
        o_arch = rng.choice(ARCHS)
        use_snap = bool(pool) and (rng.random() >= p_script)
        opp_net = None
        if use_snap:
            opp_shell.load_state_dict(pool[rng.randrange(len(pool))])
            opp_net = opp_shell
        random.seed(seed * 2_246_822_519 + ep)   # 引擎骰用全域 random,固定它讓 rollout 可重現
        env = make_env(seed * 1_000_003 + ep, a_arch, o_arch, level, opp_net=opp_net)
        obs = build_obs(env.ws, env.current_agent_id, env.resources)
        done = False
        while not done:
            aid = env.current_agent_id
            ot = {k: torch.from_numpy(v).unsqueeze(0) for k, v in obs.items()}
            with torch.no_grad():
                action, lp, val, skm, enm, grm = _sample_action(
                    net, ot, env.resources, env.ws, aid)
            a = action.numpy().tolist()
            obs2, r, term, trunc, _ = env.step(a)
            buf["obs"].append(obs); buf["act"].append(a); buf["lp"].append(lp.numpy())
            buf["rew"].append(float(r)); buf["val"].append(val)
            buf["skm"].append(skm.numpy()); buf["enm"].append(enm.numpy())
            buf["grm"].append(grm.numpy())
            done = term or trunc; buf["done"].append(bool(done))
            obs = obs2
        ep += 1
    rewards = np.array(buf["rew"], np.float32); values = np.array(buf["val"], np.float32)
    dones = np.array(buf["done"], np.float32)
    adv, ret = _compute_gae(rewards, values, dones, 0.0, gamma, lam)
    batch = {
        "obs": {k: np.stack([o[k] for o in buf["obs"]]) for k in buf["obs"][0]},
        "actions": np.array(buf["act"], np.int64),
        "log_probs": np.array(buf["lp"], np.float32),
        "skill_masks": np.stack(buf["skm"]).astype(np.bool_),
        "entity_masks": np.stack(buf["enm"]).astype(np.bool_),
        "grid_masks": np.stack(buf["grm"]).astype(np.bool_),
        "rewards": rewards, "values": values,
        "returns": ret, "advantages": adv, "dones": dones,
    }
    return batch, ep


# ── 探針(greedy;評測對手一律專家腳本 = 固定錨) ────────────────────────────────
def _greedy_action(net, env, actor):
    ob = build_obs(env.ws, actor, env.resources)
    ot = {k: torch.from_numpy(v).unsqueeze(0) for k, v in ob.items()}
    with torch.no_grad():
        el, s, e, g = net(ot)
    s = apply_resource_mask(s, env.resources, env.ws, actor)
    e = apply_entity_mask(e, ot, env.ws, actor)
    return list(pick_action(el[0], s[0], e[0], g[0], ws=env.ws, agent_id=actor))


def _play(net, seed, a_arch, o_arch, level, max_steps=200):
    """跑一局 greedy vs 專家腳本,回傳 (won, 己方末血%, 敵方末血%)。"""
    random.seed(seed)
    env = make_env(seed, a_arch, o_arch, level)      # opp_net=None → 專家
    aid = env.agent_ids[0]; oid = env.opp_ids[0]
    done = False; steps = 0
    while not done and steps < max_steps:
        steps += 1
        act = _greedy_action(net, env, env.current_agent_id)
        _, _, term, trunc, _ = env.step(act); done = term or trunc
    A = env.ws.characters[aid]; W = env.ws.characters[oid]
    won = (not W.is_alive() and A.is_alive())
    return won, max(0., A.hp) / max(1, A.max_hp), max(0., W.hp) / max(1, W.max_hp)


def probe_per_class(net, n, level, seed0=40_000):
    """便宜探針:每模型職 n 局 vs 隨機專家。回傳 {arch: (wr, 己血, 敵血)}。"""
    net.eval(); out = {}
    for a_arch in ARCHS:
        rng = random.Random(seed0 + hash(a_arch) % 9973)
        wins = 0.0; ahp = []; whp = []
        for gi in range(n):
            o_arch = rng.choice(ARCHS)
            won, a, w = _play(net, seed0 + gi * 131 + hash(a_arch) % 997,
                              a_arch, o_arch, level)
            wins += won; ahp.append(a); whp.append(w)
        out[a_arch] = (wins / n, float(np.mean(ahp)), float(np.mean(whp)))
    net.train()
    return out


def probe_matrix(net, n, level, seed0=50_000):
    """交付物:完整 5×5(模型職 × 專家對手職)勝率矩陣。回傳 wr[a][o]。"""
    net.eval(); wr = {}
    for a_arch in ARCHS:
        wr[a_arch] = {}
        for o_arch in ARCHS:
            wins = 0.0
            for gi in range(n):
                won, _, _ = _play(net, seed0 + gi, a_arch, o_arch, level)
                wins += won
            wr[a_arch][o_arch] = wins / n
    net.train()
    return wr


def _report_per_class(tag, res):
    parts = []
    for a in ARCHS:
        wr, ahp, whp = res[a]
        parts.append(f"{ARCH_NAME[a]}:WR{wr:3.0%}(己{ahp:2.0%}/敵{whp:2.0%})")
    avg = np.mean([res[a][0] for a in ARCHS])
    print(f"[{tag:<5}] 均WR={avg:4.0%} | " + " ".join(parts), flush=True)


def _report_matrix(tag, wr, n):
    print(f"\n=== {tag} 5×5 勝率矩陣 (n={n}/格;列=模型職,欄=對手職[專家]) ===", flush=True)
    header = "模型\\對手 " + "".join(f"{ARCH_NAME[o]:>7}" for o in ARCHS) + "   平均"
    print(header, flush=True)
    for a in ARCHS:
        row = [wr[a][o] for o in ARCHS]
        cells = "".join(f"{v:6.0%} " for v in row)
        print(f"{ARCH_NAME[a]:<8}" + cells + f"  {np.mean(row):5.0%}", flush=True)
    col_avg = [np.mean([wr[a][o] for a in ARCHS]) for o in ARCHS]
    print("對手均擋 " + "".join(f"{1-v:6.0%} " for v in col_avg)
          + f"  總{np.mean([wr[a][o] for a in ARCHS for o in ARCHS]):5.0%}", flush=True)


# ── 快照池(Phase 1) ───────────────────────────────────────────────────────────
def _snapshot(net):
    """凍結當前權重成一份 cpu state_dict(與 net 解耦,之後訓練不會動到它)。"""
    return {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}


def refresh_pool(pool, net, pool_size):
    """存一份當前快照、截到最近 pool_size 份。就地改 pool 並回傳(方便測試)。"""
    pool.append(_snapshot(net))
    del pool[:-pool_size]
    return pool


# ── 機制自檢 ──────────────────────────────────────────────────────────────────
def smoke():
    print("── smoke:v05 / 5 職 / 專家+快照對手 / reward / forward ──", flush=True)
    net = build_net()
    n_par = sum(p.numel() for p in net.parameters())
    print(f"v05 CombatPolicyNet(noarch+enemyskill) params={n_par} "
          f"_ent_emb_dim={net._ent_emb_dim} "
          f"skill_head.in={net.skill_heads[0].weight.shape[1]} "
          f"entity_head.in={net.entity_heads[0].weight.shape[1]}", flush=True)
    assert net._ent_emb_dim == 96 and net.ablate_archetype and net.encode_entity_skills

    # 專家對手身分正確性
    want = {"war": "WarClericPolicy", "battle_master": "BattleMasterPolicy",
            "assassin": "AssassinPolicy", "evocation": "EvocationPolicy",
            "vengeance": "VengeancePolicy"}
    for arch, cls in want.items():
        got = type(make_archetype_policy(arch)).__name__
        assert got == cls, f"{arch} 專家該是 {cls} 得到 {got}"
    print(f"  專家對應正確:{want}", flush=True)

    # 每職:專家對手一局
    for a_arch in ARCHS:
        o_arch = ARCHS[(ARCHS.index(a_arch) + 1) % len(ARCHS)]
        env = make_env(123, a_arch, o_arch, level=5)
        aid = env.agent_ids[0]; oid = env.opp_ids[0]
        A = env.ws.characters[aid]; W = env.ws.characters[oid]
        opp_pol = type(env._opp_policies[oid]).__name__
        sks = [s.skill_id for s in available_skills(A, env.ws)]
        rtot = 0.0; steps = 0; done = False
        while not done and steps < 12:
            steps += 1
            act = _greedy_action(net, env, env.current_agent_id)
            _, r, term, trunc, _ = env.step(act); rtot += r; done = term or trunc
        assert np.isfinite(rtot), "reward 該有限"
        assert opp_pol != "HeuristicCombatPolicy", f"{o_arch} 該拿到專家 policy"
        print(f"  {ARCH_NAME[a_arch]}(L{A.level} HP{A.max_hp} AC{A.ac}) vs "
              f"{ARCH_NAME[o_arch]}[{opp_pol}] | 技能{len(sks)}招 | "
              f"{steps}步 R累={rtot:+.2f} 己血{A.hp:.0f} 敵血{W.hp:.0f}", flush=True)

    # 快照對手:池非空 → collect 走 mixed 路徑(對手席 = 神經快照)
    pool = [_snapshot(net)]
    env = make_env(7, "assassin", "war", level=5, opp_net=build_net())
    oid = env.opp_ids[0]
    assert type(env._opp_policies[oid]).__name__ == "NeuralCombatPolicy", \
        "opp_net 給了就該是神經對手"
    print(f"  快照對手 wiring OK:opp={type(env._opp_policies[oid]).__name__}", flush=True)

    # 快照池刷新/截斷
    p = []
    for _ in range(7):
        refresh_pool(p, net, pool_size=5)
    assert len(p) == 5, f"池該截到 5,得 {len(p)}"
    print(f"  快照池刷新/截斷 OK:7 次 refresh → len={len(p)}(pool_size=5)", flush=True)

    # collect 兩條路徑各跑一小段(base:p_script=1;mixed:p_script=0 全快照)
    nk = dict(skill_combo_dim=8, ablate_immunity_joins=False, drop_noop_h=True,
              ablate_archetype=True, encode_entity_skills=True)
    b0, e0 = collect(net, 32, seed=1, level=5, pool=[], p_script=1.0, net_kwargs=nk)
    b1, e1 = collect(net, 32, seed=1, level=5, pool=pool, p_script=0.0, net_kwargs=nk)
    assert b0["rewards"].size >= 32 and b1["rewards"].size >= 32
    assert np.isfinite(b0["advantages"]).all() and np.isfinite(b1["advantages"]).all()
    print(f"  collect base(eps={e0}) + mixed-all-snapshot(eps={e1}) 皆有限、shape OK", flush=True)
    print("── smoke 全過:可以開訓 ──", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--eval_only", default="", help="給 checkpoint 路徑,只跑 5×5 矩陣")
    p.add_argument("--phase", choices=["base", "mixed"], default="base",
                   help="base=純專家腳本對手;mixed=腳本+凍結快照池混合")
    p.add_argument("--out_dir", default="models/exp_selfplay5_base")
    p.add_argument("--updates", type=int, default=60)
    p.add_argument("--steps", type=int, default=2048)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--ent_coef", type=float, default=0.01)
    p.add_argument("--value_warmup", type=int, default=3)
    p.add_argument("--level", type=int, default=5)
    p.add_argument("--eval_every", type=int, default=5)
    p.add_argument("--eval_games", type=int, default=48)
    p.add_argument("--matrix_games", type=int, default=48, help="最終 5×5 每格局數")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--threads", type=int, default=8)
    # 混合自我對弈參數(僅 --phase mixed)
    p.add_argument("--p_script", type=float, default=0.5,
                   help="mixed:每局對手是腳本專家的機率(強錨防漂移);其餘從快照池抽")
    p.add_argument("--k_refresh", type=int, default=10,
                   help="mixed:每幾 updates 把當前權重存一份進快照池")
    p.add_argument("--pool_size", type=int, default=5, help="mixed:快照池保留最近幾份")
    # 架構旗標(預設 = v05;一般不要動)
    p.add_argument("--skill_combo_dim", type=int, default=8)
    p.add_argument("--ablate_immunity_joins", type=int, default=0)
    p.add_argument("--drop_noop_h", type=int, default=1)
    p.add_argument("--no_arch", type=int, default=1, help="v05=1(noarch)")
    p.add_argument("--enemy_skills", type=int, default=1, help="v05=1(entity_skills)")
    p.add_argument("--init_from", default="",
                   help="warm-start:載入此 checkpoint 權重(fresh optimizer,strict=False)。"
                        "mixed 階段應指向 base 末檔——它同時是快照池的第 1 份")
    args = p.parse_args()

    if args.smoke:
        smoke(); return

    net_kwargs = dict(skill_combo_dim=args.skill_combo_dim,
                      ablate_immunity_joins=bool(args.ablate_immunity_joins),
                      drop_noop_h=bool(args.drop_noop_h),
                      ablate_archetype=bool(args.no_arch),
                      encode_entity_skills=bool(args.enemy_skills))

    if args.eval_only:
        net = build_net(**net_kwargs)
        net.load_state_dict(torch.load(args.eval_only, map_location="cpu"))
        wr = probe_matrix(net, args.matrix_games, args.level)
        _report_matrix(f"eval_only {Path(args.eval_only).name}", wr, args.matrix_games)
        return

    torch.set_num_threads(args.threads)
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    net = build_net(**net_kwargs)

    # 快照池:mixed 階段 warm-start 來源即池的第 1 份(池從不空/冷)。
    pool: list = []
    if args.init_from:
        sd = torch.load(args.init_from, map_location="cpu")
        missing, unexpected = net.load_state_dict(sd, strict=False)
        print(f"warm-start:已載入 {args.init_from}(fresh optimizer);"
              f"新增(留原始初始化)={list(missing)} 忽略={list(unexpected)}", flush=True)
        if args.phase == "mixed":
            pool.append({k: v.detach().cpu().clone() for k, v in net.state_dict().items()})
    if args.phase == "mixed" and not pool:
        # 沒給 init_from 也要有起點,否則早期是「兩個隨機網互毆」(冷啟動病態)。
        pool.append(_snapshot(net))
        print("警告:mixed 未給 --init_from → 快照池以隨機初始網起步(冷啟動,不建議)", flush=True)

    p_script = args.p_script if args.phase == "mixed" else 1.0
    optim = torch.optim.Adam(net.parameters(), lr=args.lr)
    n_par = sum(p_.numel() for p_ in net.parameters())
    opp_desc = ("純專家腳本" if args.phase == "base"
                else f"混合 p_script={p_script} + 快照池(k={args.k_refresh}/N={args.pool_size})")
    print(f"[{args.phase}] v05 params={n_par} _ent_emb_dim={net._ent_emb_dim} | "
          f"5職隨機發模型+對手 L{args.level} 1v1 open | 對手={opp_desc} | "
          f"workers={args.workers}", flush=True)

    from trpg.rl.exp_parallel import ExpParallel
    from trpg.rl import architectures as A
    pc = ExpParallel("exp_scratch_selfplay5", workers=args.workers,
                     scripts_dir=os.path.dirname(os.path.abspath(__file__)),
                     net_kwargs=net_kwargs)

    _report_per_class("u0", probe_per_class(net, args.eval_games, args.level))
    try:
        for update in range(1, args.updates + 1):
            batch, neps = pc.run(net, args.steps, args.seed * 7919 + update,
                                 args.level, pool, p_script, net_kwargs)
            net.train()
            info = ppo_update(net, batch, optim, n_epochs=args.epochs,
                              batch_size=args.batch, ent_coef=args.ent_coef,
                              device="cpu", value_only=(update <= args.value_warmup))
            mean_r = float(batch["rewards"].sum() / max(1, neps))
            snap_note = ""
            if args.phase == "mixed" and update % args.k_refresh == 0:
                refresh_pool(pool, net, args.pool_size)
                snap_note = f" [snapshot→pool({len(pool)})]"
            print(f"U{update:3d}/{args.updates}{' [wu]' if update <= args.value_warmup else ''} "
                  f"eps={neps} R/ep={mean_r:+.2f} pol={info['policy_loss']:+.3f} "
                  f"val={info['value_loss']:.2f} ent={info['entropy']:.3f}{snap_note}", flush=True)
            if update % args.eval_every == 0:
                ckpt = out_dir / f"sp5_u{update:04d}.pt"
                A.save_net(net, str(ckpt), CODENAME)      # 帶 v05 sidecar
                _report_per_class(f"u{update}",
                                  probe_per_class(net, args.eval_games, args.level))
    finally:
        pc.close()

    wr = probe_matrix(net, args.matrix_games, args.level)
    _report_matrix(f"final [{args.phase}]", wr, args.matrix_games)
    print("done.", flush=True)


if __name__ == "__main__":
    main()
