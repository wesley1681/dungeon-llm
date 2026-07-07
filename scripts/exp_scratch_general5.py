"""實驗:跨職業泛化 — 一個 fresh PPO 網能否同時學會 5 種完全不同的打法。

每局隨機發給模型 5 職之一、對手也隨機發 5 職之一(用**專家腳本**);問一個網能不能
**同時**打好 5 種職業。這不是「訓一個更強的交付模型」(通用混合獎勵會侵蝕,見 uni 線),
是一個**泛化探針**:共享 trunk 表徵能不能表達 5 種相反的最優策略。

5 職業(archetype_id / 專家 policy 見 trpg/engine/combat_policy.py):
  berserker  狂戰士     BerserkerPolicy   衝上去暴怒硬砍(暴怒對物理有抗性)
  evocation  塑能法師   EvocationPolicy   火球/魔導/火焰箭,拉距離放 AoE
  assassin   刺客       AssassinPolicy    短弓風箏 + 潛行 + assassinate
  vengeance  復仇聖騎   VengeancePolicy   額外攻擊 + 神聖斥責 smite
  war        戰爭牧師   WarClericPolicy   靈體守護/靈魂武器 + 補血續戰

設定(全部查證,非猜):
  - 架構 = codeword 全網 CombatPolicyNet(n_head_groups=1, skill_combo_dim=8),**不用
    NoArch**(照架構指令)。ablate_immunity_joins=False 保留手刻免疫/抗性 join——狂戰士
    暴怒對物理有抗性,這條 join 讀得到該留;drop_noop_h=True 拿掉 skill/entity 頭死權重。
    raw build_obs 不歸零 self 的 archetype one-hot,故網**看得到自己職業**(+ skill
    codeword)。所以這裡「泛化」= 一個 trunk 能否表達 5 種打法(多任務),不是盲猜隱藏職業。
  - env = 1v1 open,雙方同 level(預設 5:狂戰士 frenzy/暴怒、法師火球、刺客 assassinate、
    聖騎士額外攻擊+smite、牧師靈體守護+靈魂武器 都上線)。對手 policy 由 env.reset
    自動指派 make_archetype_policy(opp_arch)=專家(env_v2.py:258),對手席零額外工程。
  - 獎勵 = 引擎預設(env_v2.py:300-311,635-644):Φ = 己方HP% − 敵方HP%,dense=3.0·ΔΦ,
    終局 +5 勝 / −5 敗 / −10 逾時。**1v1 下 war cleric 補自己 → 己方HP%↑ → Φ↑ → 直接
    有正獎勵**,所以輔助職不需特製獎勵。TRPG_WASTED_MOVE_COST=0(fresh 別讓 wmc 擋探索)。

判讀:交付物 = **5×5 對局勝率矩陣**(模型職 × 對手職,含鏡像對角)。成功 = 沒有任何模型
職業塌到 ~0、5 條 WR 曲線一起爬。必盯:**狂戰士**(我的紀錄標「拒訓」,BC/DAgger 脈絡);
PPO 下未必重演,但它 WR 若塌,先查根因、別當泛化失敗。判讀一律 n≥48/格。

用法:
  python scripts/exp_scratch_general5.py --smoke                       # 先驗機制
  python scripts/exp_scratch_general5.py --updates 60 --out_dir models/exp_general5
  python scripts/exp_scratch_general5.py --eval_only models/exp_general5/g5_u0060.pt  # 只跑 5×5
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
ARCHS = ("berserker", "evocation", "assassin", "vengeance", "war")
ARCH_NAME = {"berserker": "狂戰士", "evocation": "塑能法", "assassin": "刺客",
             "vengeance": "復仇騎", "war": "戰爭牧"}


# ── 架構契約(ExpParallel 要求模組頂層有 build_net) ──────────────────────────────
def build_net(hidden=128, skill_combo_dim=8, ablate_immunity_joins=False,
              drop_noop_h=True, ablate_archetype=False, encode_entity_skills=False):
    return CombatPolicyNet(hidden=hidden, n_head_groups=1,
                           skill_combo_dim=skill_combo_dim,
                           ablate_immunity_joins=ablate_immunity_joins,
                           drop_noop_h=drop_noop_h,
                           ablate_archetype=ablate_archetype,
                           encode_entity_skills=encode_entity_skills)


def make_env(seed, agent_arch, opp_arch, level):
    """1v1 open;對手 policy 由 env.reset 自動 = make_archetype_policy(opp_arch) 專家。"""
    env = CombatEnvV2(seed=seed, n_agents=1, n_opps=1)
    env.reset(agent_archs=[agent_arch], opp_archs=[opp_arch],
              level=level, opp_level=level, layout="open")
    return env


# ── rollout(ExpParallel 契約:collect(net, n_steps, seed, *rest) -> (batch, neps)) ──
_BUF_KEYS = ("obs", "act", "lp", "rew", "val", "done", "skm", "enm", "grm")


def collect(net, n_steps, seed, level, gamma=0.99, lam=0.95):
    net.eval()
    rng = random.Random(seed)
    buf = {k: [] for k in _BUF_KEYS}
    ep = 0
    while len(buf["rew"]) < n_steps:
        a_arch = rng.choice(ARCHS)          # 每局獨立均勻抽:模型職 + 對手職
        o_arch = rng.choice(ARCHS)
        random.seed(seed * 2_246_822_519 + ep)   # 引擎骰用全域 random,固定它讓 rollout 可重現
        env = make_env(seed * 1_000_003 + ep, a_arch, o_arch, level)
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


# ── 探針(greedy) ──────────────────────────────────────────────────────────────
def _greedy_action(net, env, actor):
    ob = build_obs(env.ws, actor, env.resources)
    ot = {k: torch.from_numpy(v).unsqueeze(0) for k, v in ob.items()}
    with torch.no_grad():
        el, s, e, g = net(ot)
    s = apply_resource_mask(s, env.resources, env.ws, actor)
    e = apply_entity_mask(e, ot, env.ws, actor)
    return list(pick_action(el[0], s[0], e[0], g[0], ws=env.ws, agent_id=actor))


def _play(net, seed, a_arch, o_arch, level, max_steps=200):
    """跑一局 greedy,回傳 (won, 己方末血%, 敵方末血%)。"""
    random.seed(seed)
    env = make_env(seed, a_arch, o_arch, level)
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
    """便宜探針:每模型職 n 局,對手每局隨機抽。回傳 {arch: (wr, 己血, 敵血)}。"""
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
    """交付物:完整 5×5(模型職 × 對手職)勝率矩陣。回傳 wr[a][o]。"""
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
    print(f"\n=== {tag} 5×5 勝率矩陣 (n={n}/格;列=模型職,欄=對手職) ===", flush=True)
    header = "模型\\對手 " + "".join(f"{ARCH_NAME[o]:>7}" for o in ARCHS) + "   平均"
    print(header, flush=True)
    for a in ARCHS:
        row = [wr[a][o] for o in ARCHS]
        cells = "".join(f"{v:6.0%} " for v in row)
        print(f"{ARCH_NAME[a]:<8}" + cells + f"  {np.mean(row):5.0%}", flush=True)
    col_avg = [np.mean([wr[a][o] for a in ARCHS]) for o in ARCHS]
    print("對手均擋 " + "".join(f"{1-v:6.0%} " for v in col_avg)
          + f"  總{np.mean([wr[a][o] for a in ARCHS for o in ARCHS]):5.0%}", flush=True)


# ── 機制自檢 ──────────────────────────────────────────────────────────────────
def smoke():
    print("── smoke:5 職 env / 專家對手 / reward / forward ──", flush=True)
    net = build_net()
    n_par = sum(p.numel() for p in net.parameters())
    print(f"CombatPolicyNet(codeword,ablate=F,drop_h=T) params={n_par} "
          f"skill_head.in={net.skill_heads[0].weight.shape[1]} "
          f"entity_head.in={net.entity_heads[0].weight.shape[1]}", flush=True)

    for a_arch in ARCHS:
        o_arch = ARCHS[(ARCHS.index(a_arch) + 1) % len(ARCHS)]
        env = make_env(123, a_arch, o_arch, level=5)
        aid = env.agent_ids[0]; oid = env.opp_ids[0]
        A = env.ws.characters[aid]; W = env.ws.characters[oid]
        opp_pol = type(env._opp_policies[oid]).__name__
        sks = [s.skill_id for s in available_skills(A, env.ws)]
        # 跑幾步:agent greedy,對手專家由 env.step 內部走;確認 reward 有限、局面推進
        rtot = 0.0; steps = 0; done = False
        while not done and steps < 12:
            steps += 1
            act = _greedy_action(net, env, env.current_agent_id)
            _, r, term, trunc, _ = env.step(act); rtot += r; done = term or trunc
        assert np.isfinite(rtot), "reward 該有限"
        print(f"  {ARCH_NAME[a_arch]}(L{A.level} HP{A.max_hp} AC{A.ac}) vs "
              f"{ARCH_NAME[o_arch]}[{opp_pol}] | 技能{len(sks)}招 | "
              f"{steps}步 R累={rtot:+.2f} 己血{A.hp:.0f} 敵血{W.hp:.0f}", flush=True)
        assert opp_pol != "HeuristicCombatPolicy", f"{o_arch} 該拿到專家 policy 非 heuristic"

    # 專家對手身分正確性:make_archetype_policy 對每職都給對的專家類別
    want = {"berserker": "BerserkerPolicy", "evocation": "EvocationPolicy",
            "assassin": "AssassinPolicy", "vengeance": "VengeancePolicy",
            "war": "WarClericPolicy"}
    for arch, cls in want.items():
        got = type(make_archetype_policy(arch)).__name__
        assert got == cls, f"{arch} 專家該是 {cls} 得到 {got}"
    print(f"  專家對應正確:{want}", flush=True)
    print("── smoke 全過:可以開訓 ──", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--eval_only", default="", help="給 checkpoint 路徑,只跑 5×5 矩陣")
    p.add_argument("--out_dir", default="models/exp_general5")
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
    p.add_argument("--skill_combo_dim", type=int, default=8)
    p.add_argument("--ablate_immunity_joins", type=int, default=0,
                   help="0=保留手刻免疫/抗性 join(狂戰士暴怒物理抗性讀得到);1=拔掉")
    p.add_argument("--drop_noop_h", type=int, default=1,
                   help="1=skill/entity 頭拿掉死權重常數 h;0=保留")
    p.add_argument("--no_arch", type=int, default=0,
                   help="1=ablate_archetype:架構中結構性移除職業(FiLM整條拿掉、entity_mlp/"
                        "end_head 切掉職業欄、不讀職業路由)→ 只能靠技能 codeword 推身分;"
                        "0=保留職業(baseline)。非遮罩,是欄位/模組不存在")
    p.add_argument("--enemy_skills", type=int, default=0,
                   help="1=encode_entity_skills:讀每個實體的靜態 kit(obs v8),用共享 skill "
                        "encoder 總結進 ent_emb → 分得出聚合相同的敵人(=架構 v05);0=關(v04)")
    p.add_argument("--init_from", default="",
                   help="續訓/warm-start:訓練前載入此 checkpoint 權重(fresh optimizer);"
                        "架構須相符或為超集(v05 從 v04 warm-start 時 entity_kit_proj 留 zero-init)")
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
        _report_matrix(f"eval_only {Path(args.eval_only).name}"
                       f"{' [no_arch]' if args.no_arch else ''}", wr, args.matrix_games)
        return

    torch.set_num_threads(args.threads)
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    net = build_net(**net_kwargs)
    if args.init_from:
        sd = torch.load(args.init_from, map_location="cpu")
        missing, unexpected = net.load_state_dict(sd, strict=False)
        print(f"續訓/warm-start:已載入 {args.init_from}(fresh optimizer);"
              f"新增(留原始初始化)={list(missing)} 忽略={list(unexpected)}", flush=True)
    optim = torch.optim.Adam(net.parameters(), lr=args.lr)
    n_par = sum(p_.numel() for p_ in net.parameters())
    print(f"fresh CombatPolicyNet(codeword) params={n_par} combo_dim={args.skill_combo_dim} "
          f"ablate_imm={bool(args.ablate_immunity_joins)} drop_h={bool(args.drop_noop_h)} "
          f"no_arch={bool(args.no_arch)} | "
          f"5職隨機發模型+對手(專家腳本) L{args.level} 1v1 open | "
          f"workers={args.workers}", flush=True)

    from trpg.rl.exp_parallel import ExpParallel
    pc = ExpParallel("exp_scratch_general5", workers=args.workers,
                     scripts_dir=os.path.dirname(os.path.abspath(__file__)),
                     net_kwargs=net_kwargs)

    _report_per_class("u0", probe_per_class(net, args.eval_games, args.level))
    try:
        for update in range(1, args.updates + 1):
            batch, neps = pc.run(net, args.steps, args.seed * 7919 + update, args.level)
            net.train()
            info = ppo_update(net, batch, optim, n_epochs=args.epochs,
                              batch_size=args.batch, ent_coef=args.ent_coef,
                              device="cpu", value_only=(update <= args.value_warmup))
            mean_r = float(batch["rewards"].sum() / max(1, neps))
            print(f"U{update:3d}/{args.updates}{' [wu]' if update <= args.value_warmup else ''} "
                  f"eps={neps} R/ep={mean_r:+.2f} pol={info['policy_loss']:+.3f} "
                  f"val={info['value_loss']:.2f} ent={info['entropy']:.3f}", flush=True)
            if update % args.eval_every == 0:
                torch.save(net.state_dict(), out_dir / f"g5_u{update:04d}.pt")
                _report_per_class(f"u{update}",
                                  probe_per_class(net, args.eval_games, args.level))
    finally:
        pc.close()

    wr = probe_matrix(net, args.matrix_games, args.level)
    _report_matrix("final" + (" [no_arch]" if args.no_arch else ""), wr, args.matrix_games)
    print("done.", flush=True)


if __name__ == "__main__":
    main()
