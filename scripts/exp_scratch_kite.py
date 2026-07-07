"""實驗 02：從零 PPO — 遠程角色能不能自己學會「拉開距離、放風箏」。

戰士實驗(exp_scratch_fighter.py)的鏡像:那次是純近戰要學會「走過去打」,
這次是遠程(assassin,短弓)要學會「離開」。對手是一個**會攻擊的戰士**
(HeuristicCombatPolicy:搆得到就砍、搆不到才朝我走)——但**移動資源砍半**
(每回合最多推進 4.5m,模型是 9m)→ 只要模型肯拉開距離,戰士永遠貼不上、
挨打的是戰士。這是設計成「風箏可贏」的乾淨 regime,用來量**現行架構從零能不能
自己發現風箏**(這是之後換架構要打敗的 baseline)。

身分選 assassin:紀錄裡它是最差的風箏者(project_generalize_grid:assassin −18.7、
「model bow-only」),最有意義的 benchmark;它有短弓遠程、也有短劍近戰,所以
「有沒有學會離開」不能只看勝率(近戰互毆也可能贏),必須看**距離**。

要觀察的行為(行為探針,非只看 loss):
  - 均距 / 最近距 → 該保持大(拉開了);最近距若 <1.5 表示被戰士搆到過
  - 遠程攻擊% → 攻擊時距離 >1.5m 的比例(真的從遠處放)
  - 末 HP → 模型存活血量(風箏成功=少挨打=高血量)
  - 勝率 → 打死戰士
獎勵用引擎預設(HP-PBRS + 勝負 + 亂走懲罰;wasted-move 已豁免改變接戰距離的
移動 → 風箏後撤本來就不罰)。wmc=0 沿用戰士實驗驗證過的從零可學配置。

用法(參數與戰士實驗相同):
  TRPG_WASTED_MOVE_COST=0 python scripts/exp_scratch_kite.py --updates 60 \
      --out_dir models/exp_kite
"""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding="utf-8")
import warnings; warnings.filterwarnings("ignore")

import argparse, random
from pathlib import Path
import numpy as np
import torch

import trpg.rl.obs  # noqa: F401  freeze N_ARCHETYPES before any registry use
from trpg.rl.env_v2 import CombatEnvV2
from trpg.rl.model import (CombatPolicyNet, CombatPolicyNetNoArch,
                           apply_resource_mask, apply_entity_mask, pick_action)
from trpg.rl.train_ppo import _sample_action, _compute_gae, ppo_update
from trpg.rl.obs import build_obs
from trpg.engine.skill import available_skills, TargetType
from trpg.engine.combat_policy import HeuristicCombatPolicy
from trpg.engine.combat import MOVE_BUDGET_M

RANGED = "assassin"         # 遠程(短弓);也有短劍近戰 → 用距離判斷有沒有學會離開
WARRIOR = "champion"        # 對手:純近戰長劍戰士(搆到就砍)
MELEE_REACH_M = 1.5         # 長劍/一般近戰觸及;均距>此=戰士搆不到
_OFFENSE_TT = (TargetType.SINGLE_ENEMY, TargetType.MULTI_ENEMY,
               TargetType.POINT, TargetType.LINE, TargetType.CONE)


class SlowWarrior:
    """會朝我走近、搆得到就砍的戰士,但每回合移動預算砍半(move_frac=0.5)。

    非侵入式減速:HeuristicCombatPolicy 每回合被 env 發滿 MOVE_BUDGET_M(9m),
    我在 decide 入口把它 clamp 到一半 → 戰士每回合最多只能推進 4.5m。clamp 冪等
    (回合內 movement 只減不增;回合開始被 env 重設回 9 → 首個子動作即壓回 4.5)。
    攻擊行為完全交給 HeuristicCombatPolicy(搆得到就砍),滿足「戰士要會攻擊」。
    """
    def __init__(self, move_frac=0.5):
        self._inner = HeuristicCombatPolicy()
        self._cap = MOVE_BUDGET_M * move_frac

    def decide(self, opp_id, opp, ws, resources, round_number):
        if resources.get("movement", 0.0) > self._cap:
            resources["movement"] = self._cap
        return self._inner.decide(opp_id, opp, ws, resources, round_number)


def make_env(seed, level, opp_level, move_frac):
    env = CombatEnvV2(seed=seed, n_agents=1, n_opps=1)
    env.reset(agent_archs=[RANGED], opp_archs=[WARRIOR],
              level=level, opp_level=opp_level, layout="open")
    for oid in env.opp_ids:                       # 覆蓋掉腳本對手 → 半速戰士
        env._opp_policies[oid] = SlowWarrior(move_frac)
    return env


def _dist(env):
    a = env.ws.characters[env.agent_ids[0]].position
    o = env.ws.characters[env.opp_ids[0]].position
    return ((a.x - o.x) ** 2 + (a.y - o.y) ** 2) ** 0.5


# ── rollout(與戰士實驗完全相同)────────────────────────────────────────────────

def collect(net, n_steps, seed, level, opp_level, move_frac, gamma=0.99, lam=0.95):
    net.eval()
    obs_l, act_l, lp_l, rew_l, val_l, done_l = [], [], [], [], [], []
    skm_l, enm_l, grm_l = [], [], []
    ep = 0
    while len(rew_l) < n_steps:
        env = make_env(seed * 1_000_003 + ep, level, opp_level, move_frac)
        obs = build_obs(env.ws, env.current_agent_id, env.resources)
        done = False
        while not done:
            aid = env.current_agent_id            # 恆為 agent(戰士對手由引擎內部跑)
            ot = {k: torch.from_numpy(v).unsqueeze(0) for k, v in obs.items()}
            with torch.no_grad():
                action, lp, val, skm, enm, grm = _sample_action(
                    net, ot, env.resources, env.ws, aid)
            a = action.numpy().tolist()
            obs2, r, term, trunc, _ = env.step(a)
            obs_l.append(obs); act_l.append(a); lp_l.append(lp.numpy())
            rew_l.append(float(r)); val_l.append(val)
            skm_l.append(skm.numpy()); enm_l.append(enm.numpy())
            grm_l.append(grm.numpy())
            done = term or trunc
            done_l.append(bool(done))
            obs = obs2
        ep += 1
    rewards = np.array(rew_l, np.float32); values = np.array(val_l, np.float32)
    dones = np.array(done_l, np.float32)
    adv, ret = _compute_gae(rewards, values, dones, 0.0, gamma, lam)
    batch = {
        "obs": {k: np.stack([o[k] for o in obs_l]) for k in obs_l[0]},
        "actions": np.array(act_l, np.int64),
        "log_probs": np.array(lp_l, np.float32),
        "skill_masks": np.stack(skm_l).astype(np.bool_),
        "entity_masks": np.stack(enm_l).astype(np.bool_),
        "grid_masks": np.stack(grm_l).astype(np.bool_),
        "rewards": rewards, "values": values,
        "returns": ret, "advantages": adv, "dones": dones,
    }
    return batch, ep


# ── 行為探針(貪婪、固定種子)──風箏度量 ────────────────────────────────────────

def probe(net, n_games, level, opp_level, move_frac, max_steps=200):
    net.eval()
    wins = 0
    atk_tot = atk_ranged = 0
    mean_dists, min_dists, end_hps = [], [], []
    trace0 = None
    for gi in range(n_games):
        random.seed(10_000 + gi)                  # 種全域引擎骰 → 可重現
        env = make_env(1_000 + gi, level, opp_level, move_frac)
        aid = env.agent_ids[0]; oid = env.opp_ids[0]
        done, steps, seq = False, 0, []
        step_dists, gmin = [], 1e9
        while not done and steps < max_steps:
            steps += 1
            actor = env.current_agent_id
            d_now = _dist(env)
            step_dists.append(d_now); gmin = min(gmin, d_now)
            obs = build_obs(env.ws, actor, env.resources)
            ot = {k: torch.from_numpy(v).unsqueeze(0) for k, v in obs.items()}
            with torch.no_grad():
                el, s, e, g = net(ot)
            s = apply_resource_mask(s, env.resources, env.ws, actor)
            e = apply_entity_mask(e, ot, env.ws, actor)
            act = list(pick_action(el[0], s[0], e[0], g[0], ws=env.ws,
                                   agent_id=actor))
            sks = available_skills(env.ws.characters[actor], env.ws)
            sk = sks[act[0]] if 0 < act[0] < len(sks) else None
            seq.append(sk.skill_id if sk is not None else "END")
            if (sk is not None and getattr(sk.features, "expected_damage", 0) > 0
                    and sk.features.target_type in _OFFENSE_TT):
                atk_tot += 1
                if d_now > MELEE_REACH_M + 1e-6:  # 距離判斷:從近戰觸及外攻擊=遠程放
                    atk_ranged += 1
            _, _, term, trunc, _ = env.step(act)
            done = term or trunc
        wins += (not env.ws.characters[oid].is_alive()
                 and env.ws.characters[aid].is_alive())
        mean_dists.append(float(np.mean(step_dists)) if step_dists else 0.0)
        min_dists.append(gmin if gmin < 1e9 else 0.0)
        mc = env.ws.characters[aid]
        end_hps.append(max(0.0, mc.hp) / max(1, mc.max_hp))
        if trace0 is None:
            trace0 = seq[:16]
    net.train()
    rng = (atk_ranged / atk_tot) if atk_tot else 0.0
    return dict(wr=wins / n_games, atk=atk_tot / n_games, ranged=rng,
                mdist=float(np.mean(mean_dists)),
                mindist=float(np.mean(min_dists)),
                endhp=float(np.mean(end_hps)), trace=trace0)


def _report(tag, m):
    print(f"[{tag:<4}] WR={m['wr']:3.0%} 攻擊/局={m['atk']:.1f}(遠程{m['ranged']:3.0%}) "
          f"均距={m['mdist']:4.1f}m 最近={m['mindist']:4.1f}m 末HP={m['endhp']:3.0%}  "
          f"trace={m['trace']}", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out_dir", default="models/exp_kite")
    p.add_argument("--updates", type=int, default=60)
    p.add_argument("--steps", type=int, default=2048)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--ent_coef", type=float, default=0.01)
    p.add_argument("--value_warmup", type=int, default=5)
    p.add_argument("--level", type=int, default=3)
    p.add_argument("--opp_level", type=int, default=3)   # 對等 L3 → 戰士是真威脅、風箏才有意義
    p.add_argument("--move_frac", type=float, default=0.5)  # 戰士移動 = 模型的一半
    p.add_argument("--eval_every", type=int, default=5)
    p.add_argument("--eval_games", type=int, default=16)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no_arch", action="store_true",
                   help="用 CombatPolicyNetNoArch(無職業編碼)取代母類別")
    args = p.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    net = (CombatPolicyNetNoArch(hidden=128) if args.no_arch
           else CombatPolicyNet(hidden=128, n_head_groups=1))   # 隨機初始化,無 warm
    optim = torch.optim.Adam(net.parameters(), lr=args.lr)
    n_par = sum(p_.numel() for p_ in net.parameters())
    print(f"fresh {'NoArch' if args.no_arch else 'base'} net params={n_par} | "
          f"{RANGED} L{args.level} vs 半速戰士 {WARRIOR} L{args.opp_level} "
          f"(move×{args.move_frac}) | open", flush=True)

    m = probe(net, args.eval_games, args.level, args.opp_level, args.move_frac)
    _report("u0", m)

    for update in range(1, args.updates + 1):
        batch, neps = collect(net, args.steps, args.seed * 7919 + update,
                              args.level, args.opp_level, args.move_frac)
        net.train()
        info = ppo_update(net, batch, optim, n_epochs=args.epochs,
                          batch_size=args.batch, ent_coef=args.ent_coef,
                          device="cpu",
                          value_only=(update <= args.value_warmup))
        mean_r = float(batch["rewards"].sum() / max(1, neps))   # 每局總獎勵
        print(f"U{update:3d}/{args.updates}"
              f"{' [wu]' if update <= args.value_warmup else ''} "
              f"eps={neps} R/ep={mean_r:+.2f} pol={info['policy_loss']:+.3f} "
              f"val={info['value_loss']:.2f} ent={info['entropy']:.3f}",
              flush=True)
        if update % args.eval_every == 0:
            torch.save(net.state_dict(), out_dir / f"kite_u{update:04d}.pt")
            m = probe(net, args.eval_games, args.level, args.opp_level,
                      args.move_frac)
            _report(f"u{update}", m)
    print("done.", flush=True)


if __name__ == "__main__":
    main()
