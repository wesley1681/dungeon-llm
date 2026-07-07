"""實驗 01：從零 PPO — 戰士能不能自己學會「走過去、打站樁靶」。

最乾淨的 sanity check：隨機初始化的 CombatPolicyNet，純 PPO（無 BC / 無 warm /
無 blind），單一身分 champion（純近戰，沒有遠程可作弊）對一個「站樁」對手
（每回合什麼都不做）。地圖固定 open（先排除牆/視線）。

要觀察的行為（行為探針，非只看 loss）：
  - 平均最終距離 → 該縮小到武器射程內（走得過去）
  - 每局攻擊次數 → 該 > 0（到了會打）
  - 勝率        → 殺掉靶（打得死）
獎勵用引擎預設（HP-PBRS + 勝負 + 亂走懲罰），這套本來就塑形「靠近→攻擊」。

用法：
  python scripts/exp_scratch_fighter.py --updates 60
  python scripts/exp_scratch_fighter.py --updates 120 --lr 3e-4 --out_dir models/exp_fighter
"""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding="utf-8")
import warnings; warnings.filterwarnings("ignore")

import argparse, random
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch

import trpg.rl.obs  # noqa: F401  freeze N_ARCHETYPES before any registry use
from trpg.rl.env_v2 import CombatEnvV2
from trpg.rl.model import (CombatPolicyNet, CombatPolicyNetNoArch,
                           apply_resource_mask, apply_entity_mask, pick_action)
from trpg.rl.train_ppo import _sample_action, _compute_gae, ppo_update
from trpg.rl.obs import build_obs
from trpg.engine.skill import available_skills, TargetType

FIGHTER = "champion"        # 純近戰戰士（沒有遠程 → 一定得走過去）
DUMMY = "champion"          # 靶也用 champion，opp_level 壓低 → 好殺
_OFFENSE_TT = (TargetType.SINGLE_ENEMY, TargetType.MULTI_ENEMY,
               TargetType.POINT, TargetType.LINE, TargetType.CONE)


class StationaryPolicy:
    """站樁：每回合什麼都不做（_run_opponent_turn 見 action=None 就 break）。"""
    def decide(self, opp_id, opp, ws, resources, round_number):
        return SimpleNamespace(action=None, fled=False, ended=True)


def make_env(seed, level, opp_level):
    env = CombatEnvV2(seed=seed, n_agents=1, n_opps=1)
    env.reset(agent_archs=[FIGHTER], opp_archs=[DUMMY],
              level=level, opp_level=opp_level, layout="open")
    for oid in env.opp_ids:                       # 覆蓋掉腳本對手 → 純站樁
        env._opp_policies[oid] = StationaryPolicy()
    return env


def _dist(env):
    a = env.ws.characters[env.agent_ids[0]].position
    o = env.ws.characters[env.opp_ids[0]].position
    return ((a.x - o.x) ** 2 + (a.y - o.y) ** 2) ** 0.5


# ── rollout ───────────────────────────────────────────────────────────────────

def collect(net, n_steps, seed, level, opp_level, gamma=0.99, lam=0.95):
    net.eval()
    obs_l, act_l, lp_l, rew_l, val_l, done_l = [], [], [], [], [], []
    skm_l, enm_l, grm_l = [], [], []
    ep = 0
    while len(rew_l) < n_steps:
        env = make_env(seed * 1_000_003 + ep, level, opp_level)
        obs = build_obs(env.ws, env.current_agent_id, env.resources)
        done = False
        while not done:
            aid = env.current_agent_id            # 恆為 agent（對手站樁、引擎內部跑）
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


# ── 行為探針（貪婪、固定種子）─────────────────────────────────────────────────

def probe(net, n_games, level, opp_level, max_steps=200):
    net.eval()
    wins = atks = 0
    dists = []
    trace0 = None
    for gi in range(n_games):
        random.seed(10_000 + gi)                  # 種全域引擎骰 → 可重現
        env = make_env(1_000 + gi, level, opp_level)
        aid = env.agent_ids[0]
        done, steps, atk, seq = False, 0, 0, []
        while not done and steps < max_steps:
            steps += 1
            actor = env.current_agent_id
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
                atk += 1
            _, _, term, trunc, _ = env.step(act)
            done = term or trunc
        wins += (not env.ws.characters[env.opp_ids[0]].is_alive()
                 and env.ws.characters[aid].is_alive())
        atks += atk
        dists.append(_dist(env))
        if trace0 is None:
            trace0 = seq[:14]
    net.train()
    return (wins / n_games, atks / n_games,
            float(np.mean(dists)), trace0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out_dir", default="models/exp_fighter")
    p.add_argument("--updates", type=int, default=60)
    p.add_argument("--steps", type=int, default=2048)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--ent_coef", type=float, default=0.01)
    p.add_argument("--value_warmup", type=int, default=5)
    p.add_argument("--level", type=int, default=3)
    p.add_argument("--opp_level", type=int, default=1)
    p.add_argument("--eval_every", type=int, default=5)
    p.add_argument("--eval_games", type=int, default=16)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no_arch", action="store_true",
                   help="用 CombatPolicyNetNoArch(無職業編碼)取代母類別")
    args = p.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    net = (CombatPolicyNetNoArch(hidden=128) if args.no_arch
           else CombatPolicyNet(hidden=128, n_head_groups=1))   # 隨機初始化，無 warm
    optim = torch.optim.Adam(net.parameters(), lr=args.lr)
    n_par = sum(p_.numel() for p_ in net.parameters())
    print(f"fresh {'NoArch' if args.no_arch else 'base'} net params={n_par} | "
          f"{FIGHTER} L{args.level} vs 站樁 {DUMMY} L{args.opp_level} | open",
          flush=True)

    wr, ak, ds, tr = probe(net, args.eval_games, args.level, args.opp_level)
    print(f"[u0  ] WR={wr:3.0%} 攻擊/局={ak:.1f} 平均末距={ds:5.1f}m  "
          f"trace={tr}", flush=True)

    for update in range(1, args.updates + 1):
        batch, neps = collect(net, args.steps, args.seed * 7919 + update,
                              args.level, args.opp_level)
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
            torch.save(net.state_dict(), out_dir / f"fighter_u{update:04d}.pt")
            wr, ak, ds, tr = probe(net, args.eval_games, args.level,
                                   args.opp_level)
            print(f"[u{update:<3d}] WR={wr:3.0%} 攻擊/局={ak:.1f} "
                  f"平均末距={ds:5.1f}m  trace={tr}", flush=True)
    print("done.", flush=True)


if __name__ == "__main__":
    main()
