"""實驗：從零 PPO — 模型能不能「無指導」自己學會打贏腳本專家（先測 battle_master 鏡像）。

和 exp_scratch_fighter 的差別：對手席不再是站樁靶，而是**引擎預設掛的腳本
BattleMasterPolicy 專家**（會機動 maneuver → 普攻 → action_surge 爆發、且會還手）。
agent 席自己也打 battle_master，等級對等＝公平鏡像。問題：隨機初始化的
CombatPolicyNet（codeword、無手寫免疫）純 PPO（無 BC / 無 warm / 無 blind）能不能
自然追平／超過專家？

「打贏專家」的量尺：專家 vs 專家鏡像本身有先手優勢，不是 50%。所以開跑前先量
一次 **expert-vs-expert 基準 WR**（腳本專家坐 agent 席、對手席同樣腳本專家），
那條線才是 fresh net 要追平的目標，不是 50%。

架構（使用者指令：往後實驗預設）＝ codeword 全網 CombatPolicyNet
(n_head_groups=1, skill_combo_dim=8)，手寫免疫結構性移除
(ablate_immunity_joins + drop_noop_h + 決策層 mask_immune_null 關)。battle_master
鏡像本就無免疫，這些對行為是 no-op，只為與架構指令一致。

行為探針（貪婪、固定種子）：
  - WR / 平手率        → 追平專家基準即達標
  - 每局攻擊次數        → >0（有接戰）
  - 己方 / 敵方末血%    → 看是打贏還是被打贏
  - 划水率 idle        → 首手就結束回合（action 沒花）＝退化訊號
  - 平均末距            → 有沒有走過去貼身

用法：
  python scripts/exp_scratch_battlemaster.py --updates 120
  python scripts/exp_scratch_battlemaster.py --updates 120 --level 5 --wmc 0
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
from trpg.rl.model import (CombatPolicyNet, apply_resource_mask,
                           apply_entity_mask, pick_action)
from trpg.rl.train_ppo import _sample_action, _compute_gae, ppo_update
from trpg.rl.exp_parallel import ExpParallel
from trpg.rl.obs import build_obs
from trpg.rl.action import encode_action
from trpg.engine.skill import available_skills, TargetType
from trpg.engine.combat_policy import make_archetype_policy

AGENT = "battle_master"     # 模型駕的身分
OPP = "battle_master"       # 對手席＝引擎自動掛的腳本 BattleMasterPolicy 專家（鏡像）
_OFFENSE_TT = (TargetType.SINGLE_ENEMY, TargetType.MULTI_ENEMY,
               TargetType.POINT, TargetType.LINE, TargetType.CONE)


def build_net(skill_combo_dim=8):
    """架構指令：codeword 全網、無手寫免疫（ablate + drop_noop_h）。主程序與 worker 共用。"""
    return CombatPolicyNet(hidden=128, n_head_groups=1,
                           skill_combo_dim=skill_combo_dim,
                           ablate_immunity_joins=True, drop_noop_h=True)


def make_env(seed, level, opp_level, wmc):
    """不覆蓋 _opp_policies → 對手席保持引擎預設的腳本專家（會還手）。"""
    env = CombatEnvV2(seed=seed, n_agents=1, n_opps=1)
    env.reset(agent_archs=[AGENT], opp_archs=[OPP],
              level=level, opp_level=opp_level, layout="open")
    env._wasted_move_cost = wmc     # fresh 探索期預設 0：避免亂走懲罰擋住「學會移動」
    return env


def _dist(env):
    a = env.ws.characters[env.agent_ids[0]].position
    o = env.ws.characters[env.opp_ids[0]].position
    return ((a.x - o.x) ** 2 + (a.y - o.y) ** 2) ** 0.5


# ── rollout（對手席在 env.step 內部自動跑，迴圈只驅動 agent 席）───────────────────

def collect(net, n_steps, seed, level, opp_level, wmc, gamma=0.99, lam=0.95):
    net.eval()
    obs_l, act_l, lp_l, rew_l, val_l, done_l = [], [], [], [], [], []
    skm_l, enm_l, grm_l = [], [], []
    ep = 0
    while len(rew_l) < n_steps:
        env = make_env(seed * 1_000_003 + ep, level, opp_level, wmc)
        obs = build_obs(env.ws, env.current_agent_id, env.resources)
        done = False
        while not done:
            aid = env.current_agent_id
            ot = {k: torch.from_numpy(v).unsqueeze(0) for k, v in obs.items()}
            with torch.no_grad():
                # mask_immune_null=False：與「無手寫免疫」架構一致（鏡像無免疫，no-op）
                action, lp, val, skm, enm, grm = _sample_action(
                    net, ot, env.resources, env.ws, aid, mask_immune_null=False)
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


# 平行 rollout 用共用原語 trpg.rl.exp_parallel.ExpParallel（把本檔的 collect() 拆多進程）。
# 契約＝本模組頂層需有 build_net(**kw) 與 collect(net, n_steps, seed, *rest)，見該檔說明。


# ── 專家 vs 專家基準（腳本專家駕 agent 席，對手席引擎腳本）＝要追平的目標線 ────────

def expert_baseline(n_games, level, opp_level, max_steps=400):
    pol = make_archetype_policy(AGENT)
    wins = draws = 0
    for gi in range(n_games):
        random.seed(20_000 + gi)
        env = make_env(30_000 + gi, level, opp_level, wmc=0.0)
        aid = env.agent_ids[0]
        done, steps = False, 0
        while not done and steps < max_steps:
            steps += 1
            actor = env.current_agent_id
            a = env.ws.characters[actor]
            dec = pol.decide(actor, a, env.ws, env.resources,
                             env.ws.combat.round_number)
            if dec.action is None or getattr(dec, "fled", False):
                act = [0, 0, 0]
            else:
                act = list(encode_action(dec.action, env.ws, actor))
            _, _, term, trunc, _ = env.step(act)
            done = term or trunc
        win = (not env.ws.characters[env.opp_ids[0]].is_alive()
               and env.ws.characters[aid].is_alive())
        draw = (not win and env.ws.characters[aid].is_alive())
        wins += win; draws += draw
    return wins / n_games, draws / n_games


# ── 行為探針（貪婪、固定種子）─────────────────────────────────────────────────

def probe(net, n_games, level, opp_level, max_steps=400):
    net.eval()
    wins = draws = atks = idle = 0
    dists, my_hp, opp_hp = [], [], []
    trace0 = None
    for gi in range(n_games):
        random.seed(10_000 + gi)                  # 種全域引擎骰 → 可重現
        env = make_env(1_000 + gi, level, opp_level, wmc=0.0)
        aid = env.agent_ids[0]
        oid = env.opp_ids[0]
        done, steps, atk, seq = False, 0, 0, []
        prev_actor = None
        while not done and steps < max_steps:
            steps += 1
            actor = env.current_agent_id
            first = actor != prev_actor
            prev_actor = actor
            had_action = env.resources.get("action", 0) > 0
            obs = build_obs(env.ws, actor, env.resources)
            ot = {k: torch.from_numpy(v).unsqueeze(0) for k, v in obs.items()}
            with torch.no_grad():
                el, s, e, g = net(ot)
            s = apply_resource_mask(s, env.resources, env.ws, actor,
                                    mask_immune_null=False)
            e = apply_entity_mask(e, ot, env.ws, actor, mask_immune_null=False)
            act = list(pick_action(el[0], s[0], e[0], g[0], ws=env.ws,
                                   agent_id=actor))
            if first and act[0] == 0 and had_action:
                idle += 1
            sks = available_skills(env.ws.characters[actor], env.ws)
            sk = sks[act[0]] if 0 < act[0] < len(sks) else None
            seq.append(sk.skill_id if sk is not None else "END")
            if (sk is not None and getattr(sk.features, "expected_damage", 0) > 0
                    and sk.features.target_type in _OFFENSE_TT):
                atk += 1
            _, _, term, trunc, _ = env.step(act)
            done = term or trunc
        agent_alive = env.ws.characters[aid].is_alive()
        opp_alive = env.ws.characters[oid].is_alive()
        wins += (not opp_alive and agent_alive)
        draws += (opp_alive and agent_alive)
        atks += atk
        dists.append(_dist(env))
        my_hp.append(max(0, env.ws.characters[aid].hp) / env.ws.characters[aid].max_hp)
        opp_hp.append(max(0, env.ws.characters[oid].hp) / env.ws.characters[oid].max_hp)
        if trace0 is None:
            trace0 = seq[:16]
    net.train()
    return dict(wr=wins / n_games, draw=draws / n_games,
                atk=atks / n_games, idle=idle / n_games,
                dist=float(np.mean(dists)), my_hp=float(np.mean(my_hp)),
                opp_hp=float(np.mean(opp_hp)), trace=trace0)


def _fmt(pr):
    return (f"WR={pr['wr']:3.0%} 平手={pr['draw']:3.0%} 攻擊/局={pr['atk']:.1f} "
            f"划水={pr['idle']:.1f} 己末血={pr['my_hp']:3.0%} 敵末血={pr['opp_hp']:3.0%} "
            f"末距={pr['dist']:4.1f}m")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out_dir", default="models/exp_battlemaster")
    p.add_argument("--updates", type=int, default=120)
    p.add_argument("--steps", type=int, default=2048)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--ent_coef", type=float, default=0.01)
    p.add_argument("--value_warmup", type=int, default=5)
    p.add_argument("--level", type=int, default=5)
    p.add_argument("--opp_level", type=int, default=5)
    p.add_argument("--wmc", type=float, default=0.0,
                   help="wasted-move cost；fresh 探索期預設 0，避免擋住『學會移動』")
    p.add_argument("--eval_every", type=int, default=5)
    p.add_argument("--eval_games", type=int, default=48)  # 記憶：判讀一律 n>=48
    p.add_argument("--skill_combo_dim", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--workers", type=int, default=16,
                   help="平行 rollout 的 worker 進程數（各單執行緒）；1=退回單進程")
    p.add_argument("--threads", type=int, default=8,
                   help="主程序 torch 執行緒數（只 ppo_update 用得到；rollout 靠 workers 平行）")
    args = p.parse_args()

    # rollout 是 batch=1 小網路逐步 forward，多執行緒反而變慢；主程序只在 ppo_update
    # 用得到執行緒。真正的核心利用靠 --workers 個單執行緒 worker 平行 rollout。
    torch.set_num_threads(max(1, args.threads))

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    net = build_net(args.skill_combo_dim)   # codeword 全網、無手寫免疫
    optim = torch.optim.Adam(net.parameters(), lr=args.lr)
    pc = (ExpParallel("exp_scratch_battlemaster", workers=args.workers,
                      scripts_dir=os.path.dirname(os.path.abspath(__file__)),
                      net_kwargs=dict(skill_combo_dim=args.skill_combo_dim))
          if args.workers > 1 else None)
    n_par = sum(p_.numel() for p_ in net.parameters())
    print(f"fresh codeword net params={n_par} | {AGENT} L{args.level} vs "
          f"腳本專家 {OPP} L{args.opp_level} | open | wmc={args.wmc} | "
          f"workers={args.workers} threads={args.threads}", flush=True)

    bw, bd = expert_baseline(args.eval_games, args.level, args.opp_level)
    print(f"[基準] 專家 vs 專家鏡像：WR={bw:3.0%} 平手={bd:3.0%}  "
          f"← fresh net 要追平的目標線（非 50%）", flush=True)

    pr = probe(net, args.eval_games, args.level, args.opp_level)
    print(f"[u0  ] {_fmt(pr)}  trace={pr['trace']}", flush=True)

    for update in range(1, args.updates + 1):
        base_seed = args.seed * 7919 + update
        if pc is not None:
            batch, neps = pc.run(net, args.steps, base_seed,
                                 args.level, args.opp_level, args.wmc)
        else:
            batch, neps = collect(net, args.steps, base_seed,
                                  args.level, args.opp_level, args.wmc)
        net.train()
        info = ppo_update(net, batch, optim, n_epochs=args.epochs,
                          batch_size=args.batch, ent_coef=args.ent_coef,
                          device="cpu",
                          value_only=(update <= args.value_warmup))
        mean_r = float(batch["rewards"].sum() / max(1, neps))
        print(f"U{update:3d}/{args.updates}"
              f"{' [wu]' if update <= args.value_warmup else ''} "
              f"eps={neps} R/ep={mean_r:+.2f} pol={info['policy_loss']:+.3f} "
              f"val={info['value_loss']:.2f} ent={info['entropy']:.3f}",
              flush=True)
        if update % args.eval_every == 0:
            torch.save(net.state_dict(), out_dir / f"bm_u{update:04d}.pt")
            pr = probe(net, args.eval_games, args.level, args.opp_level)
            print(f"[u{update:<3d}] {_fmt(pr)}  trace={pr['trace']}", flush=True)
    if pc is not None:
        pc.close()
    print("done.", flush=True)


if __name__ == "__main__":
    main()
