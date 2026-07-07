"""實驗 03：混合 — 一個「無職業」網能否同時學到兩種相反的空間行為。

把實驗 01(近戰 champion 對站樁靶 → 該靠近)和實驗 02(遠程 assassin 對半速戰士
→ 該後撤)**均勻混合**在同一個訓練迴圈裡,用 `CombatPolicyNetNoArch`(無任何職業
編碼)。關鍵問題:模型拿不到職業標籤,只能從**技能池**(champion 只有長劍 / assassin
有短弓)和 HP 等資訊,推斷「這局我該貼上去砍,還是拉開放箭」。若兩種都學會=無職業
架構能靠機制自我路由出相反策略。

rollout 逐局交替兩場景(ep 偶=近戰、奇=遠程),獎勵都用引擎預設。eval 分別量:
  近戰場景 → 末距要小、攻擊>0、WR(靠得近打得死)
  遠程場景 → 均距要大、最近距>觸及、遠程%、末HP(拉得開活得下來)

用法:
  TRPG_WASTED_MOVE_COST=0 python scripts/exp_scratch_mixed.py --updates 80 \
      --out_dir models/exp_mixed_noarch
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
from trpg.rl.model import (CombatPolicyNetNoArch, apply_resource_mask,
                           apply_entity_mask, pick_action)
from trpg.rl.train_ppo import _sample_action, _compute_gae, ppo_update
from trpg.rl.obs import build_obs
from trpg.engine.skill import available_skills, TargetType
from trpg.engine.combat_policy import (HeuristicCombatPolicy, EndTurnPolicy,
                                       CombatDecision)
from trpg.engine.combat import MOVE_BUDGET_M
from trpg.engine.vec2 import Vec2

FIGHTER = "champion"        # 場景A:純近戰,對站樁靶(EndTurnPolicy)→ 該靠近
DUMMY = "champion"
RANGED = "assassin"         # 場景B:遠程,對半速戰士 → 該後撤
WARRIOR = "champion"
MELEE_REACH_M = 1.5
_OFFENSE_TT = (TargetType.SINGLE_ENEMY, TargetType.MULTI_ENEMY,
               TargetType.POINT, TargetType.LINE, TargetType.CONE)


class SlowWarrior:
    """[DEPRECATED — cap 無效] 改 resources['movement'] 但引擎執行 move 不看它
    (逐 1m sub-action 吃角色自己的 9m 預算),實測 cap 1/4.5/9 位移全同 = 全速 9m。
    真限速請用 GenuineSlowWarrior。保留僅為舊 move_frac 路徑相容。"""
    def __init__(self, move_frac=0.5, move_abs=0.0):
        self._inner = HeuristicCombatPolicy()
        self._cap = move_abs if move_abs > 0 else MOVE_BUDGET_M * move_frac

    def decide(self, opp_id, opp, ws, resources, round_number):
        if resources.get("movement", 0.0) > self._cap:
            resources["movement"] = self._cap
        return self._inner.decide(opp_id, opp, ws, resources, round_number)


class GenuineSlowWarrior:
    """真·限速戰士:每『回合』最多朝最近敵人走 move_m 米(動作層 POINT move,
    一回合只准移動一次),到武器 reach 內就攻擊。修掉 SlowWarrior 的 cap 不生效
    (見 [[project_scratch_ppo_fighter]] 07-05 更正)。每個 env 給新實例,round 追蹤才乾淨。"""
    def __init__(self, move_m=1.0):
        self._move_m = move_m
        self._moved_round: dict = {}          # actor_id -> 已移動過的 round_number

    def _nearest_enemy(self, actor_id, ws):
        actor = ws.characters[actor_id]; is_party = ws.is_party_ally(actor_id)
        best, bd = None, 1e9
        for cid, c in ws.characters.items():
            if cid == actor_id or not c.is_alive():
                continue
            if ws.is_party_ally(cid) == is_party:
                continue
            d = actor.position.distance_to(c.position)
            if d < bd:
                bd, best = d, cid
        return best, bd

    def decide(self, actor_id, actor, ws, resources, round_number):
        tid, d = self._nearest_enemy(actor_id, ws)
        if tid is None:
            return CombatDecision(ended=True)
        skills = available_skills(actor, ws)
        wpn = actor.get_weapon() if actor.weapons else None
        reach = (wpn.range_normal or 1.5) if wpn else 1.5
        if d <= reach + 1e-6 and resources.get("action", 0) > 0 and wpn is not None:
            sk = next((s for s in skills if s.skill_id == f"weapon:{wpn.name}"), None)
            if sk is not None:
                return CombatDecision(action=sk.build_action(
                    actor_id, tid, ws.characters[tid].position))
        # 一回合只准移動一次,走 move_m 米(action-level 真限速)
        if (d > reach + 1e-6 and self._moved_round.get(actor_id) != round_number
                and resources.get("movement", 0.0) > 1e-6):
            mv = next((s for s in skills if s.skill_id == "move"), None)
            if mv is not None:
                self._moved_round[actor_id] = round_number
                ap = actor.position; tp = ws.characters[tid].position
                dx, dy = tp.x - ap.x, tp.y - ap.y
                n = (dx*dx + dy*dy) ** 0.5 or 1.0
                step = min(self._move_m, max(0.0, d - reach))
                coord = Vec2(ap.x + dx/n*step, ap.y + dy/n*step)
                return CombatDecision(action=mv.build_action(actor_id, None, coord))
        return CombatDecision(ended=True)


def make_fighter_env(seed, fighter_level, dummy_level):
    env = CombatEnvV2(seed=seed, n_agents=1, n_opps=1)
    env.reset(agent_archs=[FIGHTER], opp_archs=[DUMMY],
              level=fighter_level, opp_level=dummy_level, layout="open")
    for oid in env.opp_ids:
        env._opp_policies[oid] = EndTurnPolicy()     # 站樁(什麼都不做)
    return env


def make_kite_env(seed, ranged_level, warrior_level, move_frac, warrior_hp=0,
                  warrior_move_m=0.0):
    env = CombatEnvV2(seed=seed, n_agents=1, n_opps=1)
    env.reset(agent_archs=[RANGED], opp_archs=[WARRIOR],
              level=ranged_level, opp_level=warrior_level, layout="open")
    for oid in env.opp_ids:
        # warrior_move_m>0 用真·動作層限速(GenuineSlowWarrior);否則走舊 move_frac
        # 路徑(SlowWarrior,cap 無效=全速,僅相容用)。
        env._opp_policies[oid] = (GenuineSlowWarrior(warrior_move_m)
                                  if warrior_move_m > 0
                                  else SlowWarrior(move_frac))
        if warrior_hp > 0:                       # 加厚戰士 → 互毆磨不死 → 逼後撤
            c = env.ws.characters[oid]
            c.max_hp = warrior_hp; c.hp = warrior_hp
    return env


def _dist(env):
    a = env.ws.characters[env.agent_ids[0]].position
    o = env.ws.characters[env.opp_ids[0]].position
    return ((a.x - o.x) ** 2 + (a.y - o.y) ** 2) ** 0.5


# ── rollout:按「步數」平衡兩場景(episode 長短差很大,逐局交替會 90/10 傾斜)──────

_BUF_KEYS = ("obs", "act", "lp", "rew", "val", "done", "skm", "enm", "grm")


def _run_episode(env, net, buf):
    """跑完一局,transition 存進 buf(dict of lists),回傳步數。"""
    obs = build_obs(env.ws, env.current_agent_id, env.resources)
    done = False; n = 0
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
        obs = obs2; n += 1
    return n


def collect(net, n_steps, seed, cfg, gamma=0.99, lam=0.95):
    net.eval()
    buf = {k: [] for k in _BUF_KEYS}
    half = n_steps // 2
    ep = steps_a = steps_b = 0
    while steps_a < half:                       # 場景A:近戰對站樁,收約一半步數
        env = make_fighter_env(seed * 1_000_003 + ep, cfg.level, cfg.dummy_level)
        steps_a += _run_episode(env, net, buf); ep += 1
    while steps_b < half:                       # 場景B:遠程對半速戰士,收另一半
        env = make_kite_env(seed * 1_000_003 + ep, cfg.level,
                            cfg.warrior_level, cfg.move_frac, cfg.warrior_hp,
                            cfg.warrior_move_m)
        steps_b += _run_episode(env, net, buf); ep += 1
    obs_l, act_l, lp_l = buf["obs"], buf["act"], buf["lp"]
    rew_l, val_l, done_l = buf["rew"], buf["val"], buf["done"]
    skm_l, enm_l, grm_l = buf["skm"], buf["enm"], buf["grm"]
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
    return batch, ep, steps_a, steps_b


# ── 探針 ──────────────────────────────────────────────────────────────────────

def _greedy_step(net, env, actor):
    obs = build_obs(env.ws, actor, env.resources)
    ot = {k: torch.from_numpy(v).unsqueeze(0) for k, v in obs.items()}
    with torch.no_grad():
        el, s, e, g = net(ot)
    s = apply_resource_mask(s, env.resources, env.ws, actor)
    e = apply_entity_mask(e, ot, env.ws, actor)
    act = list(pick_action(el[0], s[0], e[0], g[0], ws=env.ws, agent_id=actor))
    sks = available_skills(env.ws.characters[actor], env.ws)
    sk = sks[act[0]] if 0 < act[0] < len(sks) else None
    is_atk = (sk is not None and getattr(sk.features, "expected_damage", 0) > 0
              and sk.features.target_type in _OFFENSE_TT)
    return act, (sk.skill_id if sk is not None else "END"), is_atk


def probe_fighter(net, n_games, cfg, max_steps=200):
    """近戰場景:能否靠近。回報 WR / 攻擊 / 末距(該小)。"""
    net.eval(); wins = atks = 0; dists = []; trace0 = None
    for gi in range(n_games):
        random.seed(20_000 + gi)
        env = make_fighter_env(2_000 + gi, cfg.level, cfg.dummy_level)
        aid = env.agent_ids[0]; done = steps = atk = 0; done = False; seq = []
        while not done and steps < max_steps:
            steps += 1; actor = env.current_agent_id
            act, sid, is_atk = _greedy_step(net, env, actor)
            seq.append(sid); atk += is_atk
            _, _, term, trunc, _ = env.step(act); done = term or trunc
        wins += (not env.ws.characters[env.opp_ids[0]].is_alive()
                 and env.ws.characters[aid].is_alive())
        atks += atk; dists.append(_dist(env))
        if trace0 is None: trace0 = seq[:14]
    net.train()
    return dict(wr=wins / n_games, atk=atks / n_games,
                fdist=float(np.mean(dists)), trace=trace0)


def probe_kite(net, n_games, cfg, max_steps=200):
    """遠程場景:能否後撤。回報 WR / 敗 / 超時 / 遠程% / 均距 / 末HP(己+敵)。"""
    net.eval(); wins = losses = timeouts = atk_tot = atk_rng = 0
    mean_d, min_d, end_hp, foe_hp = [], [], [], []; trace0 = None
    for gi in range(n_games):
        random.seed(30_000 + gi)
        env = make_kite_env(3_000 + gi, cfg.level, cfg.warrior_level,
                            cfg.move_frac, cfg.warrior_hp, cfg.warrior_move_m)
        aid = env.agent_ids[0]; oid = env.opp_ids[0]
        done = False; steps = 0; seq = []; sd = []; gmin = 1e9; trunc = False
        while not done and steps < max_steps:
            steps += 1; actor = env.current_agent_id
            d_now = _dist(env); sd.append(d_now); gmin = min(gmin, d_now)
            act, sid, is_atk = _greedy_step(net, env, actor)
            seq.append(sid)
            if is_atk:
                atk_tot += 1
                if d_now > MELEE_REACH_M + 1e-6: atk_rng += 1
            _, _, term, trunc, _ = env.step(act); done = term or trunc
        A = env.ws.characters[aid]; W = env.ws.characters[oid]
        won = (not W.is_alive() and A.is_alive())
        wins += won
        if not won:
            if not A.is_alive(): losses += 1
            else: timeouts += 1              # 沒贏、自己還活著 = 超時(磨不死)
        mean_d.append(float(np.mean(sd)) if sd else 0.0)
        min_d.append(gmin if gmin < 1e9 else 0.0)
        end_hp.append(max(0.0, A.hp) / max(1, A.max_hp))
        foe_hp.append(max(0.0, W.hp) / max(1, W.max_hp))
        if trace0 is None: trace0 = seq[:16]
    net.train()
    rng = (atk_rng / atk_tot) if atk_tot else 0.0
    return dict(wr=wins / n_games, loss=losses / n_games, to=timeouts / n_games,
                ranged=rng, mdist=float(np.mean(mean_d)),
                mindist=float(np.mean(min_d)), endhp=float(np.mean(end_hp)),
                foehp=float(np.mean(foe_hp)), trace=trace0)


def _report(tag, fa, ki):
    print(f"[{tag:<4}] 近戰:WR={fa['wr']:3.0%} 攻/局={fa['atk']:.1f} 末距={fa['fdist']:4.1f}m"
          f"  ‖  遠程:WR={ki['wr']:3.0%} 敗{ki['loss']:3.0%} 超時{ki['to']:3.0%} "
          f"遠程{ki['ranged']:3.0%} 均距={ki['mdist']:4.1f}m 最近={ki['mindist']:4.1f}m "
          f"己血{ki['endhp']:3.0%} 敵血{ki['foehp']:3.0%}", flush=True)
    print(f"        近戰trace={fa['trace']}", flush=True)
    print(f"        遠程trace={ki['trace']}", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out_dir", default="models/exp_mixed_noarch")
    p.add_argument("--updates", type=int, default=80)
    p.add_argument("--steps", type=int, default=2048)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--ent_coef", type=float, default=0.01)
    p.add_argument("--value_warmup", type=int, default=5)
    p.add_argument("--level", type=int, default=3)          # 模型(兩場景同級)
    p.add_argument("--dummy_level", type=int, default=1)    # 站樁靶
    p.add_argument("--warrior_level", type=int, default=3)  # 半速戰士
    p.add_argument("--warrior_hp", type=int, default=0)     # >0 覆蓋戰士血量(逼後撤)
    p.add_argument("--move_frac", type=float, default=0.5)
    p.add_argument("--warrior_move_m", type=float, default=0.0,
                   help=">0 用絕對公尺覆蓋戰士每回合移動預算(蓋過 move_frac)")
    p.add_argument("--eval_every", type=int, default=5)
    p.add_argument("--eval_games", type=int, default=16)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--skill_combo_dim", type=int, default=0,
                   help="技能組合身分 codeword 維度(0=關閉,B 計劃雙線性項)")
    p.add_argument("--init_from", default="",
                   help="接續訓練:載入這個 checkpoint 的權重(含 critic);Adam 仍重初始化,"
                        "記得同時把 --value_warmup 設 0")
    args = p.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    net = CombatPolicyNetNoArch(hidden=128, skill_combo_dim=args.skill_combo_dim)  # 無職業
    if args.init_from:
        net.load_state_dict(torch.load(args.init_from))
        print(f"resumed weights from {args.init_from}", flush=True)
    optim = torch.optim.Adam(net.parameters(), lr=args.lr)
    n_par = sum(p_.numel() for p_ in net.parameters())
    _whp = f"血{args.warrior_hp}" if args.warrior_hp > 0 else "預設血"
    _wmv = (f"{args.warrior_move_m:g}m/回合" if args.warrior_move_m > 0
            else f"{MOVE_BUDGET_M * args.move_frac:g}m/回合")
    print(f"fresh NoArch net params={n_par} combo_dim={args.skill_combo_dim} | "
          f"MIX: {FIGHTER} vs 站樁L{args.dummy_level} ⊕ {RANGED} vs "
          f"戰士L{args.warrior_level}(移動{_wmv},{_whp}) | open", flush=True)

    fa = probe_fighter(net, args.eval_games, args)
    ki = probe_kite(net, args.eval_games, args)
    _report("u0", fa, ki)

    for update in range(1, args.updates + 1):
        batch, neps, sa, sb = collect(net, args.steps, args.seed * 7919 + update, args)
        net.train()
        info = ppo_update(net, batch, optim, n_epochs=args.epochs,
                          batch_size=args.batch, ent_coef=args.ent_coef,
                          device="cpu", value_only=(update <= args.value_warmup))
        mean_r = float(batch["rewards"].sum() / max(1, neps))
        print(f"U{update:3d}/{args.updates}{' [wu]' if update <= args.value_warmup else ''} "
              f"eps={neps} 步A/B={sa}/{sb} R/ep={mean_r:+.2f} "
              f"pol={info['policy_loss']:+.3f} val={info['value_loss']:.2f} "
              f"ent={info['entropy']:.3f}", flush=True)
        if update % args.eval_every == 0:
            torch.save(net.state_dict(), out_dir / f"mixed_u{update:04d}.pt")
            fa = probe_fighter(net, args.eval_games, args)
            ki = probe_kite(net, args.eval_games, args)
            _report(f"u{update}", fa, ki)
    print("done.", flush=True)


if __name__ == "__main__":
    main()
