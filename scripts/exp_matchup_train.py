"""Phase 3b — train a general model on matchmaker-balanced, diverse encounters
with an ONLINE strength currency (the C loop).

Loop (single-process first — correctness/observability before parallelism; v05 is
update-bound so parallel rollout only ~halves wall-clock, added later):

  bootstrap StrengthModel  (rough prior: equiv-level + EQUIV_LEVEL_1V1)
  repeat:
    for each rollout episode:
      target ~ sample_imbalance_target()               # mostly fair, imbalance tail
      A, B  = propose_matchup(model, draw_identity, target)   # currency picks a fair-ish B
      play (agent seat = learning net over A;  opp seat = scripts over B, per-entity levels)
      record outcome y (agent-side win) -> outcomes
    ppo_update(net)                                     # improve the policy
    for (a_atoms,b_atoms,y) in outcomes: model.update(...)   # C: re-fit balance to THIS policy
    periodic: fair-band realized-WR health + save

Why online: Phase 1 proved a static balance prior is crude (equal-Σlevel swings
0.11↔0.94 via action economy + L5 Extra Attack cliff); the currency must track the
moving policy. See project_matchup_balance / scripts/exp_matchup_calib.py.

Identity space = 12 classes × trainable levels + 1v1-viable monsters (natural_level).
All have scripts, so the opponent seat is scripted (self-play for script-less ids is
a later extension; the snapshot-pool machinery is proven in exp_scratch_selfplay5).
"""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout.reconfigure(encoding="utf-8")
import warnings; warnings.filterwarnings("ignore")
os.environ.setdefault("TRPG_WASTED_MOVE_COST", "0")

import argparse, math, random
from pathlib import Path
import numpy as np
import torch

import trpg.rl.obs  # noqa: F401  freeze N_ARCHETYPES before registry use
from trpg.rl.env_v2 import CombatEnvV2, ARCHETYPE_LIST
from trpg.rl.model import (CombatPolicyNet, apply_resource_mask,
                           apply_entity_mask, pick_action)
from trpg.rl.train_ppo import _sample_action, _compute_gae, ppo_update
from trpg.rl.obs import build_obs
from trpg.rl.matchup import (StrengthModel, atom_key, propose_matchup,
                             sample_imbalance_target, gap_for_target)
from trpg.scenarios.monsters import (register_monsters, onev1_viable_monsters,
                                     EQUIV_LEVEL_1V1, PARTY3_EQUIV_LEVEL,
                                     MONSTER_DEFS)

register_monsters()
CODENAME = "codeword-noarch-enemyskill"     # v05
CLASS_IDS = list(ARCHETYPE_LIST)            # 12 標準職業（皆有腳本）
# L1 included: meaningful class space is L1–8 (highest feature grant is min_level
# =7; L8 is the top HP tier, nothing new above). L1 is fragile/swingy but valid.
CLASS_LEVELS = (1, 2, 3, 4, 5, 6, 7, 8)
# Party-content monsters: 1v1-equiv is inf, but they have a MEASURED 3-party
# equiv (PARTY3_EQUIV_LEVEL) — the 'strong single' side of a boss-vs-party fight
# (e.g. hill_giant 3.2, troll 4.5). The truly-inf apex (dragons/lich/tarrasque…)
# stays out: it can't be balanced within a ≤3-body team (CR≥10 crosses nowhere
# through 3×L8). Finite-party-equiv only.
PARTY_MONSTERS = {k: v for k, v in PARTY3_EQUIV_LEVEL.items() if math.isfinite(v)}


def build_net(hidden=128, skill_combo_dim=8, ablate_immunity_joins=False,
              drop_noop_h=True, ablate_archetype=True, encode_entity_skills=True):
    return CombatPolicyNet(hidden=hidden, n_head_groups=1,
                           skill_combo_dim=skill_combo_dim,
                           ablate_immunity_joins=ablate_immunity_joins,
                           drop_noop_h=drop_noop_h,
                           ablate_archetype=ablate_archetype,
                           encode_entity_skills=encode_entity_skills)


# ── identity space (injected into the matchmaker; keeps it registry-decoupled) ──
def make_draw_identity(mon_max_level=8.0, p_monster=0.3, p_boss=0.0):
    monsters = onev1_viable_monsters(mon_max_level)
    bosses = list(PARTY_MONSTERS)          # party-content 'strong single' band

    def draw(rng):
        if bosses and rng.random() < p_boss:
            mid = rng.choice(bosses)
            return (mid, MONSTER_DEFS[mid].natural_level, True)
        if monsters and rng.random() < p_monster:
            mid = rng.choice(monsters)
            return (mid, MONSTER_DEFS[mid].natural_level, True)
        return (rng.choice(CLASS_IDS), rng.choice(CLASS_LEVELS), False)

    return draw


def bootstrap_model(mon_max_level=8.0):
    m = StrengthModel(k_max=4)
    viable = set(onev1_viable_monsters(mon_max_level))
    m.bootstrap(CLASS_IDS, CLASS_LEVELS,
                {k: v for k, v in EQUIV_LEVEL_1V1.items() if k in viable},
                party_monster_equiv=PARTY_MONSTERS)   # seeded as full-party strength
    return m


# ── env build from matchmaker specs (specs = list of (arch, level, is_monster)) ──
def env_from_specs(seed, a_specs, b_specs, opp_net=None):
    a_archs = [s[0] for s in a_specs]; a_lvls = [s[1] for s in a_specs]
    b_archs = [s[0] for s in b_specs]; b_lvls = [s[1] for s in b_specs]
    env = CombatEnvV2(seed=seed, n_agents=len(a_specs), n_opps=len(b_specs))
    if opp_net is not None:                       # self-play: snapshot drives B seat
        env.use_self_play_opponent(opp_net, blind=True)   # (before reset — see drive_episode)
    env.reset(agent_archs=a_archs, opp_archs=b_archs,
              agent_levels=a_lvls, opp_levels=b_lvls, layout="open")
    return env


def _snapshot(net):
    return {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}


def refresh_pool(pool, net, pool_size):
    pool.append(_snapshot(net)); del pool[:-pool_size]
    return pool


def selfplay_frac(seat_bias, thresh, scale, cap):
    """Self-play fraction ramps as the agent OUTGROWS scripts (seat_bias rises).
    Weak agent (seat_bias≤thresh) → 0 (learn from scripts via the curriculum);
    as seat_bias climbs past thresh, hand more episodes to same-skill snapshots so
    balance stays composition-driven instead of seat_bias running away."""
    return max(0.0, min(cap, (seat_bias - thresh) / scale))


def _outcome_y(env):
    """1 agent-side win / 0 loss / 0.5 draw, from the AGENT seat."""
    team = any(env.ws.characters[a].is_alive() for a in env.agent_ids)
    opp = any(env.ws.characters[o].is_alive() for o in env.opp_ids)
    if team and not opp:
        return 1.0
    if opp and not team:
        return 0.0
    return 0.5


# ── single-process rollout: matchmaker builds each episode, returns outcomes ────
_BUF = ("obs", "act", "lp", "rew", "val", "done", "skm", "enm", "grm")


def collect_local(net, n_steps, seed, model, draw, sizes, max_size_gap,
                  pool=None, p_selfplay=0.0, net_kwargs=None,
                  gamma=0.99, lam=0.95, tail_gap=None):
    net.eval()
    rng = random.Random(seed)
    opp_shell = build_net(**(net_kwargs or {})) if pool else None
    if opp_shell is not None:
        opp_shell.eval()
    buf = {k: [] for k in _BUF}
    outcomes = []          # (a_atoms, b_atoms, y, target, is_selfplay)
    ep = 0
    while len(buf["rew"]) < n_steps:
        use_sp = bool(pool) and rng.random() < p_selfplay
        # self-play opp skill tracks the agent → no persistent gap → seat_bias 0;
        # a script opp is fixed-skill → use the learned self.seat_bias.
        sb = 0.0 if use_sp else None
        target = sample_imbalance_target(rng)
        # fair fights stay tight; lopsided tail fights may widen to tail_gap so
        # 1-vs-many boss-vs-party can appear (default tail_gap=max_size_gap → off).
        gap = gap_for_target(target, fair_gap=max_size_gap,
                             tail_gap=max_size_gap if tail_gap is None else tail_gap)
        a_specs, b_specs, _p, _t = propose_matchup(
            model, rng, draw, sizes=sizes, n_candidates=32,
            target=target, max_size_gap=gap, seat_bias=sb)
        opp_net = None
        if use_sp:
            opp_shell.load_state_dict(pool[rng.randrange(len(pool))])
            opp_net = opp_shell
        random.seed(seed * 2_246_822_519 + ep)      # engine dice = global RNG
        env = env_from_specs(seed * 1_000_003 + ep, a_specs, b_specs, opp_net=opp_net)
        obs = build_obs(env.ws, env.current_agent_id, env.resources)
        done = False
        while not done:
            aid = env.current_agent_id
            ot = {k: torch.from_numpy(v).unsqueeze(0) for k, v in obs.items()}
            with torch.no_grad():
                action, lp, val, skm, enm, grm = _sample_action(
                    net, ot, env.resources, env.ws, aid)
            obs2, r, term, trunc, _ = env.step(action.numpy().tolist())
            buf["obs"].append(obs); buf["act"].append(action.numpy().tolist())
            buf["lp"].append(lp.numpy()); buf["rew"].append(float(r))
            buf["val"].append(val); buf["skm"].append(skm.numpy())
            buf["enm"].append(enm.numpy()); buf["grm"].append(grm.numpy())
            done = term or trunc; buf["done"].append(bool(done))
            obs = obs2
        outcomes.append(([atom_key(*s) for s in a_specs],
                         [atom_key(*s) for s in b_specs], _outcome_y(env),
                         target, use_sp))
        ep += 1
    rew = np.array(buf["rew"], np.float32); val = np.array(buf["val"], np.float32)
    done = np.array(buf["done"], np.float32)
    adv, ret = _compute_gae(rew, val, done, 0.0, gamma, lam)
    batch = {
        "obs": {k: np.stack([o[k] for o in buf["obs"]]) for k in buf["obs"][0]},
        "actions": np.array(buf["act"], np.int64),
        "log_probs": np.array(buf["lp"], np.float32),
        "skill_masks": np.stack(buf["skm"]).astype(np.bool_),
        "entity_masks": np.stack(buf["enm"]).astype(np.bool_),
        "grid_masks": np.stack(buf["grm"]).astype(np.bool_),
        "rewards": rew, "values": val, "returns": ret, "advantages": adv,
        "dones": done,
    }
    return batch, ep, outcomes


# ── ExpParallel 契約：模組頂層 collect(net, n_steps, seed, *rest) → 3-tuple ────────
# rest 全部可 pickle（spawn）：model 傳 to_dict、draw 傳參數在 worker 內重建（closure
# 不可 pickle）。回 (batch, ep, outcomes)；outcomes 由 run_aux 收回主程序做 model.update
# （model 是主程序單一 owner，worker 只讀快照提對戰、不改貨幣）。
def collect(net, n_steps, seed, model_dict, draw_params, sizes, max_size_gap,
            tail_gap, pool, p_selfplay, net_kwargs):
    model = StrengthModel.from_dict(model_dict)
    draw = make_draw_identity(*draw_params)
    return collect_local(net, n_steps, seed, model, draw, sizes, max_size_gap,
                         pool=pool, p_selfplay=p_selfplay, net_kwargs=net_kwargs,
                         tail_gap=tail_gap)


# ── health: among ~fair proposals, is the realised agent WR actually ~0.5? ──────
def fair_band_health(outcomes, band=0.08, selfplay=None):
    """Realised agent WR among ~fair (target≈0.5) proposals. selfplay=None: all;
    True/False: only self-play / only script episodes."""
    ys = [o[2] for o in outcomes if abs(o[3] - 0.5) <= band
          and (selfplay is None or o[4] == selfplay)]
    if not ys:
        return None
    return sum(ys) / len(ys), len(ys)


# ── Phase 4: guardrail probe (GREEDY policy, not the noisy rollout outcomes) ────
def _greedy_action(net, env, actor):
    ob = build_obs(env.ws, actor, env.resources)
    ot = {k: torch.from_numpy(v).unsqueeze(0) for k, v in ob.items()}
    with torch.no_grad():
        el, s, e, g = net(ot)
    s = apply_resource_mask(s, env.resources, env.ws, actor)
    e = apply_entity_mask(e, ot, env.ws, actor)
    return list(pick_action(el[0], s[0], e[0], g[0], ws=env.ws, agent_id=actor))


def _play_greedy(net, a_specs, b_specs, seed, opp_net=None, max_steps=300):
    random.seed(seed)
    env = env_from_specs(seed, a_specs, b_specs, opp_net=opp_net)
    done = False; steps = 0; term = trunc = False
    while not done and steps < max_steps:
        steps += 1
        _, _, term, trunc, _ = env.step(_greedy_action(net, env, env.current_agent_id))
        done = term or trunc
    mh = np.mean([max(0., env.ws.characters[c].hp) / max(1, env.ws.characters[c].max_hp)
                  for c in list(env.agent_ids) + list(env.opp_ids)])
    return {"y": _outcome_y(env), "steps": steps,
            "timeout": bool(trunc and not term), "mutual_hp": float(mh)}


def probe_health(net, model, draw, n, sizes, gap, pool, net_kwargs, seed):
    """Three guardrails on the GREEDY policy (rollout WR is stochastic):
      balance   — greedy WR on target=0.5 proposals should sit ~0.5;
      capability— greedy WR by team-shape + monster-involved (competent everywhere?);
      co-collapse— self-play greedy timeout%/mutual-末血 (both sides turtling?)."""
    net.eval()
    rng = random.Random(seed)
    from collections import defaultdict
    allwr = []; by_shape = defaultdict(list); by_kind = defaultdict(list)
    for i in range(n):
        a, b, _p, _t = propose_matchup(model, rng, draw, sizes=sizes,
                                       n_candidates=48, target=0.5,
                                       max_size_gap=gap, seat_bias=model.seat_bias)
        y = _play_greedy(net, a, b, seed + i * 131)["y"]
        allwr.append(y)
        by_shape[f"{len(a)}v{len(b)}"].append(y)
        by_kind["有怪" if any(s[2] for s in a + b) else "純職"].append(y)
    degen = None
    if pool:
        shell = build_net(**(net_kwargs or {})); shell.eval()
        shell.load_state_dict(pool[-1])
        K = max(8, n // 3); to = dec = 0.0; mh = []
        for i in range(K):
            a, b, _p, _t = propose_matchup(model, rng, draw, sizes=sizes,
                                           n_candidates=48, target=0.5,
                                           max_size_gap=gap, seat_bias=0.0)
            r = _play_greedy(net, a, b, seed + 9000 + i * 7, opp_net=shell)
            to += r["timeout"]; dec += (not r["timeout"]); mh.append(r["mutual_hp"])
        degen = {"timeout": to / K, "decisive": dec / K, "mutual_hp": float(np.mean(mh))}
    net.train()
    return {"balance": sum(allwr) / len(allwr), "n": len(allwr),
            "by_shape": {k: (sum(v) / len(v), len(v)) for k, v in by_shape.items()},
            "by_kind": {k: (sum(v) / len(v), len(v)) for k, v in by_kind.items()},
            "degen": degen}


def _report_health(tag, h):
    shape = " ".join(f"{k}:{v[0]:.0%}(n{v[1]})" for k, v in sorted(h["by_shape"].items()))
    kind = " ".join(f"{k}:{v[0]:.0%}" for k, v in h["by_kind"].items())
    line = (f"[{tag}] 平衡 greedyWR={h['balance']:.0%}(n{h['n']}, 越近50%越好) "
            f"| 形狀 {shape} | {kind}")
    if h["degen"]:
        d = h["degen"]
        flag = "⚠共塌" if (d["timeout"] > 0.40 or d["mutual_hp"] > 0.60) else ""
        line += (f" | 自對局 打完{d['decisive']:.0%}/逾時{d['timeout']:.0%}/"
                 f"末血{d['mutual_hp']:.0%}{flag}")
    print(line, flush=True)


def smoke():
    print("── smoke: v05 / matchmaker / online currency / multi-agent rollout ──", flush=True)
    net = build_net()
    assert net._ent_emb_dim == 96 and net.ablate_archetype and net.encode_entity_skills
    model = bootstrap_model()
    draw = make_draw_identity()
    # draw covers classes + monsters
    rng = random.Random(0)
    kinds = [draw(rng)[2] for _ in range(200)]
    assert any(kinds) and not all(kinds), "draw 應同時出職業與怪物"
    # propose respects sizes + gap
    a, b, p, t = propose_matchup(model, rng, draw, sizes=(1, 2, 3), max_size_gap=1)
    assert abs(len(a) - len(b)) <= 1
    print(f"  sample matchup: A={a} vs B={b}  model_p={p:.2f} target={t:.2f}", flush=True)
    # env builds from mixed specs (per-entity levels honoured)
    env = env_from_specs(1, [("champion", 6, False), ("goblin", 1, True)],
                         [("ogre", 2, True)])
    lv = [env.ws.characters[a_].level for a_ in env.agent_ids]
    assert lv == [6, 1], f"per-entity 等級該是 [6,1] 得 {lv}"
    print(f"  mixed env OK: agent levels={lv} (champ L6 + goblin natural L1)", flush=True)
    # collect runs, returns outcomes, model.update moves ratings
    b0, e0, outs = collect_local(net, 48, 1, model, draw, (1, 2, 3), 1)
    assert b0["rewards"].size >= 48 and len(outs) == e0
    assert np.isfinite(b0["advantages"]).all()
    r_before = dict(model.r)
    for a_at, b_at, y, _t, _sp in outs:
        model.update(a_at, b_at, y)
    changed = sum(1 for k in model.r if model.r[k] != r_before.get(k))
    print(f"  collect eps={e0} outcomes={len(outs)}; model.update 動了 {changed} 個 atom", flush=True)
    # self-play path: pool given + p_selfplay=1 → every opp is a snapshot
    pool = [_snapshot(net)]
    _b, _e, outs_sp = collect_local(net, 48, 2, model, draw, (1, 2, 3), 1,
                                    pool=pool, p_selfplay=1.0, net_kwargs={})
    assert all(o[4] for o in outs_sp), "p_selfplay=1 應每局都是 self-play"
    assert selfplay_frac(-3, 0, 1, 0.6) == 0.0 and selfplay_frac(2, 0, 1, 0.6) == 0.6
    print(f"  self-play 路徑 OK:{len(outs_sp)} 局全 self-play;"
          f" selfplay_frac 觸發器 seat<thresh→0 / 高→cap", flush=True)
    # Phase 4 guardrail probe runs + reports all three axes
    h = probe_health(net, model, draw, 12, (1, 2, 3), 1, pool, {}, seed=7)
    assert 0.0 <= h["balance"] <= 1.0 and h["by_shape"] and h["degen"] is not None
    _report_health("smoke", h)
    print("── smoke 全過 ──", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--out_dir", default="models/exp_matchup_train")
    p.add_argument("--updates", type=int, default=80)
    p.add_argument("--steps", type=int, default=2048)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--ent_coef", type=float, default=0.01)
    p.add_argument("--value_warmup", type=int, default=3)
    p.add_argument("--sizes", default="1,2,3")
    p.add_argument("--max_size_gap", type=int, default=1,
                   help="人數差上限（fair 局）")
    p.add_argument("--tail_gap", type=int, default=None,
                   help="不平衡尾巴局的人數差上限（放行 1v3 boss-vs-party；預設=max_size_gap 即關）")
    p.add_argument("--p_monster", type=float, default=0.3)
    p.add_argument("--p_boss", type=float, default=0.0,
                   help="抽 party-content 強怪（hill_giant 等）的機率；預設 0 即關")
    p.add_argument("--mon_max_level", type=float, default=8.0)
    p.add_argument("--eval_every", type=int, default=5)
    p.add_argument("--health_games", type=int, default=48,
                   help="Phase4 守門探針每次 greedy 局數")
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--workers", type=int, default=16,
                   help="平行 rollout worker 數（進程級）；0/1＝單進程。rollout＝batch=1 "
                        "單樣本 forward，靠進程平行吃多核（torch 執行緒/GPU 對它無用）")
    p.add_argument("--device", default="auto",
                   help="ppo_update 的 device（auto→有 cuda 就用）。v05 是 update-bound，"
                        "batched ppo_update 在 GPU 快 ~7.5x；rollout 仍在 CPU worker（batch=1 GPU 無用）")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--init_from", default="")
    # self-play (triggered by seat_bias = "agent has outgrown the scripts")
    p.add_argument("--sp_thresh", type=float, default=0.0,
                   help="seat_bias 超過此值才開始摻 self-play 對手")
    p.add_argument("--sp_scale", type=float, default=1.5)
    p.add_argument("--sp_cap", type=float, default=0.6, help="self-play 上限比例")
    p.add_argument("--k_refresh", type=int, default=5, help="每幾 updates 存一份快照")
    p.add_argument("--pool_size", type=int, default=5)
    args = p.parse_args()

    if args.smoke:
        smoke(); return

    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    sizes = tuple(int(x) for x in args.sizes.split(","))
    device = ("cuda" if torch.cuda.is_available() else "cpu") \
        if args.device == "auto" else args.device

    net = build_net()
    if args.init_from:
        sd = torch.load(args.init_from, map_location="cpu")
        miss, unexp = net.load_state_dict(sd, strict=False)
        print(f"warm-start {args.init_from}: +{list(miss)} ignore={list(unexp)}", flush=True)
    # net 常駐 device；optim 建在 .to(device) 後（否則 Adam state device 與 param 不符）。
    # rollout（batch=1、GPU 無用）在 CPU：parallel 由 worker 的 cpu state_dict 跑（run_aux
    # 內 .cpu() 複製，net 留 device）；單進程/probe（主程序直接餵 cpu obs）時暫移 net 回 cpu。
    net.to(device)
    model = bootstrap_model(args.mon_max_level)
    draw = make_draw_identity(args.mon_max_level, args.p_monster, args.p_boss)
    optim = torch.optim.Adam(net.parameters(), lr=args.lr)
    n_par = sum(x.numel() for x in net.parameters())
    _tg = args.max_size_gap if args.tail_gap is None else args.tail_gap
    # 進程級平行 rollout（rollout-bound；見 trpg/rl/exp_parallel）。model 留主程序單一
    # owner：worker 只讀 model 快照提對戰、回 outcomes；主程序 run_aux 收回做 update。
    pc = None
    if args.workers and args.workers > 1:
        from trpg.rl.exp_parallel import ExpParallel
        pc = ExpParallel("exp_matchup_train", workers=args.workers,
                         scripts_dir=os.path.dirname(os.path.abspath(__file__)),
                         net_kwargs={})
    draw_params = (args.mon_max_level, args.p_monster, args.p_boss)
    print(f"[matchup-train] v05 params={n_par} | sizes={sizes} gap≤{args.max_size_gap}"
          f"(tail≤{_tg}) | identity=12職×L{CLASS_LEVELS} + "
          f"{len(onev1_viable_monsters(args.mon_max_level))}怪+{len(PARTY_MONSTERS)}強怪"
          f"(p_mon={args.p_monster},p_boss={args.p_boss}) | "
          f"{'workers='+str(args.workers) if pc else '單進程'} | ppo@{device}", flush=True)

    pool = [_snapshot(net)]        # never empty; used only once seat_bias > sp_thresh
    from trpg.rl import architectures as A
    for update in range(1, args.updates + 1):
        p_sp = selfplay_frac(model.seat_bias, args.sp_thresh, args.sp_scale, args.sp_cap)
        base_seed = args.seed * 7919 + update
        if pc is not None:                       # net 留 device；run_aux 內 .cpu() 給 worker
            batch, neps, outs = pc.run_aux(
                net, args.steps, base_seed, model.to_dict(), draw_params, sizes,
                args.max_size_gap, args.tail_gap, pool, p_sp, {})
        else:                                     # 單進程 rollout：主程序餵 cpu obs → net 回 cpu
            net.to("cpu")
            batch, neps, outs = collect_local(
                net, args.steps, base_seed, model, draw, sizes,
                args.max_size_gap, pool=pool, p_selfplay=p_sp, net_kwargs={},
                tail_gap=args.tail_gap)
        net.train()
        info = ppo_update(net, batch, optim, n_epochs=args.epochs,   # ppo_update 內 net.to(device)
                          batch_size=args.batch, ent_coef=args.ent_coef,
                          device=device, value_only=(update <= args.value_warmup))
        # C: re-fit currency. Script games train seat_bias (agent-vs-fixed-skill
        # gap); self-play games DON'T (opp tracks agent → gap ~0) — they train only
        # r/g, the purest composition-strength signal at equal skill.
        for a_at, b_at, y, _t, is_sp in outs:
            model.update(a_at, b_at, y, update_seat_bias=not is_sp)
        if update % args.k_refresh == 0:
            refresh_pool(pool, net, args.pool_size)
        mean_r = float(batch["rewards"].sum() / max(1, neps))
        n_sp = sum(o[4] for o in outs)
        hs = fair_band_health(outs, selfplay=False)   # script fair-band WR
        hp = fair_band_health(outs, selfplay=True)     # self-play fair-band WR
        htxt = (f"scriptWR={hs[0]:.2f}(n{hs[1]})" if hs else "scriptWR=–")
        htxt += (f" spWR={hp[0]:.2f}(n{hp[1]})" if hp else "")
        print(f"U{update:3d}/{args.updates}{' [wu]' if update <= args.value_warmup else ''} "
              f"eps={neps} R/ep={mean_r:+.2f} pol={info['policy_loss']:+.3f} "
              f"val={info['value_loss']:.2f} ent={info['entropy']:.3f} "
              f"seat={model.seat_bias:+.2f} p_sp={p_sp:.2f}({n_sp}/{neps}) | {htxt}",
              flush=True)
        if update % args.eval_every == 0:
            net.to("cpu")            # save + probe（batch=1 greedy）在 cpu；不 step optim 故安全
            A.save_net(net, str(out_dir / f"mt_u{update:04d}.pt"), CODENAME)
            model.save(out_dir / f"strength_u{update:04d}.json")
            _report_health(f"u{update}", probe_health(
                net, model, draw, args.health_games, sizes, args.max_size_gap,
                pool, {}, seed=90_000 + update))
            net.to(device)
    if pc is not None:
        pc.close()
    print("done.", flush=True)


if __name__ == "__main__":
    main()
