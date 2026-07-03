"""Can the grid-head's per-cell features predict per-cell LINE OF SIGHT?

The grid-head scores a move cell C by grid_query(h) · spatial_feat[:,C] — a
purely POINTWISE function of cell C's own features:
    [terrain(C), self/ally/enemy markers(C), dist_to_self(C), dist_to_enemy(C)]
Whether standing at C grants LoS to the enemy depends on whether a WALL lies on
the segment C->enemy — information about OTHER cells, absent from C's feature.

This probe quantifies that gap. For many wall states it pools, per reachable
cell C, (features(C), enemy_xy, self_xy) -> label LoS(C->nearest enemy), then
fits:
  - linear (logistic)         — the grid-head's own expressive class (per-state
                                bilinear collapses to linear-in-cellfeat)
  - MLP (2x64)                — a universal pointwise fn WITH enemy/self pos
  - +true-LoS channel (linear)— sanity: must hit ~100% (no label leak / bug)
If linear AND MLP sit near the majority-class baseline while the sanity fit is
~100%, the needed info is simply NOT in the inputs -> a precomputed per-cell
LoS channel is required (architecture, not training). Mirrors probe_fire_bcfit.
"""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding="utf-8")
import warnings; warnings.filterwarnings("ignore")

import random
import numpy as np
import torch
import torch.nn as nn

from trpg.scenarios.monsters import register_monsters
register_monsters()

from trpg.engine.vec2 import Vec2
from trpg.rl.env_v2 import CombatEnvV2
from trpg.rl.obs import (terrain_obs, entity_grid_obs, distance_grid_obs,
                         partition_entities, N_GRID)
from trpg.rl.action import encode_action  # noqa
from synth_identity import STANDARD_IDS


def collect(n_states: int, seed: int):
    rng = random.Random(seed)
    feats, los_true, labels, exy, sxy = [], [], [], [], []
    layouts = ["pillar", "walls", "pillar"]
    got = 0
    attempt = 0
    while got < n_states and attempt < n_states * 5:
        attempt += 1
        ident = rng.choice(STANDARD_IDS)
        env = CombatEnvV2(seed=rng.randint(0, 1 << 30), n_agents=1, n_opps=1)
        env.reset(agent_archs=[ident], opp_archs=[rng.choice(STANDARD_IDS)],
                  level=5, opp_level=5, layout=rng.choice(layouts))
        ws = env.ws
        bf = ws.combat.battlefield
        aid = env.agent_ids[0]
        agent = ws.characters[aid]
        # randomise positions so LoS labels vary
        W, H = bf.width, bf.height
        agent.position = Vec2(rng.uniform(1, W - 1), rng.uniform(1, H - 1))
        enemy = ws.characters[env.opp_ids[0]]
        enemy.position = Vec2(rng.uniform(1, W - 1), rng.uniform(1, H - 1))
        if bf.is_blocked(agent.position) or bf.is_blocked(enemy.position):
            continue
        _, enemies = partition_entities(ws, aid)
        if not enemies:
            continue
        e = ws.characters[enemies[0]].position
        terr = terrain_obs(bf)                       # [N,N]
        eg = entity_grid_obs(ws, aid)                # [3,N,N]
        dg = distance_grid_obs(ws, aid)              # [2,N,N]
        cell = bf.width / N_GRID
        for ix in range(N_GRID):
            for iy in range(N_GRID):
                c = Vec2((ix + 0.5) * cell, (iy + 0.5) * cell)
                if bf.is_blocked(c):
                    continue                          # can't stand on a wall
                f = [terr[ix, iy], eg[0, ix, iy], eg[1, ix, iy], eg[2, ix, iy],
                     dg[0, ix, iy], dg[1, ix, iy]]
                feats.append(f)
                lab = 1.0 if bf.has_line_of_sight(c, e) else 0.0
                labels.append(lab)
                los_true.append(lab)
                exy.append([e.x / W, e.y / H]); sxy.append(
                    [agent.position.x / W, agent.position.y / H])
        got += 1
    return (np.array(feats, np.float32), np.array(labels, np.float32),
            np.array(los_true, np.float32), np.array(exy, np.float32),
            np.array(sxy, np.float32))


def fit_eval(X, y, hidden=None, epochs=300, cap=60000):
    n = len(y); idx = np.arange(n); rng = np.random.RandomState(0); rng.shuffle(idx)
    if n > cap:
        idx = idx[:cap]; n = cap
    cut = int(n * 0.8)
    tr, te = idx[:cut], idx[cut:]
    # standardise so the optimiser converges (incl. on the sanity feature)
    mu = X[tr].mean(0, keepdims=True); sd = X[tr].std(0, keepdims=True) + 1e-6
    Xs = (X - mu) / sd
    Xt = torch.from_numpy(Xs[tr]); yt = torch.from_numpy(y[tr])
    Xe = torch.from_numpy(Xs[te]); ye = torch.from_numpy(y[te])
    d = X.shape[1]
    if hidden:
        net = nn.Sequential(nn.Linear(d, hidden), nn.ReLU(),
                            nn.Linear(hidden, hidden), nn.ReLU(),
                            nn.Linear(hidden, 1))
    else:
        net = nn.Linear(d, 1)
    opt = torch.optim.Adam(net.parameters(), lr=5e-2)
    lossf = nn.BCEWithLogitsLoss()
    bs = 4096
    for _ in range(epochs):
        p = torch.randperm(len(yt))[:bs]
        opt.zero_grad()
        out = net(Xt[p]).squeeze(-1)
        lossf(out, yt[p]).backward()
        opt.step()
    with torch.no_grad():
        pred = (net(Xe).squeeze(-1) > 0).float()
        acc = (pred == ye).float().mean().item()
    return acc


def main():
    X, y, los_true, exy, sxy = collect(n_states=400, seed=7)
    base = max(y.mean(), 1 - y.mean())
    print(f"samples={len(y)}  LoS-rate={y.mean():.2%}  "
          f"majority-baseline={base:.2%}\n")

    acc_lin = fit_eval(X, y)
    Xpos = np.concatenate([X, exy, sxy], axis=1)
    acc_lin_pos = fit_eval(Xpos, y)
    acc_mlp = fit_eval(Xpos, y, hidden=64)
    Xsan = np.concatenate([X, los_true[:, None]], axis=1)
    acc_sanity = fit_eval(Xsan, y)

    print(f"linear  (6 cell feats)            acc = {acc_lin:.1%}  "
          f"(Δ vs base {100*(acc_lin-base):+.1f}pp)")
    print(f"linear  (+enemy/self xy)          acc = {acc_lin_pos:.1%}  "
          f"(Δ {100*(acc_lin_pos-base):+.1f}pp)")
    print(f"MLP 2x64(+enemy/self xy)          acc = {acc_mlp:.1%}  "
          f"(Δ {100*(acc_mlp-base):+.1f}pp)  <- universal pointwise fn")
    print(f"linear  (+TRUE los channel)       acc = {acc_sanity:.1%}  "
          f"<- sanity, must be ~100%")
    print()
    if acc_mlp - base < 0.08:
        print("=> CONFIRMED: per-cell features cannot predict LoS even with an "
              "MLP + positions. The grid-head structurally cannot aim a flank; "
              "a precomputed per-cell LoS channel is REQUIRED.")
    else:
        print("=> NOT confirmed: the info is recoverable; revisit the "
              "exploration/training hypothesis before adding a channel.")


if __name__ == "__main__":
    main()
