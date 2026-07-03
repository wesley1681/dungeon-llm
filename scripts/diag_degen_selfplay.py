"""對手席/自對打退化審計：模型同時駕駛**雙方**（agent 席 + 對手席），
驗證「模型坐對手席扮任意身分」也零退化。

monkeypatch execute_action 記錄每個 MOVE 的位移與行為者；no-op move = 位移<0.1m。
對手席經 env._run_opponent_turn → NeuralCombatPolicy → pick_action（含 no-op 抑制），
故與 agent 席共用修復。另記每場是否 truncate（凍結會導致打不完→截斷）。

用法: python scripts/diag_degen_selfplay.py [模型路徑] [--games N]
"""
import sys, os, argparse, random
from zlib import crc32
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout.reconfigure(encoding="utf-8")
import warnings; warnings.filterwarnings("ignore")
import torch
from trpg.scenarios.monsters import register_monsters, EQUIV_LEVEL_1V1, MONSTER_DEFS
register_monsters()
import trpg.engine.combat as C
import trpg.rl.env_v2 as E
from trpg.rl.env_v2 import CombatEnvV2
from trpg.rl.model import CombatPolicyNet

PAIRS = [   # (我方身分, 對手身分) — 兩席都由模型駕駛
    ("battle_master", "champion"), ("evocation", "wight"),
    ("life", "ghoul"), ("assassin", "ogre"), ("war", "owlbear"),
    ("kobold", "battle_master"), ("wight", "evocation"),
    ("manticore", "champion"), ("shadow", "vengeance"),
    ("devotion", "berserker"), ("orc", "life"), ("ghoul", "assassin"),
]

_MOVE_LOG = []
_orig_exec = C.execute_action
def _patched_exec(action, ws):
    r = _orig_exec(action, ws)
    if isinstance(r, dict) and r.get("type") == "MOVE" and not r.get("teleport"):
        _MOVE_LOG.append((action.get("character", "?"), r.get("distance", 0.0)))
    return r
C.execute_action = _patched_exec
E.execute_action = _patched_exec


def load(p):
    net = CombatPolicyNet(hidden=128); sd = torch.load(p, map_location="cpu")
    net.load_state_dict(CombatPolicyNet.adapt_state_dict_for_perarch(sd), strict=False)
    net.eval(); return net


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt", nargs="?", default="models/unified/uni_v7.pt")
    ap.add_argument("--games", type=int, default=8)
    args = ap.parse_args()
    net = load(args.ckpt)
    print(f"模型: {os.path.basename(args.ckpt)} 自對打(雙席模型) games/pair={args.games}\n")
    print(f"{'我方 vs 對手':28s} {'局':>3s} {'截斷':>4s} {'總MOVE':>6s} {'空轉':>5s}")
    print("-" * 52)
    g_trunc = g_moves = g_noop = 0
    for a_id, o_id in PAIRS:
        lvl = max(1, round(EQUIV_LEVEL_1V1.get(a_id, 5))) if a_id in MONSTER_DEFS else 5
        olvl = max(1, round(EQUIV_LEVEL_1V1.get(o_id, 5))) if o_id in MONSTER_DEFS else 5
        truncs = moves = noops = 0
        for gi in range(args.games):
            _MOVE_LOG.clear()
            k = crc32(f"{a_id}|{o_id}|{gi}".encode()); random.seed(k)
            env = CombatEnvV2(seed=k ^ 0x33, n_agents=1, n_opps=1)
            env.use_self_play_opponent(net)
            obs, _ = env.reset(agent_archs=[a_id], opp_archs=[o_id],
                               level=lvl, opp_level=olvl)
            aid = env.agent_ids[0]
            done = False; trunc = False
            while not done:
                actor = env.current_agent_id
                if actor == aid:
                    from train_population import blind_np_single
                    from trpg.rl.model import (apply_resource_mask,
                                               apply_entity_mask, pick_action)
                    ob = blind_np_single(obs)
                    ot = {kk: torch.from_numpy(v).unsqueeze(0) for kk, v in ob.items()}
                    with torch.no_grad():
                        el, s, e, g = net(ot)
                    s = apply_resource_mask(s, env.resources, env.ws, actor)
                    e = apply_entity_mask(e, ot, env.ws, actor)
                    act = list(pick_action(el[0], s[0], e[0], g[0], ws=env.ws,
                                           agent_id=actor))
                    obs, _, term, trunc, _ = env.step(act)
                else:
                    obs, _, term, trunc, _ = env.step([0, 0, 0])
                done = term or trunc
            truncs += int(trunc)
            moves += len(_MOVE_LOG)
            noops += sum(1 for _, d in _MOVE_LOG if d < 0.1)
        g_trunc += truncs; g_moves += moves; g_noop += noops
        flag = " ⚠" if noops else ""
        print(f"{a_id+' vs '+o_id:28s} {args.games:3d} {truncs:4d} "
              f"{moves:6d} {noops:5d}{flag}")
    print("-" * 52)
    print(f"{'總計':28s} {len(PAIRS)*args.games:3d} {g_trunc:4d} "
          f"{g_moves:6d} {g_noop:5d}")
    print(f"\n空轉MOVE(雙席合計)={g_noop} / {g_moves} 次MOVE  截斷={g_trunc}")


if __name__ == "__main__":
    main()
