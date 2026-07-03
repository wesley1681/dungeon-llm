"""Weight-average several checkpoints (SWA-style) into one, to damp the
per-round oscillation of DAgger snapshots into a single stable policy. Plain
mean of state_dict tensors (all share the exact same architecture/keys).

Usage:
  python scripts/avg_ckpts.py --out models/dagger2/avg_r5_8.pt \
      models/dagger2/dagger_r05.pt ... models/dagger2/dagger_r08.pt
"""
from __future__ import annotations
import argparse
import torch


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("ckpts", nargs="+")
    args = p.parse_args()
    sds = [torch.load(c, map_location="cpu") for c in args.ckpts]
    keys = sds[0].keys()
    avg = {}
    for k in keys:
        ts = [sd[k] for sd in sds]
        if ts[0].is_floating_point():
            avg[k] = sum(t.float() for t in ts) / len(ts)
        else:
            avg[k] = ts[0]  # non-float buffers (e.g. counters): take first
    torch.save(avg, args.out)
    print(f"averaged {len(sds)} ckpts -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
