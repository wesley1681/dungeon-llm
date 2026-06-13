"""Champion-only BC clone WITHOUT the inverse-freq skill weighting.

Tests whether the weighting (which over-boosts the rare action_surge) is what
makes the champion clone over-surge (1.47 vs expert 0.92) and under-attack.
"""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding="utf-8")
import warnings; warnings.filterwarnings("ignore")
import torch
from trpg.rl.bc_collect import collect_bc_dataset
from trpg.rl.train_bc import train_bc

print("collecting champion-only BC data (200 episodes vs all opps)...")
ds = collect_bc_dataset(n_episodes_per_arch=200, seed=0, only_arch="champion")
print(f"  {ds['actions'].shape[0]} transitions")
dev = "cuda" if torch.cuda.is_available() else "cpu"
out = "models/bc_champion_noweight.pt"
train_bc(ds, epochs=25, batch_size=128, lr=3e-4, device=dev,
         save_path=out, use_skill_weight=False, verbose=True)
print(f"saved {out}")
