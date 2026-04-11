#!/usr/bin/env python3
"""
Quick LIBERO tokenization sanity check for OATPolicy checkpoints.

Why this exists:
- Long one-liners break when pasted through SSH clients.
- We want a stable way to print token diversity + CE loss on a single batch.

Typical remote usage:
  cd third_party/oat
  export PYTHONPATH="$HOME/oat-early-exit/src:${PYTHONPATH:-}"
  uv run python ../../scripts/diag_libero_tokens.py \
    --ckpt /path/to/latest.ckpt \
    --zarr data/libero/libero10_N500.zarr
"""

from __future__ import annotations

import argparse
import os

import torch
from torch.utils.data import DataLoader

from oat.dataset.zarr_dataset import ZarrDataset
from oat.policy.base_policy import BasePolicy


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default=os.environ.get("CKPT", ""), help="Policy checkpoint (.ckpt)")
    p.add_argument(
        "--zarr",
        default="data/libero/libero10_N500.zarr",
        help="Zarr path relative to third_party/oat unless absolute",
    )
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--batches", type=int, default=3, help="How many random batches to probe")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    if not args.ckpt:
        raise SystemExit("Missing --ckpt (or set env CKPT=...)")

    torch.manual_seed(args.seed)

    policy, _cfg = BasePolicy.from_checkpoint(args.ckpt, return_configuration=True)

    ds = ZarrDataset(
        zarr_path=args.zarr,
        obs_keys=[
            "agentview_rgb",
            "robot0_eye_in_hand_rgb",
            "robot0_eef_pos",
            "robot0_eef_quat",
            "robot0_gripper_qpos",
            "task_uid",
        ],
        action_key="action",
        n_obs_steps=2,
        n_action_steps=32,
        seed=42,
        val_ratio=0.1,
        max_train_episodes=None,
        copy_to_memory=False,
    )

    dl = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=True,
    )

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy = policy.to(dev).train()

    it = iter(dl)
    for i in range(args.batches):
        batch = next(it)
        for k, v in list(batch["obs"].items()):
            batch["obs"][k] = v.to(dev) if torch.is_tensor(v) else v
        batch["action"] = batch["action"].to(dev)

        act = batch["action"]
        tok = policy.action_tokenizer.tokenize(act)
        t = tok.detach().long()

        u = int(torch.unique(t).numel())
        z = float((t == 0).float().mean().item())
        mn = int(t.min().item())
        mx = int(t.max().item())

        loss = policy(batch)
        lf64 = float(loss.detach().double().item())
        finite = bool(torch.isfinite(loss.detach()).item())

        print(
            f"[batch {i}] action_mean_std=({float(act.mean()):.6f},{float(act.std()):.6f}) "
            f"tokens_shape={tuple(t.shape)} zero_frac={z:.4f} uniq={u} minmax=({mn},{mx}) "
            f"loss_f64={lf64} finite={finite}"
        )


if __name__ == "__main__":
    main()
