#!/usr/bin/env python3
"""
Quick threshold sweep for early-exit proxy metrics.

Outputs per-threshold:
- early_exit_rate
- mean_generated_tokens
- mean_tokens_saved_vs_full
- proxy_action_mse_vs_gt

This is a low-cost offline evaluation for report tables when full simulator
rollouts are too expensive.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import List
from datetime import datetime

import dill
import hydra
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

_REPO_ROOT = Path(__file__).resolve().parent.parent
_OAT_ROOT = _REPO_ROOT / "third_party" / "oat"
_LAB_SRC = _REPO_ROOT / "src"
for _p in (_OAT_ROOT, _LAB_SRC):
    if _p.is_dir() and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from oat.common.hydra_util import register_new_resolvers
from oat.common.pytorch_util import dict_apply
from oat.policy.oatpolicy import OATPolicy
from oat.workspace.train_policy import TrainPolicyWorkspace
from oat_ext.early_exit import EarlyExitGate

register_new_resolvers()


def _prepare_obs_for_encoder(obs: dict) -> dict:
    fixed = {}
    for k, v in obs.items():
        if torch.is_tensor(v) and k.endswith("_rgb"):
            if not v.dtype.is_floating_point:
                v = v.to(torch.float32)
            if float(v.max()) > 1.0:
                v = v / 255.0
            fixed[k] = v
        else:
            fixed[k] = v
    return fixed


def parse_args():
    p = argparse.ArgumentParser(description="Sweep early-exit thresholds (proxy metrics).")
    p.add_argument("--checkpoint", type=str, required=True, help="OAT policy workspace .ckpt")
    p.add_argument(
        "--mode",
        type=str,
        choices=["gate", "maxprob"],
        default="gate",
        help="gate = learned EarlyExitGate, maxprob = confidence-only heuristic",
    )
    p.add_argument("--gate", type=str, default="", help="Path to early_exit_gate.pt (for mode=gate)")
    p.add_argument("--thresholds", type=float, nargs="+", default=[0.7, 0.8, 0.9])
    p.add_argument("--max-batches", type=int, default=50, help="Validation batches to evaluate")
    p.add_argument("--batch-size", type=int, default=0, help="Override val_dataloader.batch_size (0 = keep cfg)")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out-csv", type=str, default="", help="Optional CSV output path")
    return p.parse_args()


def _load_payload(path: str):
    load_kw = dict(map_location="cpu", pickle_module=dill)
    try:
        return torch.load(path, **{**load_kw, "weights_only": False})
    except TypeError:
        return torch.load(path, **load_kw)


@torch.no_grad()
def main():
    args = parse_args()
    device = torch.device(args.device)
    payload = _load_payload(args.checkpoint)
    cfg = payload["cfg"]
    if not OmegaConf.has_resolver("now"):
        OmegaConf.register_new_resolver(
            "now", lambda fmt="%Y%m%d.%H%M%S": datetime.now().strftime(fmt)
        )
    OmegaConf.resolve(cfg)

    ws = TrainPolicyWorkspace(cfg, output_dir="/tmp/oat_sweep", lazy_instantiation=False)
    ws.load_checkpoint(path=args.checkpoint)
    policy = ws.model
    if not isinstance(policy, OATPolicy):
        raise TypeError(f"Expected OATPolicy checkpoint, got {type(policy).__name__}")
    policy.eval().to(device)

    for p in policy.parameters():
        p.requires_grad_(False)

    gate = None
    if args.mode == "gate":
        if not args.gate:
            raise ValueError("--gate is required when --mode gate")
        gate_payload = _load_payload(args.gate)
        gate_state = gate_payload.get("gate_state_dict", gate_payload) if isinstance(gate_payload, dict) else gate_payload
        gate = EarlyExitGate(n_emb=policy.model.n_emb, hidden_mult=4, dropout=0.0).to(device)
        gate.load_state_dict(gate_state)
        gate.eval()

    dataset = hydra.utils.instantiate(cfg.task.policy.dataset)
    val_dataset = dataset.get_validation_dataset()
    val_cfg = cfg.val_dataloader if "val_dataloader" in cfg else cfg.dataloader
    if args.batch_size and args.batch_size > 0:
        val_cfg.batch_size = args.batch_size
    val_loader = DataLoader(val_dataset, **val_cfg)

    full_len = int(policy.max_seq_len)
    rows = []
    for th in args.thresholds:
        n_seqs = 0
        n_early = 0
        sum_len = 0.0
        sum_mse = 0.0
        n_batches = 0

        for bi, batch in enumerate(val_loader):
            if args.max_batches and bi >= args.max_batches:
                break
            batch = dict_apply(batch, lambda x: x.to(device) if torch.is_tensor(x) else x)
            B = batch["action"].shape[0]
            feats = policy.obs_encoder(_prepare_obs_for_encoder(batch["obs"]))
            prefix = torch.full((B, 1), policy.bos_id, dtype=torch.long, device=device)
            out = policy.model.generate(
                prefix=prefix,
                cond=feats,
                max_new_tokens=full_len,
                temperature=policy.temperature,
                top_k=policy.topk,
                early_exit_gate=gate if args.mode == "gate" else None,
                early_exit_threshold=float(th),
                early_exit_min_new_tokens=int(policy.early_exit_min_new_tokens),
                early_exit_max_prob=float(th) if args.mode == "maxprob" else None,
            )
            gen = out[:, 1:]
            gen_len = gen.shape[1]
            pred = policy.action_tokenizer.detokenize(gen)
            mse = ((pred - batch["action"]) ** 2).mean().item()

            n_seqs += B
            n_early += int(gen_len < full_len) * B
            sum_len += float(gen_len) * B
            sum_mse += mse * B
            n_batches += 1

        mean_len = sum_len / max(n_seqs, 1)
        row = {
            "mode": args.mode,
            "threshold": th,
            "batches": n_batches,
            "sequences": n_seqs,
            "early_exit_rate": n_early / max(n_seqs, 1),
            "mean_generated_tokens": mean_len,
            "mean_tokens_saved_vs_full": full_len - mean_len,
            "proxy_action_mse_vs_gt": sum_mse / max(n_seqs, 1),
        }
        rows.append(row)
        print(
            f"[{args.mode}] th={th:.3f} "
            f"exit_rate={row['early_exit_rate']:.3f} "
            f"mean_tokens={row['mean_generated_tokens']:.2f} "
            f"saved={row['mean_tokens_saved_vs_full']:.2f} "
            f"proxy_mse={row['proxy_action_mse_vs_gt']:.6f}"
        )

    if args.out_csv:
        out_path = Path(args.out_csv)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = list(rows[0].keys()) if rows else [
            "mode",
            "threshold",
            "batches",
            "sequences",
            "early_exit_rate",
            "mean_generated_tokens",
            "mean_tokens_saved_vs_full",
            "proxy_action_mse_vs_gt",
        ]
        with out_path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(rows)
        print(f"Saved CSV to {out_path}")


if __name__ == "__main__":
    main()

