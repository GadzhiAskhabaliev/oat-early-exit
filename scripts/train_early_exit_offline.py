#!/usr/bin/env python3
"""
Offline training of EarlyExitGate using reconstruction-based labels.

Freeze the OAT policy (and tokenizer). For each demo batch, compute whether each
token prefix yields low MSE vs ground-truth actions, then train the gate to
predict those labels from teacher-forcing LM hidden states.

Run from repo root (or anywhere) with OAT on the path::

    export PATH="$HOME/.local/bin:$PATH"
    export PYTHONPATH="/abs/path/to/mipt-lab-project/src:/abs/path/to/mipt-lab-project/third_party/oat"
    cd /abs/path/to/mipt-lab-project/third_party/oat
    uv run python ../../scripts/train_early_exit_offline.py \\
        --checkpoint /path/to/train_oatpolicy/checkpoints/latest.ckpt \\
        --mse-threshold 0.01 \\
        --epochs 5 \\
        --max-batches 100 \\
        --out-gate /path/to/early_exit_gate.pt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from datetime import datetime

import dill
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

_REPO_ROOT = Path(__file__).resolve().parent.parent
_OAT_ROOT = _REPO_ROOT / "third_party" / "oat"
_LAB_SRC = _REPO_ROOT / "src"
for _p in (_OAT_ROOT, _LAB_SRC):
    if _p.is_dir() and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import hydra

from oat.common.hydra_util import register_new_resolvers
from oat.common.pytorch_util import dict_apply
from oat.policy.oatpolicy import OATPolicy
from oat.workspace.train_policy import TrainPolicyWorkspace

register_new_resolvers()
from oat_ext.early_exit import EarlyExitGate
from oat_ext.early_exit_supervision import mse_per_prefix, reconstruction_labels


def _prepare_obs_for_encoder(obs: dict) -> dict:
    """Ensure RGB tensors match robomimic encoder dtype expectations."""
    fixed = {}
    for k, v in obs.items():
        if torch.is_tensor(v) and k.endswith("_rgb"):
            # Robomimic encoder expects floating image tensors.
            if not v.dtype.is_floating_point:
                v = v.to(torch.float32)
            if float(v.max()) > 1.0:
                v = v / 255.0
            fixed[k] = v
        else:
            fixed[k] = v
    return fixed


def parse_args():
    p = argparse.ArgumentParser(description="Train EarlyExitGate offline (reconstruction labels).")
    p.add_argument("--checkpoint", type=str, required=True, help="Policy workspace .ckpt (dill).")
    p.add_argument("--mse-threshold", type=float, default=0.01, help="MSE threshold for positive label.")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--max-batches", type=int, default=0, help="0 = full train loader.")
    p.add_argument("--out-gate", type=str, required=True, help="Output .pt state_dict for EarlyExitGate.")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)

    _load_kw = dict(map_location="cpu", pickle_module=dill)
    try:
        payload = torch.load(args.checkpoint, **{**_load_kw, "weights_only": False})
    except TypeError:
        payload = torch.load(args.checkpoint, **_load_kw)
    cfg = payload["cfg"]
    # Seed checkpoints created outside Hydra runtime can still contain
    # `${now:...}` interpolations in logging fields.
    if not OmegaConf.has_resolver("now"):
        OmegaConf.register_new_resolver(
            "now", lambda fmt="%Y%m%d.%H%M%S": datetime.now().strftime(fmt)
        )
    OmegaConf.resolve(cfg)

    ws = TrainPolicyWorkspace(cfg, output_dir="/tmp/oat_early_exit_offline", lazy_instantiation=False)
    ws.load_checkpoint(path=args.checkpoint)

    policy = ws.model
    if not isinstance(policy, OATPolicy):
        raise TypeError(
            "This script expects an OATPolicy checkpoint. Got "
            f"{type(policy).__name__}."
        )
    policy.eval()
    policy.to(device)
    for p in policy.parameters():
        p.requires_grad_(False)

    n_emb = policy.model.n_emb
    gate = EarlyExitGate(n_emb=n_emb, hidden_mult=4, dropout=0.1).to(device)
    opt = torch.optim.AdamW(gate.parameters(), lr=args.lr)

    dataset = hydra.utils.instantiate(cfg.task.policy.dataset)
    train_loader = DataLoader(dataset, **cfg.dataloader)
    val_dataset = dataset.get_validation_dataset()
    val_dl_cfg = cfg.val_dataloader if "val_dataloader" in cfg else cfg.dataloader
    val_loader = DataLoader(val_dataset, **val_dl_cfg)

    @torch.no_grad()
    def eval_gate(loader, max_batches: int = 0):
        gate.eval()
        total_loss = 0.0
        total_acc = 0.0
        n_steps = 0
        for bi, batch in enumerate(loader):
            if max_batches and bi >= max_batches:
                break
            batch = dict_apply(batch, lambda x: x.to(device) if torch.is_tensor(x) else x)
            tokens = policy.action_tokenizer.tokenize(batch["action"])
            mse_mat = mse_per_prefix(policy.action_tokenizer, batch["action"], tokens)
            y = reconstruction_labels(mse_mat, args.mse_threshold)
            feats = policy.obs_encoder(_prepare_obs_for_encoder(batch["obs"]))
            B = tokens.shape[0]
            bos = torch.full((B, 1), policy.bos_id, dtype=torch.long, device=device)
            action_tokens = torch.cat([bos, tokens], dim=1)
            _, h = policy.model(action_tokens[:, :-1], feats, return_hidden=True)
            logits = gate(h).squeeze(-1)
            loss = F.binary_cross_entropy_with_logits(logits, y)
            pred = (logits > 0).float()
            acc = (pred == y).float().mean()
            total_loss += float(loss.item())
            total_acc += float(acc.item())
            n_steps += 1
        return total_loss / max(n_steps, 1), total_acc / max(n_steps, 1)

    for epoch in range(args.epochs):
        gate.train()
        total_loss = 0.0
        total_acc = 0.0
        n_steps = 0
        for bi, batch in enumerate(train_loader):
            if args.max_batches and bi >= args.max_batches:
                break
            batch = dict_apply(batch, lambda x: x.to(device) if torch.is_tensor(x) else x)

            with torch.no_grad():
                tokens = policy.action_tokenizer.tokenize(batch["action"])
                mse_mat = mse_per_prefix(policy.action_tokenizer, batch["action"], tokens)
                y = reconstruction_labels(mse_mat, args.mse_threshold)

                feats = policy.obs_encoder(_prepare_obs_for_encoder(batch["obs"]))
                B = tokens.shape[0]
                bos = torch.full((B, 1), policy.bos_id, dtype=torch.long, device=device)
                action_tokens = torch.cat([bos, tokens], dim=1)
                _, h = policy.model(
                    action_tokens[:, :-1], feats, return_hidden=True
                )

            logits = gate(h).squeeze(-1)
            loss = F.binary_cross_entropy_with_logits(logits, y)
            pred = (logits > 0).float()
            acc = (pred == y).float().mean()
            opt.zero_grad()
            loss.backward()
            opt.step()

            total_loss += loss.item()
            total_acc += float(acc.item())
            n_steps += 1

        train_loss = total_loss / max(n_steps, 1)
        train_acc = total_acc / max(n_steps, 1)
        val_loss, val_acc = eval_gate(val_loader, max_batches=args.max_batches)
        print(
            f"epoch {epoch + 1}/{args.epochs}  "
            f"train_bce={train_loss:.4f} train_acc={train_acc:.4f}  "
            f"val_bce={val_loss:.4f} val_acc={val_acc:.4f}"
        )

    out_path = Path(args.out_gate)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "gate_state_dict": gate.state_dict(),
            "mse_threshold": args.mse_threshold,
            "n_emb": n_emb,
        },
        out_path,
    )
    print(f"Saved gate to {out_path}")


if __name__ == "__main__":
    main()
