#!/usr/bin/env python3
"""
Sanity-check an OATTok (train_oattok) checkpoint before policy training.

Exits with code 1 if encoder LinearHead projection weights are still at zero-init
(|W|_1 < 1e-5), which guarantees collapsed latents/tokens downstream.

Run from third_party/oat (same as other oat scripts):

  cd third_party/oat
  export PATH="$HOME/.local/bin:$PATH"
  uv run python ../../scripts/inspect_oattok_ckpt.py /root/oattok_libero10.ckpt
"""

from __future__ import annotations

import argparse
import sys

import torch


def main() -> None:
    p = argparse.ArgumentParser(description="Inspect OATTok checkpoint encoder head weights")
    p.add_argument("ckpt", help="Path to tokenizer .ckpt (train_oattok output)")
    args = p.parse_args()

    from oat.tokenizer.base_tokenizer import BaseTokenizer

    tok = BaseTokenizer.from_checkpoint(args.ckpt)
    enc = getattr(tok, "encoder", None)
    head = getattr(enc, "head", None) if enc is not None else None
    proj = getattr(head, "proj", None) if head is not None else None
    if proj is None:
        print("No encoder.head.proj on this tokenizer class; nothing to check.")
        sys.exit(0)
    w = proj.weight.detach().float()
    s = float(w.abs().sum())
    n = float(w.numel())
    print(f"checkpoint: {args.ckpt}")
    print(f"|encoder.head.proj.weight|_1 = {s:.6f}  (numel={int(n)})")
    if s < 1e-5:
        print("FAIL: head is all-zero — retrain train_oattok or use a different ckpt.")
        sys.exit(1)
    print("OK")
    sys.exit(0)


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    main()
