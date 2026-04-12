#!/usr/bin/env python3
"""
Upload a trained policy checkpoint (and optional JSON logs) to a Hugging Face *model* repo.

Auth (pick one):
  export HF_TOKEN=hf_...           # recommended
  export HUGGING_FACE_HUB_TOKEN=hf_...
  or:  --token hf_...            # avoid in shared shell history if possible

On the GPU box (after ./scripts/install_oat.sh), from repo root:

  third_party/oat/.venv/bin/python scripts/push_checkpoint_to_hf.py \\
    --repo-id YOUR_USERNAME/oat-libero-policy \\
    --checkpoint third_party/oat/output/manual/<run>/checkpoints/latest.ckpt

Create the repo on https://huggingface.co/new-model first, or pass --create-repo.
"""

from __future__ import annotations

import argparse
import io
import os
import subprocess
import sys
from pathlib import Path


def _git_commit_short(root: Path) -> str | None:
    try:
        out = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        )
        return out.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
        return None


def _build_model_card(
    *,
    repo_id: str,
    ckpt_name: str,
    eval_log: Path | None,
    train_log: Path | None,
    commit: str | None,
    extra_notes: str,
) -> str:
    lines = [
        "---",
        "license: mit",
        "tags:",
        "  - robotics",
        "  - vla",
        "  - oat",
        "  - libero",
        "---",
        "",
        f"# {repo_id.split('/')[-1]}",
        "",
        "Policy checkpoint from the **oat-early-exit** fork (OAT + optional early-exit decode).",
        "",
        "## Files",
        "",
        f"- `{ckpt_name}` — OAT policy weights (`latest.ckpt` from training).",
    ]
    if eval_log:
        lines.append(f"- `{eval_log.name}` — simulator eval summary (if provided).")
    if train_log:
        lines.append(f"- `{train_log.name}` — training `logs.json` (if provided).")
    lines += [
        "",
        "## Repro",
        "",
        "Train / eval instructions: https://github.com/GadzhiAskhabaliev/oat-early-exit",
        "",
    ]
    if commit:
        lines.append(f"Source git revision (when uploaded): `{commit}`.")
        lines.append("")
    if extra_notes.strip():
        lines.append("## Notes")
        lines.append("")
        lines.append(extra_notes.strip())
        lines.append("")
    lines.append("## Citation")
    lines.append("")
    lines.append("If you use OAT, cite the original OAT paper/repo; add your own citation for this fork if needed.")
    return "\n".join(lines) + "\n"


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--repo-id", required=True, help="HF model repo, e.g. username/oat-libero-policy")
    p.add_argument("--checkpoint", required=True, type=Path, help="Path to .ckpt file")
    p.add_argument(
        "--path-in-repo",
        default="oat_policy_latest.ckpt",
        help="Remote filename for the checkpoint (default: oat_policy_latest.ckpt)",
    )
    p.add_argument("--eval-log", type=Path, default=None, help="Optional eval_log.json")
    p.add_argument("--train-log", type=Path, default=None, help="Optional training logs.json")
    p.add_argument("--create-repo", action="store_true", help="Create the model repo if it does not exist")
    p.add_argument("--private", action="store_true", help="Use with --create-repo for a private repo")
    p.add_argument("--token", default=None, help="HF token (prefer env HF_TOKEN instead)")
    p.add_argument("--notes", default="", help="Extra markdown appended to the model card")
    args = p.parse_args()

    try:
        from huggingface_hub import HfApi
    except ImportError:
        print("Missing package: huggingface_hub", file=sys.stderr)
        print("Install:  uv pip install huggingface_hub", file=sys.stderr)
        print("Or from repo root after install_oat.sh: already in requirements.txt — re-run ./scripts/install_oat.sh", file=sys.stderr)
        return 1

    ckpt = args.checkpoint.expanduser().resolve()
    if not ckpt.is_file():
        print(f"Checkpoint not found: {ckpt}", file=sys.stderr)
        return 1

    token = args.token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not token:
        print(
            "No HF token. Set HF_TOKEN (https://huggingface.co/settings/tokens) or pass --token.",
            file=sys.stderr,
        )
        return 1

    root = Path(__file__).resolve().parents[1]
    api = HfApi(token=token)

    if args.create_repo:
        api.create_repo(
            repo_id=args.repo_id,
            repo_type="model",
            private=args.private,
            exist_ok=True,
        )

    repo_id = args.repo_id
    print(f"==> Uploading checkpoint to https://huggingface.co/{repo_id} as {args.path_in_repo!r} ...")
    api.upload_file(
        path_or_fileobj=str(ckpt),
        path_in_repo=args.path_in_repo,
        repo_id=repo_id,
        repo_type="model",
        commit_message="Add policy checkpoint",
    )
    print("    done.")

    for label, path, remote_name in (
        ("eval log", args.eval_log, "eval_log.json"),
        ("train log", args.train_log, "logs.json"),
    ):
        if path is None:
            continue
        path = path.expanduser().resolve()
        if not path.is_file():
            print(f"WARN: skip {label}: not a file: {path}", file=sys.stderr)
            continue
        print(f"==> Uploading {label} as {remote_name!r} ...")
        api.upload_file(
            path_or_fileobj=str(path),
            path_in_repo=remote_name,
            repo_id=repo_id,
            repo_type="model",
            commit_message=f"Add {remote_name}",
        )
        print("    done.")

    commit = _git_commit_short(root)
    card = _build_model_card(
        repo_id=repo_id,
        ckpt_name=args.path_in_repo,
        eval_log=args.eval_log.expanduser().resolve() if args.eval_log else None,
        train_log=args.train_log.expanduser().resolve() if args.train_log else None,
        commit=commit,
        extra_notes=args.notes,
    )
    print("==> Uploading README.md (model card) ...")
    api.upload_file(
        path_or_fileobj=io.BytesIO(card.encode("utf-8")),
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="model",
        commit_message="Update model card",
    )
    print("    done.")
    print(f"Open: https://huggingface.co/{repo_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
