"""Upload a built dataset folder to the Hugging Face Hub.

    hf auth login                                   # once; paste a write token
    python -m src.publish_dataset --repo <user>/index-drift-rl-bench            # private
    python -m src.publish_dataset --repo <user>/index-drift-rl-bench --public   # when ready

The folder's README.md (written by src.build_dataset) becomes the dataset card.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m src.publish_dataset", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo", required=True, help="<user or org>/<dataset name>")
    parser.add_argument("--path", type=Path, default=Path("data/dataset"))
    parser.add_argument("--public", action="store_true", help="create the repo as public (default: private)")
    args = parser.parse_args(argv)

    from huggingface_hub import HfApi

    if not (args.path / "README.md").exists():
        raise SystemExit(f"{args.path} has no README.md - run python -m src.build_dataset first")
    api = HfApi()
    api.create_repo(args.repo, repo_type="dataset", private=not args.public, exist_ok=True)
    api.upload_folder(repo_id=args.repo, repo_type="dataset", folder_path=str(args.path),
                      allow_patterns=["README.md", "*.json", "*.jsonl", "*.sql"],
                      commit_message="Update index-drift-rl benchmark")
    print(f"https://huggingface.co/datasets/{args.repo}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
