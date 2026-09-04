#!/usr/bin/env python3
"""Check that a RunPod pod is actually ready to run DRAKES.

The DRAKES training scripts fail late and cryptically -- a missing checkpoint
surfaces forty minutes in, and a wrong gReLU version raises a pickle error that
looks like a corrupt download. This checks the things that break, up front.

    python smoke_test.py --base-path /workspace/drakes_data
"""

import argparse
import importlib
import os
import pathlib
import sys

# Files the DNA experiment reads, relative to BASE_PATH. Paths come from
# dataloader_gosai.py, oracle.py, train_oracle.py and finetune_reward_bp.py.
DNA_FILES = [
    "mdlm/gosai_data/processed_data/gosai_all.csv",
    "mdlm/gosai_data/dataset.csv.gz",
    "mdlm/gosai_data/binary_atac_cell_lines.ckpt",
    "mdlm/outputs_gosai/pretrained.ckpt",
    "mdlm/outputs_gosai/lightning_logs/reward_oracle_ft.ckpt",
    "mdlm/outputs_gosai/lightning_logs/reward_oracle_eval.ckpt",
]

PROTEIN_FILES = [
    "proteindpo_data",
    "pmpnn/outputs/pretrained_if_model.pt",
    "protein_oracle/outputs/reward_oracle_ft.pt",
    "protein_oracle/outputs/reward_oracle_eval.pt",
]

PASS, FAIL, WARN = "PASS", "FAIL", "WARN"
results = []


def record(status, label, detail=""):
    results.append((status, label, detail))
    print(f"[{status}] {label}" + (f" -- {detail}" if detail else ""))


def check_gpu():
    try:
        import torch
    except ImportError:
        record(FAIL, "torch importable", "not installed; is the conda env active?")
        return

    record(PASS, "torch", torch.__version__)

    if not torch.cuda.is_available():
        record(FAIL, "CUDA available", "no GPU visible to torch")
        return

    name = torch.cuda.get_device_name(0)
    total_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
    record(PASS, "GPU", f"{name}, {total_gb:.0f} GB")

    # DRAKES backpropagates through ~50 unrolled diffusion steps. The DNA model
    # is small (a 128-dim CNN over 200bp), so 24 GB is comfortable, but below
    # that you will be trimming batch size.
    if total_gb < 20:
        record(WARN, "GPU memory", "under 20 GB; expect to lower --batch_size")

    # A real allocation catches a driver/toolkit mismatch that
    # torch.cuda.is_available() alone will happily report as fine.
    try:
        x = torch.randn(256, 256, device="cuda")
        (x @ x).sum().item()
        record(PASS, "CUDA matmul", "kernels run")
    except Exception as exc:  # noqa: BLE001 - report whatever the driver says
        record(FAIL, "CUDA matmul", str(exc)[:120])


def check_imports():
    for mod in ["lightning", "hydra", "omegaconf", "transformers", "pandas", "scipy"]:
        try:
            importlib.import_module(mod)
            record(PASS, f"import {mod}")
        except ImportError as exc:
            record(FAIL, f"import {mod}", str(exc)[:80])

    try:
        import grelu

        version = getattr(grelu, "__version__", "unknown")
        # The provided oracle checkpoints were written by gReLU 1.0.2. Later
        # releases changed LightningModel's signature, so loading them raises
        # a confusing unpickling error rather than a version complaint.
        if version.lstrip("v").startswith("1.0.2"):
            record(PASS, "grelu", version)
        else:
            record(WARN, "grelu", f"{version}; the oracles were saved with 1.0.2")
    except ImportError as exc:
        record(FAIL, "import grelu", str(exc)[:80])


def check_data(base_path, files, label):
    base = pathlib.Path(base_path)
    for rel in files:
        target = base / rel
        if target.exists():
            if target.is_dir():
                record(PASS, f"{label}: {rel}", "directory")
            else:
                size_mb = target.stat().st_size / 1024**2
                record(PASS, f"{label}: {rel}", f"{size_mb:.1f} MB")
        else:
            record(FAIL, f"{label}: {rel}", "missing")


def check_patched(repo):
    """The author's hardcoded path must be gone or nothing will resolve."""
    if not repo:
        return
    repo_path = pathlib.Path(repo)
    if not repo_path.is_dir():
        record(WARN, "repo checkout", f"{repo} not found; skipping path check")
        return

    stale = [
        p.relative_to(repo_path)
        for p in repo_path.rglob("*.py")
        if ".git" not in p.parts and "/data/scratch/wangchy/seqft/" in p.read_text(
            encoding="utf-8", errors="ignore"
        )
    ]
    if stale:
        record(FAIL, "base_path patched", f"{len(stale)} file(s) still hardcoded")
    else:
        record(PASS, "base_path patched")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base-path", default=os.environ.get("BASE_PATH", "/workspace/drakes_data"))
    ap.add_argument("--repo", default=os.environ.get("REPO_DIR", "/workspace/DRAKES"))
    ap.add_argument(
        "--experiment",
        choices=["dna", "protein", "both"],
        default="dna",
        help="which experiment's data files to check for",
    )
    args = ap.parse_args()

    print(f"BASE_PATH = {args.base_path}\n")

    check_gpu()
    print()
    check_imports()
    print()
    if args.experiment in ("dna", "both"):
        check_data(args.base_path, DNA_FILES, "dna")
    if args.experiment in ("protein", "both"):
        check_data(args.base_path, PROTEIN_FILES, "protein")
    print()
    check_patched(args.repo)

    failures = [r for r in results if r[0] == FAIL]
    warnings = [r for r in results if r[0] == WARN]
    print(
        f"\n{len(results) - len(failures) - len(warnings)} passed, "
        f"{len(warnings)} warning(s), {len(failures)} failure(s)"
    )
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
