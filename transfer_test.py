"""Do the proxy's gains transfer to a real folding model?

Every structural improvement measured during training was measured on the proxy
that training optimized. That is circular: a model can move a proxy without
moving the quantity the proxy stands for. This generates sequences from each
fine-tuned checkpoint and scores them with ViennaRNA -- a full Zuker/McCaskill
folding model with the Mathews DNA parameters, which never entered training.

Also scored: base composition, and both activity oracles, so a structural gain
that came at the cost of biology is visible rather than hidden.

    pip install ViennaRNA
    python transfer_test.py --base-path /workspace/drakes_data \\
        --checkpoints <run_dir_or_ckpt> [more...] --n 256

Run from DRAKES/drakes_dna/, or pass --drakes-dir.
"""

from __future__ import annotations

# A new tmux window or SSH session starts in conda's base environment, where
# none of this is installed, and the resulting ModuleNotFoundError reads as a
# missing package rather than a missing activation. Say which it is.
try:
    import numpy  # noqa: F401
    import torch  # noqa: F401
    import wandb  # noqa: F401
except ImportError as _exc:  # pragma: no cover
    raise SystemExit(
        f"Cannot import {getattr(_exc, 'name', 'a required module')!r}.\n\n"
        "This is almost always the wrong conda environment rather than a missing\n"
        "install -- a new shell starts in base. Run:\n\n"
        "    conda activate sedd\n\n"
        "If conda itself is not found (a fresh container wipes /root), first run:\n"
        "    source /workspace/miniconda3/etc/profile.d/conda.sh\n"
    )

import argparse
import glob
import os
import sys

import numpy as np
import torch


def pearson(a, b) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    va, vb = a - a.mean(), b - b.mean()
    d = np.linalg.norm(va) * np.linalg.norm(vb)
    return float((va * vb).sum() / d) if d else float("nan")


def spearman(a, b) -> float:
    return pearson(np.argsort(np.argsort(a)), np.argsort(np.argsort(b)))


def resolve(path: str) -> str | None:
    """Accept a checkpoint file or a run directory containing one."""
    if os.path.isfile(path):
        return path
    if os.path.isdir(path):
        for name in ("checkpoint_final.pt", "checkpoint_latest.pt"):
            candidate = os.path.join(path, name)
            if os.path.isfile(candidate):
                return candidate
        numbered = sorted(glob.glob(os.path.join(path, "checkpoint_epoch*.pt")))
        if numbered:
            return numbered[-1]
    return None


def load_state(path: str):
    """Handle both this project's checkpoints and DRAKES's own."""
    blob = torch.load(path, map_location="cpu")
    for key in ("model", "state_dict"):
        if isinstance(blob, dict) and key in blob:
            return blob[key], blob.get("args", {})
    return blob, {}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base-path", default="/workspace/drakes_data")
    ap.add_argument("--drakes-dir", default="/workspace/DRAKES/drakes_dna")
    ap.add_argument("--checkpoints", nargs="*", default=[],
                    help="run directories or .pt files; the pretrained model is always included")
    ap.add_argument("--n", type=int, default=256, help="sequences per checkpoint")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--temperature", type=float, default=37.0)
    ap.add_argument("--skip-artifact", action="store_true", default=True)
    ap.add_argument("--out", default=None, help="write per-sequence scores to this CSV")
    args = ap.parse_args()

    if os.path.isdir(args.drakes_dir) and args.drakes_dir not in sys.path:
        sys.path.insert(0, args.drakes_dir)
    if args.skip_artifact:
        import grelu_offline
        grelu_offline.enable()

    try:
        import RNA
    except ImportError:
        print("ViennaRNA not installed.  pip install ViennaRNA", file=sys.stderr)
        return 1

    from hydra import compose, initialize
    from hydra.core.global_hydra import GlobalHydra

    import dataloader_gosai
    import diffusion_gosai_update
    import oracle
    from hairpin_reward import HairpinScorer
    from physics_reward import gc_content, one_hot_from_strings

    RNA.params_load_DNA_Mathews2004()
    md = RNA.md()
    md.temperature = args.temperature

    ckpt_path = os.path.join(args.base_path, "mdlm/outputs_gosai/pretrained.ckpt")
    GlobalHydra.instance().clear()
    initialize(config_path="configs_gosai", job_name="transfer_test")
    cfg = compose(config_name="config_gosai.yaml")
    cfg.eval.checkpoint_path = ckpt_path

    print("loading models...")
    model = diffusion_gosai_update.Diffusion.load_from_checkpoint(ckpt_path, config=cfg)
    reward = oracle.get_gosai_oracle(mode="train").to(model.device).eval()
    reward_eval = oracle.get_gosai_oracle(mode="eval").to(model.device).eval()
    for m in (reward, reward_eval):
        m.requires_grad_(False)
    scorer = HairpinScorer(cfg.model.length, stem_length=10)

    targets = [("pretrained", None)]
    for path in args.checkpoints:
        resolved = resolve(path)
        if resolved is None:
            print(f"  no checkpoint found under {path}, skipping", file=sys.stderr)
            continue
        state, saved_args = load_state(resolved)
        label = saved_args.get("name") or os.path.basename(os.path.dirname(resolved))[:28]
        targets.append((label, (resolved, state)))

    rows = []
    per_seq = []
    for label, payload in targets:
        if payload is None:
            model.load_state_dict(
                torch.load(ckpt_path, map_location="cpu")["state_dict"]
            )
        else:
            resolved, state = payload
            model.load_state_dict(state)
            print(f"  loaded {resolved}")
        model.eval()

        seqs = []
        while len(seqs) < args.n:
            with torch.no_grad():
                tokens = model._sample(eval_sp_size=min(args.batch, args.n - len(seqs)))
            seqs.extend(dataloader_gosai.batch_dna_detokenize(tokens.cpu().numpy()))
        seqs = seqs[: args.n]
        print(f"{label}: generated {len(seqs)} sequences, folding with ViennaRNA...")

        mfe = np.array([RNA.fold_compound(s, md).mfe()[1] for s in seqs])
        x = one_hot_from_strings(seqs).float()
        proxy = scorer.ensemble_free_energy(x).numpy()
        gc = gc_content(x).numpy()

        xg = x.to(model.device).transpose(1, 2)
        with torch.no_grad():
            act = reward(xg).squeeze(-1)[:, 0].cpu().numpy()
            act_held = reward_eval(xg).squeeze(-1)[:, 0].cpu().numpy()

        rows.append((label, mfe, proxy, gc, act, act_held))
        for s, a, b, c, d, e in zip(seqs, mfe, proxy, gc, act, act_held):
            per_seq.append((label, s, a, b, c, d, e))

    print(f"\n{'checkpoint':<16}{'ViennaRNA MFE':>15}{'proxy dG':>11}{'GC':>8}"
          f"{'activity':>10}{'held-out':>10}")
    print("-" * 70)
    base = None
    for label, mfe, proxy, gc, act, act_held in rows:
        if base is None:
            base = mfe.mean()
        print(f"{label:<16}{mfe.mean():>15.2f}{proxy.mean():>11.3f}{gc.mean():>8.3f}"
              f"{act.mean():>10.4f}{act_held.mean():>10.4f}")

    print(f"\n{'checkpoint':<16}{'ΔMFE vs pretrained':>20}{'Δproxy':>10}   transfer?")
    print("-" * 62)
    base_mfe, base_proxy = rows[0][1].mean(), rows[0][2].mean()
    for label, mfe, proxy, *_ in rows[1:]:
        d_mfe, d_proxy = mfe.mean() - base_mfe, proxy.mean() - base_proxy
        if d_proxy <= 0.05:
            verdict = "proxy did not improve"
        elif d_mfe > 0.05:
            verdict = "YES — both improved"
        else:
            verdict = "NO — proxy moved, ViennaRNA did not"
        print(f"{label:<16}{d_mfe:>+20.2f}{d_proxy:>+10.3f}   {verdict}")

    print("\nper-sequence agreement within each checkpoint (proxy vs ViennaRNA MFE):")
    for label, mfe, proxy, *_ in rows:
        print(f"  {label:<16}pearson {pearson(proxy, mfe):>6.3f}   "
              f"spearman {spearman(proxy, mfe):>6.3f}")

    if args.out:
        import csv
        with open(args.out, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["checkpoint", "sequence", "vienna_mfe", "proxy_dg",
                        "gc", "activity", "activity_heldout"])
            w.writerows(per_seq)
        print(f"\nwrote {args.out} ({len(per_seq)} rows)")

    print(
        "\nBoth columns are in kcal/mol and less negative is less structured, so a\n"
        "positive delta is an improvement. The result worth reporting is the third\n"
        "column: a proxy gain that does not move ViennaRNA is a proxy that was\n"
        "optimized rather than a property that was improved."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
