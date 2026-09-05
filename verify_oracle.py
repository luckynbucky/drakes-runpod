"""Check that a gReLU reward oracle actually loaded, by testing what it predicts.

This exists because of --skip_grelu_artifact. That flag skips downloading
Enformer's pretrained weights on the argument that the DRAKES checkpoint
overwrites them all with strict=True. The argument is sound, but a mis-loaded
oracle does not announce itself -- it returns plausible-looking numbers that
happen to be meaningless, and every downstream result inherits the problem
silently.

So rather than trusting the argument, test the conclusion: run the oracle over
real measured enhancer sequences and see whether its predictions track the
measurements. A correctly loaded oracle correlates strongly. A randomly
initialized one correlates around zero.

    python verify_oracle.py --base-path /workspace/drakes_data [--skip-artifact]

Run it from DRAKES/drakes_dna/, which is where the oracle module lives.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd


def pearson(a, b) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    va, vb = a - a.mean(), b - b.mean()
    denom = np.linalg.norm(va) * np.linalg.norm(vb)
    return float((va * vb).sum() / denom) if denom else float("nan")


def spearman(a, b) -> float:
    return pearson(np.argsort(np.argsort(a)), np.argsort(np.argsort(b)))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base-path", default="/workspace/drakes_data")
    ap.add_argument("--n", type=int, default=2000, help="sequences to score")
    ap.add_argument("--mode", default="train", choices=["train", "eval"])
    ap.add_argument(
        "--skip-artifact",
        action="store_true",
        help="apply the grelu_offline patch before loading",
    )
    args = ap.parse_args()

    if args.skip_artifact:
        import grelu_offline

        grelu_offline.enable()

    csv = os.path.join(args.base_path, "mdlm/gosai_data/processed_data/gosai_all.csv")
    print(f"reading {csv}")
    frame = pd.read_csv(csv)
    print(f"  {len(frame)} rows, columns: {list(frame.columns)}")

    seq_col = next(
        (c for c in frame.columns if c.lower() in ("seq", "sequence", "seqs")), None
    )
    if seq_col is None:
        print("could not find a sequence column", file=sys.stderr)
        return 1

    # The measured activities are the numeric columns; the oracle emits three
    # outputs, one per cell type, in the dataset's own column order.
    label_cols = [
        c
        for c in frame.columns
        if c != seq_col and pd.api.types.is_numeric_dtype(frame[c])
    ]
    print(f"  sequence column: {seq_col}")
    print(f"  numeric columns: {label_cols}")
    if not label_cols:
        print("no numeric label columns found", file=sys.stderr)
        return 1

    sample = frame.sample(n=min(args.n, len(frame)), random_state=0)
    seqs = sample[seq_col].tolist()
    print(f"\nscoring {len(seqs)} sequences with the '{args.mode}' oracle...")

    import oracle

    preds = oracle.cal_gosai_pred(seqs, mode=args.mode)
    preds = np.asarray(preds)
    print(f"  predictions shape {preds.shape}")

    if preds.ndim == 1:
        preds = preds[:, None]

    print(f"\n{'measured column':<24}{'vs pred':<10}{'pearson':>10}{'spearman':>11}")
    print("-" * 55)
    best = 0.0
    for j in range(preds.shape[1]):
        for col in label_cols:
            r = pearson(preds[:, j], sample[col].values)
            rho = spearman(preds[:, j], sample[col].values)
            if abs(r) > abs(best):
                best = r
            print(f"{col:<24}{f'out[{j}]':<10}{r:>10.3f}{rho:>11.3f}")

    print(f"\nstrongest correlation: {best:.3f}")
    if abs(best) > 0.5:
        print(
            "\nPASS. The oracle's predictions track the measured activities, so\n"
            "the checkpoint weights loaded correctly. If this was run with\n"
            "--skip-artifact, that confirms the bypass is sound rather than\n"
            "merely argued."
        )
        return 0
    if abs(best) > 0.2:
        print(
            "\nWEAK. Some signal, but less than a properly loaded oracle should\n"
            "show. Investigate before trusting any reward computed from it."
        )
        return 1
    print(
        "\nFAIL. Predictions are uncorrelated with measurement, which is what a\n"
        "randomly initialized model looks like. The checkpoint did not load.\n"
        "Do not train against this oracle."
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
