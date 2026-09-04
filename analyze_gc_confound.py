"""Is the physics constraint secretly just measuring base composition?

A reward that correlates almost perfectly with GC content is not a constraint on
sequence *design*, because a model satisfies it by shifting composition without
learning anything about arrangement. This script measures that for both physics
rewards so the answer is a number rather than an intuition.

Two diagnostics:

  1. Correlation with GC content across the realistic composition range.
  2. A shuffle test. Shuffling preserves base composition exactly and destroys
     arrangement, so the change under shuffling is the part of the signal that
     is genuinely about order. This is the more decisive of the two.

    python analyze_gc_confound.py
"""

from __future__ import annotations

import argparse
import random

import torch

from hairpin_reward import HairpinScorer
from physics_reward import duplex_thermodynamics, gc_content, one_hot_from_strings


def sequences_at_gc(n: int, length: int, gc: float, seed: int) -> list[str]:
    """Random sequences with the requested expected GC fraction."""
    rng = random.Random(seed)
    return [
        "".join(
            rng.choice("GC") if rng.random() < gc else rng.choice("AT")
            for _ in range(length)
        )
        for _ in range(n)
    ]


def pearson_r(a: torch.Tensor, b: torch.Tensor) -> float:
    va, vb = a - a.mean(), b - b.mean()
    return ((va * vb).sum() / (va.norm() * vb.norm())).item()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--length", type=int, default=200)
    parser.add_argument("--n-per-bin", type=int, default=400)
    parser.add_argument("--n-shuffle", type=int, default=200)
    parser.add_argument("--stem-length", type=int, default=8)
    args = parser.parse_args()

    scorer = HairpinScorer(args.length, stem_length=args.stem_length)
    metrics = {
        "duplex dG/bp": lambda x: duplex_thermodynamics(x)["delta_g_per_bp"],
        "hairpin dG": scorer.ensemble_free_energy,
    }

    gc_bins = (0.30, 0.40, 0.50, 0.60, 0.70)
    collected = {name: [] for name in metrics}
    gc_values = []

    print(f"length {args.length}, {args.n_per_bin} sequences per GC bin\n")
    header = f"{'GC target':>10} {'GC actual':>10}" + "".join(
        f"{name:>18}" for name in metrics
    )
    print(header)
    print("-" * len(header))

    for gc in gc_bins:
        x = one_hot_from_strings(
            sequences_at_gc(args.n_per_bin, args.length, gc, seed=int(gc * 1000))
        )
        observed_gc = gc_content(x)
        gc_values.append(observed_gc)
        row = f"{gc:>10.2f} {observed_gc.mean():>10.3f}"
        for name, fn in metrics.items():
            values = fn(x)
            collected[name].append(values)
            row += f"{values.mean():>18.4f}"
        print(row)

    gc_all = torch.cat(gc_values)
    print(f"\n{'metric':<16}{'Pearson r':>12}{'R^2':>10}{'range':>12}")
    print("-" * 50)
    for name in metrics:
        values = torch.cat(collected[name])
        r = pearson_r(gc_all, values)
        span = (values.max() - values.min()).item()
        print(f"{name:<16}{r:>12.4f}{r ** 2:>10.4f}{span:>12.4f}")

    # --- the decisive test -------------------------------------------------
    print(
        f"\nShuffle test at 50% GC ({args.n_shuffle} sequences). Shuffling holds\n"
        "composition fixed and destroys order, so this isolates the part of each\n"
        "signal that is genuinely about arrangement.\n"
    )
    rng = random.Random(0)
    originals = sequences_at_gc(args.n_shuffle, args.length, 0.50, seed=99)
    shuffles = []
    for seq in originals:
        chars = list(seq)
        rng.shuffle(chars)
        shuffles.append("".join(chars))

    x_orig = one_hot_from_strings(originals)
    x_shuf = one_hot_from_strings(shuffles)

    print(f"{'metric':<16}{'mean |change|':>16}{'max |change|':>15}{'vs GC range':>14}")
    print("-" * 61)
    for name, fn in metrics.items():
        delta = (fn(x_orig) - fn(x_shuf)).abs()
        gc_range = (torch.cat(collected[name]).max() - torch.cat(collected[name]).min()).item()
        ratio = delta.mean().item() / gc_range if gc_range else float("nan")
        print(
            f"{name:<16}{delta.mean():>16.4f}{delta.max():>15.4f}{ratio:>13.1%}"
        )

    print(
        "\nRead the last column as: how large is the order effect relative to the\n"
        "effect of composition. A few percent means the metric is composition in\n"
        "disguise; comparable magnitude means it is a genuine constraint on design."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
