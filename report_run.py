"""Read back a run's metrics, and compare runs across a sweep.

Terminal scrollback is the least durable record of a run: a dropped SSH
connection erases it. Every epoch is written to metrics.jsonl as it completes,
so that file, not the terminal, is the record. This reads it back.

    # one run
    python report_run.py /workspace/drakes_data/mdlm/reward_bp_results_final/<run>

    # every run, summarized side by side
    python report_run.py /workspace/drakes_data/mdlm/reward_bp_results_final/*

    # with plots
    python report_run.py <runs...> --plot out/
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

COLUMNS = [
    ("epoch", "epoch", "{:>5.0f}"),
    ("bio", "bio_reward_train_oracle", "{:>8.4f}"),
    ("held-out", "bio_reward_heldout_oracle", "{:>9.4f}"),
    ("hack", "reward_hacking_gap", "{:>+8.4f}"),
    ("phys", "physics_reward", "{:>8.4f}"),
    ("hairpin", "hairpin_dg_ensemble", "{:>8.3f}"),
    ("viol", "hairpin_violation_rate", "{:>6.2f}"),
    ("GC", "gc_content", "{:>6.3f}"),
    ("arr", "arrangement_effect", "{:>+7.3f}"),
    ("spec", "hepg2_specificity", "{:>+7.3f}"),
    ("KL", "kl_loss", "{:>8.3f}"),
    ("peak", "peak_gpu_gb", "{:>6.1f}"),
]


def load(path: str):
    """Accept a run directory or a metrics.jsonl path.

    A glob over the results directory also matches loose files -- the data
    bundle ships the authors' reference finetuned.ckpt there -- so anything
    that is not a directory or a .jsonl file is skipped rather than read as
    text. Binary opened as UTF-8 raises deep inside the read loop and stops
    the whole report.
    """
    if os.path.isdir(path):
        path = os.path.join(path, "metrics.jsonl")
    elif not path.endswith(".jsonl"):
        return None, None
    if not os.path.isfile(path):
        return None, None

    records = []
    try:
        handle = open(path, encoding="utf-8", errors="replace")
    except OSError:
        return None, None
    with handle:
        for line in handle:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    pass  # a partial final line from an interrupted write
    config = None
    config_path = os.path.join(os.path.dirname(path), "config.json")
    if os.path.isfile(config_path):
        with open(config_path) as handle:
            config = json.load(handle)
    return records, config


def print_table(records) -> None:
    header = "".join(f"{name:>{max(6, len(name) + 2)}}" for name, _, _ in COLUMNS)
    print(header)
    print("-" * len(header))
    for record in records:
        row = ""
        for name, key, fmt in COLUMNS:
            value = record.get(key)
            width = max(6, len(name) + 2)
            if value is None or (isinstance(value, float) and value != value):
                row += f"{'-':>{width}}"
            else:
                row += f"{fmt.format(value):>{width}}"
        print(row)


def print_summary(records) -> None:
    first, last = records[0], records[-1]
    print(f"\n{'metric':<28}{'start':>12}{'end':>12}{'change':>12}")
    print("-" * 64)
    for name, key, _ in COLUMNS[1:]:
        a, b = first.get(key), last.get(key)
        if a is None or b is None:
            continue
        print(f"{key:<28}{a:>12.4f}{b:>12.4f}{b - a:>+12.4f}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("runs", nargs="+", help="run directories or metrics.jsonl paths")
    ap.add_argument("--plot", metavar="DIR", help="write comparison plots here")
    ap.add_argument("--quiet", action="store_true", help="summaries only, no tables")
    args = ap.parse_args()

    paths = []
    for pattern in args.runs:
        paths.extend(sorted(glob.glob(pattern)) or [pattern])

    loaded = []
    for path in paths:
        records, config = load(path)
        if not records:
            continue
        loaded.append((os.path.basename(path.rstrip("/")), records, config))

    if not loaded:
        print(
            "No runs with metrics.jsonl found in:\n  "
            + "\n  ".join(paths)
            + "\n\nEach run writes metrics.jsonl into its own directory under\n"
            "  <base_path>/mdlm/reward_bp_results_final/<run name>/\n"
            "so point this at those directories, or at the parent with a glob.",
            file=sys.stderr,
        )
        return 1

    for name, records, config in loaded:
        w_phys = config.get("w_phys") if config else None
        print(f"\n{'=' * 70}\n{name}")
        if config:
            print(
                f"  w_phys={w_phys}  alpha={config.get('alpha')}  "
                f"batch={config.get('batch_size')}x{config.get('num_accum_steps')}  "
                f"epochs completed={len(records)}"
            )
        print("=" * 70)
        if not args.quiet:
            print_table(records)
        print_summary(records)

    if len(loaded) > 1:
        print(f"\n{'=' * 70}\nFINAL EPOCH ACROSS RUNS\n{'=' * 70}")
        print(f"{'w_phys':>8}{'bio':>10}{'held-out':>10}{'hack':>9}{'hairpin':>9}"
              f"{'viol':>7}{'GC':>7}{'arr':>8}")
        print("-" * 68)
        for _, records, config in sorted(
            loaded, key=lambda x: (x[2] or {}).get("w_phys", 0)
        ):
            last = records[-1]
            w = (config or {}).get("w_phys", float("nan"))
            print(
                f"{w:>8.2f}{last.get('bio_reward_train_oracle', 0):>10.4f}"
                f"{last.get('bio_reward_heldout_oracle', 0):>10.4f}"
                f"{last.get('reward_hacking_gap', 0):>+9.4f}"
                f"{last.get('hairpin_dg_ensemble', 0):>9.3f}"
                f"{last.get('hairpin_violation_rate', 0):>7.2f}"
                f"{last.get('gc_content', 0):>7.3f}"
                f"{last.get('arrangement_effect', 0):>+8.3f}"
            )
        print(
            "\nThe Pareto frontier is held-out against hairpin, one point per\n"
            "w_phys. Read GC and arr alongside: a constraint met by dropping GC\n"
            "with arr flat is composition, not design."
        )

    if args.plot:
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            print("\nmatplotlib not installed; skipping plots", file=sys.stderr)
            return 0

        os.makedirs(args.plot, exist_ok=True)
        panels = [
            ("bio_reward_train_oracle", "training oracle"),
            ("bio_reward_heldout_oracle", "held-out oracle"),
            ("reward_hacking_gap", "reward hacking gap"),
            ("hairpin_dg_ensemble", "hairpin dG"),
            ("gc_content", "GC content"),
            ("arrangement_effect", "arrangement effect"),
        ]
        fig, axes = plt.subplots(2, 3, figsize=(15, 8))
        for ax, (key, title) in zip(axes.flat, panels):
            for name, records, config in loaded:
                w = (config or {}).get("w_phys", "?")
                xs = [r["epoch"] for r in records if r.get(key) is not None]
                ys = [r[key] for r in records if r.get(key) is not None]
                if xs:
                    ax.plot(xs, ys, label=f"w_phys={w}")
            ax.set_title(title)
            ax.set_xlabel("epoch")
            ax.grid(alpha=0.3)
        axes.flat[0].legend()
        fig.tight_layout()
        out = os.path.join(args.plot, "training_curves.png")
        fig.savefig(out, dpi=140)
        print(f"\nwrote {out}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
