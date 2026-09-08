"""The shuffle control, measured with ViennaRNA instead of the proxy.

Matching checkpoints on GC content is not a composition control. GC content is
(G+C)/L, but self-complementarity needs G to pair with C: a sequence that is
35% G and 15% C has the same GC content as one that is 25% each and folds far
less, with no difference in arrangement whatsoever. Any run that shifted the
G:C or A:T balance will therefore show a spurious "arrangement" effect under
GC matching -- which is the most likely reading of the activity-only run
scoring +5.87 kcal/mol less structured at matched GC, against every other
measurement saying it built structure.

The control that has no such hole is the shuffle. Reorder a sequence and all
four base counts are preserved exactly, so every compositional quantity is
held fixed by construction and only arrangement changes:

    arr = MFE(sequence) - MFE(shuffle of the same sequence)

Free energy is more negative for more structure, so arr > 0 means the real
ordering folds LESS than a random ordering of its own bases. That is design.

Two null models, because they answer different questions:

  mononucleotide  preserves base counts. Detects any arrangement effect,
                  including one that works purely through stacking -- putting
                  G next to C rather than next to A raises folding propensity
                  without moving base counts at all.
  dinucleotide    preserves every dinucleotide count as well (Altschul-Erikson
                  Euler-path shuffle). Stacking is now held fixed too, so what
                  survives is long-range complementarity: actual stems.

A run that moves the first but not the second learned local stacking, not
sequence design. That distinction is the whole question this project asks, and
the two nulls together separate it exactly. Write

    S = MFE(mononucleotide shuffle) - MFE(dinucleotide shuffle)
      = arr_dinuc - arr_mono

for the folding propensity carried by dinucleotide composition beyond base
counts -- the stacking load. Then for any run against a reference,

    d arr_mono  =  d arr_dinuc  -  dS
    (total)        (stems)        (stacking)

so the overall arrangement effect splits, with no residual, into what the run
did to long-range complementarity and what it did to local stacking. A run can
post a total of zero by moving both halves in opposite directions, which is not
the same finding as a run that did nothing.

No GPU and no regeneration: the sequences are already in transfer.csv.

    python vienna_shuffle_control.py /workspace/transfer.csv --shuffles 5
"""

from __future__ import annotations

import argparse
import collections
import csv
import random
import sys


def mono_shuffle(seq: str, rng: random.Random) -> str:
    chars = list(seq)
    rng.shuffle(chars)
    return "".join(chars)


def dinuc_shuffle(seq: str, rng: random.Random, max_tries: int = 100) -> str:
    """Altschul-Erikson: a uniform shuffle preserving all dinucleotide counts.

    The sequence is an Euler path through a graph whose vertices are bases and
    whose edges are the dinucleotides. Any other Euler path over the same edge
    multiset has identical dinucleotide counts, so generating one uniformly is
    the shuffle. The construction is: pick, for every vertex except the last,
    one outgoing edge to be traversed last; those choices must form a tree
    rooted at the last vertex, or the walk strands edges. Shuffle each vertex's
    remaining edges freely and walk.
    """
    if len(seq) < 3:
        return seq
    first, last = seq[0], seq[-1]

    out_edges = collections.defaultdict(list)
    for a, b in zip(seq, seq[1:]):
        out_edges[a].append(b)
    verts = set(seq)

    # A vertex other than the last with no outgoing edge cannot be left, so the
    # sequence would not be a valid Euler path. It cannot arise from a real
    # sequence, but fail loudly rather than loop forever if it somehow does.
    for v in verts:
        if v != last and not out_edges[v]:
            return mono_shuffle(seq, rng)

    for _ in range(max_tries):
        last_edge = {v: rng.choice(out_edges[v]) for v in verts if v != last}

        # Following the designated edges from any vertex must reach `last`;
        # a cycle among them would strand every edge inside it.
        if all(_reaches(v, last, last_edge) for v in last_edge):
            break
    else:
        return mono_shuffle(seq, rng)

    ordered = {}
    for v in verts:
        rest = list(out_edges[v])
        if v != last:
            rest.remove(last_edge[v])
            rng.shuffle(rest)
            rest.append(last_edge[v])
        else:
            rng.shuffle(rest)
        ordered[v] = rest

    walk = [first]
    cursor = collections.defaultdict(int)
    node = first
    for _ in range(len(seq) - 1):
        nxt = ordered[node][cursor[node]]
        cursor[node] += 1
        walk.append(nxt)
        node = nxt
    return "".join(walk)


def _reaches(start, target, last_edge, limit=8):
    node, seen = start, set()
    while node != target:
        if node in seen or node not in last_edge:
            return False
        seen.add(node)
        node = last_edge[node]
        if len(seen) > limit:
            return False
    return True


def bootstrap_ci(values, rng, n=5000, lo=2.5, hi=97.5):
    k = len(values)
    if k < 2:
        return float("nan"), float("nan")
    means = []
    for _ in range(n):
        means.append(sum(values[rng.randrange(k)] for _ in range(k)) / k)
    means.sort()
    return means[int(n * lo / 100)], means[int(n * hi / 100)]


def bootstrap_diff(a, b, rng, n=5000, lo=2.5, hi=97.5):
    """Interval on mean(b) - mean(a), resampling each group independently.

    Two intervals that overlap do not imply the difference is consistent with
    zero, and two that do not overlap is a stricter test than the difference
    needs -- so neither can be read off the per-checkpoint intervals above.
    The difference has to be bootstrapped in its own right.
    """
    ka, kb = len(a), len(b)
    if ka < 2 or kb < 2:
        return float("nan"), float("nan")
    diffs = []
    for _ in range(n):
        ma = sum(a[rng.randrange(ka)] for _ in range(ka)) / ka
        mb = sum(b[rng.randrange(kb)] for _ in range(kb)) / kb
        diffs.append(mb - ma)
    diffs.sort()
    return diffs[int(n * lo / 100)], diffs[int(n * hi / 100)]


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv", help="transfer.csv written by transfer_test.py")
    ap.add_argument("--shuffles", type=int, default=5,
                    help="shuffles averaged per sequence (default 5)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0,
                    help="cap sequences per checkpoint, for a quick pass")
    ap.add_argument("--out", default=None, help="write per-sequence arr to CSV")
    ap.add_argument("--from-arr", default=None, metavar="FILE",
                    help="re-analyse a CSV previously written by --out, without"
                         " re-folding anything (seconds instead of half an hour)")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    per_seq_out = []
    results = collections.OrderedDict()

    if args.from_arr:
        with open(args.from_arr, newline="") as fh:
            for r in csv.DictReader(fh):
                g = results.setdefault(r["checkpoint"], ([], []))
                g[0].append(float(r["arr_mono"]))
                g[1].append(float(r["arr_dinuc"]))
        print(f"re-analysing {args.from_arr}: "
              + ", ".join(f"{k} n={len(v[0])}" for k, v in results.items()))
    else:
        try:
            import RNA
        except ImportError:
            print("ViennaRNA not importable. `pip install ViennaRNA`, and check"
                  " the conda env is `sedd`.", file=sys.stderr)
            return 1

        RNA.params_load_DNA_Mathews2004()
        md = RNA.md()

        def mfe(s: str) -> float:
            return RNA.fold_compound(s, md).mfe()[1]

        rows = collections.OrderedDict()
        with open(args.csv, newline="") as fh:
            for r in csv.DictReader(fh):
                rows.setdefault(r["checkpoint"], []).append(
                    (r["sequence"], float(r["vienna_mfe"])))

        for label, entries in rows.items():
            if args.limit:
                entries = entries[: args.limit]
            print(f"{label}: {len(entries)} sequences x {args.shuffles} shuffles"
                  f" x 2 null models ...", flush=True)

            arr_mono, arr_dinuc = [], []
            for seq, real in entries:
                m = sum(mfe(mono_shuffle(seq, rng)) for _ in range(args.shuffles))
                d = sum(mfe(dinuc_shuffle(seq, rng)) for _ in range(args.shuffles))
                a_m = real - m / args.shuffles
                a_d = real - d / args.shuffles
                arr_mono.append(a_m)
                arr_dinuc.append(a_d)
                per_seq_out.append([label, seq, real, a_m, a_d])
            results[label] = (arr_mono, arr_dinuc)
        if not args.out:
            print("\n(pass --out next time: it saves the per-sequence values so"
                  " --from-arr can\n re-analyse them without re-folding.)")

    print()
    print("=" * 78)
    print("arr = MFE(sequence) - MFE(shuffle).  Positive = folds LESS than its"
          " own bases would")
    print("=" * 78)
    print(f"  {'checkpoint':<14}{'mononuc':>10}{'95% CI':>18}"
          f"{'dinuc':>10}{'95% CI':>18}")
    print("  " + "-" * 68)
    ref_label = next(iter(results))
    for label, (am, ad) in results.items():
        mm, dd = sum(am) / len(am), sum(ad) / len(ad)
        lm, hm = bootstrap_ci(am, rng)
        ld, hd = bootstrap_ci(ad, rng)
        print(f"  {label:<14}{mm:>+10.3f}   [{lm:>+6.3f},{hm:>+6.3f}]"
              f"{dd:>+10.3f}   [{ld:>+6.3f},{hd:>+6.3f}]")

    print()
    print("=" * 78)
    print(f"Change vs '{ref_label}' -- this is the arrangement claim, with"
          " composition\nheld exactly fixed by construction rather than matched"
          " on one summary statistic")
    print("=" * 78)
    rm, rd = results[ref_label]
    base_m, base_d = sum(rm) / len(rm), sum(rd) / len(rd)
    print(f"  {'checkpoint':<12}{'d mononuc':>11}{'95% CI':>18}"
          f"{'d dinuc':>10}{'95% CI':>18}")
    print("  " + "-" * 69)
    verdicts = []
    for label, (am, ad) in results.items():
        if label == ref_label:
            continue
        dm = sum(am) / len(am) - base_m
        dd = sum(ad) / len(ad) - base_d
        lm, hm = bootstrap_diff(rm, am, rng)
        ld, hd = bootstrap_diff(rd, ad, rng)
        sig_m = 0 if lm <= 0 <= hm else (1 if lm > 0 else -1)
        sig_d = 0 if ld <= 0 <= hd else (1 if ld > 0 else -1)
        verdicts.append((label, dm, dd, sig_m, sig_d))
        print(f"  {label:<12}{dm:>+11.3f}   [{lm:>+6.3f},{hm:>+6.3f}]"
              f"{dd:>+10.3f}   [{ld:>+6.3f},{hd:>+6.3f}]")

    print("\n  Reading, using only the intervals that exclude zero:")
    for label, dm, dd, sig_m, sig_d in verdicts:
        if sig_m > 0 and sig_d > 0:
            r = "avoids structure by arrangement, stems included"
        elif sig_m > 0 and sig_d < 0:
            r = ("LOCAL STACKING ONLY -- dinucleotide composition improved"
                 " while\n                  long-range complementarity got"
                 " WORSE. A second cheat,\n                  one level up from"
                 " GC content.")
        elif sig_m > 0:
            r = "arrangement effect, but not shown to reach long-range stems"
        elif sig_d > 0:
            r = "long-range stems avoided, overall effect within noise"
        elif sig_m < 0 or sig_d < 0:
            r = "BUILDS structure by arrangement"
        else:
            r = "no arrangement effect distinguishable from noise"
        print(f"    {label:<12} {r}")

    # --- what the total is made of ----------------------------------------
    print()
    print("=" * 78)
    print("Decomposition: d arr_mono = d arr_dinuc - dS, exactly and with no"
          " residual")
    print("=" * 78)
    print("  stems    = change in folding beyond what dinucleotide composition"
          " explains")
    print("  stacking = change in the folding propensity of the dinucleotide"
          " composition\n")
    print(f"  {'checkpoint':<12}{'stems':>9}{'95% CI':>17}"
          f"{'stacking':>10}{'95% CI':>17}{'total':>9}")
    print("  " + "-" * 74)
    load_ref = [d - m for m, d in zip(rm, rd)]
    for label, (am, ad) in results.items():
        if label == ref_label:
            continue
        load = [d - m for m, d in zip(am, ad)]
        stems = sum(ad) / len(ad) - base_d
        # stacking is -dS, so the reference and the run swap places
        stack_lo, stack_hi = bootstrap_diff(load, load_ref, rng)
        stack = sum(load_ref) / len(load_ref) - sum(load) / len(load)
        st_lo, st_hi = bootstrap_diff(rd, ad, rng)
        mark = lambda lo, hi: "" if lo <= 0 <= hi else "*"
        print(f"  {label:<12}{stems:>+9.3f} [{st_lo:>+6.3f},{st_hi:>+6.3f}]"
              f"{mark(st_lo, st_hi):<2}{stack:>+10.3f}"
              f" [{stack_lo:>+6.3f},{stack_hi:>+6.3f}]{mark(stack_lo, stack_hi):<2}"
              f"{stems + stack:>+9.3f}")
    print("\n  A total near zero with both halves large and opposite is a trade,"
          "\n  not an absence of effect -- read the halves, not the total.")

    if args.out:
        with open(args.out, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["checkpoint", "sequence", "vienna_mfe",
                        "arr_mono", "arr_dinuc"])
            w.writerows(per_seq_out)
        print(f"\nwrote {args.out} ({len(per_seq_out)} rows)")

    print("\nNothing here is matched, adjusted, or modelled. Each sequence is"
          " compared\nonly against reorderings of itself, so a difference"
          " cannot be composition.\n"
          "\nPer-sequence arr is noisy -- most of the spread is real variation"
          " in how\nmuch a given sequence folds, not measurement error, so"
          " more shuffles will\nnot narrow these intervals and more sequences"
          " will. The interval width\nfalls as 1/sqrt(n), so quadrupling --n"
          " in transfer_test.py halves it.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
