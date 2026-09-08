"""Was the ViennaRNA gain arrangement, or was it composition again?

transfer_test.py reports whether ViennaRNA MFE improved alongside the proxy.
That comparison is confounded in exactly the way the whole project is about:
GC content drives folding energy, so a run that collapsed composition will show
a large MFE "improvement" without having designed anything.

This partials composition out three ways, from weakest to strongest:

  1. Between-checkpoint regression of mean MFE on mean GC. If R^2 is near 1,
     the headline transfer result is composition and nothing else.
  2. ANCOVA. Fit the MFE-vs-GC slope *within* checkpoints, where it is not
     driven by the spread between them, then adjust every checkpoint's mean
     MFE to a common reference GC. Differences that survive are arrangement.
  3. Stratified matching. Compare checkpoints only on sequences that share a
     GC value, in bins of 0.005, with no model of the GC relationship at all.
     This is the one to believe; it is also the one that most often reports
     insufficient overlap, which is itself the honest answer.

Also reported: the partial correlation between the proxy and ViennaRNA with GC
held fixed. The raw within-checkpoint correlation flatters the proxy, because
both quantities track composition.

    python analyze_transfer.py /workspace/transfer.csv
"""

from __future__ import annotations

import argparse
import collections
import csv
import sys

import numpy as np

RNG = np.random.default_rng(0)


def ols(x, y):
    """Slope, intercept, R^2 for a simple linear fit."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 3 or np.ptp(x) == 0:
        return float("nan"), float("nan"), float("nan")
    slope, intercept = np.polyfit(x, y, 1)
    resid = y - (slope * x + intercept)
    ss_tot = ((y - y.mean()) ** 2).sum()
    r2 = 1.0 - (resid**2).sum() / ss_tot if ss_tot else float("nan")
    return float(slope), float(intercept), float(r2)


def pearson(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    va, vb = a - a.mean(), b - b.mean()
    d = np.linalg.norm(va) * np.linalg.norm(vb)
    return float((va * vb).sum() / d) if d else float("nan")


def partial_corr(a, b, c):
    """Correlation of a and b with c held fixed."""
    r_ab, r_ac, r_bc = pearson(a, b), pearson(a, c), pearson(b, c)
    denom = np.sqrt((1 - r_ac**2) * (1 - r_bc**2))
    return float((r_ab - r_ac * r_bc) / denom) if denom else float("nan")


def load(path):
    groups = collections.OrderedDict()
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            g = groups.setdefault(row["checkpoint"], collections.defaultdict(list))
            g["mfe"].append(float(row["vienna_mfe"]))
            g["proxy"].append(float(row["proxy_dg"]))
            g["gc"].append(float(row["gc"]))
            g["act"].append(float(row["activity"]))
    for g in groups.values():
        for k in g:
            g[k] = np.asarray(g[k], float)
    return groups


def pooled_within_slope(groups):
    """MFE-vs-GC slope fitted within checkpoints, then pooled.

    The between-checkpoint slope is fitted to five points spanning the entire
    GC range, so it absorbs any real arrangement difference into the slope
    itself. Fitting inside each checkpoint, where composition varies for
    reasons unrelated to the training run, keeps the two separable.
    """
    num = den = 0.0
    for g in groups.values():
        gc, mfe = g["gc"], g["mfe"]
        v = ((gc - gc.mean()) ** 2).sum()
        num += ((gc - gc.mean()) * (mfe - mfe.mean())).sum()
        den += v
    return num / den if den else float("nan")


def stratified_match(a, b, width=0.005, min_per_bin=3):
    """Mean MFE difference (b - a) over GC bins both checkpoints occupy."""
    bins_a = collections.defaultdict(list)
    bins_b = collections.defaultdict(list)
    for gc, mfe in zip(a["gc"], a["mfe"]):
        bins_a[int(gc / width)].append(mfe)
    for gc, mfe in zip(b["gc"], b["mfe"]):
        bins_b[int(gc / width)].append(mfe)

    shared = sorted(set(bins_a) & set(bins_b))
    diffs, weights, n_a, n_b = [], [], 0, 0
    for k in shared:
        if len(bins_a[k]) < min_per_bin or len(bins_b[k]) < min_per_bin:
            continue
        diffs.append(np.mean(bins_b[k]) - np.mean(bins_a[k]))
        w = min(len(bins_a[k]), len(bins_b[k]))
        weights.append(w)
        n_a += len(bins_a[k])
        n_b += len(bins_b[k])

    if not diffs:
        return None
    diffs, weights = np.asarray(diffs), np.asarray(weights, float)
    est = float((diffs * weights).sum() / weights.sum())

    # Bootstrap over bins: the bins are the independent units here, not the
    # sequences, because within a bin composition is held fixed by construction.
    boots = []
    idx = np.arange(len(diffs))
    for _ in range(2000):
        s = RNG.choice(idx, size=len(idx), replace=True)
        boots.append((diffs[s] * weights[s]).sum() / weights[s].sum())
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return est, float(lo), float(hi), len(diffs), n_a, n_b


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv", help="transfer.csv written by transfer_test.py")
    ap.add_argument("--reference", default=None,
                    help="checkpoint to compare against (default: first row's)")
    args = ap.parse_args()

    groups = load(args.csv)
    if not groups:
        print("no rows", file=sys.stderr)
        return 1
    ref = args.reference or next(iter(groups))
    if ref not in groups:
        print(f"no checkpoint named {ref!r}; have {list(groups)}", file=sys.stderr)
        return 1

    labels = list(groups)
    mean_gc = np.array([groups[l]["gc"].mean() for l in labels])
    mean_mfe = np.array([groups[l]["mfe"].mean() for l in labels])

    # --- 1. how much of the headline result is just composition? ----------
    slope_b, icept_b, r2_b = ols(mean_gc, mean_mfe)
    print("=" * 74)
    print("1. Between-checkpoint: is the transfer result just GC content?")
    print("=" * 74)
    print(f"  MFE = {icept_b:+.2f} {slope_b:+.2f} x GC     R^2 = {r2_b:.4f}"
          f"   (n = {len(labels)} checkpoints)")
    if r2_b > 0.9:
        print(f"  {r2_b*100:.1f}% of the between-checkpoint MFE spread is composition.")
        print("  A raw 'both improved' verdict cannot distinguish design from collapse.")
    print()

    # --- 2. ANCOVA ---------------------------------------------------------
    slope_w = pooled_within_slope(groups)
    ref_gc = groups[ref]["gc"].mean()
    print("=" * 74)
    print(f"2. ANCOVA: mean MFE adjusted to GC = {ref_gc:.3f}")
    print("=" * 74)
    print(f"  pooled within-checkpoint slope: {slope_w:+.2f} kcal/mol per unit GC")
    print(f"  (between-checkpoint slope was {slope_b:+.2f} -- if these disagree"
          " badly,\n   the between fit was absorbing real differences)\n")
    print(f"  {'checkpoint':<14}{'GC':>8}{'raw MFE':>10}{'adjusted':>11}"
          f"{'vs ' + ref:>14}")
    print("  " + "-" * 57)
    adj = {}
    for l in labels:
        g = groups[l]
        adj[l] = g["mfe"].mean() - slope_w * (g["gc"].mean() - ref_gc)
    for l in labels:
        d = adj[l] - adj[ref]
        print(f"  {l:<14}{groups[l]['gc'].mean():>8.3f}{groups[l]['mfe'].mean():>10.2f}"
              f"{adj[l]:>11.2f}{d:>+14.2f}")
    print("\n  Positive 'vs' means less structured than the reference at equal"
          " composition\n  -- that is arrangement, and it is the only column"
          " here that is.\n")

    # --- 3. stratified matching -------------------------------------------
    print("=" * 74)
    print(f"3. Stratified matching against '{ref}' (GC bins of 0.005)")
    print("=" * 74)
    print(f"  {'checkpoint':<14}{'delta MFE':>11}{'95% CI':>20}{'bins':>7}"
          f"{'n_ref':>7}{'n':>6}")
    print("  " + "-" * 65)
    for l in labels:
        if l == ref:
            continue
        res = stratified_match(groups[ref], groups[l])
        if res is None:
            gr, gl = groups[ref]["gc"], groups[l]["gc"]
            print(f"  {l:<14}{'--':>11}{'no GC overlap':>20}"
                  f"      GC {gl.min():.3f}-{gl.max():.3f} vs"
                  f" {gr.min():.3f}-{gr.max():.3f}")
            continue
        est, lo, hi, nb, na_, nb_ = res
        star = "  *" if (lo > 0) or (hi < 0) else ""
        print(f"  {l:<14}{est:>+11.2f}   [{lo:>+6.2f}, {hi:>+6.2f}]"
              f"{nb:>7}{na_:>7}{nb_:>6}{star}")
    print("\n  * = confidence interval excludes zero. Where a row says no overlap,"
          "\n  the two checkpoints share no composition and cannot be compared"
          " on\n  arrangement at all -- which is the finding, not a gap in it.\n")

    # --- 4. does the proxy see arrangement, or only composition? -----------
    print("=" * 74)
    print("4. Proxy vs ViennaRNA, before and after removing GC")
    print("=" * 74)
    print(f"  {'checkpoint':<14}{'raw r':>9}{'partial r':>12}{'GC sd':>9}")
    print("  " + "-" * 44)
    for l in labels:
        g = groups[l]
        print(f"  {l:<14}{pearson(g['proxy'], g['mfe']):>9.3f}"
              f"{partial_corr(g['proxy'], g['mfe'], g['gc']):>12.3f}"
              f"{g['gc'].std():>9.4f}")
    print("\n  The partial column is what the proxy is worth as a folding model."
          "\n  The raw column is what it is worth as a GC detector.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
