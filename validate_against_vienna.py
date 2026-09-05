"""Does the differentiable hairpin proxy agree with a real folding model?

hairpin_reward.py is a proxy. It enumerates fixed-length stems, uses an
approximate loop-closure term, ignores interior loops, bulges and multiloops,
and omits the unfolded reference state. Those simplifications are what make it
differentiable and cheap enough to sit inside a training loop, but they mean
the score is not a folding free energy and should not be presented as one.

The question that matters is narrower and answerable: does optimizing the proxy
move sequences in the same direction a real folding model would? This measures
that against ViennaRNA, which implements the full Zuker/McCaskill dynamic
program with the Mathews DNA parameter set.

Rank correlation is the figure to care about. Training does not consume the
proxy's absolute value -- it follows the gradient -- so what matters is whether
the proxy orders sequences the way the reference does.

The GC-matched section is the sharp test. Both measures respond to base
composition, so a correlation computed across varying GC could be two
thermometers agreeing about the weather. Holding GC fixed asks whether they
agree about sequence *arrangement*, which is the thing the proxy is supposed
to contribute.

    pip install ViennaRNA
    python validate_against_vienna.py
"""

from __future__ import annotations

import argparse
import random

import numpy as np
import torch

from hairpin_reward import HairpinScorer
from physics_reward import BASES, gc_content, one_hot_from_strings

try:
    import RNA
except ImportError:  # pragma: no cover
    raise SystemExit("ViennaRNA not installed. pip install ViennaRNA")


def make_model_details(temperature_c: float):
    """Explicit model settings, recorded so the comparison is reproducible."""
    RNA.params_load_DNA_Mathews2004()
    md = RNA.md()
    md.temperature = temperature_c
    return md


def vienna_energies(seqs: list[str], md) -> tuple[np.ndarray, np.ndarray]:
    """MFE and ensemble free energies from ViennaRNA, kcal/mol."""
    mfe, ensemble = [], []
    for seq in seqs:
        fc = RNA.fold_compound(seq, md)
        _, e_mfe = fc.mfe()
        _, e_ens = fc.pf()
        mfe.append(e_mfe)
        ensemble.append(e_ens)
    return np.array(mfe), np.array(ensemble)


def pearson(a: np.ndarray, b: np.ndarray) -> float:
    va, vb = a - a.mean(), b - b.mean()
    denom = np.linalg.norm(va) * np.linalg.norm(vb)
    return float((va * vb).sum() / denom) if denom else float("nan")


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    return pearson(np.argsort(np.argsort(a)).astype(float),
                   np.argsort(np.argsort(b)).astype(float))


def random_seqs(n: int, length: int, gc: float, seed: int) -> list[str]:
    rng = random.Random(seed)
    return [
        "".join(
            rng.choice("GC") if rng.random() < gc else rng.choice("AT")
            for _ in range(length)
        )
        for _ in range(n)
    ]


def report(label: str, proxy: np.ndarray, mfe: np.ndarray, ens: np.ndarray) -> None:
    print(
        f"{label:<28}"
        f"{pearson(proxy, mfe):>10.3f}{spearman(proxy, mfe):>10.3f}"
        f"{pearson(proxy, ens):>12.3f}{spearman(proxy, ens):>10.3f}"
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--length", type=int, default=200)
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--stem-length", type=int, default=8)
    ap.add_argument("--temperature", type=float, default=37.0)
    args = ap.parse_args()

    md = make_model_details(args.temperature)
    scorer = HairpinScorer(args.length, stem_length=args.stem_length)
    baseline = scorer.unstructured_baseline()

    print(
        f"ViennaRNA {RNA.__version__}, DNA parameters (Mathews 2004), "
        f"{args.temperature} C, no added salt correction\n"
        f"proxy: stem length {args.stem_length}, {scorer.n_stems} candidates, "
        f"unstructured baseline {baseline:+.3f} kcal/mol\n"
    )

    def proxy_scores(seqs):
        with torch.no_grad():
            return scorer.ensemble_free_energy(
                one_hot_from_strings(seqs)
            ).numpy()

    header = f"{'sequence set':<28}{'r(MFE)':>10}{'rho(MFE)':>10}{'r(ens)':>12}{'rho(ens)':>10}"
    print(header)
    print("-" * len(header))

    # 1. Across the composition range. Expect a strong correlation, but it is
    #    partly both measures tracking GC.
    all_seqs = []
    for gc in (0.30, 0.40, 0.50, 0.60, 0.70):
        all_seqs += random_seqs(args.n // 5, args.length, gc, seed=int(gc * 1000))
    proxy = proxy_scores(all_seqs)
    mfe, ens = vienna_energies(all_seqs, md)
    report("random, GC 0.30-0.70", proxy, mfe, ens)

    # 2. The sharp test: composition held fixed, so only arrangement varies.
    fixed = random_seqs(args.n, args.length, 0.50, seed=7)
    proxy_f = proxy_scores(fixed)
    mfe_f, ens_f = vienna_energies(fixed, md)
    report("random, GC fixed at 0.50", proxy_f, mfe_f, ens_f)

    # 3. Planted hairpins of increasing stem length, padded with poly-A which
    #    cannot pair with itself. A controlled structural gradient.
    planted = []
    rng = random.Random(11)
    for stem_len in range(4, 16):
        for _ in range(max(1, args.n // 12)):
            stem = "".join(rng.choice(BASES) for _ in range(stem_len))
            arm = stem.translate(str.maketrans("ACGT", "TGCA"))[::-1]
            core = stem + "TTTT" + arm
            pad = args.length - len(core)
            planted.append("A" * (pad // 2) + core + "A" * (pad - pad // 2))
    proxy_p = proxy_scores(planted)
    mfe_p, ens_p = vienna_energies(planted, md)
    report("planted stems, 4-15 bp", proxy_p, mfe_p, ens_p)

    # --- agreement on the decision the constraint actually makes -----------
    print(
        "\nThe training signal is a threshold, so what matters operationally is\n"
        "whether the proxy flags the same sequences a folding model would."
    )
    order_proxy = set(np.argsort(proxy_f)[: args.n // 10].tolist())
    order_mfe = set(np.argsort(mfe_f)[: args.n // 10].tolist())
    overlap = len(order_proxy & order_mfe) / max(1, len(order_proxy))
    print(
        f"  most-structured decile at fixed GC: {overlap:.0%} overlap "
        f"between proxy and ViennaRNA MFE"
    )

    # --- a worked example --------------------------------------------------
    stem, loop = "GCGGCTAG", "TTTT"
    arm = stem.translate(str.maketrans("ACGT", "TGCA"))[::-1]
    core = stem + loop + arm
    pad = args.length - len(core)
    designed = "A" * (pad // 2) + core + "A" * (pad - pad // 2)
    scrambled = "".join(random.Random(3).sample(designed, len(designed)))
    print("\nworked example (identical base composition):")
    for label, seq in (("designed hairpin", designed), ("shuffled", scrambled)):
        fc = RNA.fold_compound(seq, md)
        structure, e_mfe = fc.mfe()
        proxy_v = proxy_scores([seq])[0]
        gc = gc_content(one_hot_from_strings([seq])).item()
        print(
            f"  {label:<18} proxy {proxy_v:>8.3f}   vienna MFE {e_mfe:>8.3f}   "
            f"GC {gc:.3f}"
        )
        print(f"  {'':<18} {structure[85:120]}  (positions 85-120)")

    print(
        "\nInterpretation: a high rank correlation at FIXED GC is the result\n"
        "worth reporting. It says the proxy is not merely re-expressing base\n"
        "composition, and that following its gradient moves sequences the way\n"
        "a full folding model would."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
