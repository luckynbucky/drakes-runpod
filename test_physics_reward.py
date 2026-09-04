"""Validation for physics_reward.py.

The physics has to be right before it is worth spending GPU hours on, and a
wrong thermodynamic table produces plausible-looking numbers rather than a
crash. So the parameters and the resulting melting temperatures are checked
against Biopython's independent implementation of the same SantaLucia (1998)
model, and the differentiable path is checked separately.

    pip install biopython
    python test_physics_reward.py
"""

from __future__ import annotations

import math
import random

import torch

from physics_reward import (
    BASES,
    DELTA_H_TABLE,
    DELTA_S_TABLE,
    DuplexStabilityReward,
    dinucleotide_counts,
    duplex_thermodynamics,
    gc_content,
    melting_temperature,
    one_hot_from_strings,
)

try:
    from Bio.SeqUtils import MeltingTemp as bio_mt

    HAVE_BIOPYTHON = True
except ImportError:  # pragma: no cover
    HAVE_BIOPYTHON = False

PASSED, FAILED = [], []


def check(name: str, condition: bool, detail: str = "") -> None:
    (PASSED if condition else FAILED).append(name)
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {name}" + (f" -- {detail}" if detail else ""))


def random_sequences(n: int, length: int, seed: int = 0) -> list[str]:
    rng = random.Random(seed)
    return ["".join(rng.choice(BASES) for _ in range(length)) for _ in range(n)]


# --------------------------------------------------------------------------
# 1. Parameters match the reference table
# --------------------------------------------------------------------------
def test_parameters_match_biopython() -> None:
    if not HAVE_BIOPYTHON:
        print("[SKIP] parameter comparison (biopython not installed)")
        return

    mismatches = []
    for key, (ref_h, ref_s) in bio_mt.DNA_NN3.items():
        if "/" not in key or key.startswith("init"):
            continue
        top = key.split("/")[0]  # e.g. "AA/TT" -> "AA"
        i, j = BASES.index(top[0]), BASES.index(top[1])
        got_h = DELTA_H_TABLE[i, j].item()
        got_s = DELTA_S_TABLE[i, j].item()
        if abs(got_h - ref_h) > 1e-9 or abs(got_s - ref_s) > 1e-9:
            mismatches.append(f"{key}: got ({got_h}, {got_s}) want ({ref_h}, {ref_s})")

    check(
        "nearest-neighbor parameters match Biopython DNA_NN3",
        not mismatches,
        "; ".join(mismatches) if mismatches else "all 10 stacks agree",
    )


# --------------------------------------------------------------------------
# 2. End-to-end melting temperature matches Biopython
# --------------------------------------------------------------------------
def test_melting_temperature_matches_biopython() -> None:
    if not HAVE_BIOPYTHON:
        print("[SKIP] melting temperature comparison (biopython not installed)")
        return

    # Mixed lengths, including short oligos where the initiation terms carry
    # real weight relative to the stacking sum.
    seqs = random_sequences(12, 20, seed=1) + random_sequences(12, 60, seed=2)
    worst = 0.0
    for seq in seqs:
        reference = bio_mt.Tm_NN(seq, nn_table=bio_mt.DNA_NN3, saltcorr=0)
        ours = melting_temperature(one_hot_from_strings([seq])).item()
        worst = max(worst, abs(reference - ours))

    check(
        "melting temperature matches Biopython Tm_NN",
        worst < 1e-6,
        f"max deviation {worst:.2e} C over {len(seqs)} sequences",
    )


# --------------------------------------------------------------------------
# 3. Dinucleotide counts match a naive loop
# --------------------------------------------------------------------------
def test_dinucleotide_counts() -> None:
    seqs = random_sequences(5, 40, seed=3)
    counts = dinucleotide_counts(one_hot_from_strings(seqs))

    worst = 0.0
    for b, seq in enumerate(seqs):
        expected = torch.zeros(4, 4, dtype=torch.float64)
        for first, second in zip(seq, seq[1:]):
            expected[BASES.index(first), BASES.index(second)] += 1
        worst = max(worst, (counts[b] - expected).abs().max().item())

    check("dinucleotide counts match a naive loop", worst < 1e-9, f"max error {worst:.2e}")

    total = counts.sum(dim=(-2, -1))
    check(
        "counts sum to length - 1",
        torch.allclose(total, torch.full_like(total, 39.0)),
        f"got {total.tolist()[:3]}",
    )


# --------------------------------------------------------------------------
# 4. Physical sanity
# --------------------------------------------------------------------------
def test_physical_sanity() -> None:
    x = one_hot_from_strings(random_sequences(64, 200, seed=4))
    thermo = duplex_thermodynamics(x)
    per_bp = thermo["delta_g_per_bp"]

    check(
        "dG per bp for random 200bp DNA is physically plausible",
        bool(((per_bp > -2.2) & (per_bp < -1.0)).all()),
        f"range [{per_bp.min():.3f}, {per_bp.max():.3f}] kcal/mol/bp",
    )

    # poly-GC stacks far more strongly than poly-AT, the classic sanity check.
    gc_rich = one_hot_from_strings(["GC" * 100])
    at_rich = one_hot_from_strings(["AT" * 100])
    g_gc = duplex_thermodynamics(gc_rich)["delta_g_per_bp"].item()
    g_at = duplex_thermodynamics(at_rich)["delta_g_per_bp"].item()
    check(
        "GC-rich duplex is more stable than AT-rich",
        g_gc < g_at,
        f"GC {g_gc:.3f} vs AT {g_at:.3f} kcal/mol/bp",
    )

    # Order dependence is what separates this from a base-composition proxy.
    # Both sequences below are 100 G and 100 C, so GC content is identical at
    # 1.0; only the arrangement differs. Alternating GC/CG stacks are far
    # stronger than runs of GG and CC, so the free energies must differ.
    # (Note that (GC)n and (CG)n would NOT work here: both alternate the same
    # two stacks and differ only at one boundary.)
    blocks = one_hot_from_strings(["G" * 100 + "C" * 100])
    g_blocks = duplex_thermodynamics(blocks)["delta_g_per_bp"].item()
    check(
        "stacking is order-dependent at identical base composition",
        abs(g_gc - g_blocks) > 0.1,
        f"(GC)x100 {g_gc:.3f} vs G100-C100 {g_blocks:.3f} kcal/mol/bp, both 100% GC",
    )


# --------------------------------------------------------------------------
# 5. The differentiable path
# --------------------------------------------------------------------------
def test_gradients_flow() -> None:
    torch.manual_seed(0)
    logits = torch.randn(8, 200, 4, dtype=torch.float64, requires_grad=True)
    relaxed = torch.softmax(logits, dim=-1)

    reward = DuplexStabilityReward(target_dg_per_bp=-1.5)(relaxed)
    check("reward is per-sequence", reward.shape == (8,), f"shape {tuple(reward.shape)}")

    reward.sum().backward()
    grad = logits.grad
    check(
        "gradients reach the sampler logits",
        grad is not None and torch.isfinite(grad).all() and grad.abs().max() > 0,
        f"max |grad| {grad.abs().max():.3e}",
    )


def test_gradient_descends_toward_target() -> None:
    """Optimizing the reward should actually move dG toward the target."""
    torch.manual_seed(0)
    target = -1.90  # more stable than random DNA, so there is real work to do
    logits = torch.randn(4, 200, 4, dtype=torch.float64, requires_grad=True)
    reward_fn = DuplexStabilityReward(target_dg_per_bp=target)

    start = duplex_thermodynamics(torch.softmax(logits, -1))["delta_g_per_bp"].mean().item()
    optimizer = torch.optim.Adam([logits], lr=0.05)
    for _ in range(300):
        optimizer.zero_grad()
        (-reward_fn(torch.softmax(logits, -1)).mean()).backward()
        optimizer.step()
    end = duplex_thermodynamics(torch.softmax(logits, -1))["delta_g_per_bp"].mean().item()

    check(
        "gradient ascent moves dG toward the target",
        abs(end - target) < abs(start - target),
        f"{start:.3f} -> {end:.3f}, target {target:.3f} kcal/mol/bp",
    )


def test_relaxed_matches_hard_in_the_limit() -> None:
    """As the relaxation sharpens, the energy converges on the discrete value.

    The logits are built with a guaranteed margin rather than drawn at random.
    Random logits throw up occasional near-ties between the top two bases, and
    those positions stay genuinely mixed at any finite temperature -- correct
    behaviour, but it puts a floor under the residual that has nothing to do
    with the limit being tested here.
    """
    hard = one_hot_from_strings(random_sequences(4, 100, seed=7))
    logits = hard * 4.0
    hard_g = duplex_thermodynamics(hard)["delta_g_per_bp"]

    deviations = []
    for temperature in (1.0, 0.1, 0.01, 0.001):
        soft = torch.softmax(logits / temperature, dim=-1)
        soft_g = duplex_thermodynamics(soft)["delta_g_per_bp"]
        deviations.append((soft_g - hard_g).abs().max().item())

    # Non-increasing, not strictly decreasing: once the relaxation is sharp
    # enough the deviation hits exactly zero in float64 and stays there, which
    # is the ideal outcome rather than a failure.
    monotone = all(a >= b for a, b in zip(deviations, deviations[1:]))
    converged = deviations[0] > deviations[-1] and deviations[-1] < 1e-9
    check(
        "relaxed energy converges to the discrete energy as temperature falls",
        monotone and converged,
        " -> ".join(f"{d:.2e}" for d in deviations),
    )


# --------------------------------------------------------------------------
# 6. Batching and diagnostics
# --------------------------------------------------------------------------
def test_batch_independence() -> None:
    seqs = random_sequences(6, 120, seed=5)
    batched = duplex_thermodynamics(one_hot_from_strings(seqs))["delta_g"]
    individual = torch.cat(
        [duplex_thermodynamics(one_hot_from_strings([s]))["delta_g"] for s in seqs]
    )
    check(
        "batched and per-sequence results agree",
        torch.allclose(batched, individual, atol=1e-10),
        f"max diff {(batched - individual).abs().max():.2e}",
    )


def test_gc_content_diagnostic() -> None:
    x = one_hot_from_strings(["GC" * 50, "AT" * 50, "ACGT" * 25])
    got = gc_content(x)
    check(
        "GC content diagnostic is correct",
        torch.allclose(got, torch.tensor([1.0, 0.0, 0.5], dtype=torch.float64)),
        f"got {[round(v, 3) for v in got.tolist()]}",
    )


def main() -> int:
    print(f"torch {torch.__version__}, biopython available: {HAVE_BIOPYTHON}\n")
    test_parameters_match_biopython()
    test_melting_temperature_matches_biopython()
    print()
    test_dinucleotide_counts()
    test_physical_sanity()
    print()
    test_gradients_flow()
    test_gradient_descends_toward_target()
    test_relaxed_matches_hard_in_the_limit()
    print()
    test_batch_independence()
    test_gc_content_diagnostic()

    print(f"\n{len(PASSED)} passed, {len(FAILED)} failed")
    if FAILED:
        print("failed: " + ", ".join(FAILED))
    return 1 if FAILED else 0


if __name__ == "__main__":
    raise SystemExit(main())
