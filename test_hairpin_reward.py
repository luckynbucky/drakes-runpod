"""Validation for hairpin_reward.py.

There is no Biopython equivalent to check against here, so these tests pin down
behaviour instead: a designed hairpin must score far below an unstructured
sequence, the score must respond to *arrangement* rather than composition
(which is the entire reason this exists), and the gradient path must work.

    python test_hairpin_reward.py
"""

from __future__ import annotations

import random

import torch

from hairpin_reward import HairpinPenaltyReward, HairpinScorer
from physics_reward import BASES, gc_content, one_hot_from_strings

PASSED, FAILED = [], []

# A perfect 8bp stem closed by a 4-nucleotide loop. The arms are exact reverse
# complements, so every pair in the stem forms.
STEM = "GCGGCTAG"
LOOP = "TTTT"
ARM = "CTAGCCGC"  # reverse complement of STEM
HAIRPIN = STEM + LOOP + ARM

# Poly-A pads to full length without introducing structure of its own: A does
# not Watson-Crick pair with A, so no padding stem can form.
LENGTH = 200


def check(name: str, condition: bool, detail: str = "") -> None:
    (PASSED if condition else FAILED).append(name)
    print(f"[{'PASS' if condition else 'FAIL'}] {name}" + (f" -- {detail}" if detail else ""))


def padded(core: str, length: int = LENGTH) -> str:
    pad = length - len(core)
    left = pad // 2
    return "A" * left + core + "A" * (pad - left)


def shuffled(seq: str, seed: int = 0) -> str:
    chars = list(seq)
    random.Random(seed).shuffle(chars)
    return "".join(chars)


def test_designed_hairpin_is_detected() -> None:
    scorer = HairpinScorer(LENGTH)
    x = one_hot_from_strings([padded(HAIRPIN), "A" * LENGTH])
    dg = scorer.ensemble_free_energy(x)

    # Not zero: a partition function over many unfavourable stems still
    # contributes a small positive free energy. The scorer can state that
    # baseline exactly, and poly-A should sit right on it since no pair forms.
    baseline = scorer.unstructured_baseline()
    check(
        "unstructured poly-A sits on the computed no-pairing baseline",
        abs(dg[1].item() - baseline) < 1e-9,
        f"dG = {dg[1]:.3f}, baseline = {baseline:.3f} kcal/mol",
    )
    check(
        "designed hairpin is strongly negative",
        dg[0].item() < -4.0,
        f"dG = {dg[0]:.3f} kcal/mol",
    )
    check(
        "hairpin separates clearly from unstructured",
        (dg[1] - dg[0]).item() > 4.0,
        f"separation {(dg[1] - dg[0]).item():.3f} kcal/mol",
    )


def test_responds_to_order_not_composition() -> None:
    """The property duplex dG lacked: sensitivity to arrangement alone."""
    scorer = HairpinScorer(LENGTH)
    original = padded(HAIRPIN)
    scrambled = shuffled(original, seed=1)

    x = one_hot_from_strings([original, scrambled])
    dg = scorer.ensemble_free_energy(x)
    gc = gc_content(x)

    check(
        "shuffling preserves composition exactly",
        abs(gc[0].item() - gc[1].item()) < 1e-12,
        f"GC {gc[0]:.4f} vs {gc[1]:.4f}",
    )
    check(
        "shuffling destroys the hairpin at identical composition",
        (dg[1] - dg[0]).item() > 3.0,
        f"intact {dg[0]:.3f} -> shuffled {dg[1]:.3f} kcal/mol",
    )


def test_stem_geometry() -> None:
    scorer = HairpinScorer(length=40, stem_length=5, min_loop=3)
    loops = (scorer.pair_a_j[:, 0] - scorer.pair_a_i[:, 0]) - 2 * 5 + 1
    check(
        "every enumerated stem leaves a legal loop",
        bool((loops >= 3).all()),
        f"{scorer.n_stems} stems, smallest loop {loops.min().item()}",
    )
    check(
        "stem indices stay inside the sequence",
        bool(
            (scorer.pair_b_i.max() < 40)
            and (scorer.pair_b_j.min() >= 0)
            and (scorer.pair_a_j.max() < 40)
        ),
        "index bounds respected",
    )


def test_gradients_flow() -> None:
    """Three separate claims, because the penalty is deliberately one-sided."""
    torch.manual_seed(0)
    scorer = HairpinScorer(LENGTH)
    reward_fn = HairpinPenaltyReward(LENGTH, tolerance=-3.0)

    reward = reward_fn(torch.softmax(torch.randn(4, LENGTH, 4, dtype=torch.float64), -1))
    check("reward is per-sequence", reward.shape == (4,), f"shape {tuple(reward.shape)}")

    # 1. The underlying free energy is differentiable everywhere.
    logits = torch.randn(4, LENGTH, 4, dtype=torch.float64, requires_grad=True)
    scorer.ensemble_free_energy(torch.softmax(logits, -1)).sum().backward()
    check(
        "free energy has gradients on arbitrary sequences",
        logits.grad is not None
        and torch.isfinite(logits.grad).all()
        and logits.grad.abs().max() > 0,
        f"max |grad| {logits.grad.abs().max():.3e}",
    )

    # 2. The penalty has gradients where it is active, i.e. on a violator.
    structured = (one_hot_from_strings([padded(HAIRPIN)]) * 4.0).requires_grad_(True)
    reward_fn(torch.softmax(structured, -1)).sum().backward()
    check(
        "penalty has gradients on a sequence that violates the constraint",
        structured.grad is not None
        and torch.isfinite(structured.grad).all()
        and structured.grad.abs().max() > 0,
        f"max |grad| {structured.grad.abs().max():.3e}",
    )

    # 3. And is exactly flat where the constraint is already satisfied. This is
    # the intended semantics of a constraint rather than an objective, but it
    # matters during training: if every sampled sequence is already within
    # tolerance, this reward term contributes no gradient at all and the run
    # will look as though the physics is doing nothing.
    satisfied = (one_hot_from_strings(["A" * LENGTH]) * 4.0).requires_grad_(True)
    reward_fn(torch.softmax(satisfied, -1)).sum().backward()
    check(
        "penalty is exactly flat where the constraint is satisfied",
        satisfied.grad is not None and satisfied.grad.abs().max() == 0,
        f"max |grad| {satisfied.grad.abs().max():.3e} (zero by design)",
    )


def test_penalty_is_one_sided() -> None:
    reward_fn = HairpinPenaltyReward(LENGTH, tolerance=-3.0)
    x = one_hot_from_strings([padded(HAIRPIN), "A" * LENGTH])
    reward = reward_fn(x)

    check(
        "unstructured sequence is not penalized",
        abs(reward[1].item()) < 1e-9,
        f"reward {reward[1].item():.3e}",
    )
    check(
        "structured sequence is penalized",
        reward[0].item() < -1.0,
        f"reward {reward[0].item():.3f}",
    )


def test_optimization_removes_hairpins() -> None:
    """Descending the penalty should actually dismantle a planted hairpin."""
    torch.manual_seed(0)
    reward_fn = HairpinPenaltyReward(LENGTH, tolerance=-3.0)
    hard = one_hot_from_strings([padded(HAIRPIN)])
    logits = (hard * 4.0).clone().requires_grad_(True)

    start = reward_fn.scorer.ensemble_free_energy(torch.softmax(logits, -1)).item()
    optimizer = torch.optim.Adam([logits], lr=0.1)
    for _ in range(200):
        optimizer.zero_grad()
        (-reward_fn(torch.softmax(logits, -1)).mean()).backward()
        optimizer.step()
    end = reward_fn.scorer.ensemble_free_energy(torch.softmax(logits, -1)).item()

    check(
        "optimization relaxes the planted hairpin",
        end > start + 1.0,
        f"{start:.3f} -> {end:.3f} kcal/mol",
    )


def test_batch_independence() -> None:
    scorer = HairpinScorer(LENGTH)
    seqs = [padded(HAIRPIN), "A" * LENGTH, shuffled(padded(HAIRPIN), 2)]
    batched = scorer.ensemble_free_energy(one_hot_from_strings(seqs))
    individual = torch.cat(
        [scorer.ensemble_free_energy(one_hot_from_strings([s])) for s in seqs]
    )
    check(
        "batched and per-sequence results agree",
        torch.allclose(batched, individual, atol=1e-10),
        f"max diff {(batched - individual).abs().max():.2e}",
    )


def test_stem_length_sensitivity() -> None:
    """A longer stem requirement should find the planted 8bp stem, not more."""
    x = one_hot_from_strings([padded(HAIRPIN)])
    energies = {}
    for stem_length in (4, 6, 8):
        energies[stem_length] = HairpinScorer(
            LENGTH, stem_length=stem_length
        ).ensemble_free_energy(x).item()

    check(
        "longer stems resolve as more stable structure",
        energies[8] < energies[4],
        ", ".join(f"W={w}: {e:.2f}" for w, e in energies.items()),
    )


def main() -> int:
    print(f"torch {torch.__version__}\n")
    test_designed_hairpin_is_detected()
    test_responds_to_order_not_composition()
    print()
    test_stem_geometry()
    test_stem_length_sensitivity()
    print()
    test_gradients_flow()
    test_penalty_is_one_sided()
    test_optimization_removes_hairpins()
    print()
    test_batch_independence()

    print(f"\n{len(PASSED)} passed, {len(FAILED)} failed")
    if FAILED:
        print("failed: " + ", ".join(FAILED))
    return 1 if FAILED else 0


if __name__ == "__main__":
    raise SystemExit(main())
