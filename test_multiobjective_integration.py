"""Integration checks for the multi-objective reward path.

finetune_multiobjective.py cannot run without the DRAKES checkpoints and a GPU,
so this exercises the part that does not need them: the tensor plumbing between
the sampler, the two oracles and the physics reward. It replicates the shapes
and operations of the training loop against mock data, so a shape or dtype
mistake surfaces here instead of forty minutes into a pod session.

    python test_multiobjective_integration.py
"""

from __future__ import annotations

import time

import torch
import torch.nn.functional as F

from hairpin_reward import HairpinPenaltyReward
from physics_reward import duplex_thermodynamics, gc_content

BATCH, LENGTH = 32, 200
PASSED, FAILED = [], []


def check(name: str, condition: bool, detail: str = "") -> None:
    (PASSED if condition else FAILED).append(name)
    print(f"[{'PASS' if condition else 'FAIL'}] {name}" + (f" -- {detail}" if detail else ""))


class MockOracle(torch.nn.Module):
    """Stands in for the gReLU oracle: [B, 4, L] in, [B, 3] out."""

    def __init__(self, seed: int):
        super().__init__()
        torch.manual_seed(seed)
        self.conv = torch.nn.Conv1d(4, 3, kernel_size=5, padding=2)

    def forward(self, x):
        return self.conv(x).mean(dim=-1)


def mock_sampler_output(batch=BATCH, length=LENGTH, temperature=1.0):
    """Mimics _sample_finetune_gradient: a relaxed one-hot with live gradients."""
    logits = torch.randn(batch, length, 4, requires_grad=True)
    return torch.softmax(logits / temperature, dim=-1), logits


def test_shapes_through_the_reward_path() -> None:
    torch.manual_seed(0)
    sample, logits = mock_sampler_output()
    check("sampler output shape", sample.shape == (BATCH, LENGTH, 4), str(tuple(sample.shape)))

    oracle = MockOracle(seed=1)
    reward_bio = oracle(torch.transpose(sample, 1, 2)).squeeze(-1)[:, 0]
    check("biological reward is per-sequence", reward_bio.shape == (BATCH,), str(tuple(reward_bio.shape)))

    physics = HairpinPenaltyReward(LENGTH, tolerance=-3.0)
    reward_phys = physics(sample.float())
    check("physics reward is per-sequence", reward_phys.shape == (BATCH,), str(tuple(reward_phys.shape)))
    check(
        "physics reward is non-positive",
        bool((reward_phys <= 0).all()),
        f"max {reward_phys.max().item():.3e}",
    )


def test_combined_loss_backpropagates() -> None:
    torch.manual_seed(0)
    sample, logits = mock_sampler_output()
    oracle = MockOracle(seed=1)
    physics = HairpinPenaltyReward(LENGTH, tolerance=2.0)  # strict: always active

    reward_bio = oracle(torch.transpose(sample, 1, 2)).squeeze(-1)[:, 0]
    reward_phys = physics(sample.float())
    combined = 1.0 * reward_bio + 0.5 * reward_phys
    loss = -combined.mean()
    loss.backward()

    check(
        "combined loss reaches the sampler logits",
        logits.grad is not None
        and torch.isfinite(logits.grad).all()
        and logits.grad.abs().max() > 0,
        f"max |grad| {logits.grad.abs().max():.3e}",
    )


def test_w_phys_zero_isolates_the_baseline() -> None:
    """--w_phys 0 must give gradients identical to biology alone."""
    torch.manual_seed(0)
    oracle = MockOracle(seed=1)
    physics = HairpinPenaltyReward(LENGTH, tolerance=2.0)

    grads = {}
    for w_phys in (0.0, 0.5):
        torch.manual_seed(7)
        sample, logits = mock_sampler_output()
        reward_bio = oracle(torch.transpose(sample, 1, 2)).squeeze(-1)[:, 0]
        combined = reward_bio + w_phys * physics(sample.float())
        (-combined.mean()).backward()
        grads[w_phys] = logits.grad.clone()

    torch.manual_seed(7)
    sample, logits = mock_sampler_output()
    (-oracle(torch.transpose(sample, 1, 2)).squeeze(-1)[:, 0].mean()).backward()

    check(
        "w_phys=0 matches biology-only gradients exactly",
        torch.allclose(grads[0.0], logits.grad, atol=0),
        "identical",
    )
    check(
        "w_phys>0 changes the gradient",
        not torch.allclose(grads[0.5], grads[0.0]),
        f"max delta {(grads[0.5] - grads[0.0]).abs().max():.3e}",
    )


def test_hard_sample_diagnostics() -> None:
    """Diagnostics run on the discretized sample, as the training loop does."""
    torch.manual_seed(0)
    sample, _ = mock_sampler_output()
    hard = F.one_hot(sample.argmax(2), num_classes=4).float()

    check(
        "hard sample is a valid one-hot",
        bool(torch.allclose(hard.sum(-1), torch.ones(BATCH, LENGTH))),
        "rows sum to 1",
    )

    physics = HairpinPenaltyReward(LENGTH, tolerance=-3.0)
    diag = physics.diagnostics(hard)
    expected = {
        "hairpin_dg_ensemble",
        "hairpin_dg_best_stem",
        "hairpin_penalty",
        "fraction_over_tolerance",
    }
    check("diagnostics contain the logged keys", set(diag) == expected, str(sorted(diag)))
    check(
        "diagnostics are per-sequence",
        all(v.shape == (BATCH,) for v in diag.values()),
        "all [batch]",
    )

    gc = gc_content(hard)
    duplex = duplex_thermodynamics(hard)["delta_g_per_bp"]
    check(
        "GC and duplex diagnostics are sane on random sequences",
        bool(((gc > 0.3) & (gc < 0.7)).all() and ((duplex > -2.2) & (duplex < -1.0)).all()),
        f"GC {gc.mean():.3f}, duplex dG/bp {duplex.mean():.3f}",
    )


def test_dtype_and_device_handling() -> None:
    """The loop hands the scorer float32; tables must follow without error."""
    physics = HairpinPenaltyReward(LENGTH, tolerance=-3.0)
    for dtype in (torch.float32, torch.float64):
        sample = torch.softmax(torch.randn(4, LENGTH, 4, dtype=dtype), dim=-1)
        reward = physics(sample)
        if reward.dtype != dtype or not torch.isfinite(reward).all():
            check(f"scorer handles {dtype}", False, f"got {reward.dtype}")
            return
    check("scorer handles float32 and float64", True, "dtype preserved, values finite")


def test_cost_is_negligible() -> None:
    """The physics term must not dominate the step it is attached to."""
    physics = HairpinPenaltyReward(LENGTH, tolerance=-3.0)
    sample = torch.softmax(torch.randn(BATCH, LENGTH, 4), dim=-1).requires_grad_(True)

    physics(sample).sum().backward()  # warm up
    start = time.perf_counter()
    for _ in range(5):
        sample.grad = None
        physics(sample).sum().backward()
    elapsed = (time.perf_counter() - start) / 5

    check(
        "forward+backward is fast enough to ignore",
        elapsed < 2.0,
        f"{elapsed * 1000:.1f} ms per batch of {BATCH} on CPU "
        f"({physics.scorer.n_stems} stems); far cheaper on GPU",
    )


def main() -> int:
    print(f"torch {torch.__version__}, mock batch {BATCH} x {LENGTH}\n")
    test_shapes_through_the_reward_path()
    print()
    test_combined_loss_backpropagates()
    test_w_phys_zero_isolates_the_baseline()
    print()
    test_hard_sample_diagnostics()
    test_dtype_and_device_handling()
    print()
    test_cost_is_negligible()

    print(f"\n{len(PASSED)} passed, {len(FAILED)} failed")
    if FAILED:
        print("failed: " + ", ".join(FAILED))
    return 1 if FAILED else 0


if __name__ == "__main__":
    raise SystemExit(main())
