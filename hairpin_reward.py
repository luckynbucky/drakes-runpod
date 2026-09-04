"""Differentiable hairpin propensity for reward-guided sequence design.

Duplex free energy (physics_reward.py) turns out to be almost entirely a
restatement of GC content -- R^2 = 0.996 against base composition across the
realistic range, and shuffling a sequence, which preserves composition exactly
while destroying arrangement, moves it by only ~0.01 kcal/mol/bp against a 0.70
range from composition alone. A model asked to hit a duplex dG target satisfies
it by shifting composition and learns nothing about arrangement.

Intramolecular secondary structure does not have that problem, and it is the
property that actually governs whether a sequence can be synthesized. A stable
hairpin occludes the growing chain during phosphoramidite synthesis and blocks
annealing during assembly and PCR; it is the structure vendors flag. Crucially,
a sequence at 50% GC can be hairpin-free or badly structured depending entirely
on the order of its bases, so the constraint cannot be satisfied by composition.

Model
-----
Candidate hairpins are enumerated as a stem of fixed length pairing an upstream
arm against a downstream arm, separated by a loop. Each stem's free energy is
the sum of its nearest-neighbor stacking terms, gated by whether the flanking
base pairs can actually form, plus a loop-closure penalty:

    E(i, j) = sum_k  pair(i+k, j-k) * pair(i+k+1, j-k-1) * dG_stack(x_i+k, x_i+k+1)
              + dG_loop(loop size)

Stems are then aggregated as a restricted partition function rather than a hard
minimum, which is both smoother to optimize and closer to what folding actually
is -- an ensemble, not a single structure:

    dG_ensemble = -RT * log sum_stems exp(-E / RT)

Scope and simplifications, stated plainly because they matter if anyone asks:

* Only hairpins are enumerated. Interior loops, bulges, multiloops and
  intermolecular dimers are not. A full treatment is the McCaskill partition
  function, an O(L^3) dynamic program.
* Loop closure uses the Jacobson-Stockmayer logarithmic form with approximate
  coefficients rather than the Turner loop tables.
* Stems are a fixed length, so a long stem is counted as several overlapping
  windows rather than as one cooperative unit.

The result is therefore a well-behaved proxy for hairpin propensity that
captures the dominant term -- contiguous complementary stacking -- not a
quantitative folding free energy. It reuses the SantaLucia stacking parameters
validated in test_physics_reward.py.
"""

from __future__ import annotations

import math

import torch

from physics_reward import (
    BASES,
    DELTA_H_TABLE,
    DELTA_S_TABLE,
    GAS_CONSTANT,
    TEMP_37C,
)

# Watson-Crick pairing, in the ACGT channel order DRAKES uses.
_PAIRS = [("A", "T"), ("T", "A"), ("C", "G"), ("G", "C")]


def _pairing_matrix(dtype=torch.float64) -> torch.Tensor:
    matrix = torch.zeros(4, 4, dtype=dtype)
    for first, second in _PAIRS:
        matrix[BASES.index(first), BASES.index(second)] = 1.0
    return matrix


def stacking_free_energy_table(temperature_k: float = TEMP_37C) -> torch.Tensor:
    """Nearest-neighbor dG at the given temperature, kcal/mol per stack."""
    return DELTA_H_TABLE - temperature_k * DELTA_S_TABLE / 1000.0


# Loop closure, Jacobson-Stockmayer form: a constant initiation for the
# reference 3-nucleotide loop plus a logarithmic term for larger loops. The
# coefficients are approximate; the Turner tables are what a production folding
# program would use.
LOOP_INIT_KCAL = 5.6
LOOP_REFERENCE_SIZE = 3


def _loop_penalty(loop_sizes: torch.Tensor, temperature_k: float) -> torch.Tensor:
    slope = 1.75 * GAS_CONSTANT * temperature_k / 1000.0
    ratio = loop_sizes.to(torch.float64) / LOOP_REFERENCE_SIZE
    return LOOP_INIT_KCAL + slope * torch.log(ratio)


class HairpinScorer:
    """Enumerates candidate hairpin stems once, then scores batches cheaply.

    The stem geometry depends only on sequence length, so all index arithmetic
    is precomputed in the constructor and reused for every batch.

    Args:
        length: sequence length in bases.
        stem_length: number of base pairs per candidate stem.
        min_loop: smallest permitted hairpin loop, in nucleotides.
        temperature_k: temperature for both stacking energies and the ensemble.
    """

    def __init__(
        self,
        length: int,
        stem_length: int = 8,
        min_loop: int = 3,
        temperature_k: float = TEMP_37C,
    ):
        if stem_length < 2:
            raise ValueError("a stem needs at least two base pairs to stack")

        self.length = length
        self.stem_length = stem_length
        self.min_loop = min_loop
        self.temperature_k = temperature_k
        self.rt = GAS_CONSTANT * temperature_k / 1000.0  # kcal/mol

        # A stem with outer pair (i, j) occupies i..i+W-1 and j-W+1..j, leaving
        # a loop of j - i - 2W + 1 nucleotides between the arms.
        outer_i, outer_j = [], []
        span = 2 * stem_length + min_loop - 1
        for i in range(length):
            for j in range(i + span, length):
                outer_i.append(i)
                outer_j.append(j)

        if not outer_i:
            raise ValueError(
                f"no hairpin fits: length {length} with stem {stem_length} "
                f"and min loop {min_loop}"
            )

        i_tensor = torch.tensor(outer_i, dtype=torch.long)
        j_tensor = torch.tensor(outer_j, dtype=torch.long)
        offsets = torch.arange(stem_length - 1, dtype=torch.long)

        # Index grids of shape [n_stems, stem_length - 1], one entry per stack.
        self.pair_a_i = i_tensor[:, None] + offsets[None, :]
        self.pair_a_j = j_tensor[:, None] - offsets[None, :]
        self.pair_b_i = self.pair_a_i + 1
        self.pair_b_j = self.pair_a_j - 1
        self.stack_index = self.pair_a_i

        loop_sizes = j_tensor - i_tensor - 2 * stem_length + 1
        self.loop_penalty = _loop_penalty(loop_sizes, temperature_k)
        self.n_stems = len(outer_i)

        self._pairing = _pairing_matrix()
        self._stacking = stacking_free_energy_table(temperature_k)

    def _to(self, device, dtype):
        """Move precomputed tables onto the batch's device and dtype."""
        if self._pairing.device != device or self._pairing.dtype != dtype:
            self._pairing = self._pairing.to(device=device, dtype=dtype)
            self._stacking = self._stacking.to(device=device, dtype=dtype)
            self.loop_penalty = self.loop_penalty.to(device=device, dtype=dtype)
        if self.pair_a_i.device != device:
            for name in (
                "pair_a_i", "pair_a_j", "pair_b_i", "pair_b_j", "stack_index"
            ):
                setattr(self, name, getattr(self, name).to(device))

    def stem_free_energies(self, x: torch.Tensor) -> torch.Tensor:
        """Free energy of every candidate stem, [batch, n_stems], kcal/mol.

        Args:
            x: [batch, length, 4] one-hot or relaxed one-hot, channels ACGT.
        """
        if x.dim() != 3 or x.shape[1] != self.length or x.shape[2] != 4:
            raise ValueError(
                f"expected [batch, {self.length}, 4], got {tuple(x.shape)}"
            )
        self._to(x.device, x.dtype)

        # pairing[b, i, j] = probability that positions i and j can base-pair.
        pairing = torch.einsum("bia,ac,bjc->bij", x, self._pairing, x)
        # stacking[b, i] = nearest-neighbor dG of the stack starting at i.
        stacking = torch.einsum(
            "bia,ac,bic->bi", x[:, :-1], self._stacking, x[:, 1:]
        )

        # A stack contributes only if both of its flanking pairs can form.
        gate = (
            pairing[:, self.pair_a_i, self.pair_a_j]
            * pairing[:, self.pair_b_i, self.pair_b_j]
        )
        contributions = gate * stacking[:, self.stack_index]
        return contributions.sum(dim=-1) + self.loop_penalty

    def ensemble_free_energy(self, x: torch.Tensor) -> torch.Tensor:
        """Restricted partition function over hairpins, [batch], kcal/mol.

        More negative means more stable structure. Values near or above zero
        mean no appreciable hairpin was found.
        """
        energies = self.stem_free_energies(x)
        return -self.rt * torch.logsumexp(-energies / self.rt, dim=-1)

    def unstructured_baseline(self) -> float:
        """Ensemble free energy of a sequence in which no pair can form.

        This is not zero. A partition function sums over every candidate stem,
        so even when all of them are unfavourable the sum contributes a small
        positive free energy that grows with the number of candidates, and so
        with sequence length. That is physically reasonable -- a longer chain
        has more ways to fold -- but it means ``tolerance`` is not transferable
        between lengths unless expressed relative to this baseline.
        """
        with torch.no_grad():
            return (
                -self.rt * torch.logsumexp(-self.loop_penalty / self.rt, dim=-1)
            ).item()

    def most_stable_stem(self, x: torch.Tensor) -> torch.Tensor:
        """Free energy of the single best hairpin, [batch]. Diagnostic only."""
        with torch.no_grad():
            return self.stem_free_energies(x).min(dim=-1).values


class HairpinPenaltyReward:
    """Penalizes sequences that fold back on themselves.

    Sequences whose ensemble free energy is weaker than ``tolerance`` are free;
    beyond it the penalty grows quadratically. A one-sided penalty is the right
    shape here because there is no benefit to being *less* structured than
    "unstructured enough to synthesize" -- unlike a target-seeking reward, this
    should stop pushing once the constraint is satisfied.

    Args:
        length: sequence length in bases.
        tolerance: ensemble free energy, kcal/mol, at which penalty begins.
            Less negative is stricter.
        scale: divides the deviation before squaring, to keep the reward on a
            comparable scale to the biological oracle.
    """

    def __init__(
        self,
        length: int,
        tolerance: float = -3.0,
        scale: float = 1.0,
        **scorer_kwargs,
    ):
        self.scorer = HairpinScorer(length, **scorer_kwargs)
        self.tolerance = tolerance
        self.scale = scale

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Returns [batch] rewards in (-inf, 0], differentiable in x."""
        dg = self.scorer.ensemble_free_energy(x)
        excess = torch.relu(self.tolerance - dg) / self.scale
        return -excess.pow(2)

    def diagnostics(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        with torch.no_grad():
            dg = self.scorer.ensemble_free_energy(x)
            return {
                "hairpin_dg_ensemble": dg,
                "hairpin_dg_best_stem": self.scorer.most_stable_stem(x),
                "hairpin_penalty": -(
                    torch.relu(self.tolerance - dg) / self.scale
                ).pow(2),
                "fraction_over_tolerance": (dg < self.tolerance).to(dg.dtype),
            }
