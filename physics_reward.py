"""Differentiable nucleic-acid thermodynamics for reward-guided sequence design.

DRAKES fine-tunes a discrete diffusion model against a *learned* reward oracle
(a gReLU network predicting enhancer activity). This module supplies a second,
physically exact reward that can be optimized alongside it: the nearest-neighbor
free energy of duplex formation.

Why this is worth having:

* It is exact. There is no reward-model error, so any pathology you observe
  during fine-tuning belongs to the optimizer rather than to a mis-specified
  reward. That makes it a clean instrument for studying reward hacking.
* It is differentiable with respect to the relaxed one-hot sequences that
  DRAKES's Gumbel-softmax sampler produces, so it plugs straight into the
  paper's direct-backpropagation objective with no policy-gradient estimator.
* It encodes a real manufacturability constraint. Synthetic constructs whose
  duplex stability sits outside a workable window are harder and costlier to
  synthesize, so "functional AND synthesizable" is the real design problem.

Physics: the nearest-neighbor model treats duplex formation as a sum of
stacking interactions between adjacent base pairs, plus an initiation penalty
at each helix end.

    dH_total = sum_over_stacks dH(XY) + dH_init(5' end) + dH_init(3' end)
    dS_total = sum_over_stacks dS(XY) + dS_init(5' end) + dS_init(3' end)
    dG(T)    = dH_total - T * dS_total

Parameters are the SantaLucia (1998) unified set, in kcal/mol (enthalpy) and
cal/(mol K) (entropy). They are verified against Biopython's DNA_NN3 table in
test_physics_reward.py.

Reference:
    SantaLucia J Jr. "A unified view of polymer, dumbbell, and oligonucleotide
    DNA nearest-neighbor thermodynamics." PNAS 95(4):1460-1465, 1998.
"""

from __future__ import annotations

import torch

# Channel order must match DRAKES's dataloader_gosai.py:
#     DNA_ALPHABET = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
BASES = "ACGT"
BASE_INDEX = {b: i for i, b in enumerate(BASES)}

GAS_CONSTANT = 1.987  # cal / (mol K)
CELSIUS_TO_KELVIN = 273.15
TEMP_37C = 310.15  # K, the reference temperature for dG_37

# SantaLucia (1998) unified nearest-neighbor parameters: (dH, dS) for each of
# the ten unique stacks, keyed 5'->3' on the top strand.
_NN_UNIQUE = {
    "AA": (-7.9, -22.2),
    "AT": (-7.2, -20.4),
    "TA": (-7.2, -21.3),
    "CA": (-8.5, -22.7),
    "GT": (-8.4, -22.4),
    "CT": (-7.8, -21.0),
    "GA": (-8.2, -22.2),
    "CG": (-10.6, -27.2),
    "GC": (-9.8, -24.4),
    "GG": (-8.0, -19.9),
}

# Helix initiation, applied once per end, chosen by whether that terminal base
# pair is G/C or A/T.
INIT_GC = (0.1, -2.8)
INIT_AT = (2.3, 4.1)

_COMPLEMENT = str.maketrans("ACGT", "TGCA")


def _reverse_complement(seq: str) -> str:
    return seq.translate(_COMPLEMENT)[::-1]


def _build_nn_tables() -> tuple[torch.Tensor, torch.Tensor]:
    """Expand the ten unique stacks into full 4x4 (dH, dS) lookup tables.

    A stack read 5'-XY-3' on one strand is the same physical interaction as
    5'-comp(Y)comp(X)-3' read on the other, so the sixteen dinucleotides
    collapse onto ten parameters. Building the full table once lets the energy
    be evaluated as a single contraction against dinucleotide occupancies.
    """
    d_h = torch.zeros(4, 4, dtype=torch.float64)
    d_s = torch.zeros(4, 4, dtype=torch.float64)

    for first in BASES:
        for second in BASES:
            dimer = first + second
            if dimer in _NN_UNIQUE:
                params = _NN_UNIQUE[dimer]
            else:
                # Fall back to the reverse-complement reading of the same stack.
                params = _NN_UNIQUE[_reverse_complement(dimer)]
            d_h[BASE_INDEX[first], BASE_INDEX[second]] = params[0]
            d_s[BASE_INDEX[first], BASE_INDEX[second]] = params[1]

    return d_h, d_s


DELTA_H_TABLE, DELTA_S_TABLE = _build_nn_tables()


def dinucleotide_counts(x: torch.Tensor) -> torch.Tensor:
    """Soft occupancy of each of the sixteen stacks.

    Args:
        x: [batch, length, 4] one-hot or relaxed one-hot sequences. Relaxed
           inputs (a Gumbel-softmax sample) give expected counts under the
           per-position marginals, which is what makes the energy differentiable.

    Returns:
        [batch, 4, 4], where entry (i, j) is the expected number of times base
        i is followed by base j.
    """
    if x.dim() != 3 or x.shape[-1] != 4:
        raise ValueError(f"expected [batch, length, 4], got {tuple(x.shape)}")
    if x.shape[1] < 2:
        raise ValueError("need at least two positions to form a stack")

    return torch.einsum("bli,blj->bij", x[:, :-1], x[:, 1:])


def _initiation_terms(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Expected initiation enthalpy and entropy across both helix ends.

    The penalty depends on whether the terminal base pair is G/C or A/T. For a
    relaxed input we take the expectation over that binary choice, which keeps
    the term smooth and exact for hard one-hot inputs.
    """
    # P(terminal base is G or C) at each end. C is channel 1, G is channel 2.
    p_gc_5prime = x[:, 0, 1] + x[:, 0, 2]
    p_gc_3prime = x[:, -1, 1] + x[:, -1, 2]

    d_h = torch.zeros_like(p_gc_5prime)
    d_s = torch.zeros_like(p_gc_5prime)
    for p_gc in (p_gc_5prime, p_gc_3prime):
        d_h = d_h + p_gc * INIT_GC[0] + (1.0 - p_gc) * INIT_AT[0]
        d_s = d_s + p_gc * INIT_GC[1] + (1.0 - p_gc) * INIT_AT[1]

    return d_h, d_s


def duplex_thermodynamics(
    x: torch.Tensor, temperature_k: float = TEMP_37C
) -> dict[str, torch.Tensor]:
    """Nearest-neighbor duplex thermodynamics for a batch of sequences.

    Args:
        x: [batch, length, 4] one-hot or relaxed one-hot, channels ordered ACGT.
        temperature_k: temperature for the free energy, in Kelvin.

    Returns:
        dict with per-sequence [batch] tensors:
            delta_h        total enthalpy, kcal/mol
            delta_s        total entropy, cal/(mol K)
            delta_g        free energy at temperature_k, kcal/mol
            delta_g_per_bp free energy normalized by stack count, kcal/mol

    The symmetry correction for self-complementary duplexes is deliberately
    omitted: it is a constant -1.4 cal/(mol K) that applies only to exactly
    self-complementary sequences, a measure-zero case for the 200bp enhancers
    this is used on, and it has no meaningful relaxation.
    """
    d_h_table = DELTA_H_TABLE.to(device=x.device, dtype=x.dtype)
    d_s_table = DELTA_S_TABLE.to(device=x.device, dtype=x.dtype)

    counts = dinucleotide_counts(x)
    stack_h = (counts * d_h_table).sum(dim=(-2, -1))
    stack_s = (counts * d_s_table).sum(dim=(-2, -1))

    init_h, init_s = _initiation_terms(x)
    delta_h = stack_h + init_h
    delta_s = stack_s + init_s

    # dS is in cal/(mol K) while dH is in kcal/mol, hence the factor of 1000.
    delta_g = delta_h - temperature_k * delta_s / 1000.0
    n_stacks = x.shape[1] - 1

    return {
        "delta_h": delta_h,
        "delta_s": delta_s,
        "delta_g": delta_g,
        "delta_g_per_bp": delta_g / n_stacks,
    }


def melting_temperature(
    x: torch.Tensor, strand_conc_m: float = 12.5e-9
) -> torch.Tensor:
    """Predicted duplex melting temperature in Celsius.

    Args:
        x: [batch, length, 4] one-hot or relaxed one-hot.
        strand_conc_m: effective strand concentration. The default corresponds
            to Biopython's Tm_NN default of 25 nM of each non-self-complementary
            strand, which enters as (dnac1 - dnac2 / 2).

    No salt correction is applied, so this is the bare nearest-neighbor
    prediction. Add one downstream if you need to match a specific buffer.
    """
    thermo = duplex_thermodynamics(x)
    denominator = thermo["delta_s"] + GAS_CONSTANT * torch.log(
        torch.tensor(strand_conc_m, device=x.device, dtype=x.dtype)
    )
    return 1000.0 * thermo["delta_h"] / denominator - CELSIUS_TO_KELVIN


def gc_content(x: torch.Tensor) -> torch.Tensor:
    """Expected GC fraction per sequence, [batch].

    Report this as a diagnostic, not as a reward term. GC content and duplex
    free energy are strongly correlated, so a model can hit a dG target by
    shifting composition alone. Holding GC fixed while dG moves is the evidence
    that the model learned genuine stacking preferences, which are
    order-dependent: 5'-GC-3' and 5'-CG-3' differ by 0.8 kcal/mol in dH.
    """
    return (x[:, :, 1] + x[:, :, 2]).mean(dim=1)


class DuplexStabilityReward:
    """Reward that drives sequences toward a target duplex free energy.

    The reward is the negative squared deviation of per-base-pair free energy
    from a target, so it is zero at the target and decreases smoothly away from
    it. Normalizing per base pair keeps the target interpretable and the
    gradient scale independent of sequence length.

    Typical values for random 200bp DNA are around -1.6 kcal/mol per base pair,
    so targets in [-1.9, -1.3] span a meaningful design range.
    """

    def __init__(self, target_dg_per_bp: float, temperature_k: float = TEMP_37C):
        self.target_dg_per_bp = target_dg_per_bp
        self.temperature_k = temperature_k

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Returns [batch] rewards, differentiable in x."""
        thermo = duplex_thermodynamics(x, temperature_k=self.temperature_k)
        deviation = thermo["delta_g_per_bp"] - self.target_dg_per_bp
        return -deviation.pow(2)

    def diagnostics(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Everything worth logging during a run, detached."""
        with torch.no_grad():
            thermo = duplex_thermodynamics(x, temperature_k=self.temperature_k)
            return {
                "delta_g_per_bp": thermo["delta_g_per_bp"],
                "delta_g": thermo["delta_g"],
                "melting_temp_c": melting_temperature(x),
                "gc_content": gc_content(x),
                "reward": -(thermo["delta_g_per_bp"] - self.target_dg_per_bp).pow(2),
            }


def one_hot_from_strings(seqs: list[str], device=None, dtype=torch.float64):
    """Encode ACGT strings as [batch, length, 4], for tests and evaluation."""
    lengths = {len(s) for s in seqs}
    if len(lengths) != 1:
        raise ValueError(f"sequences must be equal length, got lengths {sorted(lengths)}")

    indices = torch.tensor(
        [[BASE_INDEX[c] for c in s] for s in seqs], device=device, dtype=torch.long
    )
    return torch.nn.functional.one_hot(indices, num_classes=4).to(dtype)
