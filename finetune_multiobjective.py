"""Multi-objective DRAKES fine-tuning: biological activity under a physics constraint.

This is DRAKES's finetune_reward_bp.py with a second reward term. The original
maximizes a learned enhancer-activity oracle subject to a KL anchor on the
pretrained model:

    loss = -E[R_bio] + alpha * KL(new || pretrained)

Here the reward is a weighted combination, so the model must produce sequences
that are both biologically active and physically synthesizable:

    loss = -E[w_bio * R_bio + w_phys * R_phys] + alpha * KL(new || pretrained)

R_phys is the hairpin penalty from hairpin_reward.py -- exact, differentiable,
and computed on the same relaxed one-hot samples the Gumbel-softmax sampler
already produces, so it needs no policy-gradient estimator.

Setting --w_phys 0 reproduces stock DRAKES through this identical code path,
which is what makes the comparison clean: one flag, everything else held fixed.

Logging is built for the two plots that matter:

  * The Pareto frontier. Sweep --w_phys and plot held-out biological activity
    against the hairpin constraint.
  * The reward-hacking gap. Every epoch logs activity under the oracle being
    trained against AND under a held-out oracle that is never optimized. The
    two curves separating is reward hacking, measured rather than asserted.

GC content is logged alongside, because hairpin propensity retains a genuine
correlation with base composition (R^2 ~ 0.69, see analyze_gc_confound.py).
Comparing hairpin scores at matched GC is what isolates the arrangement the
model actually learned.

Place this file in DRAKES/drakes_dna/ next to finetune_reward_bp.py, together
with physics_reward.py and hairpin_reward.py.
"""

import argparse
import datetime
import json
import os
import random

import numpy as np
import torch
import wandb
from hydra import compose, initialize
from hydra.core.global_hydra import GlobalHydra

import diffusion_gosai_update
import oracle
from utils import set_seed, str2bool

try:
    from hairpin_reward import HairpinPenaltyReward
    from physics_reward import duplex_thermodynamics, gc_content
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "Could not import the physics reward modules. Copy physics_reward.py "
        "and hairpin_reward.py into this directory (DRAKES/drakes_dna/).\n"
        f"Original error: {exc}"
    )


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=__doc__,
    )
    # --- unchanged from DRAKES ---
    parser.add_argument("--base_path", type=str, default="/workspace/drakes_data/")
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--num_epochs", type=int, default=1000)
    parser.add_argument("--num_accum_steps", type=int, default=4)
    parser.add_argument("--truncate_steps", type=int, default=50)
    parser.add_argument("--truncate_kl", type=str2bool, default=False)
    parser.add_argument("--gumbel_temp", type=float, default=1.0)
    parser.add_argument("--gradnorm_clip", type=float, default=1.0)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--name", type=str, default="debug")
    parser.add_argument("--total_num_steps", type=int, default=128)
    parser.add_argument("--copy_flag_temp", type=float, default=None)
    parser.add_argument("--save_every_n_epochs", type=int, default=50)
    parser.add_argument("--alpha", type=float, default=0.001)
    parser.add_argument("--alpha_schedule_warmup", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--eval_oracle_device",
        type=str,
        default=None,
        help="device for the held-out oracle, e.g. 'cpu'. It is only ever used "
        "for reporting and never enters the loss, so parking it off the GPU "
        "frees its parameters on a smaller card, at the cost of one transfer "
        "per accumulation step.",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="path to a checkpoint_*.pt to resume from, restoring model, "
        "optimizer, epoch and RNG state",
    )

    # --- multi-objective additions ---
    physics = parser.add_argument_group("physics constraint")
    physics.add_argument(
        "--w_bio", type=float, default=1.0, help="weight on the enhancer-activity oracle"
    )
    physics.add_argument(
        "--w_phys",
        type=float,
        default=0.0,
        help="weight on the hairpin penalty. 0 reproduces stock DRAKES. "
        "Sweep this for the Pareto frontier.",
    )
    physics.add_argument(
        "--hairpin_tolerance",
        type=float,
        default=-3.0,
        help="ensemble free energy (kcal/mol) at which the penalty starts. "
        "Less negative is stricter.",
    )
    physics.add_argument(
        "--hairpin_stem_length", type=int, default=10,
        help="base pairs per candidate stem; see validate_against_vienna.py"
    )
    physics.add_argument(
        "--hairpin_min_loop", type=int, default=3, help="smallest permitted hairpin loop"
    )
    physics.add_argument(
        "--hairpin_scale",
        type=float,
        default=1.0,
        help="divides the constraint violation before squaring, to bring the "
        "penalty onto a scale comparable with the biological reward",
    )
    physics.add_argument(
        "--seq_length", type=int, default=200, help="sequence length the model emits"
    )
    return parser


def summarize(values: list) -> float:
    return float(np.mean(values)) if values else float("nan")


def save_checkpoint(path, new_model, optim, epoch, args, initial_gap) -> None:
    """Write full training state, not just weights.

    Weights alone cannot resume a run: the Adam moments and the RNG streams are
    part of the trajectory, and restarting without them is a different run that
    merely begins from the same parameters.
    """
    torch.save(
        {
            "epoch": epoch,
            "model": new_model.state_dict(),
            "optimizer": optim.state_dict(),
            "args": vars(args),
            "initial_gap": initial_gap,
            "rng_torch": torch.get_rng_state(),
            "rng_cuda": (
                torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
            ),
            "rng_numpy": np.random.get_state(),
            "rng_python": random.getstate(),
        },
        path,
    )


def fine_tune(new_model, reward_model, reward_model_eval, old_model, args,
              physics_reward, log_path, metrics_path, save_path, eps=1e-5):
    new_model.config.finetuning.truncate_steps = args.truncate_steps
    new_model.config.finetuning.gumbel_softmax_temp = args.gumbel_temp
    new_model.train()
    torch.set_grad_enabled(True)
    optim = torch.optim.Adam(new_model.parameters(), lr=args.learning_rate)

    eval_device = next(reward_model_eval.parameters()).device

    start_epoch = 0
    initial_gap = [None]  # boxed so the epoch loop can set it on first pass
    if args.resume:
        state = torch.load(args.resume, map_location=new_model.device)
        new_model.load_state_dict(state["model"])
        optim.load_state_dict(state["optimizer"])
        start_epoch = state["epoch"] + 1
        initial_gap = [state.get("initial_gap")]
        torch.set_rng_state(state["rng_torch"].cpu())
        if state.get("rng_cuda") is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all([s.cpu() for s in state["rng_cuda"]])
        np.random.set_state(state["rng_numpy"])
        random.setstate(state["rng_python"])
        print(f"==> resumed from {args.resume}, continuing at epoch {start_epoch}")

    # Append rather than truncate when resuming, so the log survives.
    with open(log_path, "a" if args.resume else "w") as handle:
        handle.write(repr(args) + "\n")

    for epoch_num in range(start_epoch, args.num_epochs):
        # Everything below is accumulated across gradient-accumulation steps and
        # reported once per epoch.
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        bio_train, bio_eval = [], []
        phys_rewards, hairpin_dg, hairpin_violations = [], [], []
        gc_fractions, duplex_dg = [], []
        losses, reward_losses, kl_losses = [], [], []
        tot_grad_norm = 0.0

        for accum_step in range(args.num_accum_steps):
            (
                sample,
                last_x_list,
                condt_list,
                move_chance_t_list,
                copy_flag_list,
            ) = new_model._sample_finetune_gradient(
                eval_sp_size=args.batch_size, copy_flag_temp=args.copy_flag_temp
            )  # [bsz, seqlen, 4]; hard one-hot forward, relaxed gradient

            # --- biological reward, on the relaxed sample (differentiable) ---
            reward_bio = reward_model(torch.transpose(sample, 1, 2)).squeeze(-1)[:, 0]

            # --- physics reward, on the same relaxed sample ---
            # float32 regardless of model precision: the hairpin ensemble is a
            # logsumexp over ~16k stems and is not worth doing in bf16.
            sample_f32 = sample.float()
            reward_phys = physics_reward(sample_f32)

            # The sampler returns a straight-through estimate,
            #     x_soft + (x_hard - x_soft).detach()
            # so the forward value is ALREADY a hard one-hot and only the
            # gradient flows through the relaxation. Re-discretizing with argmax
            # would be a no-op. This also means the physics reward above is
            # evaluated on the real discrete sequence, with pairing gates that
            # are exactly 0 or 1 rather than soft blends.
            sample_hard = sample.detach()
            with torch.no_grad():
                hard_t = torch.transpose(sample_hard, 1, 2)
                bio_train.append(
                    reward_model(hard_t).squeeze(-1)[:, 0].mean().item()
                )
                # The held-out oracle is never optimized against. The gap
                # between this and bio_train is the reward-hacking signal.
                bio_eval.append(
                    reward_model_eval(hard_t.to(eval_device))
                    .squeeze(-1)[:, 0]
                    .mean()
                    .item()
                )
                diagnostics = physics_reward.diagnostics(sample_hard)
                hairpin_dg.append(diagnostics["hairpin_dg_ensemble"].mean().item())
                hairpin_violations.append(
                    diagnostics["fraction_over_tolerance"].mean().item()
                )
                gc_fractions.append(gc_content(sample_hard).mean().item())
                duplex_dg.append(
                    duplex_thermodynamics(sample_hard)["delta_g_per_bp"].mean().item()
                )
            phys_rewards.append(reward_phys.mean().item())

            # --- KL to the pretrained model, unchanged from DRAKES ---
            total_kl = []
            for random_t in range(args.total_num_steps):
                if args.truncate_kl and random_t < args.total_num_steps - args.truncate_steps:
                    continue
                last_x = last_x_list[random_t]
                condt = condt_list[random_t]
                move_chance_t = move_chance_t_list[random_t]
                copy_flag = copy_flag_list[random_t]

                log_p_x0 = new_model.forward(last_x, condt)[:, :, :-1]
                log_p_x0_old = old_model.forward(last_x, condt)[:, :, :-1]
                p_x0 = log_p_x0.exp()
                p_x0_old = log_p_x0_old.exp()

                kl_div = copy_flag * (
                    -p_x0 + p_x0_old + p_x0 * (log_p_x0 - log_p_x0_old)
                ) / move_chance_t[0, 0, 0]
                total_kl.append((kl_div * last_x[:, :, :-1]).sum((1, 2)))

            if args.alpha_schedule_warmup and epoch_num < args.alpha_schedule_warmup:
                current_alpha = (epoch_num + 1) / args.alpha_schedule_warmup * args.alpha
            else:
                current_alpha = args.alpha

            kl_loss = torch.stack(total_kl, 1).sum(1).mean()

            # --- the multi-objective loss ---
            combined = args.w_bio * reward_bio + args.w_phys * reward_phys
            reward_loss = -torch.mean(combined)
            loss = (reward_loss + kl_loss * current_alpha) / args.num_accum_steps
            loss.backward()

            if (accum_step + 1) % args.num_accum_steps == 0:
                norm = torch.nn.utils.clip_grad_norm_(
                    new_model.parameters(), args.gradnorm_clip
                )
                tot_grad_norm += float(norm)
                optim.step()
                optim.zero_grad()

            losses.append(loss.item() * args.num_accum_steps)
            reward_losses.append(reward_loss.item())
            kl_losses.append(kl_loss.item())

        gap = summarize(bio_train) - summarize(bio_eval)
        if initial_gap[0] is None:
            # The two oracles are separately trained and separately calibrated,
            # so they disagree before any optimization happens. The raw gap is
            # therefore oracle disagreement, not evidence of reward hacking.
            # Anchor on the first epoch and report the CHANGE, which is the
            # quantity that can actually indicate the policy exploiting the
            # oracle it is trained against.
            initial_gap[0] = gap
            print(f"  baseline oracle disagreement at epoch 0: {gap:+.4f}")

        record = {
            "epoch": epoch_num,
            "bio_reward_train_oracle": summarize(bio_train),
            "bio_reward_heldout_oracle": summarize(bio_eval),
            "oracle_disagreement": gap,
            "oracle_disagreement_at_epoch0": initial_gap[0],
            "reward_hacking_gap": gap - initial_gap[0],
            "physics_reward": summarize(phys_rewards),
            "weighted_bio": args.w_bio * summarize(bio_train),
            "weighted_phys": args.w_phys * summarize(phys_rewards),
            "hairpin_dg_ensemble": summarize(hairpin_dg),
            "hairpin_violation_rate": summarize(hairpin_violations),
            "gc_content": summarize(gc_fractions),
            "duplex_dg_per_bp": summarize(duplex_dg),
            "loss": summarize(losses),
            "reward_loss": summarize(reward_losses),
            "kl_loss": summarize(kl_losses),
            "grad_norm": tot_grad_norm,
            "alpha": current_alpha,
            # Peak allocation, not reserved. Backpropagating through an
            # unrolled sampler is memory-bound, so this is the number that
            # decides how far batch_size and truncate_steps can go.
            "peak_gpu_gb": (
                torch.cuda.max_memory_allocated() / 1024**3
                if torch.cuda.is_available()
                else 0.0
            ),
        }

        print(
            f"epoch {epoch_num:>4}  "
            f"bio {record['bio_reward_train_oracle']:>8.4f} "
            f"(w{record['weighted_bio']:+.3f})  "
            f"phys {record['physics_reward']:>8.4f} "
            f"(w{record['weighted_phys']:+.3f})  "
            f"held-out {record['bio_reward_heldout_oracle']:>8.4f}  "
            f"hack {record['reward_hacking_gap']:+.4f}  "
            f"hairpin {record['hairpin_dg_ensemble']:>7.3f}  "
            f"viol {record['hairpin_violation_rate']:.2f}  "
            f"GC {record['gc_content']:.3f}  "
            f"KL {record['kl_loss']:.4f}  "
            f"peak {record['peak_gpu_gb']:.1f}GB"
        )

        # JSON lines, so the Pareto plot is a two-line pandas read.
        with open(metrics_path, "a") as handle:
            handle.write(json.dumps(record) + "\n")
        with open(log_path, "a") as handle:
            handle.write(json.dumps(record) + "\n")
        if args.name != "debug":
            wandb.log(record)

        # A single rolling checkpoint, overwritten every epoch. The model is a
        # small CNN so this is cheap, and it caps the cost of any interruption
        # -- Ctrl-C, a lost pod, an OOM -- at one epoch rather than the run.
        save_checkpoint(
            os.path.join(save_path, "checkpoint_latest.pt"),
            new_model, optim, epoch_num, args, initial_gap[0],
        )

        if (epoch_num + 1) % args.save_every_n_epochs == 0:
            save_checkpoint(
                os.path.join(save_path, f"checkpoint_epoch{epoch_num}.pt"),
                new_model, optim, epoch_num, args, initial_gap[0],
            )
            print(f"  checkpoint saved at epoch {epoch_num}")

    # Always write a final checkpoint regardless of the interval. A short run --
    # the two-epoch smoke test, or anything ending before the first save
    # boundary -- would otherwise finish with nothing on disk.
    final_path = os.path.join(save_path, "checkpoint_final.pt")
    save_checkpoint(
        final_path, new_model, optim, args.num_epochs - 1, args, initial_gap[0]
    )
    print(f"final checkpoint: {final_path}")

    if args.name != "debug":
        wandb.finish()


def main() -> None:
    args = build_argparser().parse_args()
    print(args)

    # gReLU builds its LightningModel with a wandb logger, so loading an oracle
    # triggers wandb on a machine that has never logged in -- and wandb answers
    # by opening an interactive account prompt. In a debug run that is merely
    # annoying; in an unattended sweep under tmux it is a silent hang waiting
    # for a keypress. Debug runs report nowhere, so disable it outright.
    # setdefault, so an explicit WANDB_MODE in the environment still wins.
    if args.name == "debug":
        os.environ.setdefault("WANDB_MODE", "disabled")
    elif not os.environ.get("WANDB_MODE") and not os.path.exists(
        os.path.expanduser("~/.netrc")
    ):
        # A real run with no credentials on disk would hit the same prompt.
        # Fall back to offline logging, which writes locally and can be synced
        # later with `wandb sync`, rather than blocking.
        print(
            "No wandb credentials found; setting WANDB_MODE=offline. "
            "Run `wandb login` first, or export WANDB_MODE=disabled, to choose "
            "explicitly."
        )
        os.environ["WANDB_MODE"] = "offline"

    ckpt_path = os.path.join(args.base_path, "mdlm/outputs_gosai/pretrained.ckpt")
    log_base_dir = os.path.join(args.base_path, "mdlm/reward_bp_results_final")

    GlobalHydra.instance().clear()
    initialize(config_path="configs_gosai", job_name="load_model")
    cfg = compose(config_name="config_gosai.yaml")
    cfg.eval.checkpoint_path = ckpt_path

    # The sampler reads cfg.sampling.steps when called without num_steps, while
    # the KL loop iterates args.total_num_steps. Upstream DRAKES leaves these
    # independent and they agree only because both happen to default to 128.
    # Bind them, so --total_num_steps cannot silently desynchronise the two and
    # compute KL over a subset of the trajectory with no error raised.
    cfg.sampling.steps = args.total_num_steps
    assert 1 <= args.truncate_steps <= args.total_num_steps, (
        f"--truncate_steps {args.truncate_steps} must lie in "
        f"[1, {args.total_num_steps}]"
    )
    assert args.seq_length == cfg.model.length, (
        f"--seq_length {args.seq_length} does not match cfg.model.length "
        f"{cfg.model.length}; the hairpin scorer would reject the samples"
    )

    curr_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.name == "debug":
        print("Debug mode")
        save_path = os.path.join(log_base_dir, "debug")
    else:
        run_name = (
            f"wphys{args.w_phys}_tol{args.hairpin_tolerance}_alpha{args.alpha}"
            f"_bsz{args.batch_size}_{args.name}_{curr_time}"
        )
        save_path = os.path.join(log_base_dir, run_name)
    os.makedirs(save_path, exist_ok=True)

    log_path = os.path.join(save_path, "log.txt")
    metrics_path = os.path.join(save_path, "metrics.jsonl")
    with open(os.path.join(save_path, "config.json"), "w") as handle:
        json.dump(vars(args), handle, indent=2)

    if args.name != "debug":
        wandb.init(
            project="drakes_multiobjective", name=run_name, config=args, dir=save_path
        )

    set_seed(args.seed, use_cuda=True)

    new_model = diffusion_gosai_update.Diffusion.load_from_checkpoint(
        cfg.eval.checkpoint_path, config=cfg
    )
    old_model = diffusion_gosai_update.Diffusion.load_from_checkpoint(
        cfg.eval.checkpoint_path, config=cfg
    )
    reward_model = oracle.get_gosai_oracle(mode="train").to(new_model.device)
    reward_model_eval = oracle.get_gosai_oracle(mode="eval").to(
        args.eval_oracle_device or new_model.device
    )

    # Only new_model is optimized. Freeze the rest: optim.zero_grad() clears
    # gradients for the optimizer's parameters alone, so without this the
    # reference model and both oracles accumulate .grad buffers that are
    # computed on every backward pass and never read.
    #
    # Freeze the parameters rather than wrapping the forward pass in
    # torch.no_grad(). The training reward has to keep its graph back through
    # the sampler; no_grad would sever exactly the path the loss needs.
    for frozen in (old_model, reward_model, reward_model_eval):
        frozen.eval()
        frozen.requires_grad_(False)

    physics_reward = HairpinPenaltyReward(
        length=args.seq_length,
        tolerance=args.hairpin_tolerance,
        scale=args.hairpin_scale,
        stem_length=args.hairpin_stem_length,
        min_loop=args.hairpin_min_loop,
    )
    baseline = physics_reward.scorer.unstructured_baseline()
    print(
        f"hairpin scorer: {physics_reward.scorer.n_stems} candidate stems, "
        f"unstructured baseline {baseline:+.3f} kcal/mol, "
        f"tolerance {args.hairpin_tolerance:+.3f}"
    )
    if args.hairpin_tolerance > baseline:
        print(
            "  WARNING: tolerance is above the unstructured baseline, so every "
            "sequence violates it and the penalty will never reach zero."
        )
    if args.w_phys == 0:
        print("  w_phys is 0: this run reproduces stock DRAKES.")

    fine_tune(
        new_model,
        reward_model,
        reward_model_eval,
        old_model,
        args,
        physics_reward,
        log_path,
        metrics_path,
        save_path,
    )


if __name__ == "__main__":
    main()
