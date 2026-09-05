# Running DRAKES on RunPod

A setup kit for [DRAKES](https://github.com/ChenyuWang-Monica/DRAKES) (ICLR 2025),
which fine-tunes discrete diffusion models with reward optimization — the same
shape of problem as RLHF post-training, on DNA and protein sequences instead of
text.

| File | What it does |
| --- | --- |
| `setup_dna.sh` | Full pod bootstrap for the regulatory-DNA experiment |
| `setup_protein.sh` | Full pod bootstrap for the protein-stability experiment |
| `set_base_path.py` | Rewrites the author's hardcoded cluster path to yours |
| `smoke_test.py` | Verifies GPU, dependencies, and downloaded weights before you burn GPU hours |
| `physics_reward.py` | Differentiable nearest-neighbor duplex thermodynamics (dG, Tm, GC) |
| `hairpin_reward.py` | Differentiable hairpin propensity, the synthesizability constraint |
| `analyze_gc_confound.py` | Checks whether a physics reward is secretly just GC content |
| `validate_against_vienna.py` | Measures the hairpin proxy against ViennaRNA's full folding model |
| `finetune_multiobjective.py` | DRAKES fine-tuning with the physics constraint added |
| `test_*.py` | Validation suites; run them before spending GPU time |

## The physics rewards

`physics_reward.py` is exact nearest-neighbor duplex thermodynamics, validated
against Biopython's independent implementation to 4.6e-13 C. `hairpin_reward.py`
is a differentiable **proxy** for hairpin propensity -- not a folding free
energy, and the distinction is load-bearing.

Use the hairpin term as the design constraint and duplex dG as a diagnostic.
`analyze_gc_confound.py` explains why. Duplex free energy is 99.6% explained by
base composition (R^2 = 0.996), so a model hits a dG target by shifting GC
content rather than learning anything about arrangement. Hairpin propensity
drops that to R^2 = 0.69, and against each metric's own composition-driven
range its response to arrangement is 11.3% versus 1.5%. At fixed 50% GC,
shuffling alone moves hairpin dG by up to 4.3 kcal/mol against a tolerance of
-3.0, so arrangement by itself decides whether the constraint is met.

### What the proxy is, and is not

`validate_against_vienna.py` measures it against ViennaRNA's full
Zuker/McCaskill model with Mathews DNA parameters at 37 C. Spearman rank
correlation against ViennaRNA MFE:

| sequence set              | stem 6 | stem 8 | stem 10 |
| ------------------------- | ------ | ------ | ------- |
| random, GC 0.30-0.70      | 0.91   | 0.91   | 0.90    |
| random, GC fixed at 0.50  | 0.59   | 0.57   | 0.59    |
| planted stems, 4-15 bp    | 0.73   | 0.86   | **0.94**|

Read honestly:

* The 0.91 across varying GC is inflated by both measures tracking composition.
* At fixed composition, agreement is **moderate (0.57-0.59) and does not
  improve with stem length**. That makes it structural rather than a tuning
  problem: it comes from the interior loops, bulges, multiloops and unfolded
  reference state the proxy omits.
* On deliberate contiguous hairpins agreement is high and rises with stem
  length, reaching 0.94.
* Only **33% of the most-structured decile** at fixed GC is shared with
  ViennaRNA, and that figure is also invariant to stem length.

So the proxy is a **strong-hairpin detector, not a folding-energy surrogate**.
That may be the right instrument regardless, since synthesis is disrupted by a
single strong stem rather than by the diffuse marginal structure random
sequences carry -- but that is a hypothesis these numbers do not establish, and
it should be presented as one.

The experimental design that follows: **train on the proxy, evaluate final
sequences with ViennaRNA.** Whether the improvement transfers is the result
worth reporting, in either direction. A proxy that guides optimization
successfully is a useful contribution; a proxy that does not transfer is a
finding, not a failure.

## What you are actually going to run

The thing worth understanding before you spend money on a GPU is that the
DRAKES training objective is the RLHF objective, almost line for line. From
`drakes_dna/finetune_reward_bp.py`:

```python
kl_loss    = torch.stack(total_kl, 1).sum(1).mean()
reward_loss = -torch.mean(reward)
loss        = reward_loss + kl_loss * current_alpha
```

Maximize a reward, minus `alpha` times the KL divergence from the frozen
pretrained model. That is the same trade-off that PPO-based RLHF makes: chase
the reward, but do not drift so far from the reference policy that you break
the fluency it learned during pretraining. The failure mode is the same too —
turn `alpha` down far enough and you get reward hacking, where the model finds
sequences the reward oracle loves and biology does not.

The interesting difference is *how* the gradient gets computed. RLHF on a
language model normally uses a policy-gradient estimator, because you cannot
backpropagate through a sampled token. DRAKES backpropagates through the entire
sampling chain directly, using the Gumbel-softmax trick to make each discrete
denoising step differentiable. That is the paper's contribution, and it is why
the run unrolls ~50 diffusion steps and holds them all in memory.

The pieces in the DNA experiment:

- **The policy**: a masked discrete diffusion model (MDLM) over 200-base-pair
  DNA sequences. Small — a 128-dim CNN, four stacks.
- **The reward model**: a [gReLU](https://genentech.github.io/gReLU/) oracle that
  predicts enhancer activity in a cell line. Two separate copies ship with the
  data: `reward_oracle_ft.ckpt` to train against, `reward_oracle_eval.ckpt` to
  score with. Using a *held-out* oracle for evaluation is how you catch reward
  hacking, and the split is worth copying into your own projects.
- **The reference model**: a frozen copy of the pretrained diffusion model,
  used only for the KL term.

---

## Step 1 — Connect the RunPod MCP server (on your own machine)

The command you found:

```bash
npx @runpod/mcp-server@latest add
```

That is an interactive installer. It detects your MCP clients, asks which to
configure, and offers hosted mode (OAuth, no key on disk — recommended) or
local mode (stores a `RUNPOD_API_KEY`). It needs a terminal and a browser, so
run it on your laptop, not inside a cloud session. Restart Claude Code
afterwards; MCP servers are loaded at startup.

The equivalent one-liner, skipping the wizard:

```bash
claude mcp add --transport http runpod -s user https://mcp.getrunpod.io/
```

Verify with `claude mcp list`, or `/mcp` inside a session.

Once connected, you can create pods, list GPU types, check what is running, and
stop pods by asking in plain language. That is genuinely useful for the "did I
leave a pod running overnight" problem, which is the main way people lose money
on RunPod.

You do not strictly need the MCP server — the web console does the same things.
It is a convenience, not a dependency.

## Step 2 — Create the pod

Settings that matter, in order of how much they will cost you if you get them
wrong:

**Attach a network volume**, 100 GB or more, mounted at `/workspace`. This is
the single most important choice. Container disk is wiped when a pod stops; the
network volume is not. The scripts default to `/workspace` for exactly this
reason. Without a volume you re-download tens of gigabytes of weights every
session.

**GPU.** An RTX A4000 (16 GB) works but needs a smaller batch; see the
memory note below. An A5000 or better (24 GB+) is more comfortable — the
model is small, and the memory pressure comes from unrolling diffusion steps,
not from parameters. An A100 40 GB is comfortable if you want to raise
`--batch_size`. For the *protein* experiment you must pick an Ampere card
(A100, A6000, A5000, 3090): it pins torch 2.0.1+cu117, which has no kernels for
H100 or 40-series and newer.

**Template.** Any RunPod PyTorch image. The scripts install their own conda
environment on top, so the image's torch version does not matter.

**Stop the pod when you are not using it.** You are billed per second while it
runs. A stopped pod keeps its volume; you pay only the volume's storage rate.

## Step 3 — Bootstrap

Get this repo onto the pod, then run the script. From a pod terminal (web
console or SSH):

```bash
tmux new -s setup
cd /workspace
git clone https://github.com/luckynbucky/drakes-runpod.git
bash drakes-runpod/setup_dna.sh
```

**Run it inside `tmux`.** Setup takes 30-60 minutes, and without tmux the whole
thing dies the moment your laptop sleeps or the browser tab drops: the terminal
session ends, the shell sends SIGHUP, and the script goes down mid-install. The
pod keeps running regardless -- it is your *connection* that is fragile, not the
machine. Inside tmux the process is owned by the pod, so disconnecting is
harmless.

Detach with ctrl-b then d. Reattach later, from any machine, with
`tmux attach -t setup`, or `tmux ls` to see what is running. If tmux is missing
on a bare image, `apt-get install -y tmux`.

The repo is public, so this needs no credentials. If you would rather not clone
it, the kit is only four files — `scp` them over, use `runpodctl send`, or paste
them into the pod's editor. `setup_dna.sh` locates its siblings relative to
itself, so any directory works.

The script installs Miniconda, creates the `sedd` environment on python 3.9.18,
installs torch 2.3.1+cu121 and the DRAKES dependencies, clones DRAKES, installs
gReLU pinned to v1.0.2, downloads the data bundle, and rewrites the hardcoded
paths. Budget 30–60 minutes, most of it the download.

Two things it handles that the upstream README does not:

- **gReLU must be v1.0.2.** The provided oracle checkpoints were saved with it.
  Later versions changed `LightningModel`, and loading fails with an unpickling
  error that looks like a corrupt download.
- **`causal-conv1d` is optional.** Upstream's `env.sh` installs it, but that is
  inherited from MDLM's Mamba backbone. The DNA config uses `backbone: cnn`, so
  the script tries it and moves on if the CUDA build fails.

Then verify before spending GPU time:

```bash
conda activate sedd
python /workspace/drakes-runpod/smoke_test.py --base-path /workspace/drakes_data
```

You want zero failures. It checks that CUDA kernels actually run (not just that
`torch.cuda.is_available()` returns True, which lies about driver mismatches),
that gReLU is the right version, that every checkpoint the training scripts
read is present, and that no file still points at the author's cluster.

## Step 4 — Fine-tune

```bash
cd /workspace/DRAKES/drakes_dna
python finetune_reward_bp.py --name run1 --base_path /workspace/drakes_data
```

Run it under `tmux` or `nohup` — an SSH drop otherwise kills the job.

Checkpoints land in `/workspace/drakes_data/mdlm/reward_bp_results_final/`,
every 50 epochs by default. A reference fine-tuned checkpoint ships with the
data bundle, so you can compare your run against the authors'.

`wandb` is imported unconditionally. Either `wandb login` first or set
`WANDB_MODE=offline`.

### The knobs worth experimenting with

| Flag | Default | Why you would change it |
| --- | --- | --- |
| `--alpha` | `0.001` | The KL penalty weight. **Start here.** Raise it and the model stays close to the pretrained distribution but gains less reward; lower it and watch reward hacking appear — the training oracle's score climbs while the held-out eval oracle's does not. That gap is the whole lesson. |
| `--batch_size` | `32` | Lower it first if you hit OOM. |
| `--num_accum_steps` | `4` | Gradient accumulation. Raising this while lowering batch size keeps the effective batch constant on a smaller GPU. |
| `--truncate_steps` | `50` | How many of the 128 diffusion steps get backpropagated through. This is the memory/fidelity dial. |
| `--truncate_kl` | `False` | Whether the KL term is computed on the truncated window only. |
| `--alpha_schedule_warmup` | `0` | Linearly ramps `alpha` over the first N epochs. |

The single most instructive experiment: run `--alpha 0.001` and `--alpha 0` with
everything else fixed, and plot the training-oracle reward against the
eval-oracle reward for both. You will see reward hacking happen, measured, in a
setting small enough to iterate on.

## Step 5 — Evaluate

`drakes_dna/eval.ipynb` scores generated sequences with the held-out oracle and
the ATAC chromatin-accessibility classifier. To reach it from your laptop, start
Jupyter on the pod and use the port RunPod exposes:

```bash
jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root
```

The `setup_dna.sh` script registers the conda env as a Jupyter kernel named
`Python (sedd)` — select it, or the imports will fail.

---

## GPU memory

The diffusion model is small -- a 128-dim CNN, 51 MB -- but the reward oracles
are not: `reward_oracle_ft.ckpt` is 2.2 GB and `reward_oracle_eval.ckpt` is
2.6 GB. The training oracle sits inside the gradient path, so its activations
are retained for the backward pass alongside the unrolled diffusion steps.

On a 16 GB card, start small and measure rather than guessing:

```bash
--batch_size 8 --num_accum_steps 16
```

That holds the effective batch at 128, the same as the 32 x 4 default, so the
optimization is unchanged. Every epoch logs `peak_gpu_gb` (peak *allocated*,
not reserved), and the epoch line prints it, so raise `--batch_size` until the
headroom runs out rather than picking a number blind.

Two more levers if it still will not fit:

* `--eval_oracle_device cpu` moves the held-out oracle off the GPU. It is only
  used for reporting and never enters the loss, so this costs one transfer per
  accumulation step and frees its parameters.
* `--truncate_steps` controls how many of the diffusion steps are
  backpropagated through, and memory scales roughly linearly in it. Lowering it
  changes the method, not just the footprint, so treat it as a last resort --
  and if you do sweep it, the memory-versus-reward curve is itself a result
  worth plotting.

## Step 6 - The multi-objective experiment

This is the part that is yours rather than the paper's. Everything above gets
stock DRAKES running; this adds the physics constraint and produces the two
plots worth presenting.

### 6a. Put the physics modules where DRAKES can import them

`finetune_multiobjective.py` imports DRAKES internals (`diffusion_gosai_update`,
`oracle`, `utils`), so it has to live in `drakes_dna/` alongside them:

```bash
cd /workspace
cp drakes-runpod/physics_reward.py \
   drakes-runpod/hairpin_reward.py \
   drakes-runpod/finetune_multiobjective.py \
   DRAKES/drakes_dna/
```

### 6b. Validate before spending GPU time

```bash
conda activate sedd
cd /workspace/drakes-runpod
python test_physics_reward.py            # 13 checks, needs biopython
python test_hairpin_reward.py            # 16 checks
python test_multiobjective_integration.py  # 13 checks
python analyze_gc_confound.py            # the confound numbers, ~1 min
```

All 42 should pass. They run on CPU in about a minute and will catch a broken
environment before a training run does.

### 6c. Smoke run

Two epochs at a tiny batch, just to prove the wiring holds end to end. `--name
debug` skips wandb.

```bash
cd /workspace/DRAKES/drakes_dna
python finetune_multiobjective.py \
  --name debug \
  --base_path /workspace/drakes_data/ \
  --num_epochs 2 --num_accum_steps 1 --batch_size 8 --save_every_n_epochs 1 \
  --w_phys 0.5
```

`--save_every_n_epochs 1` matters here: the default interval is 50, so a
two-epoch run would otherwise write no periodic checkpoint. A final
checkpoint is always written regardless, and every epoch overwrites
`checkpoint_latest.pt`, so an interrupted run resumes with
`--resume <run_dir>/checkpoint_latest.pt` instead of starting over.

Read the first epoch line carefully before going further. It reports
`bio(train)`, `bio(held-out)`, `hairpin dG`, `viol` (fraction of sequences
violating the constraint) and `GC`. Two things to check:

* **Do the reward scales match?** If the biological reward is around 5 and the
  physics penalty around -40, then `--w_phys 1.0` means the physics term
  dominates entirely. Pick `w_phys` so the weighted terms are comparable, or
  set `--hairpin_scale` to divide the penalty down.
* **Is `viol` non-zero?** The hairpin penalty is one-sided and exactly flat
  where the constraint is already satisfied. If no sampled sequence violates
  tolerance, the physics term contributes zero gradient and the run will look
  like the physics is doing nothing. Tighten `--hairpin_tolerance` (less
  negative is stricter) until some sequences violate it.

Time this run and extrapolate before committing to a long one.

### 6d. The baseline

`--w_phys 0` reproduces stock DRAKES through this identical code path. The
gradients are bitwise identical to biology-only, so this is a genuine control
rather than an approximation of one.

```bash
tmux new -s baseline
python finetune_multiobjective.py \
  --name baseline --w_phys 0 \
  --base_path /workspace/drakes_data/ \
  --num_epochs 200
```

Use `tmux` (detach with ctrl-b d) or `nohup`. An SSH drop otherwise kills the
job. Either `wandb login` first or export `WANDB_MODE=offline`.

### 6e. The sweep

```bash
for w in 0 0.1 0.3 1.0 3.0; do
  python finetune_multiobjective.py \
    --name sweep --w_phys $w \
    --base_path /workspace/drakes_data/ \
    --num_epochs 200
done
```

Each run writes `metrics.jsonl` under
`/workspace/drakes_data/mdlm/reward_bp_results_final/<run_name>/`, one JSON
object per epoch, so both plots are a short pandas read:

* **Pareto frontier** - final `bio_reward_heldout_oracle` against final
  `hairpin_dg_ensemble`, one point per `w_phys`.
* **Reward hacking** - `bio_reward_train_oracle` and
  `bio_reward_heldout_oracle` against epoch, on one axis. The curves separating
  is reward hacking, and `reward_hacking_gap` is logged directly.

Plot `gc_content` alongside. Hairpin propensity keeps a real correlation with
base composition, so showing the constraint was met without a large GC shift is
what demonstrates the model learned arrangement rather than composition.

## Cost, roughly

A 24 GB card runs about $0.20–0.45/hour on RunPod's community cloud, and an
A100 40 GB about $1.10–1.90, though pricing moves. Setup is an hour, mostly
downloading. Fine-tuning is the real spend, so treat the first run as a short
smoke run — a handful of epochs, confirm reward moves in the right direction,
kill it — before committing to a long one.

Storage on a stopped pod's volume bills at roughly $0.05–0.10/GB/month, so a
100 GB volume is a couple of dollars a month to keep your environment alive
between sessions. That is almost always worth it versus re-running setup.

## Troubleshooting

**`CondaToSNonInteractiveError` during setup** — recent Miniconda will not use
Anaconda's `defaults` channels until their Terms of Service are accepted. The
setup script creates its environment from conda-forge with
`--override-channels` to avoid the gate entirely, so pull the latest version of
this repo and re-run. Re-running is safe: it reuses the existing Miniconda and
skips anything already done. If you would rather accept the Anaconda terms
instead, run the two `conda tos accept` commands the error prints.

**Out of disk space** — check what is actually consuming it:

```bash
df -h /workspace
du -sh /workspace/* | sort -h
```

The fixed costs are the conda environment, which is roughly 8-10 GB once
PyTorch and its CUDA libraries are in, plus whatever the data bundle extracts
to. The bundle carries data for BOTH the DNA and protein experiments, so if you
are only doing DNA you can extract just the part you need:

```bash
unzip /workspace/drakes_data/DRAKES_data.zip 'mdlm/*' -d /workspace/drakes_data
```

`setup_dna.sh` measures the archive before extracting and stops with these
instructions if the space is not there, rather than filling the volume and
failing partway. RunPod volumes can be resized upward in the console.

**Setup sits on "Installing DRAKES dependencies" for a very long time** — pip
resolver backtracking. Python 3.9 is end-of-life, and nine of the packages here
have dropped support for it, so an unpinned install searches release history for
a compatible combination and prints nothing meanwhile. The script pins the
newest 3.9-compatible release of each, which avoids the search. Pull the latest
version of this repo and re-run; the re-run skips the conda environment, the
clone and PyTorch, so it restarts at the dependency step.

**`ModuleNotFoundError` for `grelu`, `hydra`, anything** — the conda env is not
active. `conda activate sedd`. New shells on the pod do not inherit it.

**Unpickling error loading an oracle checkpoint** — wrong gReLU version.
`pip install git+https://github.com/Genentech/gReLU.git@v1.0.2`.

**`FileNotFoundError` under `/data/scratch/wangchy/`** — the path rewrite did
not cover something. Re-run `set_base_path.py --dry-run` to see what is left.

**CUDA OOM** — lower `--batch_size` to 16 or 8 and raise `--num_accum_steps` to
keep the effective batch size the same.

**Everything vanished after a restart** — it was on container disk, not
`/workspace`. Recreate the pod with a network volume attached.

## References

- Paper: [Fine-Tuning Discrete Diffusion Models via Reward Optimization](https://arxiv.org/abs/2410.13643)
- Code: [ChenyuWang-Monica/DRAKES](https://github.com/ChenyuWang-Monica/DRAKES)
- Built on [MDLM](https://github.com/kuleshov-group/mdlm) (DNA) and [MultiFlow](https://github.com/jasonkyuyim/multiflow) (protein)
- Reward oracles: [gReLU](https://genentech.github.io/gReLU/)
