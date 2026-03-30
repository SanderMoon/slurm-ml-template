# ml-template

Copier template for ML projects with SLURM cluster integration, config-driven development, and standardized data management.

## Quick start

### Prerequisites

- **Local tools:** git, [uv](https://docs.astral.sh/uv/), [copier](https://copier.readthedocs.io/), rsync
- **SSH:** key-based auth to your cluster login node, configured as an alias in `~/.ssh/config`
- **Shared storage:** mounted on your laptop and accessible from cluster nodes
- **Docker image:** base image with PyTorch, uv, and pipeline dependencies

See [docs/setup.md](docs/setup.md) for detailed setup instructions.

### Create a new project

```bash
copier copy --trust git@github.com:SanderMoon/ml-template.git ~/dev/my-project
```

Copier will prompt you for project name, cluster details (SSH host, NAS paths, Docker registry, GPU tiers), and optional features. There are no defaults for personal values — you must fill in your own.

```bash
cd ~/dev/my-project
source .venv/bin/activate
make install
make test
```

### Try the example task

The template ships with a working example (synthetic data, linear model) that runs the full pipeline in ~30 seconds:

```bash
make preprocess PREPROCESS_ARGS="task=example"               # generate synthetic data
make train TRAIN_ARGS="task=example training.epochs=5"        # train for 5 epochs
make eval EVAL_ARGS="task=example"                            # evaluate best checkpoint
```

Once this works, implement your own task in `experiments/tasks/classification.py`.

## What you get

```
my-project/
├── src/my_project/          Library (pip install my_project)
│   ├── models/              Model architectures (pure PyTorch)
│   ├── data/                Datasets, transforms, collators
│   └── utils.py             Library utilities (seed_everything)
├── experiments/             Training pipeline (NOT distributed)
│   ├── tasks/
│   │   └── classification.py  Your training recipe (fill this in)
│   ├── train.py             Dispatcher (loads task, assembles Trainer)
│   ├── trainer.py           Trainer (Accelerate, checkpointing, logging, preemption)
│   ├── eval.py              Evaluator (checkpoint loading, fold aggregation)
│   ├── preprocess.py        Preprocessing (config hashing, skip-if-exists)
│   └── config.py            OmegaConf resolvers (dataset manifest)
├── configs/
│   ├── config.yaml          All defaults (Hydra entrypoint)
│   ├── experiment/          Experiment overrides (+experiment=name)
│   └── nodes/               GPU tier configs (high, medium, low)
├── scripts/slurm/           Job scripts (train, eval, preprocess, dev)
├── tests/
├── Makefile                 All commands
├── CLAUDE.md                Development guidelines
└── .env                     Per-system config (git-ignored)
```

## Documentation

| Doc | What's in it |
|-----|-------------|
| [Tutorial](docs/tutorial.md) | Getting started — run the example, fill in your task, submit to cluster |
| [Advanced](docs/advanced.md) | Cross-validation, custom optimizers, GANs, step-based training, eval callbacks |
| [Commands](docs/commands.md) | Full Makefile reference, GPU tiers, Docker, data manifest |
| [Setup](docs/setup.md) | Local tools, SSH, shared storage, Docker image, SLURM requirements |
| [Troubleshooting](docs/troubleshooting.md) | Common failures, NAS quirks, email notifications |

## Philosophy: scaffold, don't upstream

This template is a **starting point**, not an upstream dependency. Use `copier copy`
to bootstrap new projects, then the project owns all its files.

**Don't use `copier update` on existing projects.** In practice, every file diverges
from the template over time — the Dockerfile gets custom packages, the Makefile gets
project-specific targets, configs get real hyperparameters, the README gets experiment
results. Trying to merge template updates into a diverged project creates more
problems than it solves.

When you improve the template:
1. **New projects** get the improvement automatically via `copier copy`.
2. **Existing projects** — cherry-pick manually if the fix matters. Most template
   improvements (new Make target, trainer bugfix) are small and quick to apply by hand.
   Large changes (like a Trainer redesign) need per-project review anyway.

## Tip: shell alias for your defaults

If you're a colleague on the same cluster, contact [Sander](https://github.com/SanderMoon) for
a pre-configured `newml` shell function with all institutional values filled in. You'll only
need to update the personal values (`github_user`, `author`, `nas_base_path`, etc.).

For other institutions, create your own alias in `.zshrc` (or `.bashrc`). Fill in personal
values and share the institutional values (`remote_host`, `container_mounts`, `nodes_*`,
etc.) with your team:

```bash
newml() {
    local dest="${1:?Usage: newml <project-path>}"
    copier copy --trust \
        -d github_user="YourGitHub" \
        -d author="Your Name" \
        -d base_docker_image="your-registry/your-namespace/ml-base:latest" \
        -d docker_registry="your-registry" \
        -d docker_namespace="your-namespace" \
        -d remote_host="cluster-login" \
        -d nas_base_path="/data/projects/yourname" \
        -d nas_mount_path="/mnt/shared/yourname" \
        -d container_mounts="/data/projects" \
        -d nas_name="NAS" \
        -d slurm_mail_domain="" \
        -d nodes_high="" \
        -d nodes_medium="" \
        -d nodes_low="" \
        git@github.com:SanderMoon/ml-template.git "$dest"
}
```

Then creating a new project is just `newml ~/dev/my-project`. Copier still prompts for any
values not provided via `-d` (like `project_name` and `description`).

## Cluster requirements

The template assumes a SLURM cluster with Enroot/Docker container support, shared storage
accessible from all nodes, and SSH key-based access. See [docs/setup.md](docs/setup.md)
for the full list of requirements and setup instructions.
