# Command reference

All commands are run via `make`. Every command that accepts arguments uses `*_ARGS` variables for Hydra overrides.

## Local development

| Command | Description |
|---------|-------------|
| `make install` | Install project in dev mode (`uv pip install -e ".[dev]"`) |
| `make preprocess` | Run preprocessing locally |
| `make preprocess PREPROCESS_ARGS="preprocess.patch_size=512"` | With Hydra override |
| `make train` | Run training locally |
| `make train TRAIN_ARGS="training.epochs=5"` | With Hydra override |
| `make train TRAIN_ARGS="+experiment=my_exp"` | With experiment overlay |
| `make train TRAIN_ARGS="--multirun training.lr=0.001,0.01,0.1"` | Local sweep |
| `make eval` | Evaluate best checkpoint |
| `make eval EVAL_ARGS="eval.checkpoint=latest"` | Evaluate latest checkpoint |
| `make test` | Run pytest |
| `make lint` | Check formatting (ruff) |
| `make format` | Auto-format code |

## Cluster submission

| Command | Description |
|---------|-------------|
| `make submit` | Sync code + submit training to default tier |
| `make submit TIER=high` | Submit to high-VRAM nodes (H200) |
| `make submit TIER=low GRES=gpu:2` | Override SLURM params |
| `make submit TRAIN_ARGS="+experiment=my_exp"` | With experiment config |
| `make submit-preprocess` | Submit preprocessing job (CPU-only default) |
| `make submit-preprocess PREPROCESS_GRES=gpu:1` | GPU preprocessing |
| `make submit-eval` | Sync + submit evaluation job |
| `make sweep SWEEP="training.lr=0.001,0.01,0.1"` | Parameter sweep (one job per value) |
| `make sweep SWEEP="training.lr=0.001,0.01 training.epochs=50,100"` | Grid sweep |
| `make sweep-eval` | Submit eval jobs for all sweep runs |
| `make sweep-results` | Compare sweep runs (prints sorted table) |
| `make dev` | Launch interactive dev session |
| `make dev TIER=high` | Dev session on high-VRAM node |

## Monitoring

| Command | Description |
|---------|-------------|
| `make status` | Show your SLURM jobs |
| `make logs` | Tail latest job output |
| `make logs JOB=12345` | Tail specific job |
| `make cancel` | Cancel most recent job (with confirmation) |
| `make gpu-status` | GPU availability across cluster |

## Data management

| Command | Description |
|---------|-------------|
| `make init-nas` | Create project folders on NAS |
| `make copy-data SRC=/path/to/data NAME=x` | Copy dataset into data/raw/ |
| `make snapshot NAME=x` | Hardlink-copy external dataset (zero extra disk, survives source deletion) |
| `make sync` | Rsync code to NAS (respects .syncignore) |

## Docker

| Command | Description |
|---------|-------------|
| `make build` | Build project Docker image |
| `make push` | Build + push to registry |

### Bring your own Dockerfile

The template ships with a two-layer Docker setup (base image + project layer), but you can use any image. Minimum requirements:

**Python packages** (imported by template code):
```
torch accelerate hydra-core omegaconf tqdm python-dotenv numpy tensorboard
```

Plus `wandb` if `use_wandb: true` in config.

**System tools**: `uv` (the sbatch scripts use it to create a venv and install the project editable).

Minimal custom Dockerfile:
```dockerfile
FROM pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime
RUN pip install uv && uv pip install --system \
    accelerate hydra-core omegaconf tqdm python-dotenv numpy tensorboard
```

Set `DOCKER_IMAGE=your-registry/your-image:tag` in `.env` and all `make submit*` commands will use it.

## GPU tiers

Jobs are submitted to GPU tiers defined in `configs/nodes/{high,medium,low}.sh`. Each tier specifies a node list, GPU count, CPU/memory limits, and time limit. Edit these files to match your cluster's hardware.

Override the default tier in `.env` (`TIER=medium`) or per-command (`make submit TIER=high`).

## The `.env` file

Each generated project has a `.env` file (git-ignored) that stores per-system config. Created automatically by copier from your answers.

```bash
PROJECT_SLUG=my_project
REMOTE_HOST=cluster-login
NAS_BASE=/data/projects/yourname
NAS_MOUNT=/mnt/shared/yourname
TIER=medium
QOS=high
DOCKER_IMAGE=your-registry/your-namespace/ml-base:latest
```

All Makefile commands read from `.env`. Override any value on the command line: `make submit TIER=high`.

## Data manifest

Data dependencies are declared in `configs/datasets.yaml`:

```yaml
datasets:
  my_features:
    type: external
    source: /data/projects/someone/features
    owner: someone
    description: Feature embeddings for XYZ
```

Reference in configs via the `${data:name}` OmegaConf resolver:

```yaml
data:
  feature_dir: ${data:my_features}
```

Protect against source deletion with hardlink snapshots:
```bash
make snapshot NAME=my_features   # zero extra disk space, survives source deletion
```
