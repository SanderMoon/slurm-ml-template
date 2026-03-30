# Setup guide

This guide covers everything you need before creating your first project. If you're a colleague who received a `newml` shell function, most of the institutional config is already handled — but you still need the local tools and SSH setup.

## 1. Local tools

Install these on your development machine before running `copier copy`:

| Tool | What it's for | Install |
|------|--------------|---------|
| [git](https://git-scm.com/) | Version control | `apt install git` / `brew install git` |
| [uv](https://docs.astral.sh/uv/) | Python package manager (local + in containers) | `curl -LsSf https://astral.sh/uv/install.sh \| sh` |
| [copier](https://copier.readthedocs.io/) | Project generation from template | `pip install copier` or `uv pip install copier` |
| [rsync](https://rsync.samba.org/) | Code sync to cluster | `apt install rsync` / `brew install rsync` |
| [docker](https://www.docker.com/) | Build/push project images (optional if using shared base image) | [Install Docker](https://docs.docker.com/get-docker/) |

**Verify:**
```bash
git --version && uv --version && copier --version && rsync --version
```

## 2. SSH configuration

All cluster commands (`make submit`, `make sync`, `make dev`, etc.) connect via SSH using the alias you provide as `remote_host` during project creation. This alias must be configured in `~/.ssh/config`.

### 2.1 Create an SSH key (if you don't have one)

```bash
ssh-keygen -t ed25519
ssh-copy-id your-cluster-login-node
```

### 2.2 Configure `~/.ssh/config`

Add an entry for your cluster login node:

```
Host cluster-login
    HostName login.your-cluster.edu
    User yourusername
    IdentityFile ~/.ssh/id_ed25519
    ForwardAgent yes
```

Replace `cluster-login` with whatever you'll use as `remote_host` in copier (e.g., `oaks-lab`).

**`ForwardAgent yes`** is important — it allows SSH agent forwarding, which the `make dev` interactive sessions need so you can SSH from the login node to compute nodes without re-entering credentials.

### 2.3 Compute node access

The `make dev` command starts an interactive session on a compute node and provides an SSH command to connect to it. For this to work, your SSH key must be accepted on compute nodes too. On most clusters this works automatically if your home directory is shared across nodes.

If your cluster uses a jump host pattern, add entries for the compute nodes:

```
Host gpu-node-*
    ProxyJump cluster-login
    User yourusername
    ForwardAgent yes
```

### 2.4 Verify

```bash
ssh cluster-login "echo connected"
```

## 3. Shared storage (NAS)

The template assumes a shared filesystem that is accessible from:
- Your laptop (for browsing outputs, quick syncs)
- The cluster login node (for rsync)
- All compute nodes (for training data, checkpoints, code)

### 3.1 What you need to know

During project creation, copier asks for two paths:

| Prompt | Example | What it is |
|--------|---------|------------|
| `nas_base_path` | `/data/projects/yourname` | Your personal directory on the shared filesystem, as seen from the cluster |
| `nas_mount_path` | `/Volumes/data/yourname` or `/mnt/nas/yourname` | The same directory, as seen from your laptop |
| `container_mounts` | `/data/projects,/data/archive` | Comma-separated paths to bind-mount into containers |

These must point to the **same filesystem location** from different machines. The template uses `nas_base_path` for cluster operations and `nas_mount_path` for local operations (e.g., reading logs without SSH).

### 3.2 Directory structure

After running `make init-nas`, your project folder on shared storage looks like:

```
<nas_base_path>/<project_slug>/
├── data/{raw,processed}   ← datasets
├── checkpoints/           ← model checkpoints
├── outputs/               ← training outputs, metrics
├── logs/                  ← SLURM job logs
└── code/                  ← synced source code
```

### 3.3 Mount the shared filesystem

**macOS** (typical NFS/SMB mount):
```bash
# Check if mounted
ls /Volumes/your-mount-point/
```

**Linux** (NFS):
```bash
# Add to /etc/fstab or mount manually
sudo mount -t nfs nas-server:/export/path /mnt/nas
```

If the NAS isn't mounted, `make submit` falls back to SSH-based rsync (slower but works). The Makefile's `make doctor` command checks this.

### 3.4 Filesystem limitations

Some shared filesystems (NFS, GPFS) don't support changing file permissions. This affects Python's `shutil` module — see [troubleshooting.md](troubleshooting.md#nas-filesystem-restrictions) for details. The template's Trainer handles this correctly.

## 4. Base Docker image

All SLURM jobs run inside a Docker/Enroot container. You provide the image URL as `base_docker_image` during project creation. This image must contain:

### 4.1 Required contents

| Component | Why |
|-----------|-----|
| **Python** (matching your `python_version`) | Project code runs here |
| **PyTorch** | ML framework |
| **uv** | sbatch scripts use it to create lightweight venvs |
| **sshd** (`/usr/sbin/sshd`) | `make dev` sessions need SSH daemon for interactive access |
| **nvidia-smi** | GPU detection in job logs (non-critical, but useful) |
| **Pipeline dependencies** | `accelerate`, `hydra-core`, `omegaconf`, `tqdm`, `python-dotenv`, `tensorboard` |

Plus `wandb` if you enable W&B tracking.

### 4.2 Minimal Dockerfile

If you're building your own base image:

```dockerfile
FROM pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime

# uv for fast venv creation in sbatch scripts
RUN pip install uv

# ML pipeline dependencies (installed as system packages so sbatch
# scripts can inherit them via --system-site-packages)
RUN uv pip install --system \
    accelerate hydra-core omegaconf tqdm python-dotenv \
    numpy tensorboard

# SSH daemon for interactive dev sessions
RUN apt-get update && apt-get install -y openssh-server && \
    mkdir -p /run/sshd && \
    rm -rf /var/lib/apt/lists/*
```

### 4.3 Using a shared base image

If a colleague already maintains a base image, just use their image URL as `base_docker_image`. The sbatch scripts install your project code on top via `uv pip install -e . --no-deps` at job start — so only the base dependencies need to be in the image.

### 4.4 Project-specific image

The template includes a `scripts/docker/Dockerfile` that builds on top of your base image and adds project-specific dependencies from `pyproject.toml`. Use this when you've added packages that aren't in the base image:

```bash
make build    # build project image
make push     # push to registry
```

Then update `DOCKER_IMAGE` in `.env` to point to your project image instead of the base.

### 4.5 Docker registry authentication

If your registry requires authentication:

```bash
docker login your-registry.example.com
```

This must be done locally before `make build` / `make push`. The cluster also needs access to pull the image — check with your cluster admin if images aren't pulling.

## 5. SLURM cluster requirements

The template assumes your cluster supports:

| Feature | Used by | SLURM flags |
|---------|---------|-------------|
| Enroot/Docker containers | All jobs | `--container-image`, `--container-mounts`, `--no-container-entrypoint` |
| Preemption signals | `make submit` (training) | `--requeue`, `--signal=SIGUSR1@120` |
| QoS | Job priority | `--qos` |
| GPU resources | All GPU jobs | `--gres=gpu:N` |

If your cluster doesn't use Enroot containers (e.g., uses Singularity instead), you'll need to modify the sbatch scripts in `scripts/slurm/`.

### 5.1 GPU tier configuration

After creating a project, edit `configs/nodes/{high,medium,low}.sh` with your cluster's node names. Run `sinfo -N -l` on the cluster to see available nodes and their resources:

```bash
ssh cluster-login "sinfo -N -l --partition=gpu"
```

Then fill in the node configs:

```bash
# configs/nodes/high.sh
NODELIST=gpu-node-01,gpu-node-02
GRES=gpu:1
CPUS=16
MEM=128G
TIME=72:00:00
```

If you used the `newml` shell function with `-d nodes_high=...` etc., these are already populated.

## 6. Verify your setup

After creating a project, run:

```bash
cd ~/dev/my-project
source .venv/bin/activate
make install
make doctor       # checks SSH, NAS mount, Docker image
make test         # runs pytest locally
```

Then test the full pipeline locally:

```bash
make preprocess PREPROCESS_ARGS="task=example"
make train TRAIN_ARGS="task=example training.epochs=5"
make eval EVAL_ARGS="task=example"
```

And on the cluster:

```bash
make submit TRAIN_ARGS="task=example training.epochs=5"
make status
make logs
```

If all of these work, you're ready to implement your own task.

## Quick reference: what goes where

| Value | Where it's set | Personal or institutional? |
|-------|---------------|---------------------------|
| `project_name` | copier prompt | Per-project |
| `author`, `github_user` | copier prompt | Personal |
| `nas_base_path`, `nas_mount_path` | copier prompt | Personal |
| `docker_namespace` | copier prompt | Personal |
| `remote_host` | copier prompt / `newml` | Institutional |
| `base_docker_image` | copier prompt / `newml` | Institutional |
| `docker_registry` | copier prompt / `newml` | Institutional |
| `container_mounts` | copier prompt / `newml` | Institutional |
| `nodes_high`, `nodes_medium`, `nodes_low` | copier prompt / `newml` | Institutional |
| `WANDB_API_KEY`, `HF_TOKEN` | `.env` (manual) | Personal |
| `SLURM_MAIL` | `.env` (manual) | Personal |
