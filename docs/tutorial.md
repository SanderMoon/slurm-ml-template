# Getting started

This tutorial walks you through the template step by step. By the end you'll know what to change, what to leave alone, and how to run experiments on the cluster.

## Chapter 1: See it work first

Before changing anything, run the built-in example task to verify the full pipeline:

```bash
source .venv/bin/activate
make install

# Preprocess → Train → Evaluate (takes ~30 seconds)
make preprocess PREPROCESS_ARGS="task=example"
make train TRAIN_ARGS="task=example training.epochs=5"
make eval EVAL_ARGS="task=example"
```

You should see training logs, a checkpoint saved in `checkpoints/`, and evaluation results printed at the end. If this works, your setup is correct.

## Chapter 2: What to change

Your work happens in two places:

1. **`src/<your_project>/models/`** — your model architectures (pure PyTorch, no framework deps). This is the code that gets published when you `pip install` your package.
2. **`experiments/tasks/`** — training recipes: data loading, loss functions, and evaluation logic. Tasks import your models and wire them to the training pipeline. Copy `example.py` as a starting point for your own task.

The task file is where most of the work happens. Copy the example and fill in the functions:

| Function | What it does | When it runs |
|----------|-------------|--------------|
| `preprocess(cfg)` | Raw data → processed data | `make preprocess` |
| `build_dataloader(cfg, split)` | Returns a DataLoader for train/val/test | `make train` and `make eval` |
| `build_model(cfg)` | Returns your `nn.Module` (on CPU) | `make train` and `make eval` |
| `make_compute_loss(cfg)` | Returns a `(model, batch) -> loss` callable | `make train` |
| `make_evaluate(cfg)` | Returns a fast validation callable | `make train` (runs every N epochs) |
| `make_test_evaluate(cfg)` | Returns a thorough evaluation callable | `make eval` |

Look at `experiments/tasks/example.py` for a complete working reference — it's the code behind the example you just ran.

### The minimal path

To get a new task training, you need exactly four things:

**1. A dataset class and dataloader:**

```python
def build_dataloader(cfg, split="train"):
    dataset = MyDataset(Path(cfg.paths.processed_dir) / split)
    return DataLoader(dataset, batch_size=cfg.data.batch_size, shuffle=(split == "train"))
```

The framework calls this with `split="train"`, `"val"`, or `"test"`. If your data uses different names (e.g. "tune" instead of "val"), map them with a dict in the task:

```python
def build_dataloader(cfg, split="train"):
    split_name = {"train": "train", "val": "tune", "test": "test"}[split]
    dataset = MyDataset(Path(cfg.paths.processed_dir) / split_name)
    ...
```

**2. A model** (defined in `src/<your_project>/models/`, imported by the task):

```python
from my_project.models import MyModel

def build_model(cfg):
    return MyModel(num_classes=cfg.model.num_classes, hidden_dim=cfg.model.hidden_dim)
```

**3. A loss function:**

```python
def make_compute_loss(cfg):
    def compute_loss(model, batch):
        logits = model(batch["input"])
        return F.cross_entropy(logits, batch["target"])
    return compute_loss
```

**4. A validation function:**

```python
def make_evaluate(cfg):
    @torch.no_grad()
    def evaluate(model, val_loader, accelerator):
        model.eval()
        total_loss, n = 0.0, 0
        for batch in val_loader:
            loss = F.cross_entropy(model(batch["input"]), batch["target"])
            total_loss += loss.item()
            n += 1
        return {"val_loss": total_loss / max(n, 1)}
    return evaluate
```

That's it. The Trainer handles optimizer creation (AdamW), scheduling (cosine + warmup), distributed training, checkpointing, logging, and SLURM preemption — you don't write any of that (unless you want to override, then see [advanced.md](/docs/advanced.md).

### Config: hyperparameters live in YAML, not Python

All model and training parameters are in **`configs/config.yaml`** — this is the single source of truth for your project. Don't hardcode values in Python; reference `cfg` instead:

```yaml
# configs/config.yaml
model:
  num_classes: 2       # ← change these
  hidden_dim: 256
  dropout: 0.25

training:
  epochs: 100
  lr: 0.001
  weight_decay: 0.0001
```

Override anything from the command line:

```bash
make train TRAIN_ARGS="training.lr=0.01 model.hidden_dim=512"
```

For reproducible experiments, create a file in `configs/experiment/`:

```yaml
# configs/experiment/big_model.yaml
model:
  hidden_dim: 1024
  dropout: 0.1
training:
  lr: 0.0005
```

```bash
make train TRAIN_ARGS="+experiment=big_model"
```

### Returning multiple losses

If your training step produces multiple losses (e.g., reconstruction + KL divergence), return a dict from `compute_loss` instead of a scalar:

```python
def make_compute_loss(cfg):
    def compute_loss(model, batch):
        output = model(batch["input"])
        recon_loss = F.mse_loss(output["reconstruction"], batch["input"])
        kl_loss = output["kl_divergence"]
        return {
            "loss": recon_loss + 0.1 * kl_loss,   # ← must have "loss" key (used for backprop)
            "recon_loss": recon_loss,               # ← extra metrics: logged + passed to callbacks
            "kl_loss": kl_loss,
        }
    return compute_loss
```

The extra metrics show up in W&B/TensorBoard as `train/recon_loss`, `train/kl_loss`.

## Chapter 3: What NOT to change

The `experiments/` directory contains the training pipeline. These files work out of the box and rarely need editing:

| File | Role | Why you don't touch it |
|------|------|----------------------|
| `experiments/train.py` | Dispatcher | Loads your task, assembles the Trainer. Handles cross-validation routing. |
| `experiments/trainer.py` | Training engine | Accelerate, checkpointing, logging, preemption handling. |
| `experiments/eval.py` | Evaluation engine | Checkpoint loading, fold aggregation, prediction saving. |
| `experiments/preprocess.py` | Preprocessing dispatcher | Config hashing, skip-if-exists. Calls your `preprocess(cfg)`. |
| `experiments/config.py` | OmegaConf resolvers | `${data:name}` dataset manifest resolver. |
| `scripts/slurm/*.sbatch` | SLURM job scripts | Data staging, venv setup, preemption-resilient requeuing. |
| `Makefile` | Command interface | All `make` targets. |

The key distinction: `src/<your_project>/` is your **library** (published to PyPI), `experiments/` is your **pipeline** (stays in the repo). When someone does `pip install your_project`, they get your models and data utilities — not the training infrastructure.

You interact with the pipeline through your task module and config — not by editing it.

## Chapter 4: Running on the cluster

### Submit a training job

```bash
make submit                                          # default tier (medium)
make submit TIER=high                                # H200 GPUs
make submit TRAIN_ARGS="+experiment=big_model"       # with experiment config
```

This syncs your code to the cluster, submits a SLURM job, and prints the job ID.

### Monitor

```bash
make status          # your SLURM jobs
make logs            # tail latest job output
make gpu-status      # GPU availability
```

### Parameter sweeps

```bash
make sweep SWEEP="training.lr=0.001,0.01,0.1"       # one job per value
make sweep-results                                    # compare metrics
```

### Evaluation on cluster

```bash
make submit-eval                     # eval best checkpoint
make sweep-eval                      # eval all sweep runs
```

### Preprocessing on cluster

```bash
make submit-preprocess                               # CPU-only (default)
make submit-preprocess PREPROCESS_GRES=gpu:1          # with GPU
```

## The mental model

```
src/<your_project>/                    experiments/
(library — pip install)                (pipeline — not distributed)
─────────────────────                  ────────────────────────────
models/                                tasks/
  └── your model architectures           └── example.py (copy as starting point)
data/                                        ├── preprocess()
  └── datasets, transforms, collators       ├── build_dataloader()
utils.py                                    ├── build_model()
                                             ├── make_compute_loss()
                                             ├── make_evaluate()
                                             └── make_test_evaluate()
                                       train.py (dispatcher)
                                       trainer.py (Accelerate, checkpoints, logging)
                                       eval.py (checkpoint loading, fold aggregation)
                                       preprocess.py (config hashing, skip-if-exists)

configs/config.yaml                    Makefile (all commands)
configs/experiment/*.yaml              scripts/slurm/ (job submission)
```

Start with the model, then the task file. Change the config. Run `make train`. Everything else is there when you need it.
