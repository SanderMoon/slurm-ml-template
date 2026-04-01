# Advanced features

This guide covers features you'll reach for once you're comfortable with the basics in [tutorial.md](tutorial.md). Everything here is opt-in — the default pipeline works without any of it.

## Custom optimizer or scheduler

Export `build_optimizer` or `build_scheduler` from your task to override the defaults (AdamW + cosine with warmup):

```python
def build_optimizer(model, cfg):
    # Separate learning rates for backbone and head
    return torch.optim.AdamW([
        {"params": model.backbone.parameters(), "lr": cfg.training.lr * 0.1},
        {"params": model.head.parameters(), "lr": cfg.training.lr},
    ])

def build_scheduler(optimizer, cfg):
    return torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=cfg.training.lr, total_steps=cfg.training.epochs,
    )
```

## Cross-validation

The framework supports two CV patterns. Both require your `build_dataloader` to accept a `fold` parameter:

```python
def build_dataloader(cfg, split="train", fold=None) -> DataLoader:
    dataset = MyDataset(Path(cfg.paths.processed_dir) / split, fold=fold)
    return DataLoader(dataset, batch_size=cfg.data.batch_size, shuffle=(split == "train"))
```

### Sequential CV (all folds in one job)

Add folds to your config:

```yaml
training:
  folds: [0, 1, 2, 3, 4]
```

The Trainer loops through each fold, creating separate checkpoint and output directories (`fold-0/`, `fold-1/`, ...). You can aggregate results with `on_complete`:

```python
def on_complete(cfg, fold_results):
    """Called after all folds finish. Receives list of dicts with:
    fold, checkpoint_dir, output_dir, run_id, best_metric."""
    for r in fold_results:
        log.info(f"Fold {r['fold']}: best {cfg.checkpointing.val_metric}={r['best_metric']:.4f}")
```

### Parallel CV (one fold per SLURM job)

Submit folds as a parameter sweep — faster on a cluster:

```bash
make sweep SWEEP="training.fold=0,1,2,3,4"
```

Add `fold` to your config:

```yaml
training:
  fold: null   # set per-job via CLI
```

The eval pipeline auto-discovers `fold-*` directories and aggregates results.

## Validation and checkpoint frequency

By default, validation and checkpointing happen together every N epochs. You can decouple them:

```yaml
checkpointing:
  save_every_n_epochs: 10         # checkpoint every 10 epochs
  validate_every_n_epochs: 2      # but validate every 2 epochs
```

## Gradient clipping

Enabled by default (`max_grad_norm: 1.0`). To disable:

```yaml
training:
  max_grad_norm: null
```

## Early stopping

```yaml
training:
  early_stopping:
    enabled: true
    patience: 10       # validation checks, not epochs
    min_delta: 0.001
```

Patience counts *validation checks*, not epochs. If you validate every 2 epochs with patience 10, training can continue for up to 20 epochs without improvement.

## Step-based training

For LLM pretraining, RL, or any setting where "train for 100k steps" makes more sense than epochs:

```yaml
training:
  max_steps: 100000         # stop after 100k optimizer steps
  epochs: 100               # ignored when max_steps is set

checkpointing:
  save_every_n_steps: 10000
  validate_every_n_steps: 5000
```

## Weights & Biases

```yaml
logging:
  use_wandb: true
  wandb_project: my_project
```

Then set `WANDB_API_KEY` in your environment or `.env`.

## Adding a new task

Create a new file in `tasks/`:

```bash
cp experiments/tasks/example.py experiments/tasks/segmentation.py
# Edit the new file, then:
make train TRAIN_ARGS="task=segmentation"
```

## Eval callbacks

The Evaluator supports callbacks for custom hooks during evaluation:

```python
def build_eval_callbacks(cfg):
    from experiments.eval import EvalCallback

    class MyCallback(EvalCallback):
        def on_eval_begin(self, evaluator):
            """Before any model/data loading."""
        def on_checkpoint_loaded(self, evaluator, checkpoint_path):
            """After a checkpoint is loaded."""
        def on_run_evaluated(self, evaluator, fold, metrics):
            """After each fold/run is evaluated."""
        def on_eval_end(self, evaluator, results):
            """After all evaluation is complete."""

    return [MyCallback()]
```

## Extra state (teacher-student, GANs, auxiliary models)

For training patterns that need auxiliary models (EMA teacher, discriminator, frozen tokenizer), export `build_extra_state` from your task. The returned modules are checkpointed automatically and passed to your loss function and callbacks:

```python
import copy

def build_extra_state(model, cfg) -> dict[str, torch.nn.Module]:
    """Create auxiliary modules. Called after build_model()."""
    teacher = copy.deepcopy(model)
    teacher.requires_grad_(False)
    return {"teacher": teacher}

def make_compute_loss(cfg, extra_state=None):
    """When build_extra_state is exported, extra_state is passed as second arg."""
    teacher = extra_state["teacher"]

    def compute_loss(model, batch):
        student_out = model(batch["image"])
        with torch.no_grad():
            teacher_out = teacher(batch["image"])
        loss = distillation_loss(student_out, teacher_out)
        return {"loss": loss}
    return compute_loss

def build_callbacks(cfg, extra_state) -> list:
    """Build task-specific callbacks that reference extra_state."""
    return [EMACallback(extra_state["teacher"], momentum=0.996)]
```

The framework handles:
- Moving extra_state modules to the correct device
- Saving/loading extra_state alongside checkpoints
- Passing extra_state to `make_compute_loss` and `build_callbacks`

## Full training loop override

For multi-stage training or anything that doesn't fit the standard loop even with extra_state, export `run(cfg)` from your task:

```python
from experiments.trainer import Trainer

def run(cfg):
    """Bypasses the default training loop entirely."""
    model = build_model(cfg)
    train_loader = build_dataloader(cfg, split="train")

    # You can still use the Trainer as a building block
    trainer = Trainer(model=model, train_loader=train_loader, compute_loss=..., cfg=cfg)
    trainer.train()
```

For GANs, subclass `Trainer` and override `_train_step`:

```python
class GANTrainer(Trainer):
    def _train_step(self, batch):
        # Train discriminator
        d_loss = self.compute_loss(self.model, batch, mode="discriminator")
        self.accelerator.backward(d_loss)
        # ... optimizer steps, generator pass, etc.
        return g_loss, {"d_loss": d_loss.item(), "g_loss": g_loss.item()}
```

## Full eval override

Similarly, export `evaluate(cfg)` from your task to bypass the default eval harness:

```python
from experiments.eval import Evaluator

def evaluate(cfg):
    """Bypasses the default eval harness entirely."""
    evaluator = Evaluator(model=..., test_loader=..., evaluate_fn=..., cfg=cfg)
    result = evaluator.evaluate()
```

## Distributed training (DDP / FSDP)

The Trainer uses HF Accelerate under the hood. Distributed strategy is controlled via config — no code changes needed.

### Config

```yaml
distributed:
  strategy: auto              # auto | ddp | fsdp
  fsdp:
    version: 2
    sharding_strategy: FULL_SHARD  # FULL_SHARD | SHARD_GRAD_OP | NO_SHARD
    cpu_offload: false
    state_dict_type: FULL_STATE_DICT  # FULL_STATE_DICT | SHARDED_STATE_DICT
    reshard_after_forward: true
```

`strategy: auto` (the default) lets Accelerate decide — single-GPU uses no wrapper, multi-GPU uses DDP. Set `strategy: fsdp` explicitly for FSDP.

### Running locally

```bash
make train NUM_GPUS=2                                          # 2-GPU, auto strategy
make train NUM_GPUS=4 TRAIN_ARGS="distributed.strategy=fsdp"  # 4-GPU FSDP
```

### Running on the cluster

The SLURM script auto-detects the number of GPUs and launches with `accelerate` when there are multiple:

```bash
make submit TIER=high GRES=gpu:4                               # 4-GPU job, auto strategy
make submit TIER=high GRES=gpu:4 TRAIN_ARGS="distributed.strategy=fsdp"  # FSDP
```

### FSDP auto-wrap policy

For large models that need custom FSDP wrapping (e.g., wrapping each transformer layer individually), export `build_fsdp_wrap_policy` from your task:

```python
def build_fsdp_wrap_policy(model, cfg):
    """Return an FSDP auto-wrap policy for the model."""
    from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
    from my_package.models import TransformerBlock

    return functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={TransformerBlock},
    )
```

This is optional — most models work fine without a custom wrap policy.

### What the Trainer handles automatically

- Model, optimizer, and dataloaders are wrapped via `accelerator.prepare()`
- Gradient accumulation uses `accelerator.accumulate()` for correct sync
- Validation loss is gathered across processes with `accelerator.gather()`
- Checkpoints use `accelerator.save_state()` / `load_state()` (handles FSDP state dicts)
- Logging, metrics, and checkpoint metadata are written only on the main process
- Distributed sampler gets `set_epoch()` each epoch for proper shuffling

No task code needs to know about distributed training.

## Optional task overrides summary

All of these are optional exports from your task module:

| Function | What it overrides |
|----------|-------------------|
| `build_optimizer(model, cfg)` | Default AdamW |
| `build_scheduler(optimizer, cfg)` | Default cosine + warmup |
| `build_extra_state(model, cfg)` | No auxiliary modules |
| `build_callbacks(cfg, extra_state)` | No task-specific callbacks |
| `build_fsdp_wrap_policy(model, cfg)` | No auto-wrap policy |
| `build_eval_callbacks(cfg)` | No eval callbacks |
| `on_complete(cfg, fold_results)` | No-op after sequential CV |
| `run(cfg)` | Entire training loop |
| `evaluate(cfg)` | Entire eval pipeline |

## Flexibility levels

| Level | What you implement | What you get for free |
|-------|---|---|
| **Standard** | Required functions only | Trainer + Evaluator, optimizer, scheduler, logging |
| **Teacher-student** | + `build_extra_state`, `build_callbacks` | EMA teacher, auxiliary model checkpointing |
| **Cross-validation** | + `fold` param on `build_dataloader` | Fold loop, per-fold checkpoints/outputs, metric aggregation |
| **Customize** | + optional overrides | Control optimizer, scheduler, eval callbacks |
| **Full control** | `run(cfg)` / `evaluate(cfg)` | Use Trainer/Evaluator as building blocks |
