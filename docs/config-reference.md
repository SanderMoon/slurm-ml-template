# Configuration Reference

All parameters live in `configs/config.yaml`. Override per-experiment with
`+experiment=name` or bare `key=value` on the CLI.

```bash
make train                                                   # defaults
make train TRAIN_ARGS="+experiment=my_exp"                   # experiment overlay
make train TRAIN_ARGS="training.lr=0.01 model.dropout=0.5"  # CLI overrides
make train TRAIN_ARGS="--multirun training.lr=0.001,0.01"   # parameter sweep
```

---

## task

| Key | Default | Description |
|-----|---------|-------------|
| `task` | `example` | Which task module to load from `experiments/tasks/`. |

---

## paths

Overridden by SLURM scripts for cluster runs.

| Key | Default | Description |
|-----|---------|-------------|
| `paths.data_dir` | `data` | Raw input data directory. Never written by code. |
| `paths.processed_dir` | `data/processed` | Preprocessed output. Uses config-hash subdirs. |
| `paths.checkpoint_dir` | `checkpoints` | Checkpoint storage (run-hash subdirs). |
| `paths.output_dir` | `outputs` | Metrics, configs, logs (run-hash subdirs). |

---

## resume_from

| Key | Default | Description |
|-----|---------|-------------|
| `resume_from` | `null` | `null` = auto-resume from latest checkpoint. `false` = fresh start. Path = explicit checkpoint. |

---

## model

Task-specific. The framework does not interpret these — they are passed to
`task.build_model(cfg)`. Define all model hyperparameters here.

| Key | Default | Description |
|-----|---------|-------------|
| `model.*` | *(task-defined)* | All model architecture parameters. |

---

## data

Task-specific. Passed to `task.build_dataloader(cfg, split)`.

| Key | Default | Description |
|-----|---------|-------------|
| `data.batch_size` | `32` | Per-device batch size. |
| `data.num_workers` | `4` | DataLoader workers. |
| `data.*` | *(task-defined)* | Additional dataset parameters (paths, transforms, etc.). |

---

## preprocess

Task-specific. Passed to `task.preprocess(cfg)`. The framework computes a
config hash from this section + `paths.data_dir` to create a unique output
directory under `processed_dir`. Multiple preprocessed versions coexist.

| Key | Default | Description |
|-----|---------|-------------|
| `preprocess.*` | *(task-defined)* | Preprocessing parameters (patch size, feature extractor, etc.). |

---

## training

| Key | Default | Description |
|-----|---------|-------------|
| `training.seed` | `42` | Random seed (torch, numpy, random, CUDA). |
| `training.epochs` | `100` | Number of training epochs. Ignored if `max_steps` is set. |
| `training.max_steps` | `null` | Step-based training. `null` = epoch-based. |
| `training.lr` | `0.001` | Learning rate (for default AdamW optimizer). |
| `training.weight_decay` | `0.0001` | Weight decay (for default AdamW optimizer). |
| `training.warmup_epochs` | `0` | Linear warmup epochs before cosine decay. Not in default config — add when needed. |
| `training.max_grad_norm` | `1.0` | Gradient clipping. `null` to disable. |
| `training.grad_accum_steps` | `1` | Gradient accumulation steps. Not in default config — add when needed. |
| `training.folds` | *(unset)* | List of fold indices for sequential cross-validation (e.g., `[0,1,2,3,4]`). |
| `training.fold` | *(unset)* | Single fold index for parallel CV via sweep. |

### training.early_stopping

| Key | Default | Description |
|-----|---------|-------------|
| `training.early_stopping.enabled` | `false` | Enable early stopping. |
| `training.early_stopping.patience` | `10` | Validation checks without improvement before stopping. |
| `training.early_stopping.min_delta` | `0.0` | Minimum change to qualify as improvement. |

---

## checkpointing

| Key | Default | Description |
|-----|---------|-------------|
| `checkpointing.save_every_n_epochs` | `5` | Save checkpoint every N epochs. `0` to disable. |
| `checkpointing.validate_every_n_epochs` | `null` | Run validation every N epochs. `null` = same as `save_every_n_epochs`. |
| `checkpointing.save_every_n_steps` | `null` | Step-based checkpointing (use with `max_steps`). |
| `checkpointing.validate_every_n_steps` | `null` | Step-based validation (use with `max_steps`). |
| `checkpointing.keep_top_k` | `3` | Retain top-K checkpoints by validation metric. |
| `checkpointing.val_metric` | `val_loss` | Metric name to optimize for checkpoint retention. |
| `checkpointing.val_metric_mode` | `min` | `min` for loss, `max` for accuracy/AUROC. |

Checkpoint naming: `epoch{N:03d}_{run_id}/`. Each checkpoint includes
`metadata.json` with epoch, step, metric value, and date. A `best/` symlink
points to the best checkpoint.

---

## eval

| Key | Default | Description |
|-----|---------|-------------|
| `eval.checkpoint` | `best` | Which checkpoint to evaluate: `best`, `latest`, or explicit path. |
| `eval.output_file` | `eval_results.json` | Output filename for evaluation results. |
| `eval.overrides` | `{}` | Config overrides applied at eval time (e.g., larger batch size, TTA). |

---

## distributed

| Key | Default | Description |
|-----|---------|-------------|
| `distributed.strategy` | `auto` | `auto` = Accelerate decides. `ddp` = DistributedDataParallel. `fsdp` = FullyShardedDataParallel. |

### distributed.fsdp

Ignored unless `distributed.strategy: fsdp`.

| Key | Default | Description |
|-----|---------|-------------|
| `distributed.fsdp.version` | `2` | FSDP version (1 or 2). |
| `distributed.fsdp.sharding_strategy` | `FULL_SHARD` | `FULL_SHARD`, `SHARD_GRAD_OP`, or `NO_SHARD`. |
| `distributed.fsdp.cpu_offload` | `false` | Offload parameters to CPU. |
| `distributed.fsdp.state_dict_type` | `FULL_STATE_DICT` | `FULL_STATE_DICT` or `SHARDED_STATE_DICT`. |
| `distributed.fsdp.reshard_after_forward` | `true` | Re-shard parameters after forward pass. |

Multi-GPU launch:
```bash
make train NUM_GPUS=4 TRAIN_ARGS="distributed.strategy=fsdp"
```

---

## logging

| Key | Default | Description |
|-----|---------|-------------|
| `logging.log_every_n_steps` | `10` | Log training metrics every N optimizer steps. |
| `logging.use_wandb` | `true` | Enable Weights & Biases logging. |
| `logging.wandb_project` | *(project slug)* | W&B project name. |

Logging outputs:
- **W&B** — per-step loss/lr/grad_norm, per-epoch validation metrics
- **TensorBoard** — fallback, stored in output directory
- **JSON** — `metrics.jsonl` with one object per epoch
- **Console** — tqdm progress bars

---

## Task interface

Tasks are Python modules in `experiments/tasks/` with a standard interface.
The framework loads the task based on the `task` config key.

### Required functions

| Function | Signature | Called by |
|----------|-----------|----------|
| `preprocess` | `(cfg) -> None` | `make preprocess` |
| `build_dataloader` | `(cfg, split="train", fold=None) -> DataLoader` | train + eval |
| `build_model` | `(cfg) -> nn.Module` | train + eval |
| `make_compute_loss` | `(cfg) -> Callable[[model, batch], Tensor]` | train |
| `make_evaluate` | `(cfg) -> Callable[[model, loader, accelerator], dict]` | train (periodic validation) |
| `make_test_evaluate` | `(cfg) -> Callable[[model, loader, accelerator], dict \| tuple]` | eval |

### Optional overrides

| Function | Signature | Default behavior |
|----------|-----------|-----------------|
| `build_optimizer` | `(model, cfg) -> Optimizer` | AdamW with `cfg.training.lr` and `cfg.training.weight_decay` |
| `build_scheduler` | `(optimizer, cfg) -> LRScheduler` | Linear warmup + cosine decay |
| `build_fsdp_wrap_policy` | `(model, cfg) -> Callable` | No auto-wrap policy |
| `on_complete` | `(cfg, fold_results) -> None` | No-op (sequential CV post-hook) |
| `run` | `(cfg) -> None` | Standard Trainer flow |
| `evaluate` | `(cfg) -> None` | Standard Evaluator flow |

---

## Full config with all options

Copy-pasteable YAML with every framework-recognized key and its default value.
Keys under `model`, `data`, and `preprocess` are task-specific — the examples
below are placeholders.

```yaml
defaults:
  - _self_

task: example

paths:
  data_dir: data
  processed_dir: data/processed
  checkpoint_dir: checkpoints
  output_dir: outputs

resume_from: null                 # null = auto-resume, false = fresh, /path = explicit

# --- Task-specific (examples — replace with your own) ---
model:
  num_classes: 2
  hidden_dim: 256
  dropout: 0.25

data:
  batch_size: 32
  num_workers: 4

preprocess: {}

# --- Framework-interpreted ---
training:
  seed: 42
  epochs: 100
  max_steps: null                 # null = epoch-based; set for step-based
  lr: 0.001
  weight_decay: 0.0001
  warmup_epochs: 0                # linear warmup before cosine decay
  max_grad_norm: 1.0              # null to disable
  grad_accum_steps: 1             # gradient accumulation steps
  # folds: [0, 1, 2, 3, 4]       # sequential cross-validation
  # fold: 0                       # single fold (parallel CV via sweep)

  early_stopping:
    enabled: false
    patience: 10
    min_delta: 0.0

checkpointing:
  save_every_n_epochs: 5          # 0 to disable
  validate_every_n_epochs: null   # null = same as save_every_n_epochs
  save_every_n_steps: null        # step-based (use with max_steps)
  validate_every_n_steps: null    # step-based (use with max_steps)
  keep_top_k: 3
  val_metric: val_loss
  val_metric_mode: min            # min or max

eval:
  checkpoint: best                # best, latest, or /path/to/checkpoint
  output_file: eval_results.json
  overrides: {}                   # config overrides applied at eval time

distributed:
  strategy: auto                  # auto | ddp | fsdp
  fsdp:
    version: 2
    sharding_strategy: FULL_SHARD # FULL_SHARD | SHARD_GRAD_OP | NO_SHARD
    cpu_offload: false
    state_dict_type: FULL_STATE_DICT  # FULL_STATE_DICT | SHARDED_STATE_DICT
    reshard_after_forward: true

logging:
  log_every_n_steps: 10
  use_wandb: true
  wandb_project: my-project

hydra:
  run:
    dir: .
  job:
    chdir: false
  sweep:
    dir: outputs/multirun/${now:%Y-%m-%d}/${now:%H-%M-%S}
    subdir: ${hydra.job.num}
```
