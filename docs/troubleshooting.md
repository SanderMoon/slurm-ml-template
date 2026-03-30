# Troubleshooting

## Where are my logs?

SLURM logs go to `logs/slurm-<jobid>.out` and `.err` in the NAS code directory. View them with:

```bash
make logs                    # tail latest job
make logs JOB=12345          # specific job
```

## My job failed — now what?

1. **Check the error**: `make logs` — scroll to the bottom for the stack trace
2. **Check SLURM exit code**: `ssh <cluster> "sacct -j <jobid> --format=JobID,State,ExitCode,MaxRSS"`
3. **Reproduce locally**: run `make train TRAIN_ARGS="training.epochs=2"` with the same experiment config — most bugs reproduce without a GPU
4. **Common failures**:
   - **OOM** (exit code 137, `CANCELLED` with `OUT_OF_MEMORY`): reduce `data.batch_size` or use a higher tier (`TIER=high`)
   - **NAS timeout** (`OSError: [Errno 116]`): re-sync with `make sync` and resubmit
   - **Import error**: the Docker image is missing a dependency — `make build && make push`
   - **Config error** (`omegaconf.errors.ConfigAttributeError`): a config key is missing — check your experiment overlay

## Email notifications

Set `SLURM_MAIL=your.email@yourdomain.com` in `.env` to get notified when jobs finish or fail.

## NAS filesystem restrictions

Some NAS/shared filesystems do **not** support changing file or directory permissions. This affects Python's `shutil` module:

| Function | Works? | Why |
|----------|--------|-----|
| `shutil.copyfile()` | Yes | Copies data only |
| `shutil.copy()` | **No** | Copies data + mode bits |
| `shutil.copy2()` | **No** | Copies data + metadata |
| `shutil.copytree()` | **No** | Calls `copy2` + `copystat` |
| `os.chmod()` | **No** | Operation not permitted |
| Hardlinks (`os.link`, `cp -al`) | Yes | |
| Symlinks (`os.symlink`) | Yes | |

When copying files on NAS, always use `shutil.copyfile()` or hardlinks. The Trainer's checkpoint code handles this correctly.
