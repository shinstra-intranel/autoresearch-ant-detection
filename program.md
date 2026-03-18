# Program

<!-- ============================================================
     ENTRY POINT — point your agent here to start.
     Prompt: "Read program.md and let's kick off a new experiment."

     This file is IMMUTABLE — the agent must never modify it.
     Enforced via .vscode/settings.json (files.readonlyInclude).
     ============================================================ -->

## Goal

**Metric:** `mAP`
**Direction:** maximize
**Reported by:** `evaluate()` function in `prepare.py`

## Budget

**Budget type:** time
**Budget value:** 300 seconds (5 minutes)
**Enforced by:** `prepare.py`

## Constraints

### What the agent CAN do

- Modify `train.py` — this is the only editable code.
  Everything is fair game: architecture, hyperparameters, optimizer,
  training loop, feature engineering, etc. excluding `# KEEP: ...` lines.

### What the agent CANNOT do

- Modify `prepare.py`. It is read-only. It contains the fixed evaluation
  harness, data loading, and budget enforcement.
- Modify this file (`program.md`).
- Install new packages or add dependencies beyond what's in `pyproject.toml`.
- Modify the evaluation function. `evaluate()` in `prepare.py` is ground truth.
- Circumvent the budget constraint (time, compute, or cost).
- Edit lines in `train.py` marked with `# KEEP: ...` comments.

### Simplicity criterion

All else being equal, simpler is better. A small improvement that adds
ugly complexity is not worth it. Removing something and getting equal or
better results is a great outcome. When evaluating whether to keep a change,
weigh the complexity cost against the improvement magnitude.

## Files

| File | Who edits | Purpose |
|------|-----------|---------|
| `program.md` | Nobody | This file — goal, constraints, process |
| `prepare.py` | Nobody | Fixed harness, eval, data loading |
| `train.py` | **Agent** | The code being optimized |
| `strategy.md` | **Agent** | Current hypotheses, phase, direction |
| `journal.md` | **Agent** | Append-only experiment narrative |
| `results.tsv` | **Agent** | Structured metric log |

Read all files for full context before starting.

## Setup

To set up a new experiment run, work with the human to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `mar15`).
   The branch `autoresearch/<tag>` must not already exist.
2. **Create the branch**: `git checkout -b autoresearch/<tag>`
3. **Read context**: Read all files listed above for full context.
4. **Verify prerequisites**: Check that any required data/models exist.
   If not, tell the human what to run.
5. **Initialize tracking**: Create `results.tsv` with just the header row.
6. **Read strategy.md**: Check if there's prior context from previous runs.
7. **Confirm and go**: Confirm setup looks good.

Once you get confirmation, kick off the experimentation.

## The experiment loop

LOOP FOREVER:

1. **Read state**: Check `results.tsv`, `strategy.md`, and `journal.md`
   to understand where you are.
2. **Form hypothesis**: Based on prior results and your strategy, decide
   what single change to try. Write a one-line hypothesis before coding.
3. **Implement**: Edit `train.py`.
4. **Commit**: `git add -A && git commit -m "experiment: <description>"`
5. **Run**: `uv run train.py`
   (redirect everything — do NOT let output flood your context)
6. **Read results**: Extract the metric from the `run.log` file.
7. **Decide**: Keep or discard based on the rules below.
8. **Log**: Append to `results.tsv` and `journal.md`.
9. **Update strategy**: If this changes your understanding, update `strategy.md`.

### Run command

```bash
uv run train.py
```

### Reading results

```bash
uv run python -c "from pathlib import Path; print(Path('run.log').read_text())"
```

### Keep / discard rules

- **Keep** if the metric improves (even slightly) without excessive complexity.
- **Keep** if the metric is unchanged but the code is simpler.
- **Discard** if the metric gets worse. Revert: `git checkout HEAD~1 -- train.py`
- **Crash** if the run fails. Revert and log with status `crash`.

When discarding, still log the result — failed experiments are information.

### First run

Your very first run should always establish the baseline. Run the code
as-is with no modifications.

## Experiment phases

The agent should recognize and declare transitions between these phases:

1. **Baseline**: First run. No changes. Establish the starting metric.
2. **Coarse search**: Big structural changes (architecture, algorithm choice).
   Expect large variance. Goal: find the right ballpark.
3. **Fine tuning**: Hyperparameter sweeps on the best structure from phase 2.
   Expect diminishing returns. Goal: squeeze out remaining gains.
4. **Ablation**: Remove components one at a time from the current best.
   Goal: simplify without regressing. Changes that can be removed without
   hurting should be removed.

Write the current phase in `strategy.md`. When transitioning, state why.

## Output format

The training script should print a summary block at the end:

```
run_id:           20260318_230213
model_name:       PekingU/rtdetr_v2_r18vd
device:           cuda
val_metric:       0.000004
training_seconds: 15.1
total_seconds:    23.6
peak_vram_mb:     2110.6
num_steps:        1
num_epochs:       0.02
```

## Logging results

When an experiment is done, log it to `results.tsv` (tab-separated).

Header and columns:

```
commit	val_metric	memory_gb num_steps	num_epochs	status	description
```

1. git commit hash (short, 7 chars)
2. metric value (e.g. 1.234567) — use 0.000000 for crashes
3. peak memory in GB, round to .1f — use 0.0 for crashes
4. status: `keep`, `discard`, or `crash`
5. short text description of what this experiment tried

## Journaling

After each experiment, append an entry to `journal.md` (at the top).
Include:
- What you tried and **why** (not just what changed)
- The result (metric value)
- What you learned from it
- What this suggests for next steps

Keep entries concise — 3-5 sentences, not paragraphs.

## Safety

- If a run crashes, log it with status `crash` and metric value `0.0`.
- If the metric is NaN or nonsensical, treat it as a crash.
- Never delete previous results, journal entries, or git history.
