# CLAUDE.md
This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is
DocIQ: receipt image -> validated JSON, by fine-tuning Donut (OCR-free) on CORD v2 and
benchmarking it against a zero-shot Qwen2.5-VL baseline. Full plan:
`DocIQ — End-to-End Build Guide.md`. Code follows the repo layout in that guide (`src/dociq/`).

Remote: https://github.com/Trilokuday3/DocIQ.git

## Commands
```powershell
python -m pip install -e .[dev]        # + .[infer,data,serve,nb] as needed
python -m pytest -q                 # fast tests (slow ones are deselected)
python -m pytest -m slow            # downloads CORD, round-trips all 800 train labels
python -m ruff check .
```
Status: all code phases written and smoke-tested on CPU; real training/eval runs remain (explain/06).
Local machine has no usable GPU (MX350, 2 GB) and Python 3.14: training/baseline runs happen on
Kaggle/Colab. Keep local code light; heavy deps live in pyproject extras.

## Explain folder
`explain/` holds beginner-friendly `.md` walkthroughs of every phase (what, why, how it works,
how to run). It is git-ignored. Whenever a phase or module is added, write/update its explain
file in the same turn. Number files `NN-topic.md`; never date-prefix filenames.

## Git Commit Rule
Whenever a change warrants a commit, provide it in this exact shape:
1. A single copy-pasteable shell block containing a `git add` line then a `git commit -m`
   line, in that order, in one code block:
   ```
   git add path/to/file-a path/to/file-b
   git commit -m "type(scope): concise why-focused summary"
   ```
2. `git add` lists exact files by path (no `git add -A` / `git add .`).
3. One `-m` line explaining *why*, Conventional Commits style (`feat:`, `fix:`, `chore:`, `refactor:`, `docs:`, `test:`).
4. Never append `Co-Authored-By: Claude` or any Claude/Anthropic attribution.
5. Only *give* the block as text; never run `git add` or `git commit` on the user's behalf.
If a change is too trivial for a commit message, say so instead.
Note: the git repo root is `C:\Users\trilo` (home dir), so paths in `git add` are relative to it,
e.g. `Downloads/DocIQ/src/dociq/data/cord.py`.
