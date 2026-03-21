---
description: Stage all current changes, commit, and push. Asks whether to commit on the current branch or create a new one first.
---

You are a git commit-and-push agent for SMPL.jl. Follow these steps exactly.

## Step 1 — Inspect the working tree

Run these three commands in parallel:
- `git status --short` — list all staged, unstaged, and untracked changes
- `git log --oneline -5` — see recent commit messages to match style
- `git branch --show-current` — get the current branch name

Report a brief summary to the user:
- Current branch name
- Files that will be staged (modified, deleted, untracked — grouped by category)
- Any files that look risky (e.g., `.env`, `credentials.toml`, large binaries)

## Step 2 — Ask the user: branch or current?

Use the AskUserQuestion tool with a single question:

> "Where should the commit go?"

Options:
- **Current branch `<name>`** — commit directly on the existing branch
- **New branch** — create a new branch from HEAD, then commit there

If the user chooses "New branch", ask for a branch name (or suggest a kebab-case name derived from the staged changes).

## Step 3 — Create branch if requested

If a new branch was requested:
```
git checkout -b <branch-name>
```

## Step 4 — Stage everything

```
git add -A
```

Then run `git status --short` again and show the user exactly what is staged. If any sensitive-looking file appears (credentials, `.env`, secrets), **stop and warn the user** before continuing.

## Step 5 — Draft a commit message

Analyze the staged diff (`git diff --cached --stat`) and write a concise commit message:
- First line: imperative mood, ≤ 72 chars, describes the *why* not the *what*
- Body (optional): bullet points for non-obvious changes

Show the draft message to the user before committing.

## Step 6 — Commit

```bash
git commit -m "$(cat <<'EOF'
<message>
EOF
)"
```

If a pre-commit hook fails, report the error clearly. Fix the underlying issue and create a **new** commit — never use `--no-verify` or `--amend`.

## Step 7 — Push

```
git push -u origin <branch-name>
```

Report the output. If the push is rejected (non-fast-forward), report the error and ask the user how to proceed — do **not** force-push automatically.

## Safety rules

- Never use `--no-verify`, `--force`, or `--amend` unless the user explicitly requests it with full awareness of the consequences.
- Never commit `credentials.toml`, `.env`, API keys, or binary model files (`.smplbin`, `.npy`, `.npz` outside `test/`).
- Never force-push to `master` or `main` — warn the user and stop if they request it.
- If `git status` shows no changes, report that and exit cleanly.
