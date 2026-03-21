Stage all current changes, commit, and push to GitHub. Follow these steps:

1. Run `git status --short`, `git log --oneline -5`, and `git branch --show-current` in parallel. Show the user: current branch, list of files that will be staged, and any risky files.

2. Ask the user (using AskUserQuestion) whether to commit on the current branch or create a new branch. If they choose a new branch, ask for a name.

3. If a new branch was requested, run `git checkout -b <branch-name>`.

4. Run `git add -A`, then `git status --short` again. If any sensitive file appears (credentials, `.env`, API keys, binary model files outside `test/`), stop and warn before continuing.

5. Run `git diff --cached --stat` to understand the changes, then draft a concise commit message (imperative mood, ≤ 72 chars). Show it to the user before committing.

6. Commit using the drafted message.

7. Push with `git push -u origin <branch>` and report the result. Do not force-push under any circumstances without explicit user confirmation.
