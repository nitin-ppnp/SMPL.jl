# Meta-instructions for Claude Code

At the end of every work session on this repo:

- Review `.claude/rules/architecture.md` and update it to reflect any architectural changes.
  Remove stale information. Keep it concise.
- Whenever a public API, function signature, docstring, or module is added or changed,
  update the corresponding page in `docs/src/` to keep the hosted documentation in sync.
  Use `/update-docs` to do this efficiently.
- After writing or modifying any Julia source file, run `/review` to catch style and
  type-stability issues before committing.
