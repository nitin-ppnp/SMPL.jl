Read all recently changed source files in this SMPL.jl repo, identify any public API changes (new functions, changed signatures, new types, removed exports), and update the corresponding pages in `docs/src/` to keep the documentation in sync.

Steps:
1. Run `git diff --name-only HEAD` and `git status --short` to identify changed files.
2. For each changed `.jl` file, read it and note: new exports, changed function signatures, new struct fields, removed symbols.
3. Use the mapping table in `.claude/agents/docs_updater.md` to determine which `docs/src/` pages need updating.
4. Read the current content of each affected docs page.
5. Apply the minimum necessary edits: update `@docs` blocks in `api.md`, fix code examples, update argument descriptions.
6. Confirm that every symbol in the SMPL module export list appears in `docs/src/api.md`.
