Review all recently changed Julia source files in this repo for SMPL.jl code style compliance, type stability, docstring completeness, GPU compatibility, extension safety, and whether documentation needs updating.

Steps:
1. Run `git diff --name-only HEAD` and `git status --short` to identify changed files.
2. Read each changed `.jl` file.
3. Apply every check defined in `.claude/agents/review.md`.
4. Report findings as a numbered list with `file:line` references and issue category labels (Style / Type Stability / Docstring / GPU / Extension / Tests / Docs).
5. If any docs pages need updating, list them explicitly at the end under "Documentation gaps".
