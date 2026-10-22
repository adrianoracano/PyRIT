# CLAUDE.md — personal fork of microsoft/PyRIT

This file lives only on the fork (`adrianoracano/PyRIT`); upstream has no `CLAUDE.md`,
so it never conflicts on merge. For PyRIT coding conventions, see
`.github/copilot-instructions.md`.

## Remotes and syncing

- `origin` = fork, `https://github.com/adrianoracano/PyRIT.git`
- `upstream` = `https://github.com/microsoft/PyRIT.git`
  (on a fresh clone: `git remote add upstream https://github.com/microsoft/PyRIT.git`)
- Keep fork `main` in sync with `upstream/main`. A SessionStart hook in
  `.claude/settings.json` fetches upstream and, when on a clean `main`, runs
  `git merge --no-edit upstream/main`. On a conflict, it aborts the merge and prints
  "upstream merge has conflicts". If you see that message, do the merge manually and flag the
  conflicts to the user. The hook never pushes. After a merge that brought new commits, push
  `main` to `origin`.
- Open PRs to upstream from feature branches, never from fork `main`, because fork `main`
  carries fork-only files (`CLAUDE.md`, `.claude/settings.json`, `scripts/run_red_teaming.py`).

## Pushing

- Use the `gh` login (account `adrianoracano`) as the credential helper:
  `git -c credential.helper= -c 'credential.https://github.com.helper=!gh auth git-credential' push origin main`
- The token needs the `workflow` scope, because upstream merges touch `.github/workflows/`.
  If a push is rejected for that reason, ask the user to run
  `gh auth refresh -h github.com -s workflow`.

## Network quirk

The proxy often fails with `CONNECT tunnel failed, response 500`. This is transient: retry
fetches and pushes in a loop (e.g., up to 15 tries with a short sleep). Never treat it as a
reason to stop.

## Environment

- Use the project venv: `.venv/bin/python` (there is no `python` on PATH, only `python3`).

## Fork-only additions

- `scripts/run_red_teaming.py` is a standalone multi-turn `RedTeamingAttack` runner:
  - attacker: llama.cpp at `127.0.0.1:8090`
  - target: llama.cpp at `127.0.0.1:8091`
  - judge: OpenRouter, using `$OPENROUTER_API_KEY`

  It adds a resilient retry layer for 5xx and connection errors, and results persist to the
  default SQLite DB, so runs show up in the CoPyRIT GUI. See the module docstring for the
  `llama-server` launch commands.

- `scripts/run_attack_dataset.py` builds the adversarial-attack dataset (single- and multi-turn attacks
  scored 1-10 by an LLM judge, thresholded at 0.7):
  - target: llama.cpp at `127.0.0.1:8091` (8 slots)
  - attacker and judge: Google Gemini API (OpenAI-compatible endpoint), using `$GEMINI_API_KEY`;
    override with `--attacker` / `--judge`
  - plans (`--plan pilot|trial|stage1|stage2`) are resumable per `run_id`: finished non-error jobs
    are skipped. Output goes to `/home/user/attack_dataset/attacks_{run_id}.jsonl` (outside the repo);
    each row records its attacker, judge and target, so rows from different judges can be told apart.
