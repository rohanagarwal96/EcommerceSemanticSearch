# Project rules for Claude Code

## No AI attribution — hard requirement

No commit or PR in this repository may ever contain "Co-Authored-By: Claude",
"Generated with Claude Code", or any other AI-attribution trailer. Every
commit must show only the repo owner (rohanagarwal96) as author and
committer. This is enforced by `.claude/settings.json` (`attribution.commit`
and `attribution.pr` set to empty strings), which is committed to the repo
and must never be reverted or overridden. After the first commit in any
session, verify with `git log -1 --format=full` that no Claude/Anthropic
reference appears anywhere in the output. If one is ever found, amend it
immediately before proceeding with anything else.

## Commit and README workflow — required for every phase, no exceptions

After every major addition or change (each numbered build phase, or any
meaningful sub-unit of one):

1. Update `README.md` to reflect the current state of the project — what's
   built, what's not yet, how to run it, current eval/latency numbers once
   they exist.
2. Stage and commit with a clear, conventional commit message (e.g.
   `feat: add BM25 keyword search`, `docs: update README with eval
   results`). No AI attribution anywhere in the commit.
3. Push to `origin main` (or open a PR if branch protection is set up —
   ask before switching away from direct-to-main).

Never batch multiple unrelated phases into a single commit. Small,
reviewable, working commits over large ones.

Never commit `.env`, `kaggle.json`, model weight files, FAISS index files,
or raw downloaded datasets — verify against `.gitignore` before every
commit, and if something large or sensitive is about to be committed, stop
and ask instead of proceeding.
