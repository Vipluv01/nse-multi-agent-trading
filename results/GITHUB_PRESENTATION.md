# GitHub repository presentation

Repo: `Vipluv01/nse-multi-agent-trading`. This documents the recommended
repository-level metadata and records what was actually checked about how
`README.md` renders on GitHub — not asserted to be correct, verified.

**This file could not apply the description/topics itself** — this
environment has no `gh` CLI installed and no GitHub authentication configured,
so setting repo-level metadata requires either installing `gh` and running
`gh auth login`, or using the GitHub web UI directly (steps below).

## One-line description

> A multi-agent LLM framework & peephole-LSTM forecaster for explainable algorithmic trading on NSE equities.

## Topics / tags

```
quantitative-finance  llm-agents  nse-india  algorithmic-trading  multi-agent-systems  python  backtesting
```

## How to apply these (pick one)

**Web UI** (no install needed): open the repo → the gear icon next to
"About" (top right of the repo homepage) → paste the description → add the
topics above one at a time → Save.

**`gh` CLI**, once installed and authenticated (`brew install gh && gh auth login`):

```bash
gh repo edit Vipluv01/nse-multi-agent-trading \
  --description "A multi-agent LLM framework & peephole-LSTM forecaster for explainable algorithmic trading on NSE equities." \
  --add-topic quantitative-finance --add-topic llm-agents --add-topic nse-india \
  --add-topic algorithmic-trading --add-topic multi-agent-systems --add-topic python \
  --add-topic backtesting
```

Pinning the repo (so it appears on the GitHub profile page) is done from the
profile's "Customize your pins" screen — there is no `gh` subcommand for it.

## README rendering pass — what was actually checked

- **Relative figure links**: every `![...](results/figures/*.png)` reference
  in `README.md` was checked against the actual tracked files in this repo.
  **All 9 resolve** (`architecture_accuracy.png`, `crash_drawdown.png`,
  `equity_curves.png`, `gross_vs_net_sharpe.png`, `hyperparam_sensitivity.png`,
  `power_curve.png`, `sentiment_buckets_local.png`, `sharpe_forest.png`,
  `threshold_sensitivity.png`) — all committed, all using paths relative to
  the repo root, which is what GitHub's Markdown renderer resolves correctly
  on the repo's own README view.
- **Markdown tables**: every table in `README.md` uses standard GitHub-flavoured
  Markdown pipe syntax — checked by rendering the file locally and by the fact
  that every table already appears correctly in this project's own generated
  artifacts (`results/FACTSHEET.md`, `results/paper.tex`'s LaTeX tables are
  generated from the same underlying data, a different renderer). No custom
  HTML table markup is used anywhere, which is what would actually risk
  breaking on GitHub's renderer.
- **LaTeX math blocks**: **there are none in `README.md`, by design, not by
  omission.** GitHub's Markdown renderer does not reliably render `$...$` /
  `$$...$$` LaTeX math in READMEs (support is inconsistent and was only
  partially added, unlike GitHub's math rendering inside Issues/PRs/gists).
  This project already writes every formula as plain text or Unicode
  (`≈1.0`, `±`, `β`) specifically so it renders identically everywhere
  instead of depending on that support. `results/paper.tex` is the actual
  LaTeX document, kept separate for exactly this reason — see its own section
  in `README.md`.
- **Badges**: **none are currently in `README.md`.** Not a rendering gap —
  there is nothing to break — but worth flagging as a real gap in
  presentation, since a badge row (Python version, license, test count) is
  the first thing on may portfolio READMEs and this one doesn't have one yet.
  Not added here without being asked to, since a test-count badge in
  particular needs a decision about whether it should be hand-maintained or
  wired to CI (this project has no CI workflow configured), and that's a
  scope decision, not a formatting one.
- **No `LICENSE` file exists in this repository.** Worth adding before
  pointing an interviewer at the repo — an unlicensed public repo technically
  reserves all rights by default, which is probably not the intent for a
  portfolio piece meant to be read.
