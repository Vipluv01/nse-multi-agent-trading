"""Explainability evaluation and the accuracy-explainability trade-off (final evaluation).

"Explainable" is a claim, so it is tested, not asserted. Four measurements over
every logged decision:

1. **Provenance** -- does each rationale name the model that actually produced
   its numbers? (This audit caught the mid-semester logs crediting PLSTM-TAL for
   LSTM-TAL predictions in all 19,110 rows; KNOWN_ISSUES #17.)
2. **Audit-trail integrity** -- does the rationale text state the same action,
   score and per-agent stances as the logged numbers? A rationale that drifts
   from the computation is not an explanation of it.
3. **Reconstruction (decomposability)** -- can the logged score be recomputed
   *exactly* from the logged agent opinions (+ debate residual)? If so the
   explanation *is* the computation, the strongest form of faithfulness.
4. **Deletion faithfulness** (ERASER-style comprehensiveness) -- remove the
   agent the attribution names as most influential and re-decide; the decision
   should flip far more often than when a randomly chosen *other* agent is
   removed. Attribution that fails this is decorative.

Then the trade-off: directional accuracy vs an explicitly defined, ordinal
explainability rubric, for black-box models and each agent configuration.

Writes results/final/explainability/*.csv and figures/accuracy_explainability.png.
Usage: .venv/bin/python scripts/final_explainability.py
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats as sps  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nse_agents.agents.trader import TraderConfig  # noqa: E402
from nse_agents.config import RESULTS  # noqa: E402
from nse_agents.final.config import FINAL  # noqa: E402

OUT = FINAL / "explainability"
OUT.mkdir(parents=True, exist_ok=True)
FIG = FINAL / "figures"
FIG.mkdir(parents=True, exist_ok=True)
AGENTS = ("technical", "sentiment", "regime", "fundamentals")
CFG = TraderConfig()
HEAD = re.compile(r"^(BUY|HOLD|FLAT) (\S+) \(score ([+-]\d+\.\d+), confidence (\d+\.\d+)\)")
DEBATE_NET = re.compile(r"\[debate\].*?net ([+-]\d+\.\d+)")
ARCH = re.compile(r"\[technical [+-]\d+\.\d+\] (\S+) assigns")


def agents_in(frame: pd.DataFrame) -> list[str]:
    return [a for a in AGENTS if f"{a}_stance" in frame]


def recompute(frame: pd.DataFrame, drop: str | None = None) -> np.ndarray:
    """Re-run Trader.combine from logged opinions (optionally deleting one agent)."""
    num = np.zeros(len(frame))
    den = np.zeros(len(frame))
    for agent in agents_in(frame):
        if agent == drop:
            continue
        active = ~frame[f"{agent}_abstained"].astype(bool).to_numpy()
        conf = frame[f"{agent}_conf"].to_numpy(dtype=float) * active
        num += conf * frame[f"{agent}_stance"].to_numpy(dtype=float)
        den += conf
    score = np.where(den > 1e-9, num / np.where(den > 1e-9, den, 1.0), 0.0)
    if "debate_net" in frame:
        has = frame["debate_net"].notna().to_numpy()
        score = np.where(has, (1 - CFG.debate_weight) * score + CFG.debate_weight * frame["debate_net"].fillna(0).to_numpy(), score)
    return np.clip(score, -1.0, 1.0)


def attributions(frame: pd.DataFrame) -> pd.DataFrame:
    """Each agent's share of the (pre-debate) weighted-mean numerator."""
    contrib = {}
    den = np.zeros(len(frame))
    for agent in agents_in(frame):
        active = ~frame[f"{agent}_abstained"].astype(bool).to_numpy()
        conf = frame[f"{agent}_conf"].to_numpy(dtype=float) * active
        contrib[agent] = conf * frame[f"{agent}_stance"].to_numpy(dtype=float)
        den += conf
    out = pd.DataFrame({a: v / np.where(den > 1e-9, den, 1.0) for a, v in contrib.items()})
    return out


def audit(frame: pd.DataFrame, expected_arch: str | None) -> dict:
    heads = frame["rationale"].str.extract(HEAD)
    action_ok = heads[0] == frame["action"].where(frame["action"] != "HOLD", "HOLD")
    # The portfolio cap can demote BUY->HOLD after the rationale header is written;
    # the cap note is then appended, so a BUY header + HOLD action + cap note is consistent.
    demoted = (heads[0] == "BUY") & (frame["action"] == "HOLD") & frame["rationale"].str.contains("over limit|within no-trade", regex=True)
    action_ok = action_ok | demoted
    score_ok = (heads[2].astype(float) - frame["score"]).abs() <= 5e-4
    stance_ok = pd.Series(True, index=frame.index)
    for agent in agents_in(frame):
        pat = re.compile(rf"\[{agent} ([+-]\d+\.\d+)\]")
        stated = frame["rationale"].str.extract(pat)[0].astype(float)
        active = ~frame[f"{agent}_abstained"].astype(bool)
        ok = (stated - frame[f"{agent}_stance"]).abs() <= 5e-3
        stance_ok &= ~active | ok
        abst_ok = ~frame[f"{agent}_abstained"].astype(bool) | frame["rationale"].str.contains(f"[{agent}] abstained", regex=False)
        stance_ok &= abst_ok
    size_ok = ~((frame["action"] == "BUY") & (frame["size"] <= 0))
    tech = frame["rationale"].str.extract(ARCH)[0]
    prov = None
    if expected_arch is not None and tech.notna().any():
        prov = float((tech.dropna() == expected_arch).mean())
    return {
        "n": len(frame),
        "action_consistent": float(action_ok.mean()),
        "score_consistent": float(score_ok.mean()),
        "stances_consistent": float(stance_ok.mean()),
        "buy_has_size": float(size_ok.mean()),
        "provenance_correct": prov,
        "architecture_named": None if tech.dropna().empty else tech.dropna().mode().iloc[0],
        "mean_rationale_words": float(frame["rationale"].str.split().str.len().mean()),
    }


def deletion_test(frame: pd.DataFrame, rng: np.random.Generator) -> dict:
    thr = CFG.buy_threshold
    base_buy = frame["score"].to_numpy() > thr
    attr = attributions(frame).abs()
    names = list(attr.columns)
    top = attr.to_numpy().argmax(axis=1)
    has_signal = attr.to_numpy().max(axis=1) > 0
    deleted = {a: recompute(frame, drop=a) > thr for a in names}
    flip_top = np.array([deleted[names[t]][i] != base_buy[i] for i, t in enumerate(top)])
    flip_rand = []
    for i, t in enumerate(top):
        others = [j for j in range(len(names)) if j != t and attr.iat[i, j] > 0]
        if not others:
            flip_rand.append(np.nan)
            continue
        j = rng.choice(others)
        flip_rand.append(float(deleted[names[j]][i] != base_buy[i]))
    flip_rand = np.array(flip_rand, dtype=float)
    mask = has_signal & ~np.isnan(flip_rand)
    decisive = pd.Series([names[t] for t in top[has_signal]]).value_counts(normalize=True).to_dict()
    table = np.array([[np.sum(flip_top[mask] & (flip_rand[mask] == 1)), np.sum(flip_top[mask] & (flip_rand[mask] == 0))],
                      [np.sum(~flip_top[mask] & (flip_rand[mask] == 1)), np.sum(~flip_top[mask] & (flip_rand[mask] == 0))]])
    b, c = table[0, 1], table[1, 0]
    mcnemar_p = float(sps.binomtest(int(min(b, c)), int(b + c), 0.5).pvalue) if b + c > 0 else np.nan
    return {
        "n_tested": int(mask.sum()),
        "flip_rate_delete_top": float(flip_top[mask].mean()),
        "flip_rate_delete_random_other": float(np.nanmean(flip_rand[mask])),
        "mcnemar_p": mcnemar_p,
        "decisive_agent_share": json.dumps({k: round(v, 3) for k, v in decisive.items()}),
    }


def accuracy(frame: pd.DataFrame) -> dict:
    fwd = frame["fwd_ret"]
    buys = frame["action"] == "BUY"
    ic = sps.spearmanr(frame["score"], fwd, nan_policy="omit")[0]
    return {"buy_precision": float((fwd[buys] > 0).mean()) if buys.any() else np.nan,
            "base_rate_up": float((fwd > 0).mean()),
            "directional_accuracy": float(((frame["score"] > 0) == (fwd > 0))[frame["score"] != 0].mean()),
            "ic_spearman": float(ic)}


# Ordinal explainability rubric. Each criterion is binary and checkable:
RUBRIC = {
    "E1 per-decision natural-language rationale": lambda c: c != "black-box",
    "E2 exact additive attribution (score reconstructable)": lambda c: c != "black-box",
    "E3 explicit abstention ('no evidence' != 'neutral')": lambda c: c != "black-box",
    "E4 risk overrides stated in the audit trail": lambda c: c in ("agents+risk", "agents+risk+debate"),
    "E5 adversarial argument (bull vs bear case)": lambda c: c == "agents+risk+debate",
}


def main() -> int:
    rng = np.random.default_rng(20260909)
    sets = []
    meta = json.loads((FINAL / "core" / "meta.json").read_text()) if (FINAL / "core" / "meta.json").exists() else {}
    for path in sorted((RESULTS / "agents").glob("decisions_*.csv")):
        sets.append(("mid-sem", path.stem.replace("decisions_", ""), path, "LSTM-TAL"))
    for exp in ("core", "extended"):
        for path in sorted((FINAL / exp).glob("decisions_*.csv")):
            sets.append((f"final-{exp}", path.stem.replace("decisions_", ""), path, meta.get("architecture", "LSTM-TAL")))

    audits, deletions, accs = [], [], []
    for run, name, path, arch in sets:
        frame = pd.read_csv(path, parse_dates=["date"])
        frame["debate_net"] = frame["rationale"].str.extract(DEBATE_NET)[0].astype(float)
        recon = np.abs(recompute(frame) - frame["score"].to_numpy())
        # The debate residual is only logged in the rationale, printed to 2 dp,
        # so a debated row can only be reconstructed to within 0.35 * 0.005.
        tol = np.where(frame["debate_net"].notna(), CFG.debate_weight * 0.005 + 1e-6, 1e-6)
        a = {"run": run, "config": name, **audit(frame, arch),
             "reconstruction_max_abs_err": float(recon.max()),
             "reconstruction_within_tolerance": float((recon <= tol).mean())}
        audits.append(a)
        if len(agents_in(frame)) >= 2:
            deletions.append({"run": run, "config": name, **deletion_test(frame, rng)})
        accs.append({"run": run, "config": name, **accuracy(frame)})
        print(f"{run:15s} {name:34s} provenance={a['provenance_correct']}  stances={a['stances_consistent']:.4f} "
              f"recon={a['reconstruction_within_tolerance']:.4f} (max err {a['reconstruction_max_abs_err']:.1e})", flush=True)

    audits, deletions, accs = pd.DataFrame(audits), pd.DataFrame(deletions), pd.DataFrame(accs)
    audits.to_csv(OUT / "audit_integrity.csv", index=False)
    deletions.to_csv(OUT / "deletion_faithfulness.csv", index=False)
    print("\n=== deletion faithfulness ===")
    print(deletions.round(4).to_string(index=False))

    # ---- accuracy vs explainability -------------------------------------
    tech = pd.read_csv(RESULTS / "technical" / "architecture_summary.csv")
    gbm = pd.read_csv(RESULTS / "improvements" / "gbm_variants.csv")
    rows = []
    for r in tech.itertuples():
        rows.append({"model": f"{r.architecture} (black box)", "class": "black-box", "accuracy": r.accuracy_mean})
    g = gbm.iloc[0]
    rows.append({"model": "GBM (black box)", "class": "black-box", "accuracy": g["accuracy"]})
    final_core = accs[accs["run"] == "final-core"].set_index("config")
    for cfg, cls in (("Tech+Regime", "agents+risk"), ("Tech+Regime+Fund", "agents+risk"),
                     ("Tech+Sent+Regime+Fund", "agents+risk"), ("All4+CostAware+Band", "agents+risk"),
                     ("Full+Debate(conviction)", "agents+risk+debate"), ("All4+Debate(conviction)", "agents+risk+debate")):
        if cfg in final_core.index:
            rows.append({"model": cfg, "class": cls, "accuracy": final_core.loc[cfg, "directional_accuracy"]})
    trade = pd.DataFrame(rows)
    for crit, rule in RUBRIC.items():
        trade[crit] = trade["class"].map(rule).astype(int)
    trade["explainability_score"] = trade[list(RUBRIC)].sum(axis=1)
    trade.to_csv(OUT / "accuracy_explainability.csv", index=False)
    accs.to_csv(OUT / "decision_accuracy.csv", index=False)
    print("\n=== accuracy vs explainability ===")
    print(trade[["model", "class", "accuracy", "explainability_score"]].round(4).to_string(index=False))
    rho = sps.spearmanr(trade["explainability_score"], trade["accuracy"])[0]
    print(f"Spearman(explainability, accuracy) = {rho:+.3f} over {len(trade)} models")

    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    colors = {"black-box": "#8a8a8a", "agents+risk": "#2a6f97", "agents+risk+debate": "#c97c3a"}
    jitter = np.linspace(-0.18, 0.18, len(trade))
    for (r, j) in zip(trade.itertuples(), jitter):
        ax.scatter(r.explainability_score + j, r.accuracy, s=46, color=colors[r._2], zorder=3)
        ax.annotate(r.model.replace(" (black box)", ""), (r.explainability_score + j, r.accuracy),
                    fontsize=6.5, xytext=(4, 2), textcoords="offset points")
    ax.axhline(0.5, color="#b33", lw=0.8, ls="--")
    ax.text(-0.3, 0.5, "chance", color="#b33", fontsize=7, va="bottom")
    ax.set_xlabel("Explainability rubric score (0-5, criteria E1-E5)")
    ax.set_ylabel("Out-of-sample directional accuracy")
    ax.set_xlim(-0.5, 5.6)
    ax.set_title("No accuracy is traded away for explainability: every model sits at chance", fontsize=9)
    for cls, col in colors.items():
        ax.scatter([], [], color=col, label=cls)
    ax.legend(fontsize=7, frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(FIG / "accuracy_explainability.png", dpi=180)
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
