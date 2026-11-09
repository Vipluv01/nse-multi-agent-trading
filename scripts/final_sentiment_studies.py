"""Final-evaluation sentiment studies.

1. **Earnings-announcement study.** Is LLM sentiment more informative on
   results days than on ordinary news days? (Lopez-Lira & Tang find
   predictability concentrated in information-rich events.) Splits every
   (symbol, session) sentiment view by whether a results/earnings headline
   was published, and measures next-day and 5-day return predictability.
   Also checks the classifier picks genuinely information-rich days: |return|
   should be larger on earnings sessions.
2. **Model-size comparison (1.5B vs 3B).** On the identical headlines, with the
   identical prompt and constrained-logit scoring: label distribution (the
   1.5B's documented 3:1 bullish prior), agreement, and return predictability.

Writes results/final/earnings_study.csv, results/final/llm_comparison.csv,
results/final/llm_comparison_labels.csv and figures in results/final/figures.

Usage: .venv/bin/python scripts/final_sentiment_studies.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats as sps  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nse_agents.config import RESULTS, SETTINGS  # noqa: E402
from nse_agents.data.prices import load_prices  # noqa: E402
from nse_agents.final.config import FINAL  # noqa: E402
from nse_agents.final.sentiment_eval import daily_views, predictability, returns_panel  # noqa: E402

FIG = FINAL / "figures"
FIG.mkdir(parents=True, exist_ok=True)


def holm(pvalues: pd.Series) -> pd.Series:
    p = pvalues.to_numpy(dtype=float)
    order = np.argsort(np.where(np.isnan(p), np.inf, p))
    adjusted = np.full(len(p), np.nan)
    running = 0.0
    m = int(np.sum(~np.isnan(p)))
    for rank, i in enumerate(order):
        if np.isnan(p[i]):
            continue
        running = max(running, min(1.0, (m - rank) * p[i]))
        adjusted[i] = running
    return pd.Series(adjusted, index=pvalues.index)


def earnings_study(daily: pd.DataFrame, model: str) -> pd.DataFrame:
    rows = []
    subsets = {"all news days": daily, "earnings days": daily[daily["earnings"]],
               "non-earnings days": daily[~daily["earnings"]]}
    for subset, frame in subsets.items():
        for score_col in ("sentiment", "sentiment_debiased"):
            for ret_col in ("fwd_1d", "fwd_5d"):
                stats = predictability(frame, score_col, ret_col)
                rows.append({"model": model, "subset": subset, "score": score_col,
                             "horizon": ret_col, **stats})
    return pd.DataFrame(rows)


def attention_check(daily: pd.DataFrame) -> dict:
    e = daily.loc[daily["earnings"], "abs_fwd_1d"].dropna()
    o = daily.loc[~daily["earnings"], "abs_fwd_1d"].dropna()
    t, p = sps.ttest_ind(e, o, equal_var=False)
    return {"mean_abs_ret_earnings_bps": e.mean() * 1e4, "mean_abs_ret_other_bps": o.mean() * 1e4,
            "t": float(t), "p": float(p), "n_earnings": len(e), "n_other": len(o)}


def label_stats(scored: pd.DataFrame, model: str) -> dict:
    argmax = scored[["p_good", "p_bad", "p_unknown"]].to_numpy().argmax(axis=1)
    share = np.bincount(argmax, minlength=3) / len(argmax)
    return {"model": model, "n": len(scored),
            "mean_p_good": scored["p_good"].mean(), "mean_p_bad": scored["p_bad"].mean(),
            "mean_p_unknown": scored["p_unknown"].mean(),
            "share_good": share[0], "share_bad": share[1], "share_unknown": share[2],
            "good_to_bad_ratio": share[0] / max(share[1], 1e-9),
            "mean_signed_score": scored["score"].mean()}


def cohen_kappa(a: np.ndarray, b: np.ndarray, k: int = 3) -> float:
    confusion = np.zeros((k, k))
    for x, y in zip(a, b):
        confusion[x, y] += 1
    n = confusion.sum()
    po = np.trace(confusion) / n
    pe = (confusion.sum(0) * confusion.sum(1)).sum() / n**2
    return float((po - pe) / (1 - pe)) if pe < 1 else 1.0


def main() -> int:
    sessions = load_prices("RELIANCE", SETTINGS.start, SETTINGS.end)["date"]
    rets = returns_panel()

    def prepare(scored: pd.DataFrame) -> pd.DataFrame:
        daily = daily_views(scored, sessions)
        return daily.merge(rets, on=["symbol", "date"], how="left")

    local = pd.read_csv(RESULTS / "sentiment" / "scored_local.csv")
    daily_local = prepare(local)
    print(f"1.5B: {len(local):,} headlines -> {len(daily_local):,} sentiment views; "
          f"{int(daily_local['earnings'].sum()):,} are earnings sessions")

    # ---- 1. earnings study (1.5B, full corpus) ---------------------------
    study = [earnings_study(daily_local, "Qwen2.5-1.5B")]
    attention = {"model": "Qwen2.5-1.5B", **attention_check(daily_local)}
    print("\nattention check (are earnings sessions information-rich?):")
    print({k: round(v, 3) if isinstance(v, float) else v for k, v in attention.items()})

    # ---- 2. model comparison ---------------------------------------------
    path_3b = FINAL / "scored_qwen3b.csv"
    labels, comparison = [label_stats(local, "Qwen2.5-1.5B (full corpus)")], []
    if path_3b.exists():
        big = pd.read_csv(path_3b)
        key = ["symbol", "date", "title"]
        matched = local.merge(big, on=key, suffixes=("_s", "_l"))
        print(f"\n3B: {len(big):,} headlines scored; {len(matched):,} matched to 1.5B rows")
        if len(matched) >= 200:
            small = matched[key + [c for c in matched if c.endswith("_s")]].rename(columns=lambda c: c[:-2] if c.endswith("_s") else c)
            large = matched[key + [c for c in matched if c.endswith("_l")]].rename(columns=lambda c: c[:-2] if c.endswith("_l") else c)
            labels += [label_stats(small, "Qwen2.5-1.5B (matched)"), label_stats(large, "Qwen2.5-3B (matched)")]
            a = small[["p_good", "p_bad", "p_unknown"]].to_numpy().argmax(1)
            b = large[["p_good", "p_bad", "p_unknown"]].to_numpy().argmax(1)
            kappa = cohen_kappa(a, b)
            agree = float((a == b).mean())
            print(f"label agreement {agree:.3f}, Cohen's kappa {kappa:.3f}")
            d_small, d_large = prepare(small), prepare(large)
            study.append(earnings_study(d_large, "Qwen2.5-3B"))
            for name, d in (("Qwen2.5-1.5B", d_small), ("Qwen2.5-3B", d_large)):
                for subset, frame in (("all", d), ("earnings", d[d["earnings"]])):
                    for score_col in ("sentiment", "sentiment_debiased"):
                        comparison.append({"model": name, "subset": subset, "score": score_col,
                                           "n_headlines": len(matched), "label_agreement": agree,
                                           "cohen_kappa": kappa,
                                           **predictability(frame, score_col, "fwd_1d")})
            # Full-coverage daily file so the 3B model can drive the agent ablation.
            if len(big) >= 0.95 * len(local):
                from nse_agents.agents.sentiment import aggregate_daily
                from nse_agents.data.news import align_to_trading_days
                aligned = align_to_trading_days(big.assign(date=pd.to_datetime(big["date"])), sessions)
                aggregate_daily(aligned).to_csv(FINAL / "daily_qwen3b.csv", index=False)
                print("wrote full-coverage daily_qwen3b.csv for the ablation")
    else:
        print("\nno 3B scores yet; model comparison skipped")

    study = pd.concat(study, ignore_index=True)
    # Many cells are tested (2 models x 3 subsets x 2 scores x 2 horizons, plus the
    # matched comparison); Holm across the whole family before any is called real.
    study["ic_p_holm"] = holm(study["ic_p"])
    if comparison:
        comparison = pd.DataFrame(comparison)
        comparison["ic_p_holm"] = holm(comparison["ic_p"])
        comparison = comparison.to_dict("records")
    study.to_csv(FINAL / "earnings_study.csv", index=False)
    pd.DataFrame([attention]).to_csv(FINAL / "earnings_attention.csv", index=False)
    pd.DataFrame(labels).to_csv(FINAL / "llm_comparison_labels.csv", index=False)
    if comparison:
        pd.DataFrame(comparison).to_csv(FINAL / "llm_comparison.csv", index=False)

    show = study[(study["score"] == "sentiment_debiased")]
    cols = ["model", "subset", "horizon", "n", "ic_spearman", "ic_p", "ic_p_holm", "spread_pos_minus_neg_bps", "spread_p", "hit_rate"]
    print("\n=== earnings study (debiased sentiment) ===")
    print(show[cols].round(4).to_string(index=False))
    print("\n=== label distributions ===")
    print(pd.DataFrame(labels).round(3).to_string(index=False))
    if comparison:
        print("\n=== 1.5B vs 3B predictability, matched headlines ===")
        print(pd.DataFrame(comparison)[["model", "subset", "score", "n", "ic_spearman", "ic_p", "ic_p_holm",
                                        "spread_pos_minus_neg_bps", "spread_p", "hit_rate"]].round(4).to_string(index=False))

    # ---- figure: IC by subset/horizon ------------------------------------
    fig, ax = plt.subplots(figsize=(7, 3.6))
    plot = show.copy()
    plot["label"] = plot["model"].str.replace("Qwen2.5-", "") + " · " + plot["subset"]
    for i, h in enumerate(("fwd_1d", "fwd_5d")):
        part = plot[plot["horizon"] == h].reset_index(drop=True)
        y = np.arange(len(part)) + i * 0.38
        ax.barh(y, part["ic_spearman"], height=0.36, label={"fwd_1d": "next day", "fwd_5d": "next 5 days"}[h],
                color=["#2a6f97", "#c97c3a"][i])
        if i == 0:
            ax.set_yticks(y + 0.19)
            ax.set_yticklabels(part["label"], fontsize=8)
    ax.axvline(0, color="#444", lw=0.8)
    ax.set_xlabel("Spearman IC: debiased sentiment vs forward open-to-open return")
    ax.set_title("Is sentiment more informative around earnings?", fontsize=10)
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(FIG / "earnings_ic.png", dpi=180)
    print(f"\nwrote {FINAL}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
