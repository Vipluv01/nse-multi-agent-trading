"""Sentiment analyst: LLM headline scoring, adapted to Indian equities.

The prompt follows Lopez-Lira & Tang's design -- ask whether the headline is
good or bad *for the stock's price*, not whether it is good news in general,
and give the model an explicit "unknown" escape so it is not forced into a
direction it does not have. Their finding is that this signal predicts returns
for capable models and not for small ones, so the point of running it here on a
1.5B local model is to test that boundary on NSE data rather than assume it.

Labels are chosen so their first tokens differ ("Good"/"Bad"/"Unknown"), which
is what makes the single-forward-pass scoring well posed.

Two anti-lookahead rules are enforced upstream and relied on here:
* headlines are attributed to the first trading day they could be acted on
  (``news.actionable_date``), so an after-close headline informs the next day;
* a day with no headline yields an abstention, never a neutral score.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..data.news import NewsItem, is_noise
from .base import Opinion

LABELS = ["Good", "Bad", "Unknown"]

SYSTEM_PROMPT = (
    "You are a financial analyst covering Indian equities listed on the NSE. "
    "Judge whether a news headline is good or bad for the company's share price "
    "over the next trading day. Answer with exactly one word: Good, Bad, or "
    "Unknown. Answer Unknown if the headline carries no directional information "
    "about the share price."
)

USER_TEMPLATE = (
    "Company: {company} (NSE: {symbol})\n"
    "Headline: {headline}\n\n"
    "Is this headline good or bad for {symbol}'s share price tomorrow? "
    "Answer Good, Bad, or Unknown."
)


@dataclass
class ScoredHeadline:
    symbol: str
    date: str
    title: str
    p_good: float
    p_bad: float
    p_unknown: float

    @property
    def score(self) -> float:
        """Signed sentiment in [-1, 1]. Unknown mass reduces magnitude."""
        return self.p_good - self.p_bad


def score_headlines(
    backend,
    items: list[NewsItem],
    company_names: dict[str, str],
    batch_size: int = 16,
    drop_noise: bool = True,
    progress: bool = True,
) -> pd.DataFrame:
    """Score a whole corpus in batches. Returns one row per headline."""
    if drop_noise:
        items = [i for i in items if not is_noise(i.title)]
    if not items:
        return pd.DataFrame(columns=["symbol", "date", "title", "p_good", "p_bad", "p_unknown", "score"])

    prompts = [
        USER_TEMPLATE.format(
            company=company_names.get(i.symbol, i.symbol), symbol=i.symbol, headline=i.title
        )
        for i in items
    ]

    rows = []
    step = max(batch_size * 20, 200)
    for start in range(0, len(prompts), step):
        chunk_items = items[start : start + step]
        chunk_prompts = prompts[start : start + step]
        scores = backend.classify_batch(
            SYSTEM_PROMPT, chunk_prompts, LABELS, batch_size=batch_size
        )
        for item, label_scores in zip(chunk_items, scores):
            probs = label_scores.as_dict()
            rows.append(
                {
                    "symbol": item.symbol,
                    "date": item.date,
                    "title": item.title,
                    "p_good": probs["Good"],
                    "p_bad": probs["Bad"],
                    "p_unknown": probs["Unknown"],
                    "score": probs["Good"] - probs["Bad"],
                }
            )
        if progress:
            print(f"  scored {min(start + step, len(prompts)):,}/{len(prompts):,}", flush=True)

    frame = pd.DataFrame(rows)
    frame["date"] = pd.to_datetime(frame["date"])
    return frame


def aggregate_daily(scored: pd.DataFrame, min_headlines: int = 1) -> pd.DataFrame:
    """Collapse headline scores to one view per (symbol, trading day)."""
    if scored.empty:
        return pd.DataFrame(columns=["symbol", "date", "sentiment", "n_headlines", "dispersion", "unknown_share"])
    grouped = (
        scored.groupby(["symbol", "date"])
        .agg(
            sentiment=("score", "mean"),
            n_headlines=("score", "size"),
            dispersion=("score", "std"),
            unknown_share=("p_unknown", "mean"),
        )
        .reset_index()
    )
    grouped["dispersion"] = grouped["dispersion"].fillna(0.0)
    return grouped[grouped["n_headlines"] >= min_headlines].reset_index(drop=True)


def debias_daily(
    daily: pd.DataFrame, method: str = "none", trailing_window: int = 60
) -> pd.DataFrame:
    """Remove the model's own prior from the aggregated sentiment score.

    The 1.5B backend scores P(Good)=0.458 against P(Bad)=0.154 across the whole
    corpus -- a near-constant bullish offset that has nothing to do with any
    particular headline. Left in, it makes the agent's stance track *coverage
    volume* rather than tone, which is what drove the 123.9x turnover in the
    Tech+Sentiment ablation.

    Both corrections are causal:

    * ``cross_sectional`` subtracts the mean sentiment across the symbols
      covered on that same day. Every one of those headlines is already
      published by the decision point, so this uses no future information; it
      converts an absolute score into "positive *relative to* today's news".
    * ``trailing`` subtracts a per-symbol rolling mean that is shifted by one
      day, so a symbol's own baseline coverage tone is estimated only from days
      strictly before the decision.
    """
    if method == "none":
        return daily

    frame = daily.sort_values(["symbol", "date"]).copy()

    if method in ("trailing", "both"):
        baseline = (
            frame.groupby("symbol")["sentiment"]
            .transform(lambda s: s.rolling(trailing_window, min_periods=10).mean().shift(1))
        )
        # Before a baseline exists, fall back to the global prior rather than
        # to zero -- zero would assert neutrality the model never expressed.
        frame["sentiment"] = frame["sentiment"] - baseline.fillna(frame["sentiment"].expanding().mean().shift(1)).fillna(0.0)

    if method in ("cross_sectional", "both"):
        frame["sentiment"] = frame["sentiment"] - frame.groupby("date")["sentiment"].transform("mean")

    return frame.reset_index(drop=True)


class SentimentAgent:
    name = "sentiment"

    def __init__(self, daily: pd.DataFrame, saturation: float = 0.35):
        frame = daily.copy()
        frame["date"] = pd.to_datetime(frame["date"])
        self._lookup = frame.set_index(["symbol", "date"]).sort_index()
        # Raw mean sentiment rarely exceeds ~0.35 in magnitude once the corpus
        # is averaged, so a linear map to [-1, 1] would leave the agent
        # permanently timid relative to the technical agent. Saturating here
        # puts the two on a comparable scale.
        self.saturation = saturation

    def opine(self, symbol: str, date: pd.Timestamp) -> Opinion:
        try:
            row = self._lookup.loc[(symbol, pd.Timestamp(date))]
        except KeyError:
            return Opinion.abstain(self.name, "no headlines attributable to this trading day")
        if isinstance(row, pd.DataFrame):
            row = row.iloc[0]

        raw = float(row["sentiment"])
        n = int(row["n_headlines"])
        stance = float(np.clip(raw / self.saturation, -1.0, 1.0))

        # Confidence grows with headline count and falls with disagreement
        # between headlines and with the model's own "unknown" mass.
        count_term = min(n / 5.0, 1.0)
        agreement = 1.0 - min(float(row["dispersion"]), 1.0)
        informative = 1.0 - float(row["unknown_share"])
        confidence = float(np.clip(count_term * agreement * informative, 0.0, 1.0))

        tone = "positive" if raw > 0.05 else "negative" if raw < -0.05 else "mixed"
        return Opinion(
            agent=self.name,
            stance=stance,
            confidence=confidence,
            rationale=(
                f"{n} headline(s) for {symbol}; mean tone {raw:+.3f} ({tone}), "
                f"dispersion {float(row['dispersion']):.2f}, "
                f"{float(row['unknown_share']):.0%} judged non-directional."
            ),
            evidence={
                "n_headlines": n,
                "mean_score": raw,
                "dispersion": float(row["dispersion"]),
                "unknown_share": float(row["unknown_share"]),
            },
        )
