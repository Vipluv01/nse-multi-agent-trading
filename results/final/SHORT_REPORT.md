# MTP final evaluation: four-line summary

**A Multi-Agent LLM Framework for Explainable Algorithmic Trading on NSE-Listed Equities** — Vipluv (22MF3IM22), supervisor Prof. S. P. Sarmah.

1. **Built:** a working multi-agent trading system for NSE equities (LSTM-TAL technical agent, Qwen2.5 news-sentiment agent, regime agent, point-in-time fundamentals agent, bull/bear LLM debate, deterministic risk manager), evaluated by purged walk-forward on 19,110 out-of-sample decisions for 10 stocks (Dec 2018 – Aug 2026) and 58,620 for 30 stocks (to Nov 2026), under the full Indian transaction-cost model.
2. **Result (negative, and reported as such):** no configuration beats Buy-and-Hold after costs; the best agent configuration has a net Sharpe of 0.35 against 0.59 (10 stocks) and 0.36 against 0.74 (30 stocks), where four configurations are significantly worse after Holm correction.
3. **What the system does deliver:** roughly half the drawdown (−20.7% against −38.6%), a no-trade band that cuts turnover from 24× to 16× a year, and a conviction gate that cuts debate calls from 97.4% of decisions to 10.5% with no change in outcome.
4. **Explainability:** every decision carries a rationale whose score can be reconstructed exactly; deleting the decisive agent flips 37–71% of decisions against 0–3% for a random agent, and the audit caught a real provenance bug in the mid-sem rationales.
