import numpy as np
import pandas as pd
import pytest

from nse_agents.agents.base import Opinion
from nse_agents.agents.orchestrator import Orchestrator, OrchestratorConfig
from nse_agents.agents.researchers import DebateOutcome, format_findings
from nse_agents.agents.risk import RiskLimits, RiskManager, RiskState
from nse_agents.agents.sentiment import SentimentAgent, aggregate_daily
from nse_agents.agents.technical import TechnicalAgent
from nse_agents.agents.trader import Trader, TraderConfig


def test_abstention_is_not_the_same_as_a_neutral_stance():
    """The distinction the whole sentiment design rests on."""
    abstained = Opinion.abstain("sentiment", "no headlines")
    neutral = Opinion("sentiment", 0.0, 0.8, "genuinely mixed news")
    assert abstained.abstained and not neutral.abstained
    assert abstained.confidence == 0.0 and neutral.confidence == 0.8


def test_abstaining_agents_are_excluded_from_the_combination():
    """An abstention must not dilute the agents that do have a view."""
    trader = Trader(TraderConfig(use_debate=False))
    with_abstention = [Opinion("technical", 0.8, 1.0, ""), Opinion.abstain("sentiment", "")]
    alone = [Opinion("technical", 0.8, 1.0, "")]
    assert trader.combine(with_abstention, None)[0] == pytest.approx(
        trader.combine(alone, None)[0]
    )


def test_all_agents_abstaining_yields_no_position():
    trader = Trader()
    score, confidence = trader.combine([Opinion.abstain("a", ""), Opinion.abstain("b", "")], None)
    assert score == 0.0 and confidence == 0.0


def test_combination_is_confidence_weighted():
    """A confident agent must outweigh an unconfident one with the opposite view."""
    trader = Trader(TraderConfig(use_debate=False))
    opinions = [Opinion("technical", 1.0, 0.9, ""), Opinion("regime", -1.0, 0.1, "")]
    score, _ = trader.combine(opinions, None)
    assert score > 0.5


def test_contested_debate_reduces_confidence_but_not_direction():
    """Both cases standing up means act smaller, not act differently."""
    trader = Trader(TraderConfig(use_debate=True))
    opinions = [Opinion("technical", 0.6, 1.0, "")]
    calm = DebateOutcome(0.8, 0.1, "Strong", "None")
    contested = DebateOutcome(0.8, 0.8, "Strong", "Strong")
    assert not calm.contested and contested.contested
    _, calm_conf = trader.combine(opinions, calm)
    _, contested_conf = trader.combine(opinions, contested)
    assert contested_conf < calm_conf


def test_risk_manager_refuses_to_go_long_on_a_negative_score():
    manager = RiskManager()
    size, notes = manager.size(-0.5, 0.9, 0.2, RiskState())
    assert size == 0.0 and "long-only" in notes[0]


def test_risk_manager_flattens_the_book_past_the_drawdown_limit():
    manager = RiskManager(RiskLimits(drawdown_flat_at=-0.30))
    state = RiskState(equity=0.65, equity_peak=1.0)  # -35% drawdown
    size, notes = manager.size(0.9, 0.9, 0.2, state)
    assert size == 0.0
    assert any("halted" in n for n in notes)


def test_vol_scaling_shrinks_positions_in_high_volatility():
    manager = RiskManager(RiskLimits(target_annual_vol=0.20))
    calm, _ = manager.size(0.9, 0.9, 0.10, RiskState())
    wild, _ = manager.size(0.9, 0.9, 0.60, RiskState())
    assert wild < calm


def test_risk_notes_record_every_adjustment():
    """The audit trail is the deliverable; silent sizing changes defeat it."""
    manager = RiskManager()
    _, notes = manager.size(0.9, 0.9, 0.40, RiskState(equity=0.82, equity_peak=1.0))
    assert any("vol scaling" in n for n in notes)
    assert any("drawdown brake" in n for n in notes)


def test_technical_agent_abstains_off_its_out_of_sample_dates():
    oos = pd.DataFrame(
        {"date": pd.to_datetime(["2024-01-02"]), "symbol": ["RELIANCE"],
         "prob_up": [0.62], "attn_recent5": [0.4]}
    )
    agent = TechnicalAgent(oos)
    assert not agent.opine("RELIANCE", pd.Timestamp("2024-01-02")).abstained
    # A date the model has no OOS prediction for must not be guessed at.
    assert agent.opine("RELIANCE", pd.Timestamp("2024-01-03")).abstained
    assert agent.opine("TCS", pd.Timestamp("2024-01-02")).abstained


def test_technical_stance_is_signed_and_confidence_peaks_away_from_a_coin_flip():
    oos = pd.DataFrame(
        {"date": pd.to_datetime(["2024-01-02"] * 3),
         "symbol": ["A", "B", "C"], "prob_up": [0.50, 0.75, 0.25],
         "attn_recent5": [0.4, 0.4, 0.4]}
    )
    agent = TechnicalAgent(oos)
    flat = agent.opine("A", pd.Timestamp("2024-01-02"))
    up = agent.opine("B", pd.Timestamp("2024-01-02"))
    down = agent.opine("C", pd.Timestamp("2024-01-02"))
    assert flat.stance == 0.0 and flat.confidence == 0.0
    assert up.stance > 0 and down.stance < 0
    assert up.confidence == pytest.approx(down.confidence)


def test_sentiment_agent_abstains_when_there_are_no_headlines():
    daily = pd.DataFrame(
        {"symbol": ["RELIANCE"], "date": pd.to_datetime(["2024-01-02"]),
         "sentiment": [0.3], "n_headlines": [4], "dispersion": [0.1], "unknown_share": [0.2]}
    )
    agent = SentimentAgent(daily)
    assert not agent.opine("RELIANCE", pd.Timestamp("2024-01-02")).abstained
    assert agent.opine("RELIANCE", pd.Timestamp("2024-01-03")).abstained


def test_sentiment_confidence_falls_with_disagreement_and_with_unknown_mass():
    def make(dispersion, unknown):
        daily = pd.DataFrame(
            {"symbol": ["X"], "date": pd.to_datetime(["2024-01-02"]), "sentiment": [0.3],
             "n_headlines": [8], "dispersion": [dispersion], "unknown_share": [unknown]}
        )
        return SentimentAgent(daily).opine("X", pd.Timestamp("2024-01-02")).confidence

    assert make(0.8, 0.1) < make(0.1, 0.1)     # headlines disagree
    assert make(0.1, 0.9) < make(0.1, 0.1)     # model says it cannot tell


def test_aggregate_daily_collapses_headlines_per_symbol_day():
    scored = pd.DataFrame(
        {
            "symbol": ["A", "A", "B"],
            "date": ["2024-01-02", "2024-01-02", "2024-01-02"],
            "title": ["h1", "h2", "h3"],
            "p_good": [0.8, 0.2, 0.5],
            "p_bad": [0.1, 0.7, 0.3],
            "p_unknown": [0.1, 0.1, 0.2],
            "score": [0.7, -0.5, 0.2],
        }
    )
    daily = aggregate_daily(scored)
    assert len(daily) == 2
    row = daily.loc[daily["symbol"] == "A"].iloc[0]
    assert row["n_headlines"] == 2
    assert row["sentiment"] == pytest.approx(0.1)


def test_debate_residual_is_bounded_and_signed():
    assert DebateOutcome(1.0, 0.0, "Strong", "None").residual == pytest.approx(1.0)
    assert DebateOutcome(0.0, 1.0, "None", "Strong").residual == pytest.approx(-1.0)
    assert DebateOutcome(0.5, 0.5, "Weak", "Weak").residual == pytest.approx(0.0)


def test_findings_brief_names_every_agent_including_abstainers():
    text = format_findings([Opinion("technical", 0.4, 0.6, "up"), Opinion.abstain("sentiment", "quiet")])
    assert "technical" in text and "sentiment" in text and "no view" in text


class _StubAgent:
    name = "stub"

    def __init__(self, stance):
        self.stance = stance

    def opine(self, symbol, date):
        return Opinion(self.name, self.stance, 1.0, "stub")


def test_orchestrator_never_exceeds_the_gross_exposure_limit():
    """The regression test for the unfunded-leverage bug."""
    dates = pd.bdate_range("2024-01-01", periods=5)
    pairs = pd.DataFrame(
        [{"date": d, "symbol": s, "fwd_ret": 0.001}
         for d in dates for s in list("ABCDEFGHIJ")]
    )
    orchestrator = Orchestrator(
        agents=[_StubAgent(1.0)],
        trader=Trader(TraderConfig(use_debate=False), RiskManager(RiskLimits(max_gross_exposure=1.0))),
        backend=None,
        config=OrchestratorConfig(debate_mode="never", verbose=False),
    )
    frame, _ = orchestrator.run(pairs)
    per_day = frame.groupby("date")["weight"].sum()
    assert (per_day <= 1.0 + 1e-9).all(), f"max gross exposure {per_day.max()}"


def test_orchestrator_decisions_carry_a_rationale():
    dates = pd.bdate_range("2024-01-01", periods=2)
    pairs = pd.DataFrame([{"date": d, "symbol": "A", "fwd_ret": 0.001} for d in dates])
    orchestrator = Orchestrator(
        agents=[_StubAgent(0.9)],
        backend=None,
        config=OrchestratorConfig(debate_mode="never", verbose=False),
    )
    frame, decisions = orchestrator.run(pairs)
    assert len(decisions) == 2
    assert all(d.rationale and "[stub" in d.rationale for d in decisions)
    assert set(frame["action"]).issubset({"BUY", "HOLD", "FLAT"})


def test_random_control_matches_the_reference_selection_count_each_day():
    """The control must hold exposure and turnover fixed, randomising only picks."""
    from nse_agents.backtest.baselines import random_control

    dates = pd.bdate_range("2024-01-01", periods=8)
    symbols = list("ABCDEFGHIJ")
    rng = np.random.default_rng(0)
    reference = pd.DataFrame(
        [{"date": d, "symbol": s, "fwd_ret": 0.001, "prob_up": rng.random()}
         for d in dates for s in symbols]
    )
    control = random_control(reference, seed=1)

    ref_counts = (reference["prob_up"] > 0.5).groupby(reference["date"]).sum()
    ctl_counts = (control["prob_up"] > 0.5).groupby(control["date"]).sum()
    pd.testing.assert_series_equal(ref_counts, ctl_counts, check_names=False)
    # And it must actually differ from the reference's choices somewhere.
    assert not (reference["prob_up"] > 0.5).equals(control["prob_up"] > 0.5)


def test_gross_cap_is_recorded_in_the_decision_rationale():
    """The audit trail must not state a size the decision no longer has."""
    dates = pd.bdate_range("2024-01-01", periods=2)
    pairs = pd.DataFrame(
        [{"date": d, "symbol": s, "fwd_ret": 0.001} for d in dates for s in list("ABCDEFGHIJ")]
    )
    orchestrator = Orchestrator(
        agents=[_StubAgent(1.0)],
        trader=Trader(TraderConfig(use_debate=False), RiskManager(RiskLimits(max_gross_exposure=1.0))),
        backend=None,
        config=OrchestratorConfig(debate_mode="never", verbose=False),
    )
    frame, decisions = orchestrator.run(pairs)
    capped = [d for d in decisions if any("gross exposure" in n for n in d.risk_notes)]
    assert capped, "expected the gross cap to fire with ten names at 20% each"
    for decision in capped:
        assert "gross exposure" in decision.rationale


def test_cached_backend_proxies_and_caches_classify_batch(tmp_path):
    """Regression: CachedBackend exposed only complete(), so the debate crashed."""
    from nse_agents.llm.base import EchoBackend
    from nse_agents.llm.cache import CachedBackend

    inner = EchoBackend()
    backend = CachedBackend(inner, path=tmp_path / "c.sqlite")
    prompts = ["a", "b", "c"]

    first = backend.classify_batch("sys", prompts, ["Good", "Bad", "Unknown"])
    assert len(inner.classify_calls) == 3 and backend.misses == 3

    second = backend.classify_batch("sys", prompts, ["Good", "Bad", "Unknown"])
    assert len(inner.classify_calls) == 3, "second call must not reach the model"
    assert backend.hits == 3
    for a, b in zip(first, second):
        assert a.labels == b.labels
        assert a.probabilities == pytest.approx(b.probabilities)


def test_cached_classify_batch_forwards_only_the_misses(tmp_path):
    """A partially-overlapping corpus must only pay for the new items."""
    from nse_agents.llm.base import EchoBackend
    from nse_agents.llm.cache import CachedBackend

    inner = EchoBackend()
    backend = CachedBackend(inner, path=tmp_path / "c.sqlite")
    labels = ["Good", "Bad", "Unknown"]

    backend.classify_batch("sys", ["a", "b"], labels)
    inner.classify_calls.clear()
    out = backend.classify_batch("sys", ["a", "b", "c"], labels)

    assert inner.classify_calls == ["c"], "only the uncached item should be scored"
    assert len(out) == 3
    # Order must be the caller's, not cache-hits-first.
    fresh = EchoBackend().classify_batch("sys", ["a", "b", "c"], labels)
    for cached, direct in zip(out, fresh):
        assert cached.probabilities == pytest.approx(direct.probabilities)
