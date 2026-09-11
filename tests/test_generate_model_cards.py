"""Tests for scripts/generate_model_cards.py."""

import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nse_agents.config import RESULTS
from scripts.generate_model_cards import (
    debate_engine_card,
    regime_agent_card,
    sentiment_agent_card,
    technical_agent_card,
)

ROOT = Path(__file__).resolve().parent.parent
PY = str(ROOT / ".venv" / "bin" / "python")


def test_regime_agent_card_runs_with_no_cached_reports_and_still_produces_text():
    """RegimeAgent's card is pure prose (no CSV dependency) -- must never
    raise regardless of what result files happen to exist on disk."""
    card = regime_agent_card()
    assert "RegimeAgent" in card
    normalized = " ".join(card.lower().split())
    assert "not** a fundamental" in normalized or "not a fundamental" in normalized


def test_technical_agent_card_falls_back_gracefully_when_architecture_summary_is_missing(monkeypatch, tmp_path):
    import scripts.generate_model_cards as mod

    monkeypatch.setattr(mod, "RESULTS", tmp_path)
    card = mod.technical_agent_card()
    assert "not available" in card


@pytest.mark.skipif(
    not (RESULTS / "technical" / "architecture_summary.csv").exists(),
    reason="requires the main study's cached architecture ablation",
)
def test_technical_agent_card_includes_every_architecture_row():
    import pandas as pd
    arch = pd.read_csv(RESULTS / "technical" / "architecture_summary.csv")
    card = technical_agent_card()
    for name in arch["architecture"]:
        assert name in card


@pytest.mark.skipif(
    not (RESULTS / "agents" / "decisions_Full+Debate.csv").exists(),
    reason="requires the main study's cached flagship decisions",
)
def test_debate_engine_card_states_the_raw_gate_was_never_replaced_by_the_fix():
    """Regression for a real self-contradiction this script's own build
    caught: the flagship decisions file was produced with the ORIGINAL raw
    disagreement gate (97.4% escalation), not the conviction-gate fix, and
    the card must say so plainly rather than reporting the raw escalation
    rate next to text implying the fix was adopted."""
    card = debate_engine_card()
    assert "raw" in card.lower()
    assert "never" in card.lower() or "deliberately" in card.lower()


@pytest.mark.skipif(
    not (RESULTS / "sentiment" / "scored_local.csv").exists(),
    reason="requires the cached sentiment scoring corpus",
)
def test_sentiment_agent_card_reports_the_bullish_prior():
    card = sentiment_agent_card()
    assert "bullish" in card.lower()
    assert "Good" in card and "Bad" in card


@pytest.mark.skipif(
    not (RESULTS / "agents" / "decisions_Tech-only.csv").exists(),
    reason="requires the main study's cached results",
)
def test_generate_model_cards_runs_end_to_end_and_writes_all_four_cards():
    result = subprocess.run(
        [PY, str(ROOT / "scripts" / "generate_model_cards.py")],
        capture_output=True, text=True, timeout=60, cwd=ROOT,
    )
    assert result.returncode == 0, result.stderr

    out_path = RESULTS / "MODEL_CARDS.md"
    assert out_path.exists()
    text = out_path.read_text()
    for heading in ("## TechnicalAgent", "## SentimentAgent", "## RegimeAgent", "## Debate Engine"):
        assert heading in text
