"""Final-evaluation extensions: no-trade band, earnings classifier, explainability checks,
sentiment predictability helper, price-cache override."""

from __future__ import annotations

import importlib
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from nse_agents.agents.base import Opinion
from nse_agents.agents.orchestrator import Orchestrator, OrchestratorConfig
from nse_agents.agents.risk import RiskLimits, RiskManager
from nse_agents.agents.trader import Trader, TraderConfig
from nse_agents.final.sentiment_eval import is_earnings, predictability

ROOT = Path(__file__).resolve().parent.parent


class _Scripted:
    """An agent whose stance per date is given in advance."""

    name = "scripted"

    def __init__(self, stances: dict):
        self.stances = stances

    def opine(self, symbol, date):
        s = self.stances[pd.Timestamp(date)]
        return Opinion(self.name, s, 1.0, f"scripted {s:+.2f}")


def _run(stances, band, vols=None):
    dates = pd.to_datetime(list(stances))
    vols = vols or [0.20] * len(dates)
    pairs = pd.DataFrame({"date": dates, "symbol": "AAA", "fwd_ret": 0.0})
    volt = pd.DataFrame({"date": dates, "symbol": "AAA", "vol_20": vols})
    orch = Orchestrator([_Scripted(dict(zip(dates, stances.values())))],
                        Trader(TraderConfig(use_debate=False), RiskManager(RiskLimits())),
                        config=OrchestratorConfig(debate_mode="never", rebalance_band=band, verbose=False))
    frame, _ = orch.run(pairs, volt)
    return frame


DAYS = {"2020-01-01": 0.9, "2020-01-02": 0.9, "2020-01-03": 0.9, "2020-01-06": -0.5}
VOLS = [0.20, 0.21, 0.40, 0.40]   # vol scaling: 0.200 -> 0.190 (small) -> 0.100 (large) -> exit


def test_band_zero_is_mid_semester_behaviour():
    sizes = _run(DAYS, 0.0, VOLS)["size"].round(4).tolist()
    assert sizes == [0.2, 0.1905, 0.1, 0.0]


def test_band_holds_small_resizes_but_not_large_ones_or_exits():
    frame = _run(DAYS, 0.05, VOLS)
    sizes = frame["size"].round(4).tolist()
    assert sizes == [0.2, 0.2, 0.1, 0.0]        # held, resized, exited
    assert "no-trade band" in frame["rationale"].iloc[1]
    assert "no-trade band" not in frame["rationale"].iloc[2]


def test_earnings_classifier():
    assert is_earnings("TCS Q2 results: Net profit up 1.4%")
    assert is_earnings("Axis Bank shares nosedive 7% post Q1")
    assert is_earnings("HDFC Bank quarterly earnings beat estimates")
    assert not is_earnings("Sebi penalises 8 entities for disclosure violation")
    assert not is_earnings("Reliance Jio adds 3 million subscribers")


def test_predictability_detects_planted_signal_and_not_noise():
    rng = np.random.default_rng(0)
    n = 4000
    score = rng.normal(size=n)
    noise = pd.DataFrame({"sentiment": score, "fwd_1d": rng.normal(size=n) * 0.01})
    signal = pd.DataFrame({"sentiment": score, "fwd_1d": 0.003 * score + rng.normal(size=n) * 0.01})
    assert predictability(noise)["ic_p"] > 0.01
    s = predictability(signal)
    assert s["ic_spearman"] > 0.2 and s["ic_p"] < 1e-6 and s["spread_pos_minus_neg_bps"] > 0


def test_explainability_reconstruction_and_deletion():
    sys.path.insert(0, str(ROOT / "scripts"))
    sys.argv = ["final_explainability.py"]
    mod = importlib.import_module("final_explainability")
    frame = pd.DataFrame({
        "score": [0.0] * 3, "action": ["BUY", "FLAT", "BUY"], "size": [0.1, 0.0, 0.1],
        "technical_stance": [0.02, -0.04, 0.01], "technical_conf": [0.1, 0.1, 0.1], "technical_abstained": [False] * 3,
        "regime_stance": [0.8, -0.6, 0.3], "regime_conf": [0.8, 0.7, 0.5], "regime_abstained": [False] * 3,
        "fwd_ret": [0.01, -0.01, 0.0],
    })
    frame["score"] = mod.recompute(frame)
    assert np.allclose(mod.recompute(frame), frame["score"])
    attr = mod.attributions(frame)
    assert np.allclose(attr.sum(axis=1), frame["score"])     # additive by construction
    assert (attr["regime"].abs() > attr["technical"].abs()).all()


def test_price_cache_override_is_opt_in(tmp_path):
    code = ("import nse_agents.data.prices as p; print(p.PRICE_CACHE)")
    env = dict(os.environ)
    env.pop("NSE_AGENTS_PRICE_CACHE", None)
    default = subprocess.run([sys.executable, "-c", code], cwd=ROOT, env=env, capture_output=True, text=True)
    assert default.stdout.strip().endswith("data_cache/prices")
    env["NSE_AGENTS_PRICE_CACHE"] = str(tmp_path)
    over = subprocess.run([sys.executable, "-c", code], cwd=ROOT, env=env, capture_output=True, text=True)
    assert over.stdout.strip() == str(tmp_path)


def test_study_price_cache_refuses_refresh(monkeypatch):
    import nse_agents.data.prices as prices

    monkeypatch.delenv("NSE_AGENTS_PRICE_CACHE", raising=False)
    called = []
    monkeypatch.setattr(prices, "_fetch", lambda *a, **k: called.append(a))
    with pytest.raises(prices.FrozenCacheError):
        prices.load_prices("RELIANCE", "2015-01-01", "2100-01-01", refresh=True)
    assert not called            # nothing fetched, nothing overwritten


def test_live_cache_refresh_is_allowed(monkeypatch, tmp_path):
    import nse_agents.data.prices as prices

    monkeypatch.setenv("NSE_AGENTS_PRICE_CACHE", str(tmp_path))
    fake = pd.DataFrame({"date": pd.to_datetime(["2026-11-06"]), "open": [1.0], "high": [1.0],
                         "low": [1.0], "close": [1.0], "volume": [0.0]})
    monkeypatch.setattr(prices, "_fetch", lambda *a, **k: fake)
    out = prices.load_prices("XYZ", "2015-01-01", "2100-01-01", refresh=True)
    assert len(out) == 1 and (tmp_path / "XYZ.csv").exists()
