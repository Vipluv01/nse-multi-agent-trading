"""Tests for scripts/generate_paper.py.

**No LaTeX toolchain is available in this environment** (no pdflatex/xelatex),
so real compilation is never checked here -- only structural sanity (balanced
braces and environments, table row/column consistency, no un-escaped special
characters leaking through from data). A human with a TeX installation is
still the one who verifies this actually typesets, per the script's own
docstring.
"""

import re
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nse_agents.config import RESULTS
from scripts.generate_paper import (
    REFERENCES,
    _booktabs_table,
    _escape,
    architecture_table,
    build_document,
    gross_vs_net_table,
)

ROOT = Path(__file__).resolve().parent.parent
PY = str(ROOT / ".venv" / "bin" / "python")


def test_escape_handles_every_documented_special_character():
    assert _escape("Buy&Hold") == r"Buy\&Hold"
    assert _escape("100%") == r"100\%"
    assert _escape("cost_threshold") == r"cost\_threshold"
    assert _escape("$100") == r"\$100"
    assert _escape("#1") == r"\#1"


def test_escape_leaves_ordinary_text_untouched():
    assert _escape("Tech+Regime") == "Tech+Regime"


def test_booktabs_table_has_matching_column_counts_in_every_row():
    tex = _booktabs_table(["A", "B", "C"], [["1", "2", "3"], ["4", "5", "6"]], "cap", "tab:x")
    body_rows = [
        line for line in tex.splitlines()
        if line.endswith(r"\\") and "toprule" not in line and "midrule" not in line
    ]
    for row in body_rows:
        cells = row[: -len(r"\\")].split(" & ")
        assert len(cells) == 3


def test_booktabs_table_has_balanced_environments():
    tex = _booktabs_table(["A"], [["1"]], "cap", "tab:x")
    assert tex.count(r"\begin{tabular}") == tex.count(r"\end{tabular}")
    assert r"\begin{table}" in tex and r"\end{table}" in tex


def test_architecture_table_escapes_and_formats_every_row():
    frame = pd.DataFrame([
        {"architecture": "LSTM-TAL", "accuracy_mean": 0.5024, "accuracy_std": 0.0005,
         "auc_mean": 0.5017, "mcc_mean": 0.0056, "z_vs_chance": 0.67, "p_vs_chance": 0.506},
    ])
    tex = architecture_table(frame)
    assert "LSTM-TAL" in tex
    assert "0.5024" in tex


def test_gross_vs_net_table_computes_cost_drag_as_gross_minus_net():
    frame = pd.DataFrame([
        {"strategy": "Buy&Hold", "Sharpe(gross)": 0.58, "Sharpe(net,excess)": 0.50},
    ])
    tex = gross_vs_net_table(frame)
    assert r"Buy\&Hold" in tex
    assert "+0.080" in tex  # 0.58 - 0.50


def test_references_are_exactly_the_five_papers_and_invent_no_extra_metadata():
    """No DOI, volume, issue, or page-number pattern anywhere -- this project's
    own README does not record that level of detail for these citations, and
    generate_paper.py must not invent it to look more complete."""
    assert len(REFERENCES) == 5
    joined = " ".join(REFERENCES)
    assert not re.search(r"\bdoi\s*:", joined, re.IGNORECASE)
    assert not re.search(r"\bpp\.\s*\d+", joined)
    assert not re.search(r"\bvol\.\s*\d+", joined, re.IGNORECASE)


def test_build_document_is_well_formed_even_with_no_tables_available():
    """A run before any of the underlying result CSVs exist must still
    produce a structurally valid, compilable-shaped document, with the
    missing sections simply omitted -- not a crash."""
    tex = build_document({})
    assert tex.count("{") == tex.count("}")
    assert tex.count(r"\begin{document}") == 1 == tex.count(r"\end{document}")
    assert r"\begin{abstract}" in tex and r"\end{abstract}" in tex


@pytest.mark.skipif(
    not (RESULTS / "technical" / "architecture_summary.csv").exists(),
    reason="requires the main study's cached result tables to be present",
)
def test_generate_paper_runs_end_to_end_and_writes_a_balanced_tex_file():
    result = subprocess.run(
        [PY, str(ROOT / "scripts" / "generate_paper.py")],
        capture_output=True, text=True, timeout=60, cwd=ROOT,
    )
    assert result.returncode == 0, result.stderr

    tex_path = RESULTS / "paper.tex"
    assert tex_path.exists()
    text = tex_path.read_text()

    assert text.count("{") == text.count("}")
    assert text.count(r"\begin{document}") == 1 == text.count(r"\end{document}")

    begins = sorted(re.findall(r"\\begin\{(\w+)\}", text))
    ends = sorted(re.findall(r"\\end\{(\w+)\}", text))
    assert begins == ends

    # Every table row has the number of "&"-separated cells its header declares.
    for table_match in re.finditer(
        r"\\begin\{tabular\}\{(\w+)\}(.*?)\\end\{tabular\}", text, re.DOTALL
    ):
        align, body = table_match.groups()
        expected_cols = len(align)
        for line in body.splitlines():
            line = line.strip()
            if line.endswith(r"\\") and line not in (r"\toprule", r"\midrule", r"\bottomrule"):
                cells = line[: -len(r"\\")].split(" & ")
                assert len(cells) == expected_cols, f"row {line!r} has {len(cells)} cells, expected {expected_cols}"
