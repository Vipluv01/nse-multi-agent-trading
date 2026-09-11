"""Tests for scripts/build_distribution.py.

The unit tests build synthetic wheel/sdist archives (fast, no real build) to
check that verify_wheel/verify_sdist actually catch a real violation -- not
just that they pass on an already-correct archive, which would be true of a
check that never fires at all. The one end-to-end test performs a real build
via `uv build` / `python -m build` and is skipped if neither is available.
"""

import shutil
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.build_distribution import (
    FORBIDDEN_PATH_FRAGMENTS,
    render_changelog,
    sdist_contents,
    verify_sdist,
    verify_wheel,
    wheel_contents,
)

ROOT = Path(__file__).resolve().parent.parent
PY = str(ROOT / ".venv" / "bin" / "python")


def _make_wheel(tmp_path: Path, files: dict[str, str]) -> Path:
    path = tmp_path / "fake-0.0.0-py3-none-any.whl"
    with zipfile.ZipFile(path, "w") as zf:
        for name, content in files.items():
            zf.writestr(name, content)
    return path


def _make_sdist(tmp_path: Path, names: list[str]) -> Path:
    path = tmp_path / "fake-0.0.0.tar.gz"
    with tarfile.open(path, "w:gz") as tf:
        for name in names:
            info = tarfile.TarInfo(name=f"fake-0.0.0/{name}")
            data = b"x"
            info.size = len(data)
            import io
            tf.addfile(info, io.BytesIO(data))
    return path


def _real_source_modules() -> list[str]:
    return [str(p.relative_to(ROOT)) for p in (ROOT / "nse_agents").rglob("*.py")]


def test_wheel_contents_lists_every_file_in_a_real_zip(tmp_path):
    wheel = _make_wheel(tmp_path, {"a/b.py": "x", "c.py": "y"})
    assert set(wheel_contents(wheel)) == {"a/b.py", "c.py"}


def test_sdist_contents_lists_every_file_in_a_real_tarball(tmp_path):
    sdist = _make_sdist(tmp_path, ["pkg/a.py", "results/paper.tex"])
    names = sdist_contents(sdist)
    assert any(n.endswith("pkg/a.py") for n in names)
    assert any(n.endswith("results/paper.tex") for n in names)


def test_verify_wheel_flags_a_missing_core_subdirectory(tmp_path):
    """A wheel containing only some of the real source modules (agents/ but
    not backtest/, data/, models/, report/) must be flagged, not silently
    accepted -- this is the check's whole point."""
    files = {f"nse_agents/agents/{Path(m).name}": "x" for m in _real_source_modules()
             if "/agents/" in m}
    files["nse_agents/config.py"] = "x"
    wheel = _make_wheel(tmp_path, files)
    problems = verify_wheel(wheel)
    assert any("backtest" in p for p in problems)
    assert any("missing" in p.lower() and "source module" in p.lower() for p in problems)


@pytest.mark.parametrize("fragment", FORBIDDEN_PATH_FRAGMENTS)
def test_verify_wheel_flags_every_forbidden_fragment(tmp_path, fragment):
    files = {f"nse_agents/{m.split('/', 1)[1]}": "x" for m in _real_source_modules()}
    files[f"nse_agents/{fragment}leaked.bin"] = "x"
    wheel = _make_wheel(tmp_path, files)
    problems = verify_wheel(wheel)
    assert any(fragment in p for p in problems)


def test_verify_wheel_on_a_wheel_with_every_real_module_and_core_dir_is_clean(tmp_path):
    files = {m: "x" for m in _real_source_modules()}
    wheel = _make_wheel(tmp_path, files)
    assert verify_wheel(wheel) == []


def test_verify_sdist_flags_a_missing_paper_artifact(tmp_path):
    sdist = _make_sdist(tmp_path, [
        "results/agents/summary.csv", "scripts/run_agents.py",
        # results/paper.pdf deliberately omitted
        "results/paper.tex",
    ])
    problems = verify_sdist(sdist)
    assert any("paper.pdf" in p for p in problems)


def test_verify_sdist_flags_a_missing_result_table(tmp_path):
    sdist = _make_sdist(tmp_path, ["results/paper.tex", "results/paper.pdf", "scripts/run_agents.py"])
    problems = verify_sdist(sdist)
    assert any("summary.csv" in p for p in problems)


def test_verify_sdist_flags_a_missing_scripts_directory(tmp_path):
    sdist = _make_sdist(tmp_path, [
        "results/paper.tex", "results/paper.pdf", "results/agents/summary.csv",
    ])
    problems = verify_sdist(sdist)
    assert any("scripts" in p for p in problems)


@pytest.mark.parametrize("fragment", FORBIDDEN_PATH_FRAGMENTS)
def test_verify_sdist_flags_every_forbidden_fragment(tmp_path, fragment):
    sdist = _make_sdist(tmp_path, [
        "results/paper.tex", "results/paper.pdf", "results/agents/summary.csv",
        "scripts/run_agents.py", f"nse_agents/{fragment}leaked",
    ])
    problems = verify_sdist(sdist)
    assert any(fragment in p for p in problems)


def test_verify_sdist_on_a_correctly_shaped_archive_is_clean(tmp_path):
    sdist = _make_sdist(tmp_path, [
        "results/paper.tex", "results/paper.pdf", "results/agents/summary.csv",
        "scripts/run_agents.py", "nse_agents/config.py",
    ])
    assert verify_sdist(sdist) == []


def test_render_changelog_includes_every_real_commit_subject():
    log = subprocess.run(
        ["git", "log", "--pretty=format:%s"], capture_output=True, text=True, cwd=ROOT, timeout=30,
    )
    subjects = [s for s in log.stdout.splitlines() if s.strip()]
    changelog = render_changelog("0.1.0")
    for subject in subjects:
        assert subject in changelog


def test_render_changelog_strips_the_co_authored_by_trailer():
    changelog = render_changelog("0.1.0")
    assert "Co-Authored-By" not in changelog


def test_render_changelog_includes_the_version_header():
    changelog = render_changelog("0.1.0")
    assert "[0.1.0]" in changelog


@pytest.mark.skipif(
    not (shutil.which("uv") or shutil.which("python") and _real_source_modules()),
    reason="requires a usable Python build frontend",
)
def test_build_distribution_runs_end_to_end_and_writes_a_clean_changelog(tmp_path):
    result = subprocess.run(
        [PY, str(ROOT / "scripts" / "build_distribution.py")],
        capture_output=True, text=True, timeout=180, cwd=ROOT,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "All checks passed." in result.stdout

    changelog_path = ROOT / "results" / "CHANGELOG.md"
    assert changelog_path.exists()
    assert len(changelog_path.read_text()) > 500

    dist_dir = ROOT / "dist"
    assert list(dist_dir.glob("*.whl"))
    assert list(dist_dir.glob("*.tar.gz"))
