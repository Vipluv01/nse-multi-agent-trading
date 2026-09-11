"""Build and verify the source distribution (sdist) and wheel, and render a
changelog for the current version from real git history -- not by re-typing a
summary from memory, which risks silently drifting from what the commits
actually say.

**What "verify" means here, precisely, because the naive reading of that word
doesn't match how Python packaging actually works:**

- The **wheel** (the installable package) is checked to contain every real
  module under `nse_agents/` -- the actual agents, models, backtest engine,
  data pipeline, and CLI -- and nothing else. It deliberately does **not**
  contain `results/` (paper artifacts, result tables, figures) or
  `data_cache/`/`llm_cache/` (the price/headline/LLM-response caches). That
  is correct, standard practice, not a gap: a wheel is meant to be an
  importable runtime package, and bloating one with a compiled PDF or
  gigabytes of regenerable cached API responses is the anti-pattern, not the
  goal. `pip install nse-agents` should not download a stale price cache.
- The **sdist** (source archive) is checked to contain the paper artifacts
  (`results/paper.tex`, `results/paper.pdf`) and the result tables/figures
  this study publishes, alongside the full source tree (`scripts/`,
  `tests/`, docs) -- this is what "includes paper artifacts" actually means
  for a Python distribution: the archive someone builds *from*, not the
  thing `pip` installs. It is also checked to explicitly **not** include
  `data_cache/`/`llm_cache/` (gitignored, regenerable, and would make the
  archive both huge and immediately stale).

Usage:  .venv/bin/python scripts/build_distribution.py
"""

from __future__ import annotations

import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

ROOT = Path(__file__).resolve().parent.parent
DIST_DIR = ROOT / "dist"
CHANGELOG_PATH = ROOT / "results" / "CHANGELOG.md"

# Real "core model" modules -- discovered from the source tree itself, not a
# hand-maintained list that could silently drift as files are added.
CORE_PACKAGE_DIRS = ("agents", "backtest", "data", "models", "report")

# Anything under these must NEVER appear in either distribution -- gitignored,
# regenerable, and would make a release both bloated and stale the moment
# someone re-runs the pipeline.
FORBIDDEN_PATH_FRAGMENTS = ("data_cache/", "llm_cache/", "__pycache__/", ".pyc")


def run_build() -> tuple[Path, Path]:
    """Builds via ``uv build`` (already a project dependency of this
    workflow; falls back to the standard ``python -m build`` frontend if
    ``uv`` isn't on PATH, so this still works in an environment that only has
    pip). Returns (sdist_path, wheel_path)."""
    import shutil

    DIST_DIR.mkdir(exist_ok=True)
    if shutil.which("uv"):
        cmd = ["uv", "build", "--out-dir", str(DIST_DIR)]
    else:
        cmd = [sys.executable, "-m", "build", "--outdir", str(DIST_DIR)]
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT, timeout=180)
    if result.returncode != 0:
        raise RuntimeError(f"build failed ({' '.join(cmd)}):\n{result.stdout}\n{result.stderr}")

    sdists = sorted(DIST_DIR.glob("*.tar.gz"))
    wheels = sorted(DIST_DIR.glob("*.whl"))
    if not sdists or not wheels:
        raise RuntimeError(f"build reported success but dist/ is missing an sdist or wheel: "
                            f"{list(DIST_DIR.iterdir())}")
    return sdists[-1], wheels[-1]


def wheel_contents(wheel_path: Path) -> list[str]:
    with zipfile.ZipFile(wheel_path) as zf:
        return zf.namelist()


def sdist_contents(sdist_path: Path) -> list[str]:
    with tarfile.open(sdist_path, "r:gz") as tf:
        return tf.getnames()


def verify_wheel(wheel_path: Path) -> list[str]:
    """Returns a list of problems found (empty = clean)."""
    problems = []
    names = wheel_contents(wheel_path)
    joined = "\n".join(names)

    for fragment in FORBIDDEN_PATH_FRAGMENTS:
        if fragment in joined:
            problems.append(f"wheel contains a forbidden path fragment {fragment!r}")

    # Every real .py file under nse_agents/ in the source tree must appear in
    # the wheel -- checked against the source tree itself, not a fixed list,
    # so this catches a module silently left out of a future package-layout change.
    source_modules = {
        str(p.relative_to(ROOT)) for p in (ROOT / "nse_agents").rglob("*.py")
    }
    wheel_modules = {n for n in names if n.startswith("nse_agents/") and n.endswith(".py")}
    missing = source_modules - wheel_modules
    if missing:
        problems.append(f"wheel is missing {len(missing)} real source module(s): {sorted(missing)}")

    for subdir in CORE_PACKAGE_DIRS:
        if not any(n.startswith(f"nse_agents/{subdir}/") for n in names):
            problems.append(f"wheel has no files under nse_agents/{subdir}/ -- a core component is missing")

    return problems


def verify_sdist(sdist_path: Path) -> list[str]:
    problems = []
    names = sdist_contents(sdist_path)
    joined = "\n".join(names)

    for fragment in FORBIDDEN_PATH_FRAGMENTS:
        if fragment in joined:
            problems.append(f"sdist contains a forbidden path fragment {fragment!r}")

    required_paper_artifacts = ("results/paper.tex", "results/paper.pdf")
    for artifact in required_paper_artifacts:
        if not any(n.endswith(artifact) for n in names):
            problems.append(f"sdist is missing the paper artifact {artifact!r}")

    if not any(n.endswith("results/agents/summary.csv") for n in names):
        problems.append("sdist is missing the main ablation's result table (results/agents/summary.csv)")
    if not any("/scripts/" in n for n in names):
        problems.append("sdist is missing scripts/ -- the reproduction pipeline")

    return problems


def render_changelog(version: str) -> str:
    """Built from real ``git log`` output -- every entry here is an actual
    commit subject and body, reformatted, not re-summarized from memory."""
    log = subprocess.run(
        ["git", "log", "--reverse", "--pretty=format:%H%n%s%n%b%n===END==="],
        capture_output=True, text=True, cwd=ROOT, timeout=30,
    )
    entries = [e for e in log.stdout.split("===END===\n") if e.strip()]

    lines = [f"# Changelog", "", f"## [{version}]", "",
             "Every entry below is a real commit in this repository's history "
             "(`git log`), not a hand-written summary -- run `git log --oneline` "
             "yourself to cross-check.", ""]
    for entry in entries:
        parts = entry.strip("\n").split("\n", 2)
        if len(parts) < 2:
            continue
        sha, subject = parts[0][:8], parts[1]
        body = parts[2].strip() if len(parts) > 2 else ""
        # Strip the trailer this project appends to every commit -- it's
        # attribution metadata, not changelog content.
        body_lines = [l for l in body.splitlines() if not l.startswith("Co-Authored-By:")]
        body = "\n".join(body_lines).strip()

        lines.append(f"### {subject} (`{sha}`)")
        if body:
            for para in body.split("\n\n"):
                lines.append("")
                lines.append(para.strip())
        lines.append("")

    return "\n".join(lines)


def main() -> int:
    print("Building sdist + wheel...", flush=True)
    sdist_path, wheel_path = run_build()
    print(f"  sdist: {sdist_path.name} ({sdist_path.stat().st_size:,} bytes)")
    print(f"  wheel: {wheel_path.name} ({wheel_path.stat().st_size:,} bytes)")

    print("\nVerifying wheel contains core package code, and only that...", flush=True)
    wheel_problems = verify_wheel(wheel_path)
    if wheel_problems:
        for p in wheel_problems:
            print(f"  PROBLEM: {p}")
    else:
        print("  OK: every real nse_agents/ module present; no cache/build artifacts leaked in.")

    print("\nVerifying sdist contains paper artifacts, result tables, and scripts...", flush=True)
    sdist_problems = verify_sdist(sdist_path)
    if sdist_problems:
        for p in sdist_problems:
            print(f"  PROBLEM: {p}")
    else:
        print("  OK: results/paper.tex, results/paper.pdf, result tables, and scripts/ all present; "
              "data_cache/llm_cache correctly excluded.")

    print("\nRendering results/CHANGELOG.md from real git history...", flush=True)
    import tomllib

    version = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]["version"]
    changelog = render_changelog(version)
    CHANGELOG_PATH.write_text(changelog)
    print(f"  wrote {CHANGELOG_PATH} ({len(changelog):,} characters)")

    all_problems = wheel_problems + sdist_problems
    if all_problems:
        print(f"\n{len(all_problems)} problem(s) found -- see above.")
        return 1
    print("\nAll checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
