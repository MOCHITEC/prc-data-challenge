"""Mutation audit for C1 and C3: break the feature on purpose, check that something fails.

    POLARS_MAX_THREADS=2 .venv/bin/python docs/competitor_research_2026-09-21/scripts/c1_c3_mutation_audit.py

Why this exists. Three silent no-ops have shipped in this repository, and the most recent one was caught only
by an audit of this kind after every hand-written test had passed: the code computed the corrected values and
then handed the model the uncorrected ones. A test suite that passes proves nothing about a defect nobody
wrote a test for, so each mutant below is a defect a reviewer could plausibly have written, and the audit
asks the one question that matters: does the suite notice?

Each mutant is applied as a textual patch to a COPY of the source tree, the suite is run against that copy,
and the mutant is reported as KILLED if at least one test fails. Nothing under src/ is modified; the copy is
made in a temporary directory and deleted afterwards. Read-only with respect to the repository.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
TESTS = ["tests/test_nominal_proxy_and_wind.py", "tests/test_features.py", "tests/test_weather.py"]

# (name, file, what it stands for, before, after)
MUTANTS = [
    (
        "C1-pair-builder-fits-per-frame",
        "src/prc/features.py",
        "the pair builder fitting one table per frame, so the same column means two things",
        "    table = proxy_mode_table([train, rank])\n    return (\n        build_departure_features(train, proxy_mode=table),\n        build_departure_features(rank, proxy_mode=table),\n    )",
        "    return (\n        build_departure_features(train, proxy_mode=proxy_mode_table([train])),  # mutant\n        build_departure_features(rank, proxy_mode=proxy_mode_table([rank])),  # mutant\n    )",
    ),
    (
        "C1-key-airport-only",
        "src/prc/features.py",
        "the modal minute taken per airport, losing the per-runway allowance",
        'PROXY_MODE_KEYS = ("ADEP_mvt", "RUNWAY_mvt")',
        'PROXY_MODE_KEYS = ("ADEP_mvt",)  # mutant',
    ),
    (
        "C1-hand-over-the-raw-minute",
        "src/prc/features.py",
        "the flag computed and the uncorrected minute handed to the model in its place",
        '(pl.col("_pm") == pm).cast(pl.Int8).alias("proxy_at_mode"),',
        'pl.col("_pm").cast(pl.Int8).alias("proxy_at_mode"),  # mutant',
    ),
    (
        "C1-columns-never-joined",
        "src/prc/features.py",
        "the three columns computed and then dropped before the model sees them",
        "            proxy_nominal_features(row, proxy_mode),",
        "            proxy_nominal_features(row, None),  # mutant",
    ),
    (
        "C3-heading-in-degrees",
        "src/prc/weather.py",
        "the runway designator read as degrees rather than tens of degrees",
        "then(n.cast(pl.Float64) * 10.0)",
        "then(n.cast(pl.Float64))  # mutant",
    ),
    (
        "C3-cos-sin-swapped",
        "src/prc/weather.py",
        "headwind and crosswind swapped - the classic sign-and-axis error",
        '            (speed_kt * pl.col("_delta").cos()).cast(pl.Float64).alias("headwind_kt"),\n            (speed_kt * pl.col("_delta").sin().abs()).cast(pl.Float64).alias("crosswind_kt"),',
        '            (speed_kt * pl.col("_delta").sin().abs()).cast(pl.Float64).alias("headwind_kt"),  # mutant\n            (speed_kt * pl.col("_delta").cos()).cast(pl.Float64).alias("crosswind_kt"),  # mutant',
    ),
    (
        "C3-null-bearing-becomes-north",
        "src/prc/weather.py",
        "a missing bearing filled with 0, which is a legal heading and so cannot be told from an observation",
        '    delta = (pl.col(direction_col).cast(pl.Float64) - pl.col("_rwy_heading")).radians().alias("_delta")',
        '    delta = (pl.col(direction_col).cast(pl.Float64).fill_null(0.0) - pl.col("_rwy_heading").fill_null(0.0)).radians().alias("_delta")  # mutant',
    ),
    (
        "C3-raw-bearing-left-in",
        "src/prc/weather.py",
        "the raw bearing left in the frame, where it becomes a model column meaning ten different things",
        '    drop = ["_rwy_heading", "_delta", direction_col, f"{direction_col}_prev1h"]',
        '    drop = ["_rwy_heading", "_delta"]  # mutant',
    ),
    (
        "C3-gust-excess-not-a-difference",
        "src/prc/weather.py",
        "`gust_excess` left equal to `gust_filled`, so the repair says nothing new",
        '[filled.alias("gust_filled"), (filled - pl.col("wind_ms")).alias("gust_excess")]',
        '[filled.alias("gust_filled"), filled.alias("gust_excess")]  # mutant',
    ),
    (
        "C3-flags-default-on",
        "src/prc/pipeline.py",
        "the new weather columns turned on by default, changing every archived recipe",
        '        bool(spec.get("wind_dir", False)),',
        '        bool(spec.get("wind_dir", True)),  # mutant',
    ),
    (
        "C3-one-cache-for-two-recipes",
        "src/prc/pipeline.py",
        "the weather cache path blind to the new flags, so a drct-less frame is served to a run that asked",
        '    name = "metar_hourly" + ("_wxcodes" if wx_codes else "") + ("_winddir" if wind_dir else "")\n    name += "_gust" if gust_repair else ""',
        '    name = "metar_hourly" + ("_wxcodes" if wx_codes else "")  # mutant',
    ),
]


def run(tree: Path) -> tuple[bool, str]:
    r = subprocess.run(
        [str(ROOT / ".venv/bin/python"), "-m", "pytest", "-x", "-q", *TESTS],
        cwd=tree,
        capture_output=True,
        check=False,
        text=True,
        env={"PATH": "/usr/bin:/bin", "POLARS_MAX_THREADS": "2", "PYTHONPATH": str(tree / "src")},
    )
    return r.returncode == 0, (r.stdout + r.stderr).strip().splitlines()[-1]


def main() -> int:
    killed, survived = 0, []
    with tempfile.TemporaryDirectory() as td:
        base = Path(td) / "tree"
        shutil.copytree(ROOT / "src", base / "src")
        shutil.copytree(ROOT / "tests", base / "tests", ignore=shutil.ignore_patterns("__pycache__"))
        shutil.copy(ROOT / "pyproject.toml", base / "pyproject.toml")
        ok, line = run(base)
        print(f"{'unmutated tree':<34} {'PASS' if ok else 'FAIL'}  {line}")
        if not ok:
            print("the unmutated tree does not pass; the audit would be meaningless")
            return 1
        for name, rel, why, before, after in MUTANTS:
            path = base / rel
            src = path.read_text()
            if src.count(before) != 1:
                print(f"{name:<34} SKIPPED - the patch target appears {src.count(before)} times")
                survived.append(name)
                continue
            path.write_text(src.replace(before, after))
            ok, line = run(base)
            path.write_text(src)
            print(f"{name:<34} {'SURVIVED' if ok else 'KILLED  '}  {why}\n{'':<36}{line}")
            if ok:
                survived.append(name)
            else:
                killed += 1
    print(f"\n{killed}/{len(MUTANTS)} mutants killed")
    if survived:
        print("SURVIVED (a defect the suite does not see): " + ", ".join(survived))
    return 0 if not survived else 2


if __name__ == "__main__":
    raise SystemExit(main())
