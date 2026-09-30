"""Run one fitting command with its output in a log file, and record its peak cgroup `total_rss`.

    .venv/bin/python docs/competitor_research_2026-09-21/scripts/p10_run.py --tag NAME -- <command...>

Why a wrapper. The limit that has OOM-killed five builds here is a cgroup limit (13,681 MB) that `free -m` does
not see, and `memory.usage_in_bytes` includes page cache and always reads the limit. The number that means
something is `total_rss` in the cgroup's `memory.stat`, sampled over the run - so this samples it once a second
and writes the peak. The command's stdout/stderr go to a FILE, never a pipe: a piped build once reported exit 0
after being OOM-killed. The exit code recorded is the child's own.

A watchdog terminates the child if `total_rss` passes --kill-at-mb (default 13,300 of the 13,681 MB limit),
so a run that would have been OOM-killed is stopped and recorded as such instead.

Writes <out>/<tag>.log (the command's output) and <out>/<tag>.mem.json (peak, wall clock, exit code).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path


def cgroup_stat_path() -> Path:
    rel = ""
    for line in Path("/proc/self/cgroup").read_text().splitlines():
        parts = line.split(":")
        if len(parts) >= 3 and "memory" in parts[1].split(","):
            rel = parts[2]
    return Path(f"/sys/fs/cgroup/memory{rel}/memory.stat")


def read_stat(p: Path) -> dict[str, int]:
    out = {}
    for line in p.read_text().splitlines():
        k, v = line.split()
        out[k] = int(v)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out", type=Path, default=Path("docs/competitor_research_2026-09-21/results/p10-runs"))
    ap.add_argument(
        "--kill-at-mb",
        type=int,
        default=13_300,
        help="watchdog: terminate the child if cgroup total_rss passes this (the OOM killer fires at 13,681)",
    )
    ap.add_argument("cmd", nargs=argparse.REMAINDER)
    a = ap.parse_args()
    cmd = a.cmd[1:] if a.cmd and a.cmd[0] == "--" else a.cmd
    a.out.mkdir(parents=True, exist_ok=True)
    stat = cgroup_stat_path()
    limit = read_stat(stat).get("hierarchical_memory_limit")
    base = read_stat(stat)["total_rss"]
    t0 = time.time()
    peak, peak_t, samples, killed = base, 0.0, [], False
    with open(a.out / f"{a.tag}.log", "w") as log:
        proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
        while proc.poll() is None:
            rss = read_stat(stat)["total_rss"]
            if rss > peak:
                peak, peak_t = rss, time.time() - t0
            if rss // 2**20 > a.kill_at_mb and not killed:
                proc.terminate()
                killed = True
            if int(time.time() - t0) % 30 == 0:
                samples.append((round(time.time() - t0), rss // 2**20))
            time.sleep(1.0)
    rec = {
        "tag": a.tag,
        "cmd": cmd,
        "exit_code": proc.returncode,
        "killed_by_watchdog": killed,
        "wall_seconds": round(time.time() - t0, 1),
        "peak_total_rss_mb": peak // 2**20,
        "peak_at_seconds": round(peak_t, 1),
        "baseline_total_rss_mb": base // 2**20,
        "cgroup_limit_mb": None if limit is None else limit // 2**20,
        "samples_mb": samples,
    }
    (a.out / f"{a.tag}.mem.json").write_text(json.dumps(rec, indent=1))
    return proc.returncode


if __name__ == "__main__":
    raise SystemExit(main())
