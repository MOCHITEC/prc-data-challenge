"""ADS-B ingest of the labelled training days of January and July 2025, for an ADS-B model trained on labelled rows.

A thin driver around `adsb_ingest_all.py` (per-day fetch -> extract -> purge -> analyse, resumable state in
`$SCRATCH/ingest/state/<D>.json`, one raw day on disk at a time). Nothing in the existing scripts is changed; the
filters are exactly those of `adsb_stage1_feasibility.py` (fixes within 8 km of the ARP of the 10 challenge airports,
on the ground or at <= 4,000 ft; `analyse` joins them to our departures with MVT_TIME in [D 02:00, D+1 00:00) UTC).

What this wrapper adds, and only this:

  * the 2025 archive: `adsb_ingest_all.fetch_day` names the `globe_history_2026` release repository; for a v2025.*
    tag the repository is switched to `globe_history_2025` (by wrapping `s1.sizes` / `s1.part_url`).
  * unsplit days: five January 2025 releases (01-01, 01-11, 01-18, 01-19, 01-21) hold one `<tag>.tar` (< 2 GB)
    instead of `<tag>.tar.aa, .ab`. It is downloaded to `<tag>.tar.aa` so the stage-1 extractor, which streams the
    local `<tag>.tar.*` parts back to back as one tar, reads it unchanged.
  * memory gate: before each day it waits until the cgroup's `total_rss` is below 9,000 MB (a training job shares it).
  * a per-day status line (bytes streamed, fixes kept, elapsed, peak RSS of this process tree) appended to
    `$SCRATCH/ingest/status_2025.jsonl`.

    POLARS_MAX_THREADS=2 POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR=load_as_storage \
    .venv/bin/python scripts/adsb_ingest_2025.py [run | status | table] [--days D,D,...]

Data: adsb.lol, Open Database License (ODbL) 1.0. "Contains data from adsb.lol, made available under the ODbL 1.0."
"""

from __future__ import annotations

import datetime as dt
import fcntl
import json
import os
import re
import subprocess
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import adsb_ingest_all as ia

s1 = ia.s1
REPO25 = "globe_history_2025"
STATUS = os.path.join(ia.ING, "status_2025.jsonl")
RSS_MAX_MB = 9000
_sizes0, _url0 = s1.sizes, s1.part_url
_single: dict[str, bool] = {}


def _head(url: str) -> tuple[str, int | None]:
    r = subprocess.run(["curl", "-sS", "-m", "60", "-o", "/dev/null", "-I", "-L", "-w",
                        "%{http_code} %header{content-length}", url], capture_output=True, text=True, check=False)
    code, *cl = r.stdout.split() or ["000"]
    return code, (int(cl[-1]) if cl else None)


def _repo(repo: str, tag: str) -> str:
    return REPO25 if tag.startswith("v2025.") else repo


def _is_single(repo: str, tag: str) -> bool:
    if tag not in _single:
        _single[tag] = (_head(_url0(repo, tag, "aa"))[0] != "200"
                        and _head(f"{s1.BASE}/{repo}/releases/download/{tag}/{tag}.tar")[0] == "200")
    return _single[tag]


def sizes(repo: str, tag: str) -> list[tuple[str, int]]:
    repo = _repo(repo, tag)
    if _is_single(repo, tag):
        code, n = _head(f"{s1.BASE}/{repo}/releases/download/{tag}/{tag}.tar")
        return [(f"{tag}.tar.aa", n)] if code == "200" and n else []
    return _sizes0(repo, tag)


def part_url(repo: str, tag: str, sfx: str) -> str:
    repo = _repo(repo, tag)
    if _is_single(repo, tag):
        assert sfx == "aa", (tag, sfx)
        return f"{s1.BASE}/{repo}/releases/download/{tag}/{tag}.tar"
    return _url0(repo, tag, sfx)


s1.sizes, s1.part_url = sizes, part_url


def all_days() -> list[str]:
    out = []
    for m in (1, 7):
        d = dt.date(2025, m, 1)
        while d.month == m:
            out.append(d.isoformat())
            d += dt.timedelta(days=1)
    return out


def cgroup_rss_mb() -> float:
    with open("/proc/self/cgroup") as f:
        path = next(ln.split(":", 2)[2].strip() for ln in f if ln.split(":")[1] == "memory")
    with open(f"/sys/fs/cgroup/memory{path}/memory.stat") as f:
        for ln in f:
            k, v = ln.split()
            if k == "total_rss":
                return int(v) / 1e6
    return 0.0


def wait_memory(day: str) -> float:
    t0, last = time.time(), 0.0
    while (rss := cgroup_rss_mb()) > RSS_MAX_MB:
        if time.time() - last >= 600:
            ia.log(f"{day}: cgroup total_rss {rss:.0f} MB > {RSS_MAX_MB} MB; waiting "
                   f"({(time.time() - t0) / 60:.0f} min so far)")
            last = time.time()
        time.sleep(30)
    return rss


def tree_rss_mb(root: int) -> float:
    kids, rss = {}, {}
    for p in os.listdir("/proc"):
        if not p.isdigit():
            continue
        try:
            with open(f"/proc/{p}/stat") as f:
                st = f.read().rsplit(")", 1)[1].split()
            kids.setdefault(int(st[1]), []).append(int(p))
            rss[int(p)] = int(st[21]) * os.sysconf("SC_PAGE_SIZE")
        except (OSError, IndexError, ValueError):
            continue
    tot, todo = 0, [root]
    while todo:
        p = todo.pop()
        tot += rss.get(p, 0)
        todo += kids.get(p, [])
    return tot / 1e6


class Peak(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self.peak, self.stop = 0.0, threading.Event()

    def run(self):
        while not self.stop.wait(3):
            self.peak = max(self.peak, tree_rss_mb(os.getpid()))


def record(day: str, result: str, t0: float, rss0: float, peak: float) -> None:
    st = ia.load_state(day)
    summ = st.get("extract_summary", "")
    xlog = os.path.join(ia.SCR, f"extract_{day}.log")
    if not summ and os.path.exists(xlog):  # fixes reused from the validation run: its extract log holds the counts
        with open(xlog) as f:
            summ = f.read().strip().splitlines()[-1]
    m = re.search(r"(\d+) trace files, (\d+) near an airport, (\d+) fixes", summ)
    rec = {"day": day, "result": result, "started": f"{dt.datetime.fromtimestamp(t0, dt.UTC):%Y-%m-%dT%H:%M:%SZ}",
           "elapsed_s": round(time.time() - t0, 1), "cgroup_rss_mb_at_start": round(rss0),
           "peak_tree_rss_mb": round(peak), "parts": len(st.get("parts", [])),
           "bytes_streamed": sum(n for _, n in st.get("parts", [])),
           "trace_files": int(m[1]) if m else None, "traces_near_airport": int(m[2]) if m else None,
           "fixes_kept": int(m[3]) if m else None, "fixes_mb": st.get("fixes_mb"), "joined_mb": st.get("joined_mb"),
           "reused_fixes": bool(st.get("reused_fixes")), "single_tar": _single.get(st["tag"]),
           "error": st.get("fetch_error") or st.get("step_error")}
    joined = os.path.join(ia.SCR, f"{st['tag']}_joined.parquet")
    if result == "done" and os.path.exists(joined):
        import pyarrow.parquet as pq
        rec["departures_joined_rows"] = pq.ParquetFile(joined).metadata.num_rows
    with open(STATUS, "a") as f:
        f.write(json.dumps(rec) + "\n")
    ia.log(f"{day} status: {json.dumps(rec)}")


def safe(day: str, counter: list[int]) -> str:
    """adsb_ingest_all.main's per-day handler, plus the memory gate and the status record."""
    rss0 = cgroup_rss_mb() if ia.load_state(day).get("analysed") else wait_memory(day)
    t0, pk = time.time(), Peak()
    pk.start()
    try:
        res = ia.do_day(day, None, counter)
    except RuntimeError as e:
        st = ia.load_state(day)
        st["step_error"] = str(e)
        ia.save_state(st)
        ia.log(f"{day}: step failed: {e}; purging raw parts")
        s1.purge(st["tag"])
        st["fetched"] = False
        ia.save_state(st)
        res = "failed"
    finally:
        pk.stop.set()
    if res != "done" or not ia.load_state(day).get("_2025_recorded"):
        record(day, res, t0, rss0, pk.peak)
        if res == "done":
            st = ia.load_state(day)
            st["_2025_recorded"] = True
            ia.save_state(st)
    return res


def table(days: list[str]) -> None:
    last = {}
    if os.path.exists(STATUS):
        with open(STATUS) as f:
            for ln in f:
                r = json.loads(ln)
                last[r["day"]] = r
    print("| day | result | parts | GB streamed | fixes kept | fixes MB | departures joined | elapsed s | peak RSS MB |")
    print("|---|---|---|---|---|---|---|---|---|")
    for d in days:
        r = last.get(d)
        if r is None:
            print(f"| {d} | not run | | | | | | | |")
            continue
        gb = "reused" if r["reused_fixes"] else f"{r['bytes_streamed'] / 1e9:.3f}"
        tag = " (1 tar)" if r.get("single_tar") else ""
        print(f"| {d} | {r['result']} | {r['parts']}{tag} | {gb} | {r['fixes_kept'] or ''} | {r['fixes_mb'] or ''} | "
              f"{r.get('departures_joined_rows', '')} | {r['elapsed_s']:.0f} | {r['peak_tree_rss_mb']} |")


def main() -> None:
    a = sys.argv[1:]
    cmd = a[0] if a and not a[0].startswith("--") else "run"
    days = a[a.index("--days") + 1].split(",") if "--days" in a else all_days()
    os.makedirs(ia.STATE, exist_ok=True)
    if cmd == "status":
        ia.status(days)
        return
    if cmd == "table":
        table(days)
        return
    lock = open(os.path.join(ia.ING, "lock"), "w")  # noqa: SIM115
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        sys.exit("another ingest is running (lock held)")
    ia.log(f"=== 2025 run start: {len(days)} days; free {ia.free_b() / 1e9:.2f} GB")
    with open(os.path.join(ia.ING, "pid"), "w") as f:
        f.write(str(os.getpid()))
    counter, failed = [0], []
    for d in days:
        if safe(d, counter) == "failed":
            failed.append(d)
    if failed:
        ia.log(f"2025 second pass over failed days: {failed}")
        failed = [d for d in failed if safe(d, counter) == "failed"]
    done = [d for d in days if ia.load_state(d).get("analysed")]
    ia.log(f"=== 2025 run end: {len(done)}/{len(days)} days analysed; MISSING {failed or 'none'}; "
           f"free {ia.free_b() / 1e9:.2f} GB")


if __name__ == "__main__":
    main()
