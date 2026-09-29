"""ADS-B full ingest of the 62 scored days (January and July 2026), resumable, one raw day on disk at a time.

Read-only with respect to the repo: it writes only to the scratch dir (`adsb_stage1_feasibility.SCRATCH`). Detection
is the stage-1 script, unchanged: this driver only orders its steps and makes them restartable.

Per day D:  fetch (every part of the adsb.lol `globe_history_2026` release of D)  ->  extract (stage-1 `extract`,
filtered fixes near the 10 airports)  ->  purge (delete the raw parts)  ->  analyse (stage-1 `analyse`, the per-departure
joined file). Each finished step is recorded in `$SCRATCH/ingest/state/<D>.json` (written atomically), so a restart
continues from the first unfinished step of the first unfinished day:

  * a part already on disk at its HEAD size is not downloaded again; a shorter one is resumed (`curl -C -`);
  * a day marked `extracted` is never re-extracted, and any raw parts still present for it are deleted first;
  * `extract` and `analyse` write their outputs whole, and the state is updated only after they return 0, so a kill
    mid-step just repeats that step.

Disk: before a day's download, the bytes still to fetch + 3 GB margin must fit in the free space, or the run STOPS
(exit 2) rather than skipping. A download that keeps failing (5 tries per part, backoff 30-480 s) marks the day
`failed`, deletes its parts, logs the proxy status, and moves on; failed days get one more pass at the end. Days
still failed are MISSING: their flights keep the base prediction.

    POLARS_MAX_THREADS=2 POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR=load_as_storage \
    .venv/bin/python scripts/adsb_ingest_all.py [run | status] [--days D,D,...] [--stop-after-step N]

Data: adsb.lol, Open Database License (ODbL) 1.0. "Contains data from adsb.lol, made available under the ODbL 1.0."
"""

from __future__ import annotations

import datetime as dt
import fcntl
import json
import os
import shutil
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import adsb_stage1_feasibility as s1

ROOT = s1.REPO_DIR
SCR = s1.SCRATCH
ING = os.path.join(SCR, "ingest")
STATE = os.path.join(ING, "state")
LOG = os.path.join(ING, "ingest.log")
PY = os.path.join(ROOT, ".venv/bin/python")
S1 = os.path.join(ROOT, "scripts/adsb_stage1_feasibility.py")
MARGIN = 3e9
TRIES, BACKOFF0 = 5, 30
ENV = {**os.environ, "POLARS_MAX_THREADS": "2", "POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR": "load_as_storage"}


def all_days() -> list[str]:
    out = []
    for m in (1, 7):
        d = dt.date(2026, m, 1)
        while d.month == m:
            out.append(d.isoformat())
            d += dt.timedelta(days=1)
    return out


def tag_of(day: str) -> str:
    return f"v{day.replace('-', '.')}-planes-readsb-prod-0"


def log(msg: str) -> None:
    line = f"{dt.datetime.now(dt.timezone.utc):%Y-%m-%dT%H:%M:%SZ} {msg}"
    print(line, flush=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")


def free_b() -> int:
    return shutil.disk_usage(SCR).free


def load_state(day: str) -> dict:
    p = os.path.join(STATE, f"{day}.json")
    if os.path.exists(p):
        with open(p) as f:
            return json.load(f)
    return {"day": day, "tag": tag_of(day)}


def save_state(st: dict) -> None:
    p = os.path.join(STATE, f"{st['day']}.json")
    tmp = p + ".tmp"
    with open(tmp, "w") as f:
        json.dump(st, f, indent=1)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, p)


def proxy_status() -> str:
    r = subprocess.run(["bash", "-c", 'curl -sS -m 20 "$HTTPS_PROXY/__agentproxy/status"'],
                       capture_output=True, text=True, check=False)
    return (r.stdout or r.stderr)[:3000]


def fetch_day(st: dict) -> bool:
    tag = st["tag"]
    if "parts" not in st:
        parts = s1.sizes("globe_history_2026", tag)
        if not parts:
            st["fetch_error"] = "no release parts found (HEAD)"
            return False
        st["parts"] = parts
        save_state(st)
    parts = st["parts"]
    have = {n: (os.path.getsize(os.path.join(SCR, n)) if os.path.exists(os.path.join(SCR, n)) else 0)
            for n, _ in parts}
    for n, size in parts:  # a file larger than its HEAD size is corrupt: start that part again
        if have[n] > size:
            os.remove(os.path.join(SCR, n))
            have[n] = 0
    todo = sum(size - have[n] for n, size in parts)
    fb = free_b()
    log(f"{st['day']} fetch: {len(parts)} parts, {sum(s for _, s in parts) / 1e9:.3f} GB, still to fetch "
        f"{todo / 1e9:.3f} GB; free {fb / 1e9:.2f} GB")
    if todo + MARGIN > fb:
        log(f"STOP: {st['day']} needs {todo / 1e9:.2f} GB + {MARGIN / 1e9:.0f} GB margin, only {fb / 1e9:.2f} GB free")
        sys.exit(2)
    for n, size in parts:
        dst = os.path.join(SCR, n)
        for k in range(TRIES):
            if os.path.exists(dst) and os.path.getsize(dst) == size:
                break
            if k:
                wait = BACKOFF0 * 2 ** (k - 1)
                log(f"  {n}: retry {k}/{TRIES - 1} in {wait} s")
                time.sleep(wait)
            r = subprocess.run(["curl", "-sS", "-L", "--fail", "-C", "-", "--retry", "2", "--retry-delay", "10",
                                "--speed-limit", "200000", "--speed-time", "120", "-o", dst,
                                s1.part_url("globe_history_2026", tag, n[-2:])],
                               capture_output=True, text=True, check=False)
            got = os.path.getsize(dst) if os.path.exists(dst) else 0
            if r.returncode != 0:
                log(f"  {n}: curl rc {r.returncode} ({r.stderr.strip()[:200]}), have {got}/{size}")
            if got > size:
                os.remove(dst)
        if not (os.path.exists(dst) and os.path.getsize(dst) == size):
            st["fetch_error"] = f"{n} incomplete after {TRIES} tries"
            log(f"  {st['day']}: {st['fetch_error']}; proxy status:\n{proxy_status()}")
            return False
    st["fetched"] = True
    st["fetched_at"] = f"{dt.datetime.now(dt.timezone.utc):%Y-%m-%dT%H:%M:%SZ}"
    st.pop("fetch_error", None)
    save_state(st)
    log(f"{st['day']} fetched; raw on disk {sum(os.path.getsize(os.path.join(SCR, n)) for n, _ in parts) / 1e9:.3f} GB")
    return True


def run_step(args: list[str], logfile: str) -> None:
    with open(logfile, "w") as f:
        r = subprocess.run([PY, S1, *args], stdout=f, stderr=subprocess.STDOUT, env=ENV, cwd=ROOT, check=False)
    if r.returncode != 0:
        raise RuntimeError(f"{args[0]} failed rc {r.returncode}, see {logfile}")


def do_day(day: str, stop_after: int | None, counter: list[int]) -> str:
    st = load_state(day)
    tag = st["tag"]
    fixes = os.path.join(SCR, f"{tag}.parquet")
    joined = os.path.join(SCR, f"{tag}_joined.parquet")

    def tick():
        counter[0] += 1
        if stop_after is not None and counter[0] >= stop_after:
            log(f"--stop-after-step {stop_after} reached (after {day}); exiting as if killed")
            sys.exit(3)

    if st.get("analysed"):
        return "done"
    # the five validation census days: filtered fixes kept on disk, extracted by the same stage-1 script
    if not st.get("extracted") and os.path.exists(fixes) and not s1.local_parts(tag) and \
            os.path.exists(os.path.join(SCR, f"extract_{day}.log")):
        st.update(extracted=True, purged=True, reused_fixes=True, fixes_mb=round(os.path.getsize(fixes) / 1e6, 1))
        save_state(st)
        log(f"{day}: filtered fixes already on disk from the validation run; reused, not re-downloaded")
    if not st.get("extracted"):
        if not st.get("fetched"):
            if not fetch_day(st):
                st["failed"] = st.get("failed", 0) + 1
                save_state(st)
                s1.purge(tag)
                return "failed"
            tick()
        log(f"{day} extract ...")
        run_step(["extract", tag], os.path.join(SCR, f"extract_{day}.log"))
        st.update(extracted=True, fixes_mb=round(os.path.getsize(fixes) / 1e6, 1))
        with open(os.path.join(SCR, f"extract_{day}.log")) as f:
            st["extract_summary"] = f.read().strip().splitlines()[-1][:300]
        save_state(st)
        log(f"{day} extracted: {st['extract_summary']}")
        tick()
    if not st.get("purged") or s1.local_parts(tag):
        for p in s1.local_parts(tag):
            os.remove(p)
        st["purged"] = True
        save_state(st)
        log(f"{day} purged; free {free_b() / 1e9:.2f} GB")
    run_step(["analyse", tag, day], os.path.join(SCR, f"analyse_{day}.log"))
    st["analysed"] = True
    st["joined_mb"] = round(os.path.getsize(joined) / 1e6, 2)
    save_state(st)
    log(f"{day} analysed")
    tick()
    return "done"


def status(days: list[str]) -> None:
    for d in days:
        st = load_state(d)
        print(d, {k: st.get(k) for k in ("fetched", "extracted", "purged", "analysed", "reused_fixes", "failed",
                                          "fetch_error")})


def main() -> None:
    a = sys.argv[1:]
    cmd = a[0] if a and not a[0].startswith("--") else "run"
    days = a[a.index("--days") + 1].split(",") if "--days" in a else all_days()
    stop_after = int(a[a.index("--stop-after-step") + 1]) if "--stop-after-step" in a else None
    os.makedirs(STATE, exist_ok=True)
    if cmd == "status":
        status(days)
        return
    lock = open(os.path.join(ING, "lock"), "w")  # noqa: SIM115
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        sys.exit("another ingest is running (lock held)")
    log(f"=== run start: {len(days)} days; free {free_b() / 1e9:.2f} GB")
    counter = [0]
    failed = []
    for d in days:
        if do_day(d, stop_after, counter) == "failed":
            failed.append(d)
    if failed:
        log(f"second pass over failed days: {failed}")
        failed = [d for d in failed if do_day(d, stop_after, counter) == "failed"]
    done = [d for d in days if load_state(d).get("analysed")]
    log(f"=== run end: {len(done)}/{len(days)} days analysed; MISSING {failed or 'none'}; free {free_b() / 1e9:.2f} GB")


if __name__ == "__main__":
    main()
