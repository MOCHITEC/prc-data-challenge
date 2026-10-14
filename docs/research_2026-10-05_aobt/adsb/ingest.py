"""Fetch -> extract (stage-1 filter: 8 km of ARP, ground or <=4000 ft) -> purge, for a few 2025 days.
Uses the team's unchanged adsb_stage1_feasibility.py (copied from origin/claude/model-b-and-arrival-features).
Data: adsb.lol globe_history_2025, ODbL 1.0."""
import os, sys, time, json
A = os.path.dirname(os.path.abspath(__file__))
os.environ["SCRATCH"] = os.path.join(A, "raw")
sys.path.insert(0, A)
import adsb_stage1_feasibility as s1
s1.SCRATCH = os.environ["SCRATCH"]
for day in sys.argv[1:]:
    tag = f"v{day.replace('-', '.')}-planes-readsb-prod-0"
    if os.path.exists(os.path.join(s1.SCRATCH, f"{tag}.parquet")):
        print("skip", day); continue
    t0 = time.time()
    parts = s1.sizes("globe_history_2025", tag)
    s1.fetch("globe_history_2025", tag)
    t1 = time.time()
    s1.extract(tag)
    s1.purge(tag)
    with open(os.path.join(A, "ingest_status.jsonl"), "a") as f:
        f.write(json.dumps({"day": day, "bytes": sum(n for _, n in parts), "fetch_s": round(t1 - t0),
                            "extract_s": round(time.time() - t1)}) + "\n")
    print("done", day, flush=True)
