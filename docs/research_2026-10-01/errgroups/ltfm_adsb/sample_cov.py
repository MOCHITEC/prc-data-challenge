"""Range-read the first MB bytes of an adsb.lol globe_history day and count fixes near each airport.
usage: sample_cov.py YYYY.MM.DD MB   -> prints one JSON line. No data saved except the summary line."""
import sys, json, subprocess, tarfile, gzip, collections
import numpy as np
sys.path.insert(0, "/tmp/claude-0/-home-user-prc-data-challenge/bd99a3bf-146d-5c0a-be6a-11279605ec47/scratchpad/errgroups/ltfm_adsb")
from s1_ref import AIRPORTS, dist_m
day, mb = sys.argv[1], int(sys.argv[2])
yr = day[:4]; tag = f"v{day}-planes-readsb-prod-0"
from walk import walk
ps, mem = walk(day)
if not mem or "/traces/" not in mem[-1][0]: print(json.dumps({"day":day,"err":"no traces found","n":len(mem)})); sys.exit()
off = mem[-1][2]; size = sum(z for _,z in ps)
for url, sz in ps:
    if off < sz: break
    off -= sz
end = min(off + mb*1024*1024, sz) - 1
p = subprocess.Popen(["curl","-sS","-L","-r",f"{off}-{end}",url], stdout=subprocess.PIPE)
fix=collections.Counter(); gfix=collections.Counter(); gac=collections.defaultdict(set); ac=collections.defaultdict(set)
nfiles=0; first=last=None
try:
    with tarfile.open(fileobj=p.stdout, mode="r|") as tf:
        for m in tf:
            if not (m.isfile() and "/traces/" in m.name and m.name.endswith(".json")): continue
            raw = tf.extractfile(m).read(); nfiles += 1
            first = first or m.name; last = m.name
            try: d = json.loads(gzip.decompress(raw) if raw[:2]==b"\x1f\x8b" else raw)
            except Exception: continue
            tr = d.get("trace") or []
            if not tr: continue
            lat = np.array([q[1] if q[1] is not None else np.nan for q in tr],float)
            lon = np.array([q[2] if q[2] is not None else np.nan for q in tr],float)
            gr = np.array([q[3]=="ground" for q in tr])
            for apt,(a,b) in AIRPORTS.items():
                box=(np.abs(lat-a)<0.08)&(np.abs(lon-b)<0.12)
                if not box.any(): continue
                mm = box & (dist_m(a,b,lat,lon)<=8000)
                if mm.any():
                    fix[apt]+=int(mm.sum()); ac[apt].add(d["icao"])
                    g=int((mm&gr).sum())
                    if g: gfix[apt]+=g; gac[apt].add(d["icao"])
except Exception as e:
    pass
p.kill()
print(json.dumps({"day":day,"asset_GB":round(size/1e9,2),"MB":mb,"files":nfiles,"first":first,"last":last,
  "fix":dict(fix),"gfix":dict(gfix),"ac":{k:len(v) for k,v in ac.items()},"gac":{k:len(v) for k,v in gac.items()}}))
