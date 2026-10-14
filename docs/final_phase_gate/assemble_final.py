"""One step from the old-board scores to the final file (v6 base, +1x dither), following the pre-registered rules.

    python -I docs/final_phase_gate/assemble_final.py <R_gate|-> <R_gate_q15|-> <R_control|-> <P1|-> <P2|-> <P3|-> <P4|-> <out_dir> [--base=v7a]

R_gate / R_gate_q15: old-board scores of handoff/2026-10-14/oldboard/v48_plus_gate.parquet and v48_plus_gate_q15.parquet.
  gate kept if R_gate <= 234.4099 (v48), q15 kept if R_gate_q15 <= R_gate (or <= 234.4099 when the gate test is missing);
  a missing test keeps the gate and drops q15 (README: "旧ボードで試せない" -> final_v6_gatefull_plus1delta).
R_control, P1..P4: the bias probes (handoff/2026-10-14/probes). R_control defaults to 234.4099 when '-'.
Writes <out_dir>/final_selected.parquet, prints its md5 and the decisions."""
import hashlib
import os
import subprocess
import sys

V48 = 234.4099
a = sys.argv[1:]
rg, rq, rc, p1, p2, p3, p4, out = a[:8]
TAG = next((x.split("=", 1)[1] for x in a if x.startswith("--base=")), "v6")  # --base=v7a when the integrator switches
H = f"handoff/2026-10-14/{TAG}/"
gate = (rg == "-") or (float(rg) <= V48)
if rq == "-":
    q15 = False
else:
    q15 = float(rq) <= (float(rg) if rg != "-" else V48)
base = {(True, True): f"final_{TAG}_gatefull_q15_plus1delta.parquet", (True, False): f"final_{TAG}_gatefull_plus1delta.parquet",
        (False, True): f"final_{TAG}_q15_plus1delta.parquet", (False, False): None}[(gate, q15)]
if base:
    base = H + base
elif TAG == "v6":
    base = "/home/user/lc-final/handoff/2026-10-13/optional_janjul_reanchor_plus1delta.parquet"
else:
    base = H + f"{TAG}_plus1delta_built_here.parquet"
print(f"gate {'kept' if gate else 'dropped'} (R_gate {rg}), q15 {'kept' if q15 else 'dropped'} (R_gate_q15 {rq}) -> base {base}")
os.makedirs(out, exist_ok=True)
dst = os.path.join(out, "final_selected.parquet")
if all(x == "-" for x in (p1, p2, p3, p4)):
    import shutil
    shutil.copy(base, dst)
    print("no probe results: base file copied unchanged")
else:
    r0 = rc if rc != "-" else str(V48)
    cmd = [sys.executable, "-I", "docs/final_phase_gate/probe_fix.py", "handoff/2026-10-14/probes/v48_control.parquet", r0,
           p1, p2, p3, p4, base, dst, "--dithered"] + [x for x in a[8:] if x.startswith(("--p5=", "--p6=", "--stage2="))]
    print("$", " ".join(cmd))
    subprocess.run(cmd, check=True)
subprocess.run([sys.executable, "-m", "prc.validate_submission", dst, "data/raw/final_submitting.parquet"], check=True)
print("md5", hashlib.md5(open(dst, "rb").read()).hexdigest(), dst)
