"""Expected LB effect of a posterior-mean overlay pred = pi*R + (1-pi)*ms on the 2026 LIRF unmatched ms>=12h rows,
assuming the current prediction is ~ms (documented: v10/v16 model B predicts ~ms, mean |pred-ms| 988 s) and the
alternative label is the fill (y = ms). RMSE base 252.43 on N = 344,841."""
import numpy as np
N, B = 344841, 252.43
SSE0 = N*B*B
R = 86400 + 1020.5   # median y-86400 of the 12 LIRF BLOCK-shifted 2025 rows
rows = {"ITYTY674":47337,"WMTMT4SG":47705,"RYR4852":48539,"TARAR753":48776,"WMT402":48841,"TAY9LA":53520,
        "WMTT3508":56520,"SVII3006":60656,"ITYTY672":61322,"QFAA7006":87484,"NOSOS431":111654}
def lb(dsse): return np.sqrt((SSE0+dsse)/N)-B
for name, sel, pi in [("v11 replay: all 11, pi=.379", lambda m: True, 0.379),
                      ("ms>=50,400 (6 rows), pi=11/20=.55", lambda m: m>=50400, 0.55),
                      ("ms>=57,600 (4 rows), pi=9/13=.692", lambda m: m>=57600, 0.692)]:
    d = np.array([R-m for m in rows.values() if sel(m)])
    exp_gain = -(pi**2)*(d**2).sum()      # E[dSSE] when true rate = pi
    floor = (pi**2)*(d**2).sum()          # dSSE when no row is a rollover (all fills)
    ceil = -((d**2)-((1-pi)*d)**2).sum()  # all rows rollovers
    print(f"{name}: n={len(d)} E[dLB]={lb(exp_gain):+.2f}s  floor(no roll)={lb(floor):+.2f}s  ceiling(all roll)={lb(ceil):+.2f}s")
# what if alternative is 'ordinary' (y~1000) instead of fill, for the 5 low-ms rows (cost of v11-type move)
d = np.array([m for m in rows.values() if m<50400]); pred=d; new=0.379*R+0.621*d
print("5 low-ms rows if y~1000 instead of fill: dLB of v11 move =", round(lb(((new-1000)**2-(pred-1000)**2).sum()),2))
