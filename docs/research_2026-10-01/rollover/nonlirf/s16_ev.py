# L2 expected effect on LB RMSE (252.43, 344,841 rows) of the posterior-mean shift s = p*E[D] on N flagged ranking DEP rows
import numpy as np, polars as pl
O='/tmp/claude-0/-home-user-prc-data-challenge/bd99a3bf-146d-5c0a-be6a-11279605ec47/scratchpad/rollover/nonlirf/'
r=pl.read_csv(O+'s13_rules.csv')
R0=252.43; M=344841; D=85000.
def d(sse): return np.sqrt(R0**2+sse/M)-R0
for row in r.iter_rows(named=True):
    N=row['rank_dep_n']; s=row['shift_s']; p=row['p_tail']
    if row['rule'].startswith('R4'): N=6  # 2 of the 8 flagged rows are month-start edge artefacts (gap>3000 h)
    Dbar = s/p if p>0 else D
    exp_sse = N*s*s - 2*s*(N*p)*Dbar
    print(f"{row['rule'][:45]:45s} N={N:5d} s={s:7.1f}  E[K]={N*p:5.2f}  dRMSE: K=0 {d(N*s*s):+.3f}  K=1 {d(N*s*s-2*s*D):+.3f}  K=2 {d(N*s*s-4*s*D):+.3f}  expected(train rate) {d(exp_sse):+.4f}  hard-swap K=1 {d((N-1)*D*D - D*D):+.1f}")
print('reference: one fully-missed 85,000 s row = MSE', D*D/M, ' -> fixing it alone: dRMSE', d(-D*D))
