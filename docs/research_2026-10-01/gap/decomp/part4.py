import polars as pl, numpy as np
S='/tmp/claude-0/-home-user-prc-data-challenge/bd99a3bf-146d-5c0a-be6a-11279605ec47/scratchpad/gap/decomp/'
d=pl.read_parquet(S+'cv_rows.parquet'); N=d.height
x=d.filter(~pl.col('rollover')); se=np.sort(x['se'].to_numpy())[::-1]; tot=se.sum()
print('non-rollover MSE per row of N', tot/N, 'RMSE', (tot/x.height)**.5)
for k in [10,30,100,300,1000,3000,10000,30000]:
    print(f'top{k:6d}: {se[:k].sum()/tot*100:5.1f}%  MSE/row(N)={se[:k].sum()/N:7.0f}  min|e| in set={se[k-1]**.5:6.0f}')
for lane,mask in [('matched',pl.col('matched')),('unmatched',~pl.col('matched'))]:
    s=np.sort(x.filter(mask)['se'].to_numpy())[::-1]
    print(lane, 'n',len(s), ' '.join(f'top{k}={s[:k].sum()/N:.0f}' for k in [10,30,100,300,1000,3000]))
# how many top rows needed to reach 15128/1.03 MSE
need=15128/1.03; c=np.cumsum(se)/N; print('rows to remove for 14,700 s2 (if zeroed):', int(np.searchsorted(c,need))+1)
# halving errors instead of zeroing: gain = 0.75*se
c2=np.cumsum(se*0.75)/N; print('rows to halve error for 14,700:', int(np.searchsorted(c2,need))+1)
# uniform shrink of matched errors: factor needed
m=x.filter(pl.col('matched'))['se'].sum()/N; print('matched mse contrib',m)
