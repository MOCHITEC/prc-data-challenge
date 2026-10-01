import polars as pl, numpy as np
from sklearn.linear_model import LogisticRegression
pl.Config.set_tbl_rows(100); pl.Config.set_tbl_cols(40); pl.Config.set_tbl_width_chars(300)
d=pl.read_parquet('dep5.parquet')
t=d.filter((pl.col('src')=='train')&pl.col('unm')&(pl.col('ms')>=28800)).with_columns(r=(pl.col('cls')=='roll'))
F=lambda df,g=True: np.c_[df['ms'].to_numpy()/3600-16, df['glitch'].to_numpy()] if g else np.c_[df['ms'].to_numpy()/3600-16]
y=t['r'].to_numpy()
c1=LogisticRegression(C=1.0,max_iter=1000).fit(F(t),y); c0=LogisticRegression(C=1.0,max_iter=1000).fit(F(t,False),y)
rk=d.filter((pl.col('src')=='rank')&pl.col('unm')&(pl.col('ms')>=28800))
rk=rk.with_columns(p_ms=pl.Series(c0.predict_proba(F(rk,False))[:,1]),p_msg=pl.Series(c1.predict_proba(F(rk))[:,1]))
# empirical band rates (2025): 8-12h 0/21, 12-14h 0/9, 14-18h 5/13, >=18h 7/7(incl NOSOS739 fill-type)
rk=rk.with_columns(mu=pl.col('ms')+pl.col('p_msg')*(87300-pl.col('ms')))
main={'WMTT3508':33444,'TAY9LA':30644,'WMT402':36782,'WMTMT4SG':35434,'NOSOS431':39636,'ITYTY672':66037,'RYR4852':33485,'TARAR753':35683,'SVII3006':31551,'ITYTY674':66693,'QFAA7006':32688}
rk=rk.with_columns(main=pl.col('FLIGHT_mvt').replace_strict(main,default=None).cast(pl.Float64))
out=rk.select('FLIGHT_mvt','sl','ml','ms','dcross_loc0','scls','RUNWAY_mvt','ADES_mvt','STAND_mvt','p_ms','p_msg','mu','main').sort('ms')
print(out)
N=344841; RM=252.43
s=out.filter(pl.col('main').is_not_null())
gain_2025=((s['main']-s['mu'])**2).sum()   # expected SSE reduction if 2025 mixture holds
for nm,yy in [('all fill y=ms',s['ms']),('all norm y=1000',pl.Series([1000.0]*s.height))]:
    d_sse=((s['mu']-yy)**2).sum()-((s['main']-yy)**2).sum()
    print(nm,'dSSE %.3e  dRMSE %.2f'%(d_sse, np.sqrt(RM**2+d_sse/N)-RM))
print('expected (2025 mixture) dSSE %.3e dRMSE %.2f'%(-gain_2025, np.sqrt(RM**2-gain_2025/N)-RM))
# expected number of rollovers in ranking under 2025 model
print('E[#roll] ms>=8h:', out['p_msg'].sum(), ' p_ms:', out['p_ms'].sum())
# 2-8h crossing candidates
c=d.filter((pl.col('src')=='rank')&pl.col('unm')&(pl.col('ms')>=7200)&(pl.col('ms')<28800)&(pl.col('dcross_loc0')>=1))
print('rank 2-8h crossing unmatched:',c.height)
tt=d.filter((pl.col('src')=='train')&pl.col('unm')&(pl.col('ms')>=7200)&(pl.col('ms')<28800)&(pl.col('dcross_loc0')>=1))
print('train 2-8h crossing', tt.height,'roll',(tt['cls']=='roll').sum())
out.write_csv('rank_candidates_ms8h.csv')
