import polars as pl, numpy as np
from sklearn.linear_model import LogisticRegression
pl.Config.set_tbl_rows(100); pl.Config.set_tbl_cols(40); pl.Config.set_tbl_width_chars(300)
d=pl.read_parquet('dep5.parquet')
# mechanism roll = y>72000 and not y==ms
d=d.with_columns(mroll=(pl.col('y')>72000)&((pl.col('y')-pl.col('ms')).abs()>60))
t=d.filter((pl.col('src')=='train')&pl.col('unm')&(pl.col('ms')>=28800))
print('ms>=8h train n',t.height,'mech roll',t['mroll'].sum(),'norm',(t['cls']=='norm').sum(),' non-cross rolls',(t['mroll']&~t['cross']).sum(),'non-cross n',(~t['cross']).sum())
def rule(df,expr,name):
    f=df.filter(expr); tp=f['mroll'].sum(); P=df['mroll'].sum()
    print(f'{name:40s} flagged {f.height:3d} TP {tp} prec {tp/max(f.height,1):.2f} recall {tp}/{P}')
tu=d.filter((pl.col('src')=='train')&pl.col('unm'))
c=pl.col
rule(tu,c('cross')&(c('ms')>=50400),'cross & ms>=14h')
rule(tu,c('cross')&(c('ms')>=50400)&c('glitch'),'cross & ms>=14h & glitch')
rule(tu,c('cross')&(c('ms')>=50400)&(c('RUNWAY_mvt')!='16R'),'cross & ms>=14h & rwy!=16R')
rule(tu,c('cross')&(c('ms')>=50400)&c('glitch')&(c('RUNWAY_mvt')!='16R'),'cross & ms>=14h & glitch & !16R')
rule(tu,c('cross')&(c('ms')>=64800),'cross & ms>=18h')
rule(tu,c('cross')&(c('ms')>=72000),'cross & ms>=20h')
# per-month listing for LOMO-free rules (rules have no fitted params except thresholds)
# LOMO logistic on crossing rows ms>=8h, features ms(+glitch)
tc=t.filter(c('cross'))
y=tc['mroll'].to_numpy(); ms=tc['ms'].to_numpy().astype(float); Y=tc['y'].to_numpy().astype(float); mon=tc['mon'].to_numpy(); g=tc['glitch'].to_numpy()
R=87300.
def mu(p): return ms+p*(R-ms)
def sse(p): return ((Y-mu(p))**2).sum()
print('cross ms>=8h n',len(y),'roll',y.sum(),'SSE pred=ms %.3e'%sse(np.zeros(len(y))))
for nm,X in [('ms',np.c_[ms/3600-16]),('ms+glitch',np.c_[ms/3600-16,g])]:
  for C in [0.3,1.0]:
    p=np.zeros(len(y))
    for m in np.unique(mon):
        tr=mon!=m; p[~tr]=LogisticRegression(C=C,max_iter=1000).fit(X[tr],y[tr]).predict_proba(X[~tr])[:,1]
    print(f'LOMO {nm:10s} C={C} SSE {sse(p):.3e}  logloss {-np.mean(y*np.log(p)+(1-y)*np.log(1-p)):.3f}  sumP {p.sum():.1f}')
    if nm=='ms+glitch' and C==1.0: 
        tc=tc.with_columns(p_lomo=pl.Series(p)); print(tc.select('FLIGHT_mvt','mon','ms','glitch','cls','p_lomo').sort('ms'))
# final model fit on all crossing rows, apply to ranking
X=np.c_[ms/3600-16,g]; clf=LogisticRegression(C=1.0,max_iter=1000).fit(X,y); print('coef',clf.coef_,clf.intercept_)
rk=d.filter((c('src')=='rank')&c('unm')&(c('ms')>=28800))
p=clf.predict_proba(np.c_[rk['ms'].to_numpy()/3600-16,rk['glitch'].to_numpy()])[:,1]
rk=rk.with_columns(p_roll=pl.when(c('cross')).then(pl.Series(p)).otherwise(0.0))
# ms>=20h: 2025 empirical 4 roll/1 fill(y=ms)
rk=rk.with_columns(mu=c('ms')+c('p_roll')*(R-c('ms')))
main={56520:33444,53520:30644,48841:36782,47705:35434,111654:39636,61322:66037,48539:33485,48776:35683,60656:31551,47337:66693,87484:32688}
rk=rk.with_columns(main=c('ms').replace_strict(main,default=None).cast(pl.Float64))
o=rk.select('FLIGHT_mvt',c('sl').dt.strftime('%m-%d %H:%M').alias('sched_loc'),c('ml').dt.strftime('%m-%d %H:%M').alias('mvt_loc'),'ms','cross','glitch','RUNWAY_mvt','p_roll','mu','main').sort('ms')
print(o); print('sum p_roll', o['p_roll'].sum())
N=344841; RM=252.43
s=o.filter(c('main').is_not_null())
for nm,yy in [('all fill y=ms',s['ms'].cast(pl.Float64)),('all norm y=1000',pl.Series([1000.0]*s.height)),('rolls where p>0.5 else fill',pl.Series(np.where(s['p_roll']>0.5,87300.0,s['ms'])))]:
    dsse=((s['mu']-yy)**2).sum()-((s['main']-yy)**2).sum(); print(f'{nm:30s} dSSE {dsse:.3e} dRMSE {np.sqrt(RM**2+dsse/N)-RM:+.2f}')
# 'pred=ms' alternative (p=0)
for nm,yy in [('all fill',s['ms'].cast(pl.Float64)),('all norm',pl.Series([1000.0]*s.height))]:
    dsse=((s['ms']-yy)**2).sum()-((s['main']-yy)**2).sum(); print(f'pred=ms vs main, {nm}: dRMSE {np.sqrt(RM**2+dsse/N)-RM:+.2f}')
o.write_csv('rank_candidates_ms8h.csv')
