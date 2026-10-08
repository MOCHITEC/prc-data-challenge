import pandas as pd, numpy as np
f1='data/work_flseq/feat_train.parquet'; f2='data/work_flseq/feat_rank.parquet'
flt=[('ADEP_mvt','=','LFPG')]
a=pd.read_parquet(f1,filters=flt); b=pd.read_parquet(f2,filters=flt)
for d in (a,b): d['m']=pd.to_datetime(d.MVT_TIME_UTC_mvt).dt.month
num=[c for c in a.columns if pd.api.types.is_numeric_dtype(a[c]) and c not in ('MVT_ID_mvt','FLIGHT_ID_mvt','TAXITIME_SEC_mvt','m')]
out=[]
for mo in (1,7):
    x=a[a.m==mo]; y=b[b.m==mo]; ally=a
    for c in num:
        sd=a[c].std()
        if not sd or np.isnan(sd): continue
        out.append((mo,c,(y[c].mean()-x[c].mean())/sd, x[c].isna().mean(), y[c].isna().mean(), x[c].mean(), y[c].mean()))
o=pd.DataFrame(out,columns=['m','feat','smd','na25','na26','mean25','mean26'])
p=o.pivot(index='feat',columns='m',values='smd')
p['gap']=p[7].abs()-p[1].abs()
pd.set_option('display.width',250)
print(p.sort_values(7,key=abs,ascending=False).head(40).round(2).to_string())
print(o[o.m==7].assign(nad=lambda d:(d.na26-d.na25).abs()).sort_values('nad',ascending=False).head(15).round(3).to_string())
