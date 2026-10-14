from common import *
import numpy as np
d=pl.read_parquet('train_dep_cls.parquet').filter(pl.col('APT')=='LIRF').sort('MVT_TIME_UTC_mvt')
# rank of MVT_ID vs rank of time within month
d=d.with_columns(rid=pl.col('MVT_ID_mvt').rank().over('month'), rt=pl.col('M').rank().over('month'), rs=pl.col('S').rank().over('month'))
d=d.with_columns(dr=(pl.col('rid')-pl.col('rt')), drs=(pl.col('rid')-pl.col('rs')))
print(d.select(pl.corr('rid','rt'), pl.corr('rid','rs').alias('cs')))
u=d.filter(pl.col('unm'))
print(u.group_by('cls').agg(n=pl.len(), dr_med=pl.col('dr').median(), drs_med=pl.col('drs').median(), dr_abs=pl.col('dr').abs().median(),drs_abs=pl.col('drs').abs().median()))
print(d.filter(~pl.col('unm')).group_by('cls').agg(n=pl.len(), dr_med=pl.col('dr').median(), drs_med=pl.col('drs').median(),drs_abs=pl.col('drs').abs().median()))
