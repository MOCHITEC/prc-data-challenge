from common import *
t=load('train_ext.parquet').filter((pl.col('APT')=='LIRF')&pl.col('month').is_in([1,7]))
r=load('rank_ext.parquet').filter(pl.col('APT')=='LIRF')
for nm,x in [('2025 J+J',t),('2026 rank',r)]:
    u=x.filter(pl.col('unm'))
    print(nm, 'n',x.height,'unm',u.height, 'unm ms>=4h',u.filter(pl.col('ms')>=14400).height,'ms>=12h',u.filter(pl.col('ms')>=43200).height,
      'garb %.3f'%u.select((~pl.col('FLIGHT_mvt').fill_null('').str.contains(r'^[A-Z]{3}\d')).mean()).item(),
      'actype null %.3f'%u.select(pl.col('AIRCRAFT_TYPE_mvt').is_null().mean()).item())
