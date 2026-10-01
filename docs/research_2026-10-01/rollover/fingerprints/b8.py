import polars as pl, numpy as np
from scipy.stats import mannwhitneyu
pl.Config.set_tbl_rows(100); pl.Config.set_tbl_cols(40); pl.Config.set_tbl_width_chars(300)
d=pl.read_parquet('dep5.parquet'); a=pl.read_parquet('arr.parquet').with_columns(op=pl.col('FLIGHT_mvt').str.slice(0,3))
c=pl.col
q=d.filter(c('unm')&(c('ms')>=28800)&c('cross')).with_columns(op=c('FLIGHT_mvt').str.slice(0,3))
rows=[]
for r in q.iter_rows(named=True):
    S,M=r['SCHED_TIME_UTC_mvt'],r['MVT_TIME_UTC_mvt']
    st=a.filter((c('STAND_mvt')==r['STAND_mvt'])&(c('BLOCK_TIME_UTC_mvt')<M)&(c('BLOCK_TIME_UTC_mvt')>S-pl.duration(hours=30)))
    ops=a.filter((c('op')==r['op'])&(c('BLOCK_TIME_UTC_mvt')<M)&(c('BLOCK_TIME_UTC_mvt')>S-pl.duration(hours=30)))
    so=st.filter(c('op')==r['op'])
    lst=st['BLOCK_TIME_UTC_mvt'].max(); lso=so['BLOCK_TIME_UTC_mvt'].max()
    hrs=(M-S).total_seconds()/3600
    rows.append(dict(src=r['src'],cls=r['cls'],fl=r['FLIGHT_mvt'],ms_h=round(hrs,1),
        stand_arr_rate=st.filter(c('BLOCK_TIME_UTC_mvt')>S).height/hrs,
        h_since_last_stand_inb=None if lst is None else (M-lst).total_seconds()/3600,
        h_since_sameop_stand_inb=None if lso is None else (M-lso).total_seconds()/3600,
        sameop_stand_inb_after_S=so.filter(c('BLOCK_TIME_UTC_mvt')>S).height))
o=pl.DataFrame(rows)
print(o.sort('src','cls','ms_h'))
t=o.filter(c('src')=='train')
for col in ['stand_arr_rate','h_since_last_stand_inb','h_since_sameop_stand_inb','sameop_stand_inb_after_S']:
    x=t.filter(c('cls')=='roll')[col].drop_nulls().to_numpy(); z=t.filter(c('cls')=='fill')[col].drop_nulls().to_numpy()
    print(f'{col:28s} roll med {np.median(x):.2f} (n{len(x)})  fill med {np.median(z):.2f} (n{len(z)})  MWU p {mannwhitneyu(x,z).pvalue:.2f}')
