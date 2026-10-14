import polars as pl, numpy as np
pl.Config.set_tbl_rows(100); pl.Config.set_tbl_cols(40); pl.Config.set_tbl_width_chars(300)
# all LIRF rows (DEP+ARR) to see ID sequencing per file
a=pl.read_parquet('lirf.parquet')
d=pl.read_parquet('dep2.parquet')
# ID sequencing: within src, sort by MVT_ID over DEP rows only, neighbor SCHED median
def nb(df):
    df=df.sort('MVT_ID_mvt')
    s=df['SCHED_TIME_UTC_mvt'].dt.epoch('s').to_numpy().astype(float)
    n=len(s); k=5; med=np.empty(n)
    for i in range(n):
        lo,hi=max(0,i-k),min(n,i+k+1)
        w=np.concatenate([s[lo:i],s[i+1:hi]]); med[i]=np.median(w)
    return df.with_columns(id_sched_gap_h=pl.Series((s-med)/3600))
d=pl.concat([nb(d.filter(pl.col('src')==x)) for x in ['train','rank']])
d.write_parquet('dep3.parquet')
t=d.filter(pl.col('src')=='train')
print('all train DEP |gap|>2h share:', (t['id_sched_gap_h'].abs()>2).mean())
u=t.filter(pl.col('unm'))
u=u.with_columns(mb=pl.col('ms').cut([7200,14400,43200]), off=pl.col('id_sched_gap_h').abs()>2)
print(u.group_by(['mb','cls']).agg(pl.len(),pl.col('off').sum(),pl.col('id_sched_gap_h').median().alias('med_gap')).sort(['mb','cls']))
print(u.filter(pl.col('ms')>=43200).select('FLIGHT_mvt','cls','ms','id_sched_gap_h').sort('cls','ms'))
m=t.filter(~pl.col('unm')); print('matched off share', (m['id_sched_gap_h'].abs()>2).mean())
r=d.filter((pl.col('src')=='rank')&pl.col('unm')&(pl.col('ms')>=43200)).select('FLIGHT_mvt','ms','id_sched_gap_h'); print(r)
