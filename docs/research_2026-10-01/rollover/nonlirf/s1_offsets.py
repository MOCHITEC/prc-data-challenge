# Exhaustive search: BLOCK - clock == offset (+-N) for all training DEP/ARR rows; spike vs background
import polars as pl
O='/tmp/claude-0/-home-user-prc-data-challenge/bd99a3bf-146d-5c0a-be6a-11279605ec47/scratchpad/rollover/nonlirf/'
df=pl.read_parquet(O+'train.parquet')
clocks={'SCHED':'SCHED_TIME_UTC_mvt','MVT':'MVT_TIME_UTC_mvt','AOBT3':'AOBT_3_flt','EOBT1':'EOBT_1_flt','LOBT':'LOBT_flt','IOBT':'IOBT_flt','ARVT1':'ARVT_1_flt','ARVT3':'ARVT_3_flt'}
df=df.with_columns([(pl.col('BLOCK_TIME_UTC_mvt')-pl.col(c)).dt.total_seconds().alias('d_'+k) for k,c in clocks.items()])
offs=[86400,-86400,172800,-172800,43200,-43200,3600,-3600,7200,-7200]
rows=[]
for ph in ['DEP','ARR']:
    s=df.filter(pl.col('PHASE_mvt')==ph)
    for k in clocks:
        d=s['d_'+k]
        nn=d.drop_nulls().len()
        for off in offs:
            r={'phase':ph,'clock':k,'nonnull':nn,'off':off}
            for N in [0,10,60,600]:
                r[f'n{N}']=int(((d-off).abs()<=N).sum())
            # background: same window width 1201 s centred at off+-1800 (avg)
            bg=(int(((d-off-1800).abs()<=600).sum())+int(((d-off+1800).abs()<=600).sum()))/2
            r['bg600']=bg
            rows.append(r)
t=pl.DataFrame(rows)
pl.Config.set_tbl_rows(400); pl.Config.set_tbl_width_chars(200)
print(t.filter(pl.col('n600')>0))
t.write_csv(O+'s1_offsets.csv')
