import pandas as pd, numpy as np
cols=['ADEP_mvt','MVT_TIME_UTC_mvt','seq_arr_prev_s','seq_arr_next_s','seq_arr_pm10m','occ_g_arr_prev1','occ_g_arr_prev2','occ_g_dep_prev','occ_aobt_minus_arr_prev2','STAND_mvt']
a=pd.read_parquet('data/work_flseq/feat_train.parquet',columns=cols); b=pd.read_parquet('data/work_flseq/feat_rank.parquet',columns=cols)
a['y']=2025; b['y']=2026; d=pd.concat([a,b]); d['m']=pd.to_datetime(d.MVT_TIME_UTC_mvt).dt.month
g=d.groupby(['ADEP_mvt','y','m']).agg(na_prev=('seq_arr_prev_s',lambda s:s.isna().mean()),prev_med=('seq_arr_prev_s','median'),pm10=('seq_arr_pm10m','mean'),occ_arr2=('occ_g_arr_prev2','median'),occ_dep=('occ_g_dep_prev','median'),stand_na=('STAND_mvt',lambda s:s.isna().mean()))
g=g.reset_index(); g=g[g.m.isin([1,6,7,8,12])]
print(g.pivot_table(index=['ADEP_mvt'],columns=['y','m'],values='na_prev').round(2).to_string())
print(g.pivot_table(index=['ADEP_mvt'],columns=['y','m'],values='occ_arr2').round(0).to_string())
print(g.pivot_table(index=['ADEP_mvt'],columns=['y','m'],values='pm10').round(2).to_string())
