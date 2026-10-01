import polars as pl, numpy as np
from scipy.stats import fisher_exact, mannwhitneyu
pl.Config.set_tbl_rows(100); pl.Config.set_tbl_cols(40); pl.Config.set_tbl_width_chars(300)
d=pl.read_parquet('dep4.parquet')
d=d.with_columns(glitch=pl.col('scls').is_in(['dup1','dup2']), r16R=pl.col('RUNWAY_mvt')=='16R', r34L=pl.col('RUNWAY_mvt')=='34L',
   ades_eu=pl.col('ADES_mvt').str.slice(0,1).is_in(['E','L']), sh=pl.col('sl').dt.hour(), mh=pl.col('ml').dt.hour(),
   st1=pl.col('STAND_mvt').str.slice(0,1), wd=pl.col('sl').dt.weekday(), mon=pl.col('sl').dt.month(), suff=pl.col('suff_letter'),
   flen=pl.col('flen'), cross=pl.col('dcross_loc0')>=1, ms_h=pl.col('ms')/3600, summer=pl.col('sl').dt.month().is_in([4,5,6,7,8,9,10]))
d.write_parquet('dep5.parquet')
t=d.filter((pl.col('src')=='train')&pl.col('unm')&(pl.col('ms')>=43200))
y=(t['cls']=='roll').to_numpy()
print('n',len(y),'roll',y.sum())
for c in ['glitch','r16R','r34L','ades_eu','suff','cross','summer']:
    x=t[c].fill_null(False).to_numpy()
    tab=[[int((x&y).sum()),int((x&~y).sum())],[int((~x&y).sum()),int((~x&~y).sum())]]
    print(f'{c:8s} roll|x={tab[0][0]}/{tab[0][0]+tab[0][1]}  roll|~x={tab[1][0]}/{tab[1][0]+tab[1][1]}  fisher p={fisher_exact(tab)[1]:.3f}')
for c in ['ms_h','sh','mh','flen','ndig','wd']:
    x=t[c].to_numpy().astype(float)
    print(f'{c:6s} roll med {np.median(x[y]):.1f} fill med {np.median(x[~y]):.1f}  MWU p={mannwhitneyu(x[y],x[~y]).pvalue:.3f}')
print(t.group_by('st1').agg(pl.len(),(pl.col('cls')=='roll').sum()).sort('st1'))
print(t.group_by('p3').agg(pl.len(),(pl.col('cls')=='roll').sum()).sort('len',descending=True))
