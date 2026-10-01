from common import *
df=pl.read_parquet('train_dep_cls.parquet')
print(df.group_by('APT','unm').agg(n=pl.len(), fill=(pl.col('cls')=='fill').mean()).sort('APT','unm'))
x=df.filter((pl.col('APT')=='LIRF')&pl.col('unm')).with_columns(msh=(pl.col('ms')/3600).floor().clip(-3,40))
print(x.group_by('msh').agg(n=pl.len(), fill=(pl.col('cls')=='fill').sum(), roll=(pl.col('cls')=='roll').sum(), big=(pl.col('cls')=='big').sum()).sort('msh'))
print(x.group_by('month').agg(n=pl.len(), fill=(pl.col('cls')=='fill').mean(), fill12=((pl.col('cls')=='fill')&(pl.col('ms')>43200)).sum(), roll=(pl.col('cls')=='roll').sum()).sort('month'))
# seconds of minute
x=x.with_columns(ms_sec=pl.col('M').dt.second(), s_sec=pl.col('S').dt.second(), b_sec=pl.col('B').dt.second())
print(x.group_by('cls').agg(n=pl.len(), msec_le6=((pl.col('ms_sec')<=6)|(pl.col('ms_sec')>=54)).mean(), bsec_le6=((pl.col('b_sec')<=6)|(pl.col('b_sec')>=54)).mean(), ssec0=(pl.col('s_sec')==0).mean()))
