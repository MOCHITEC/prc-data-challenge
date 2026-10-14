import pandas as pd, numpy as np, warnings; warnings.filterwarnings('ignore')
cols=['MVT_ID_mvt','ADEP_mvt','MVT_TIME_UTC_mvt','aobt3_minus_eobt1','aobt3_minus_sched','mvt_minus_aobt3','TAXITIME_SEC_mvt']
a=pd.read_parquet('data/work_flseq/feat_train.parquet',columns=cols)
a=a[a.mvt_minus_aobt3.notna()&a.aobt3_minus_eobt1.notna()]
a['res']=(a.TAXITIME_SEC_mvt-a.mvt_minus_aobt3).clip(-1200,1200)
a['de']=a.aobt3_minus_eobt1.clip(-1800,3600); a['ds']=a.aobt3_minus_sched.clip(-1800,3600)
import numpy.linalg as la
for ap,x in a.groupby('ADEP_mvt'):
    y=x[(x.de!=x.ds)]
    X=np.c_[np.ones(len(y)),y.de,y.ds]; c=la.lstsq(X,y.res,rcond=None)[0]
    print(ap,len(x),'diff share',round(len(y)/len(x),2),'coef eobt',round(c[1],3),'sched',round(c[2],3))
