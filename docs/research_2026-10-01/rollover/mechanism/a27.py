import numpy as np
N=344841; base=284.36; YR=87300.
rows=[('WMTT3508',56520,33444),('TAY9LA',53520,30644),('WMT402',48841,36782),('WMTMT4SG',47705,35434),('NOSOS431',111654,39636),
('ITYTY672',61322,66037),('RYR4852',48539,33485),('TARAR753',48776,35683),('SVII3006',60656,31551),('ITYTY674',47337,66693),('QFAA7006',87484,32688)]
def band_p(ms):
    h=ms/3600; return 0.05 if h<14 else 0.31 if h<16 else 0.73 if h<28 else 0.25
def ese(c,p,ms,pn=0, yn=1000): return (1-pn)*(p*(YR-c)**2+(1-p)*(ms-c)**2)+pn*(yn-c)**2
for pn in [0,0.25,0.5,0.75,1.0]:
    d=0
    for f,ms,mp in rows:
        o=0.379*YR+0.621*ms; p=band_p(ms)
        d+=ese(o,p,ms,pn)-ese(mp,p,ms,pn)
    print('P(normal taxi)=%.2f  expected dRMSE of the failed overlay vs main: %+.2f s'%(pn, np.sqrt(base**2+d/N)-base))
