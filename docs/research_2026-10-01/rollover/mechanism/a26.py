import numpy as np
N=344841; base=252.43
YR=87300.
rows=[('WMTT3508',56520,1,33444),('TAY9LA',53520,0,30644),('WMT402',48841,0,36782),('WMTMT4SG',47705,1,35434),('NOSOS431',111654,1,39636),
('ITYTY672',61322,1,66037),('RYR4852',48539,0,33485),('TARAR753',48776,1,35683),('SVII3006',60656,1,31551),('ITYTY674',47337,1,66693),('QFAA7006',87484,1,32688)]
P={'[12,14)':(0.05,0.0,0.19),'[14,16)':(0.31,0.09,0.59),'[16,28)':(0.73,0.52,0.90),'>=28':(0.25,0.0,0.77)}
def band(ms):
    h=ms/3600
    return '[12,14)' if h<14 else '[14,16)' if h<16 else '[16,28)' if h<28 else '>=28'
tot={'ms':0,'main':0}; totlo=0; tothi=0
print('flight ms band dd p yhat E[SE|ms] E[SE|main] E[SE|yhat] gain_vs_ms gain_vs_main')
def ese(c,p,ms): return p*(YR-c)**2+(1-p)*(ms-c)**2
dSSE_ms=[];dSSE_main=[]
for f,ms,dd,mp in rows:
    b=band(ms); p,lo,hi=P[b]
    yh=p*YR+(1-p)*ms
    g1=ese(ms,p,ms)-ese(yh,p,ms); g2=ese(mp,p,ms)-ese(yh,p,ms)
    dSSE_ms.append(g1); dSSE_main.append(g2)
    print(f,ms,b,dd,p,round(yh),'%.2e %.2e %.2e'%(ese(ms,p,ms),ese(mp,p,ms),ese(yh,p,ms)),'%.2e'%g1,'%.2e'%g2)
def drmse(dsse): 
    mse=base**2; return np.sqrt(mse-dsse/N)-base
print('expected dRMSE vs ms-pred: %.2f s; vs doc-main-pred: %.2f s'%(drmse(sum(dSSE_ms)),drmse(sum(dSSE_main))))
# only band [16,28) rows overlay
s=[g for (r,g) in zip(rows,dSSE_main) if band(r[1])=='[16,28)']
print('only [16,28) rows vs main: %.2f s'%drmse(sum(s)))
# worst/best case for [16,28) with p at CI ends: realized outcomes enumeration for the 3 rows using p=0.73 overlay
import itertools
r3=[r for r in rows if band(r[1])=='[16,28)']
for outc in itertools.product([0,1],repeat=3):
    d=0
    for (f,ms,dd,mp),o in zip(r3,outc):
        yt=YR if o else ms; yh=0.73*YR+0.27*ms
        d+=(mp-yt)**2-(yh-yt)**2
    print(outc,'realized dRMSE %.2f'%drmse(d))
# ITY340
for p in [0.5,0.75,0.9]:
    yr=86460; yf=94560; yn=800; pf=0.01
    pr=p*(1-pf); pn=1-p
    yh=pr*yr+pf*yf+pn*yn
    for c in [800, 86340]:
        e_c=pr*(yr-c)**2+pf*(yf-c)**2+pn*(yn-c)**2; e_h=pr*(yr-yh)**2+pf*(yf-yh)**2+pn*(yn-yh)**2
        print('ITY340 p_bigday=%.2f yhat=%d current=%d expected dRMSE %.2f'%(p,yh,c,drmse(e_c-e_h)))
