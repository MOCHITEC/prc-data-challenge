import os; os.environ["POLARS_MAX_THREADS"]="2"
import polars as pl, numpy as np, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
B,O,A="#2a78d6","#eb6834","#1baf7a"; INK="#0b0b0b"; INK2="#52514e"; GRID="#e4e3df"
plt.rcParams.update({"font.size":9,"axes.edgecolor":INK2,"axes.labelcolor":INK,"xtick.color":INK2,"ytick.color":INK2,
  "axes.grid":True,"grid.color":GRID,"grid.linewidth":0.6,"axes.spines.top":False,"axes.spines.right":False,"figure.facecolor":"#fcfcfb","axes.facecolor":"#fcfcfb"})
os.makedirs("fig",exist_ok=True)
D=pl.read_parquet("slim.parquet"); dep=D.filter(pl.col("PHASE_mvt")=="DEP")
APTS=sorted(dep["APT"].unique().to_list())
cs=pl.read_csv("out/cell_signature_by_airport.csv")
# ---- copy-excess metric: count |d_ba|<=6 vs same-width windows at +-60,+-120
rows=[]
for a in APTS:
    x=dep.filter(pl.col("APT")==a)["d_ba"].drop_nulls().to_numpy()
    c0=(np.abs(x)<=6).sum(); nb=np.mean([(np.abs(x-k)<=6).sum() for k in (-120,-60,60,120)])
    rows.append(dict(APT=a,n=len(x),within6=c0/len(x),neighbour_mean=nb/len(x),ratio=c0/nb))
ce=pl.DataFrame(rows); ce.write_csv("out/copy_excess.csv"); print(ce)
# ---- Fig1: histograms of BLOCK-AOBT3
fig,ax=plt.subplots(2,5,figsize=(13,5),sharex=True)
for i,a in enumerate(APTS):
    x=dep.filter(pl.col("APT")==a)["d_ba"].drop_nulls().to_numpy()/60
    h=ax.flat[i]; h.hist(x,bins=np.arange(-15.5,15.6,1/3),color=B,edgecolor="#fcfcfb",linewidth=0.3)
    q=np.percentile(x,[25,50,75]); h.axvline(0,color=INK2,lw=0.8,ls=":")
    h.set_title(f"{a}  med {q[1]*60:+.0f}s  IQR {(q[2]-q[0])*60:.0f}s",fontsize=8.5,color=INK)
    h.set_yticks([])
fig.supxlabel("BLOCK − AOBT_3 (min)  [airport off-block minus NM off-block; 20-s bins, |Δ|≤15 min shown]")
fig.suptitle("DEP: airport off-block vs NM AOBT_3 — no copy spike at 0 at any airport",color=INK)
fig.tight_layout(); fig.savefig("fig/fig1_block_minus_aobt3_hist.png",dpi=150); plt.close()
# ---- Fig2: 2D density y (airport taxi) vs d_mt (NM taxi), within plausible range
fig,ax=plt.subplots(2,5,figsize=(14,6),sharex=True,sharey=True)
for i,a in enumerate(APTS):
    x=dep.filter((pl.col("APT")==a)&pl.col("d_mt").is_not_null()&pl.col("y").is_not_null())
    h=ax.flat[i]; H,xe,ye=np.histogram2d(x["y"].to_numpy()/60,x["d_mt"].to_numpy()/60,bins=[np.arange(0,45.25,0.5),np.arange(0,45.25,0.5)])
    h.imshow(np.log1p(H.T),origin="lower",extent=[0,45,0,45],cmap="Blues",aspect="auto"); h.grid(False)
    h.plot([0,45],[0,45],color=O,lw=0.8,ls="--")
    r=cs.filter(pl.col("APT")==a)
    h.set_title(f"{a}  within-cell r={r['within_corr'][0]:.2f}",fontsize=8.5,color=INK)
fig.supxlabel("airport taxi-out  MVT − BLOCK (min)"); fig.supylabel("NM taxi-out  MVT − AOBT_3 (min)")
fig.suptitle("Horizontal bands = NM taxi independent of airport taxi (AOBT_3 = take-off − model taxi time); dashed = identity",color=INK)
fig.tight_layout(); fig.savefig("fig/fig2_airport_vs_nm_taxi_density.png",dpi=150); plt.close()
# ---- Fig3: cell-mode concentration
cs=cs.sort("APT"); xs=np.arange(len(cs)); w=0.38
fig,ax=plt.subplots(1,2,figsize=(12,4))
ax[0].bar(xs-w/2-0.01,cs["inNM"],w,color=B,label="NM: MVT − AOBT_3"); ax[0].bar(xs+w/2+0.01,cs["inAPT"],w,color=O,label="airport: MVT − BLOCK")
ax[0].set_xticks(xs,cs["APT"]); ax[0].set_ylim(0,1); ax[0].set_ylabel("share in modal 2-min pair of its cell"); ax[0].legend(frameon=False,loc="upper left",fontsize=8)
ax[0].set_title("Concentration within stand×runway×month cells",color=INK)
for xx,v in zip(xs,cs["inNM"]): ax[0].text(xx-w/2,v+0.015,f"{v:.2f}",ha="center",fontsize=7,color=INK2)
ax[1].bar(xs,cs["within_corr"],0.6,color=B); ax[1].axhline(0,color=INK2,lw=0.8)
ax[1].set_xticks(xs,cs["APT"]); ax[1].set_ylabel("corr(NM taxi, airport taxi), cell-demeaned"); ax[1].set_title("Does NM taxi track the airport-measured taxi?",color=INK)
for xx,v in zip(xs,cs["within_corr"]): ax[1].text(xx,v+(0.01 if v>=0 else -0.04),f"{v:.2f}",ha="center",fontsize=7,color=INK2)
fig.tight_layout(); fig.savefig("fig/fig3_derived_signature.png",dpi=150); plt.close()
# ---- Fig4: monthly
mc=pl.read_csv("out/monthly_copy_classes.csv"); cm=pl.read_csv("out/cell_signature_by_month.csv")
fig,ax=plt.subplots(2,5,figsize=(14,5.5),sharex=True,sharey=True)
for i,a in enumerate(APTS):
    h=ax.flat[i]; c=cm.filter(pl.col("APT")==a).sort("month")
    h.plot(c["month"],c["inNM"],color=B,lw=2,marker="o",ms=3,label="NM taxi in cell mode (derived signature)")
    h.plot(c["month"],c["within_corr"],color=A,lw=2,marker="o",ms=3,label="within-cell corr NM vs airport taxi")
    g=mc.filter((pl.col("APT")==a)&(pl.col("cls")=="agree_60s")).sort("month")
    h.plot(g["month"],g["sh"],color=O,lw=2,marker="o",ms=3,label="|BLOCK − AOBT_3| ≤ 60 s")
    h.set_title(a,fontsize=9,color=INK); h.set_ylim(-0.35,1.02); h.set_xticks([1,4,7,10,12])
ax.flat[0].legend(frameon=False,fontsize=7,loc="lower left")
fig.supxlabel("month 2025"); fig.suptitle("Monthly clock signatures (DEP). Winter rise of correlation at EDDF/LFPG/LSZH/EHAM/EGLL consistent with de-icing time entering NM taxi (inference)",color=INK)
fig.tight_layout(); fig.savefig("fig/fig4_monthly_signatures.png",dpi=150); plt.close()
# ---- Fig5: ARR vs DEP agreement
t=pl.read_csv("out/arr_vs_dep_clock_agreement.csv").sort("APT"); xs=np.arange(len(t))
fig,ax=plt.subplots(figsize=(10,3.8))
ax.bar(xs-w/2-0.01,t["w60"],w,color=B,label="ARR: |MVT(landing) − ARVT_3| ≤ 60 s"); ax.bar(xs+w/2+0.01,t["dep_w60"],w,color=O,label="DEP: |BLOCK − AOBT_3| ≤ 60 s")
ax.set_xticks(xs,t["APT"]); ax.set_ylim(0,1.05); ax.set_ylabel("share of rows"); ax.legend(frameon=False,fontsize=8,loc="upper right",ncol=2)
for xx,v in zip(xs,t["w60"]): ax.text(xx-w/2,v+0.01,f"{v:.2f}",ha="center",fontsize=7,color=INK2)
for xx,v in zip(xs,t["dep_w60"]): ax.text(xx+w/2,v+0.01,f"{v:.2f}",ha="center",fontsize=7,color=INK2)
ax.set_title("Arrival (landing) clocks agree; departure off-block clocks do not",color=INK)
fig.tight_layout(); fig.savefig("fig/fig5_arr_vs_dep_agreement.png",dpi=150); plt.close()
