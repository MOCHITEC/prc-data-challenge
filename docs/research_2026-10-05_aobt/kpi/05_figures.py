import polars as pl, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.rcParams.update({"font.size":9,"axes.spines.top":False,"axes.spines.right":False,"axes.edgecolor":"#888","axes.grid":True,
                     "grid.color":"#e6e6e3","grid.linewidth":0.6,"axes.axisbelow":True,"figure.facecolor":"#fcfcfb","axes.facecolor":"#fcfcfb"})
C1,C2,C3,G="#2a78d6","#eb6834","#1baf7a","#8a8985"
c=pl.col
A=pl.read_csv("out/scenarios_annual_deice.csv")
rep=pl.read_csv("out/replication_monthly.csv").filter(c("variant")=="FY_deice")
pru_an=rep.group_by("ADEP_mvt").agg(pru=(c("pru_add")*c("pru_nref")).sum()/c("pru_nref").sum())
# Fig1: per-airport KPI: PRU published, replication, cleaned (ALL_fill both), REPAIR
b=A.filter(c("scen")=="BASE").select("ADEP_mvt",c("add").alias("base"))
cl=A.filter((c("scen")=="ALL_fill")&(c("mode")=="both")).select("ADEP_mvt",c("add").alias("clean"))
rp=A.filter(c("scen")=="REPAIR_fill_with_AOBT3").select("ADEP_mvt",c("add").alias("repair"))
d=b.join(cl,on="ADEP_mvt").join(rp,on="ADEP_mvt").join(pru_an,on="ADEP_mvt").sort("base",descending=True)
x=np.arange(d.height); w=0.2
fig,ax=plt.subplots(figsize=(9,3.8))
for i,(col,lab,colr) in enumerate([("pru","PRU published 2025",G),("base","Replication (this data)",C1),("clean","Excl. fills/suspect plan copies",C2),("repair","Fills repaired with NM AOBT",C3)]):
    ax.bar(x+(i-1.5)*w, d[col], w-0.02, label=lab, color=colr)
ax.set_xticks(x, d["ADEP_mvt"]); ax.set_ylabel("Additional taxi-out time (min/dep)")
ax.set_title("Annual 2025 additional taxi-out time per airport: original vs cleaned AOBT", loc="left")
for i,(bv,cv) in enumerate(zip(d["base"],d["clean"])):
    if abs(cv-bv)>=0.1: ax.annotate(f"{cv-bv:+.2f}", (i, max(bv,cv)+0.15), ha="center", fontsize=8, color="#333")
ax.set_ylim(0,8.2); ax.legend(frameon=False, ncol=4, loc="upper right", fontsize=8); fig.tight_layout(); fig.savefig("fig1_kpi_original_vs_cleaned.png",dpi=160)
# Fig2: share of suspect records per airport (stacked, mutually exclusive priority)
df=pl.read_csv("out/flag_counts_airport.csv").sort("ADEP_mvt")
fc=pl.read_csv("out/flag_share_pct_airport.csv").sort("ADEP_mvt")
ex=pl.read_csv("out/copy_excess_mass_pct.csv").filter(c("src")=="SCHED").sort("ADEP_mvt")
fig,axs=plt.subplots(1,2,figsize=(10,3.8),gridspec_kw=dict(width_ratios=[1.4,1]))
ax=axs[0]; x=np.arange(fc.height)
ax.bar(x-0.2, fc["F_PLAN_6"], 0.38, color=C1, label="BLOCK = SCHED/EOBT/LOBT/IOBT (±6 s)")
ax.bar(x+0.2, fc["F_PLANFILL"], 0.38, color=C2, label="…and NM AOBT >5 min later (fill)")
ax.plot(x-0.2, ex["excess"], "D", color="#0b0b0b", ms=5, label="Excess spike over neighbours (SCHED)")
ax.set_xticks(x, fc["ADEP_mvt"]); ax.set_ylabel("% of 2025 departures"); ax.set_ylim(0,30); ax.legend(frameon=False,fontsize=8,loc="upper left")
ax.set_title("Planned-time copies in reported AOBT\n(at minute-grid airports ±6 s = the whole 0-min bin, so bars include natural on-time pushes)", loc="left", fontsize=8.5)
ax=axs[1]
ax.bar(x-0.2, fc["F_A3_60"], 0.38, color=G, label="|BLOCK − NM AOBT| ≤ 60 s")
ax.bar(x+0.2, fc["F_A3_5"], 0.38, color=C1, label="≤ 5 s")
ax.set_xticks(x, fc["ADEP_mvt"], rotation=45); ax.legend(frameon=False,fontsize=8); ax.set_title("Agreement with NM AOBT", loc="left")
fig.tight_layout(); fig.savefig("fig2_suspect_share_per_airport.png",dpi=160)
# Fig3: replication vs PRU monthly
fig,ax=plt.subplots(figsize=(4.6,4.4))
r0=pl.read_csv("out/replication_monthly.csv")
for v,colr,lab in [("FY",G,"no de-icing proxy"),("FY_deice",C1,"with METAR de-icing proxy")]:
    s=r0.filter(c("variant")==v); ax.scatter(s["pru_add"],s["add"],s=10,color=colr,label=lab,alpha=0.8)
lim=[2,9]; ax.plot(lim,lim,color="#333",lw=0.8); ax.set_xlim(lim); ax.set_ylim(lim)
ax.set_xlabel("PRU published (min/dep)"); ax.set_ylabel("Replication (min/dep)"); ax.legend(frameon=False,fontsize=8)
ax.set_title("Monthly 2025, 10 airports × 12 months", loc="left"); fig.tight_layout(); fig.savefig("fig3_replication_vs_pru.png",dpi=160)
# Fig4: LIRF monthly
M=pl.read_csv("out/scenarios_monthly_deice.csv").filter(c("ADEP_mvt")=="LIRF")
fig,ax=plt.subplots(figsize=(7,3.4))
p=rep.filter(c("ADEP_mvt")=="LIRF").sort("lmon"); ax.plot(p["lmon"],p["pru_add"],color=G,lw=2,label="PRU published")
for s,mo,colr,lab in [("BASE","-",C1,"Replication"),("ALL_fill","both",C2,"Excl. fills"),("REPAIR_fill_with_AOBT3","both",C3,"Fills repaired")]:
    q=M.filter((c("scen")==s)&(c("mode")==mo)).sort("lmon"); ax.plot(q["lmon"],q["add"],color=colr,lw=2,marker="o",ms=4,label=lab)
ax.set_xticks(range(1,13)); ax.set_xlabel("Month 2025"); ax.set_ylabel("min/dep"); ax.legend(frameon=False,fontsize=8,ncol=2,loc="upper left")
ax.set_title("LIRF additional taxi-out time by month", loc="left"); fig.tight_layout(); fig.savefig("fig4_LIRF_monthly.png",dpi=160)
print("ok")
