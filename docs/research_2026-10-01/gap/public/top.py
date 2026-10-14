import json,collections
R=json.load(open("lb.json"))
R=[r for r in R if r.get("score") is not None]
best={};n=collections.Counter();b929={}
for r in R:
    t=r["teamName"];n[t]+=1
    if t not in best or r["score"]<best[t]["score"]: best[t]=r
    if r["processedAt"]<"2026-09-29T00:31" and (t not in b929 or r["score"]<b929[t]): b929[t]=r["score"]
print("max processedAt",max(r["processedAt"] for r in R))
L=sorted(best.values(),key=lambda r:r["score"])
for i,r in enumerate(L[:40],1):
    t=r["teamName"]; 
    print(i,t,r["score"],n[t],r["filename"],r["processedAt"][:16],"09-29:",b929.get(t),"gain",round(b929.get(t,999)-r["score"],2))
print([ (i+1,r["score"]) for i,r in enumerate(L) if r["teamName"]=="resilient-hedgehog"], len(L))
