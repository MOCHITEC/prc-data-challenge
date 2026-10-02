import json
R=[r for r in json.load(open("lb.json")) if r.get("score") is not None]
for t in ["adventurous-radio","bold-ladder","strong-moose","charming-eel","versatile-watermelon","strong-crane","jolly-lobster","vigorous-whistle","gracious-tornado","kind-mango"]:
    s=sorted([r for r in R if r["teamName"]==t],key=lambda r:r["processedAt"])
    best=1e9; out=[]
    for r in s:
        if r["processedAt"]<"2026-09-27": 
            best=min(best,r["score"]); continue
        mark="*" if r["score"]<best else ""
        best=min(best,r["score"])
        out.append(f'{r["processedAt"][5:16]} {r["filename"].split("_")[-1][:-8]} {r["score"]:.2f}{mark}')
    print(t, len(s)); print("   "+" | ".join(out))
