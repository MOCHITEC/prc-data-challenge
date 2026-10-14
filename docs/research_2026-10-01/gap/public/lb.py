import requests, json, collections
U="https://datacomp.opensky-network.org/api/competitions/bb3693e1-26bc-4a9e-8619-4fe78b4eab0c/leaderboard"
rows=[];cur=None
while True:
    p={"limit":200}
    if cur: p["cursor"]=cur
    r=requests.get(U,params=p,timeout=60).json()
    items=r.get("items") or r.get("data") or r.get("entries") or r.get("submissions")
    if items is None: print(list(r.keys())); break
    rows+=items; cur=r.get("nextCursor")
    if not cur: break
json.dump(rows,open("lb.json","w"))
print(len(rows), rows[0])
