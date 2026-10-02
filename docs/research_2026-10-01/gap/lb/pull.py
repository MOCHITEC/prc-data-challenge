import requests, json, polars as pl, time
U="https://datacomp.opensky-network.org/api/competitions/bb3693e1-26bc-4a9e-8619-4fe78b4eab0c/leaderboard"
items=[];cur=None;pages=0
while True:
    p={"limit":200}
    if cur: p["cursor"]=cur
    r=requests.get(U,params=p,timeout=60);r.raise_for_status();j=r.json()
    items+=j["items"];pages+=1;cur=j.get("nextCursor")
    if not cur or not j["items"]: break
json.dump(items,open("lb_raw.json","w"))
df=pl.DataFrame(items).with_columns(pl.col("processedAt").str.to_datetime(time_zone="UTC"))
df.write_parquet("lb_raw.parquet")
print(pages,df.shape,df["teamName"].n_unique(),df["processedAt"].max(), df["submissionId"].n_unique())
