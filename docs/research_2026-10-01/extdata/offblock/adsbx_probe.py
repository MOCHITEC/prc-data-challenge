import requests, gzip, json, math, sys, concurrent.futures as cf, collections
AP={'EDDF':(50.0333,8.5706),'EDDM':(48.3538,11.7861),'EGLL':(51.4706,-0.4619),'EHAM':(52.3086,4.7639),'LEBL':(41.2971,2.0785),
    'LEMD':(40.4719,-3.5626),'LFPG':(49.0097,2.5479),'LIRF':(41.8003,12.2389),'LSZH':(47.4647,8.5492),'LTFM':(41.2753,28.7519)}
def dist(a,b,c,d):
    return 6371*2*math.asin(math.sqrt(math.sin(math.radians(c-a)/2)**2+math.cos(math.radians(a))*math.cos(math.radians(c))*math.sin(math.radians(d-b)/2)**2))
def get(day,hh):
    u=f"https://samples.adsbexchange.com/readsb-hist/{day}/{hh:02d}0000Z.json.gz"
    r=requests.get(u,timeout=120)
    if r.status_code!=200: return day,hh,None
    b=r.content
    try: b=gzip.decompress(b)
    except Exception: pass
    j=json.loads(b)
    out={}
    for ap,(la,lo) in AP.items():
        g=s=0
        for a in j.get('aircraft',[]):
            if 'lat' not in a: continue
            if dist(la,lo,a['lat'],a['lon'])<4:
                if a.get('alt_baro')=='ground':
                    g+=1; s+= (a.get('gs') or 0)>1
        out[ap]=(g,s)
    return day,hh,out
days=sys.argv[1:]
res=collections.defaultdict(lambda: collections.Counter())
with cf.ThreadPoolExecutor(8) as ex:
    for day,hh,out in ex.map(lambda t:get(*t),[(d,h) for d in days for h in range(5,22,2)]):
        if out is None: print(day,hh,'missing'); continue
        for ap,(g,s) in out.items(): res[day][ap]+=g; res[day][ap+'_mv']+=s
for d in days:
    print(d,' '.join(f"{ap}:{res[d][ap]}/{res[d][ap+'_mv']}" for ap in AP))
