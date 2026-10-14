import requests,base64,zipfile,io,json,re,polars as pl,math,collections
W='/home/user/prc-data-challenge/data/work_flseq/feat_rank.parquet'
rk=pl.read_parquet(W,columns=['ADEP_mvt','STAND_mvt'])
norm=lambda s: re.sub(r'^0+','',re.sub(r'[^A-Z0-9]','',str(s).upper()))
for ap in ['EDDF','EDDM','EGLL','EHAM','LEBL','LEMD','LFPG','LIRF','LSZH','LTFM']:
    a=requests.get(f'https://gateway.x-plane.com/apiv1/airport/{ap}',timeout=60).json()['airport']
    sid=a.get('recommendedSceneryId')
    s=requests.get(f'https://gateway.x-plane.com/apiv1/scenery/{sid}',timeout=120).json()['scenery']
    z=zipfile.ZipFile(io.BytesIO(base64.b64decode(s['masterZipBlob'])))
    name=[n for n in z.namelist() if n.endswith('.dat')][0]
    txt=z.read(name).decode('latin-1')
    open(f'xp/{ap}.dat','w').write(txt)
    stands={}
    for line in txt.splitlines():
        p=line.split()
        if p and p[0]=='1300':
            nm=' '.join(p[6:]) if len(p)>6 else ''
            stands[norm(nm.split()[-1] if nm else '')]=(float(p[1]),float(p[2]))
            # try full name normalized too
            stands.setdefault(norm(nm),(float(p[1]),float(p[2])))
    st=rk.filter(pl.col('ADEP_mvt')==ap)['STAND_mvt'].cast(pl.Utf8).to_list()
    m=sum(1 for x in st if x and norm(x) in stands)
    n1300=sum(1 for l in txt.splitlines() if l.startswith('1300 '))
    ex=[l for l in txt.splitlines() if l.startswith('1300 ')][:2]
    print(ap,'scenery',sid,s.get('dateUploaded','')[:10],'n1300',n1300,'rank-row match',round(m/len(st),3),ex)
