"""Walk tar headers of a split adsb.lol day by 512-byte range requests until the first ./traces/ member."""
import sys, subprocess, json
def head(u):
    r = subprocess.run(["curl","-sS","-m","60","-o","/dev/null","-I","-L","-w","%{http_code} %header{content-length}",u],capture_output=True,text=True)
    c,*cl = r.stdout.split(); return int(cl[-1]) if c=="200" else None
def parts(day):
    yr=day[:4]; tag=f"v{day}-planes-readsb-prod-0"; base=f"https://github.com/adsblol/globe_history_{yr}/releases/download/{tag}/{tag}"
    out=[]
    for s in ["aa","ab","ac","ad"]:
        n=head(f"{base}.tar.{s}")
        if n is None: break
        out.append((f"{base}.tar.{s}",n))
    if not out:
        n=head(f"{base}.tar"); out=[(f"{base}.tar",n)] if n else []
    return out
def rng(ps, off, n):
    for u,sz in ps:
        if off < sz:
            return subprocess.run(["curl","-sS","-L","-r",f"{off}-{off+n-1}",u],capture_output=True).stdout
        off -= sz
def walk(day, maxsteps=400):
    ps=parts(day); off=0; longname=None; members=[]
    for _ in range(maxsteps):
        h=rng(ps,off,512)
        if not h or h==b"\0"*512: break
        name=h[:100].split(b"\0")[0].decode(); size=int(h[124:136].split(b"\0")[0].strip() or b"0",8); typ=h[156:157]
        if typ==b"L":
            longname=rng(ps,off+512,size).split(b"\0")[0].decode(); off+=512+(size+511)//512*512; continue
        if longname: name,longname=longname,None
        members.append((name,size,off))
        if "/traces/" in name and name.endswith(".json"): break
        off+=512+(size+511)//512*512
    return ps, members
if __name__=="__main__":
    ps,m=walk(sys.argv[1])
    print(json.dumps({"parts":[(u.rsplit('/',1)[1],s) for u,s in ps],"n":len(m),"tail":m[-5:]}))
