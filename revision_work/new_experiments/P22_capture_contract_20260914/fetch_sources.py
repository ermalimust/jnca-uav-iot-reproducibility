from pathlib import Path
import concurrent.futures, datetime, hashlib, json, urllib.request
R=Path(__file__).resolve().parent;O=R/'sources';O.mkdir(exist_ok=True)
targets={
 'wichoose_readme.md':'https://gitlab.com/rui-meireles/wichoose/-/raw/main/README.md',
 'wichoose_project.json':'https://gitlab.com/api/v4/projects/rui-meireles%2Fwichoose',
 'author_homepage.html':'https://www.cs.vassar.edu/~rpachecomeireles/'
}
def get(item):
 name,url=item;receipt=dict(file=name,url=url,accessed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
 try:
  req=urllib.request.Request(url,headers={'User-Agent':'Research-source-verification/1.0'})
  with urllib.request.urlopen(req,timeout=25) as response:data=response.read(5_000_001)
  assert len(data)<=5_000_000
  (O/name).write_bytes(data)
  receipt.update(status='retrieved',bytes=len(data),sha256=hashlib.sha256(data).hexdigest())
 except Exception as e:receipt.update(status='unavailable',error=str(e))
 return receipt
with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:result=list(pool.map(get,targets.items()))
(O/'fetch_receipt.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(result,ensure_ascii=False),flush=True)
