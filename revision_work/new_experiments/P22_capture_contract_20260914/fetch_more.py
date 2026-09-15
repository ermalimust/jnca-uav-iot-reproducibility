from pathlib import Path
import concurrent.futures, datetime, hashlib, json, urllib.request
R=Path(__file__).resolve().parent;O=R/'sources';O.mkdir(exist_ok=True)
targets={
 'wichoose_tree_page1.json':'https://gitlab.com/api/v4/projects/58506285/repository/tree?recursive=true&per_page=100&page=1',
 'wichoose_tree_page2.json':'https://gitlab.com/api/v4/projects/58506285/repository/tree?recursive=true&per_page=100&page=2',
 'wichoose_commit.json':'https://gitlab.com/api/v4/projects/58506285/repository/commits/main',
 'wiperf_zenodo.json':'https://zenodo.org/api/records/6761916',
 'comcom2024.pdf':'https://www.cs.vassar.edu/~rpachecomeireles/research/papers/comcom-2024-preprint.pdf'
}
def get(item):
 name,url=item;receipt=dict(file=name,url=url,accessed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
 try:
  req=urllib.request.Request(url,headers={'User-Agent':'Research-source-verification/1.0'})
  with urllib.request.urlopen(req,timeout=30) as response:data=response.read(10_000_001)
  assert len(data)<=10_000_000
  (O/name).write_bytes(data);receipt.update(status='retrieved',bytes=len(data),sha256=hashlib.sha256(data).hexdigest())
 except Exception as e:receipt.update(status='unavailable',error=str(e))
 return receipt
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:result=list(pool.map(get,targets.items()))
(O/'fetch_more_receipt.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(result,ensure_ascii=False),flush=True)
