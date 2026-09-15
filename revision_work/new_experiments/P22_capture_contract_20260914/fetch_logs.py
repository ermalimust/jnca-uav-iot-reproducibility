from pathlib import Path
from urllib.parse import quote
import concurrent.futures, datetime, hashlib, json, urllib.request
R=Path(__file__).resolve().parent;O=R/'sources'
commit=json.loads((O/'wichoose_commit.json').read_text())['id']
tree=sum([json.loads(p.read_text()) for p in sorted(O.glob('wichoose_fixed_tree_*.json'))],[])
targets=[x for x in tree if x['type']=='blob' and x['path'].startswith('exp-logs/') and x['path'].endswith('.csv')]
def fetch(x):
 path=x['path'];url=f'https://gitlab.com/api/v4/projects/58506285/repository/files/{quote(path,safe="")}/raw?ref={commit}'
 dest=(O/'wichoose'/path).resolve();assert dest.is_relative_to((O/'wichoose').resolve())
 with urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'Research-source-verification/1.0'}),timeout=30) as response:data=response.read(5000001)
 assert len(data)<=5000000
 dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(data)
 return dict(path=path,url=url,bytes=len(data),sha256=hashlib.sha256(data).hexdigest(),blob_id=x['id'])
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:result=list(pool.map(fetch,targets))
(O/'logs_receipt.json').write_text(json.dumps(dict(commit=commit,files=result,accessed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat()),indent=2),encoding='utf-8')
print(json.dumps(dict(files=len(result),bytes=sum(x['bytes'] for x in result))),flush=True)
