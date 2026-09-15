from pathlib import Path
from urllib.parse import quote
import concurrent.futures, datetime, hashlib, json, urllib.request
R=Path(__file__).resolve().parent;O=R/'sources'
commit=json.loads((O/'wichoose_commit.json').read_text())['id']
def download(url,limit=10000000):
 with urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'Research-source-verification/1.0'}),timeout=30) as response:data=response.read(limit+1)
 assert len(data)<=limit,(url,'size cap')
 return data
tree=[]
for page in range(1,5):
 url=f'https://gitlab.com/api/v4/projects/58506285/repository/tree?recursive=true&per_page=100&page={page}&ref={commit}'
 data=download(url);(O/f'wichoose_fixed_tree_{page}.json').write_bytes(data)
 items=json.loads(data);tree.extend(items)
 if len(items)<100:break
else:raise RuntimeError('Tree exceeds declared bounded source scan')
targets=[p for p in tree if p['type']=='blob' and (Path(p['path']).suffix.lower() in ['.cpp','.hpp','.tpp','.sh','.md','.conf','.ipynb'] or p['path']=='LICENSE')]
base=O/'wichoose';base.mkdir(exist_ok=True)
def one(item):
 path=item['path'];dest=(base/path).resolve();assert dest.is_relative_to(base.resolve())
 url=f'https://gitlab.com/api/v4/projects/58506285/repository/files/{quote(path,safe="")}/raw?ref={commit}'
 receipt=dict(path=path,url=url,blob_id=item['id'])
 try:
  data=download(url,10000000);dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(data)
  receipt.update(status='retrieved',bytes=len(data),sha256=hashlib.sha256(data).hexdigest())
 except Exception as e:receipt.update(status='unavailable',error=str(e))
 return receipt
with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:receipts=list(pool.map(one,targets))
meta=json.loads((O/'wiperf_zenodo.json').read_text())
file=next(p for p in meta['files'] if p['key']=='wiperf_2022_06_14_data.csv')
data=download(file['links']['self']);assert 'md5:'+hashlib.md5(data).hexdigest()==file['checksum']
(O/file['key']).write_bytes(data)
result=dict(commit=commit,files_in_tree=len(tree),selected_files=len(targets),
    files=receipts,wiperf=dict(file=file['key'],url=file['links']['self'],bytes=len(data),md5=file['checksum'],sha256=hashlib.sha256(data).hexdigest()),
    accessed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),third_party_code_executed=False)
(O/'repository_receipt.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(dict(commit=commit,tree_entries=len(tree),downloaded=sum(x['status']=='retrieved' for x in receipts),
      failed=[x for x in receipts if x['status']!='retrieved'],wiperf_bytes=len(data)),ensure_ascii=False),flush=True)
