"""Public-source archival only: no model API, credentials or private data."""
from pathlib import Path
import hashlib, json, urllib.request
from datetime import datetime, timezone
HERE=Path(__file__).resolve().parent
COMMIT='6bdb3a1fd38b8188fc7ba4102969fe483df8fdc9'
files={name:'https://raw.githubusercontent.com/ysymyth/ReAct/'+COMMIT+'/'+name for name in ['hotpotqa.ipynb','README.md','LICENSE','wikienv.py']}
def main():
    folder=HERE/'reference_source';folder.mkdir(exist_ok=True)
    rows=[]
    for name,url in files.items():
        p=folder/name
        if not p.exists():
            data=urllib.request.urlopen(url,timeout=30).read();p.write_bytes(data)
        rows.append({'file':p.relative_to(HERE).as_posix(),'url':url,'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'bytes':p.stat().st_size})
    (HERE/'source_provenance.json').write_text(json.dumps({'retrieved_utc':datetime.now(timezone.utc).isoformat(),'official_repository':'https://github.com/ysymyth/ReAct','commit':COMMIT,'paper_version':'https://arxiv.org/abs/2210.03629v3','files':rows},indent=2)+'\n',encoding='utf-8')
    print('Archived four pinned public reference files; no model API called.')
if __name__=='__main__':main()
