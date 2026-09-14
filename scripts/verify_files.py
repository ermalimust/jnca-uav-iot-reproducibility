"""Verify every scientific file in each installed public release part."""
from pathlib import Path
import argparse,hashlib,json

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1])
    args=parser.parse_args();root=args.root.resolve()
    manifests=sorted((root/'manifests').glob('files_*.json'))
    if not manifests:raise SystemExit('No installed part manifest. Run scripts/download_release.py first.')
    total=0
    for p in manifests:
        manifest=json.loads(p.read_text(encoding='utf-8'))
        for x in manifest['files']:
            source=(root/x['path']).resolve()
            assert source.is_relative_to(root) and source.is_file(),x['path']
            assert source.stat().st_size==x['bytes'],x['path']
            h=hashlib.sha256()
            with source.open('rb') as f:
                for block in iter(lambda:f.read(1048576),b''):h.update(block)
            assert h.hexdigest()==x['sha256'],x['path']
            total+=1
        print(manifest['part']+': PASS ('+str(len(manifest['files']))+' files)',flush=True)
    print(json.dumps({'status':'PASS','parts':len(manifests),'files_verified':total}))

if __name__=='__main__':main()
