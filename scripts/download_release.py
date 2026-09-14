"""Download the pinned scientific archives, verify SHA-256, and safely extract."""
from pathlib import Path, PurePosixPath
import argparse, hashlib, json, shutil, urllib.request, zipfile

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda:stream.read(1048576),b''):h.update(chunk)
    return h.hexdigest()

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--part',choices=['all','core','ns3_action','ns3_policy','measured_service'],default='all')
    parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1])
    args=parser.parse_args();root=args.root.resolve();root.mkdir(parents=True,exist_ok=True)
    catalogue=json.loads((Path(__file__).resolve().parents[1]/'manifests/assets.json').read_text(encoding='utf-8'))
    cache=root/'.downloads';cache.mkdir(exist_ok=True)
    for asset in catalogue['assets']:
        if args.part not in ('all',asset['part']):continue
        target=cache/asset['name']
        if not target.is_file() or target.stat().st_size!=asset['bytes'] or sha(target)!=asset['sha256']:
            print('Downloading '+asset['name'],flush=True)
            temp=target.with_suffix('.partial')
            with urllib.request.urlopen(asset['url'],timeout=120) as response,temp.open('wb') as out:
                shutil.copyfileobj(response,out,1048576)
            if temp.stat().st_size!=asset['bytes'] or sha(temp)!=asset['sha256']:
                raise ValueError('Archive integrity mismatch: '+asset['name'])
            temp.replace(target)
        with zipfile.ZipFile(target) as archive:
            for item in archive.infolist():
                rel=PurePosixPath(item.filename)
                if rel.is_absolute() or '..' in rel.parts or '\\' in item.filename or ':' in item.filename:
                    raise ValueError('Unsafe archive path: '+item.filename)
                dest=(root/Path(*rel.parts)).resolve()
                if not dest.is_relative_to(root):raise ValueError('Path escapes extraction root')
                if item.is_dir():dest.mkdir(parents=True,exist_ok=True);continue
                data=archive.read(item)
                if dest.exists():
                    if not dest.is_file() or sha(dest)!=hashlib.sha256(data).hexdigest():
                        raise ValueError('Existing file differs; extract into a fresh clone: '+item.filename)
                else:
                    dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(data)
        print('Verified and extracted '+asset['name'],flush=True)

if __name__=='__main__':main()
