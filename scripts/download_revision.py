"""Download pinned v1.0.0 records and the v1.1.0 P25 supplement separately."""
from pathlib import Path, PurePosixPath
import argparse
import hashlib
import json
import shutil
import stat
import urllib.request
import zipfile

REPO = Path(__file__).resolve().parents[1]


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1048576), b''):
            h.update(chunk)
    return h.hexdigest()


def extract_checked(archive_path, root):
    root = root.resolve()
    with zipfile.ZipFile(archive_path) as archive:
        for item in archive.infolist():
            rel = PurePosixPath(item.filename)
            if (rel.is_absolute() or '..' in rel.parts or '\\' in item.filename
                    or ':' in item.filename or stat.S_ISLNK(item.external_attr >> 16)):
                raise ValueError('Unsafe archive path: ' + item.filename)
            dest = (root / Path(*rel.parts)).resolve()
            if not dest.is_relative_to(root):
                raise ValueError('Path escapes extraction root')
            if item.is_dir():
                dest.mkdir(parents=True, exist_ok=True)
                continue
            data = archive.read(item)
            if dest.exists():
                if not dest.is_file() or sha(dest) != hashlib.sha256(data).hexdigest():
                    raise ValueError('Existing file differs; choose a fresh --root: ' + str(dest))
            else:
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(data)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--part', choices=['all', 'core', 'ns3_action', 'ns3_policy', 'measured_service', 'p25'], default='all')
    parser.add_argument('--root', type=Path, default=REPO / 'data')
    parser.add_argument('--cache', type=Path, default=REPO / '.downloads', help='Reuse already downloaded, hash-verified archives')
    parser.add_argument('--offline', action='store_true', help='Fail if a valid archive is missing from the cache; never access the network')
    args = parser.parse_args()
    cache = args.cache.resolve()
    cache.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((REPO / 'manifests/revision_v1.1.0.json').read_text(encoding='utf-8'))
    for asset in manifest['assets']:
        if args.part not in ('all', asset['part']):
            continue
        target = cache / asset['name']
        if not target.is_file() or target.stat().st_size != asset['bytes'] or sha(target) != asset['sha256']:
            if args.offline:
                raise SystemExit('Missing or invalid cached archive: ' + str(target))
            print('Downloading ' + asset['name'], flush=True)
            partial = target.with_suffix('.partial')
            with urllib.request.urlopen(asset['url'], timeout=120) as response, partial.open('wb') as out:
                shutil.copyfileobj(response, out, 1048576)
            if partial.stat().st_size != asset['bytes'] or sha(partial) != asset['sha256']:
                raise ValueError('Archive integrity mismatch: ' + asset['name'])
            partial.replace(target)
        destination = args.root.resolve() / asset['extract_to']
        if not destination.resolve().is_relative_to(args.root.resolve()):
            raise ValueError('Invalid extraction destination')
        extract_checked(target, destination)
        print('Verified and extracted ' + asset['part'] + ': ' + str(destination), flush=True)


if __name__ == '__main__':
    main()
