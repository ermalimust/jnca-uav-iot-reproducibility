"""Fetch only the pinned, publicly released DUCC v1.0.1 source files."""
from pathlib import Path
import hashlib, json, urllib.request, datetime, time

ROOT = Path(__file__).resolve().parent
SOURCES = {
    'Parameter_Description.csv': '536468cf46f6fbfa94ed5ab2ceb10b23',
    'load.py': '6b81db493714e12b748f4e623b2e49ed',
    'Dataset.zip': 'cb40e78749dc5703e815cc87388ea55a',
}
out = ROOT / 'sources'
out.mkdir(exist_ok=True)
records = []
for filename, expected in SOURCES.items():
    target = out / filename
    url = f'https://zenodo.org/records/10526786/files/{filename}?download=1'
    if not target.exists() or hashlib.md5(target.read_bytes()).hexdigest() != expected:
        req = urllib.request.Request(url, headers={'User-Agent': 'Public academic reproducibility audit'})
        tmp = target.with_suffix(target.suffix + '.part')
        for attempt in range(3):
            try:
                with urllib.request.urlopen(req, timeout=90) as response, tmp.open('wb') as dest:
                    total = 0
                    while chunk := response.read(1024 * 1024):
                        dest.write(chunk)
                        total += len(chunk)
                        if total % (10 * 1024 * 1024) == 0:
                            print(f'{filename}: {total} bytes', flush=True)
                break
            except Exception as exc:
                print(f'{filename}: attempt {attempt + 1} failed: {exc}', flush=True)
                if attempt == 2:
                    raise
                time.sleep(2)
        tmp.replace(target)
    raw = target.read_bytes()
    actual = hashlib.md5(raw).hexdigest()
    if actual != expected:
        raise ValueError(f'{filename}: MD5 mismatch {actual}')
    records.append({'filename': filename, 'url': url, 'bytes': len(raw),
                    'md5': actual, 'expected_md5': expected,
                    'sha256': hashlib.sha256(raw).hexdigest()})
    print(f'{filename}: verified {len(raw)} bytes', flush=True)
(ROOT / 'source_manifest.json').write_text(json.dumps({
    'dataset': 'DUCC', 'version': '1.0.1',
    'record': 'https://zenodo.org/records/10526786',
    'doi': '10.5281/zenodo.10526786',
    'license': 'CC BY-NC-SA 4.0',
    'retrieved_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
    'files': records,
}, indent=2), encoding='utf-8')
