"""Read-only public HTTPS alternatives; save only checksum-verified official files."""
from pathlib import Path
import concurrent.futures, urllib.request, hashlib, json, datetime
ROOT = Path(__file__).resolve().parent
CASES = [
 ('Parameter_Description.csv', 'https://www.zenodo.org/records/10526786/files/Parameter_Description.csv?download=1', '536468cf46f6fbfa94ed5ab2ceb10b23'),
 ('Dataset.zip', 'https://zenodo.org/records/10526786/files/Dataset.zip?download=1', 'cb40e78749dc5703e815cc87388ea55a'),
 ('load.py', 'https://zenodo.org/records/10526786/files/load.py?download=1', '6b81db493714e12b748f4e623b2e49ed'),
]
def fetch(case):
    filename, url, md5 = case
    result = dict(filename=filename, url=url, expected_md5=md5)
    try:
        req = urllib.request.Request(url, headers={'User-Agent':'Mozilla/5.0 academic data verification'})
        with urllib.request.urlopen(req, timeout=40) as response:
            result['status'] = response.status
            result['final_url'] = response.url
            raw = response.read()
        result.update(bytes=len(raw), md5=hashlib.md5(raw).hexdigest(), sha256=hashlib.sha256(raw).hexdigest())
        result['checksum_match'] = result['md5'] == md5
        if result['checksum_match']:
            (ROOT / 'sources' / filename).write_bytes(raw)
    except Exception as exc:
        result['error'] = str(exc)
    print(json.dumps(result), flush=True)
    return result
with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
    records = list(pool.map(fetch, CASES))
(ROOT/'download_attempts.json').write_text(json.dumps({'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'attempts': records},indent=2),encoding='utf-8')
