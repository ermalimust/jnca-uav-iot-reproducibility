"""Read-only saved-evidence checks for both completed revision rounds."""
from pathlib import Path,PurePosixPath
import hashlib,json,subprocess,sys
HERE=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
count=0
def check(folder,name,want):
    global count
    rel=PurePosixPath(name)
    assert not rel.is_absolute() and '..' not in rel.parts and ':' not in name and '\\' not in name,name
    p=folder/Path(*rel.parts)
    assert p.is_file() and sha(p)==want,str(p)
    count+=1
subprocess.run([sys.executable,str(HERE/'verify_new_experiments.py')],check=True)
p=HERE/'P2P7_completion_20260912'
for r in json.loads((p/'manifest.json').read_text(encoding='utf-8')):check(p,r['file'],r['sha256'])
p=HERE/'P3_external_completion_20260912'
for name,h in json.loads((p/'run_manifest.json').read_text(encoding='utf-8'))['files'].items():check(p,name,h)
p=HERE/'P4_prompt_completion_20260912'
for name in ['generation_manifest.json','evaluation_manifest.json']:
    for r in json.loads((p/name).read_text(encoding='utf-8'))['files']:check(p,r['path'],r['sha256'])
c=p/'component_check_20260912'
component=json.loads((c/'artifact_manifest.json').read_text(encoding='utf-8'))
for r in component['replay_sources']:check(HERE.parents[1],r['file'],r['sha256'])
for r in component['addendum_files']:check(c,r['file'],r['sha256'])
p=HERE/'P6_temporal_completion_20260912'
for name,h in json.loads((p/'completion_manifest.json').read_text(encoding='utf-8'))['hashes'].items():check(p,name,h)
print(f'PASS: {count} additional completion-round file hashes; earlier generation/calibration evidence also passed.')
