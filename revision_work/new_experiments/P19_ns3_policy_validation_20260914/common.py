"""Shared file formats and constants for the independent P19 experiment."""
from pathlib import Path
import csv
import hashlib
import json
import os

HERE = Path(__file__).resolve().parent
W = HERE.parents[2]
P18 = HERE.parent / 'P18_ns3_action_effects_20260914'
P12 = HERE.parent / 'P12_context_threshold_followup_20260912'
ACTIONS = ['Observe', 'WiFiRelief', 'LinkAdapt', 'VideoShape', 'FallbackProtect']
METHODS = ['qwen_service', 'qwen_direct', 'embedding_service', 'broad_service',
           'full_service', 'tool_service', 'qwen_numeric', 'embedding_numeric',
           'broad_numeric', 'full_numeric', 'tool_numeric']
NS3 = Path(os.environ.get('P19_NS3_ROOT', 'D:/tools/ns3/ns-allinone-3.47/ns-3.47'))
MINGW = Path(os.environ.get('P19_MINGW_BIN', 'D:/tools/ns3/msys64/mingw64/bin'))

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()

def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))

def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')

def write_csv(path, records):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)

def environment():
    env = os.environ.copy()
    env['PATH'] = os.pathsep.join(map(str, [NS3 / 'build/lib', MINGW])) + os.pathsep + env.get('PATH', '')
    env['TMP'] = '.'
    env['TEMP'] = '.'
    return env

def scenario_records():
    return read(HERE / 'protocol.json')['scenarios']

def run_folder(split, scenario, run, action='Observe', probe=False):
    return HERE / 'runs' / split / f"{scenario}_r{run:04d}_{'probe' if probe else action}"

def verify_binding(path):
    binding = read(path)
    for name, digest in binding['files'].items():
        assert sha(HERE / name) == digest, f'Frozen file changed: {name}'
    return binding
