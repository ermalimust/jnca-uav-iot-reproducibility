"""Run new ns-3 prefixes and action branches with phase and hash gates."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import gzip
import shutil
import subprocess
import tempfile
import time
from common import *

if os.name == 'nt':
    import ctypes
    ctypes.windll.kernel32.SetErrorMode(0x0001 | 0x0002 | 0x8000)

def build():
    dest = HERE / 'build'
    dest.mkdir(exist_ok=True)
    # MinGW's assembler needs an ASCII working directory on this Windows host.
    with tempfile.TemporaryDirectory(prefix='p19_compile_') as temp:
        folder = Path(temp).resolve()
        assert folder.is_relative_to(Path(tempfile.gettempdir()).resolve())
        assert folder.name.startswith('p19_compile_') and str(folder).isascii()
        shutil.copy2(HERE / 'policy_replay.cc', folder / 'policy_replay.cc')
        libs = sorted((NS3 / 'build/lib').glob('*.dll.a'))
        cmd = [str(MINGW / 'g++.exe'), '-std=c++20', '-O2', '-DNS3_LOG_ENABLE', '-DNS3_ASSERT_ENABLE',
               '-I', str(NS3 / 'build/include'), 'policy_replay.cc', '-o', 'policy_replay.exe',
               '-Wl,--start-group', *map(str, libs), '-Wl,--end-group']
        r = subprocess.run(cmd, cwd=folder, env=environment(), capture_output=True, timeout=180)
        (dest / 'compile.stdout.txt').write_bytes(r.stdout)
        (dest / 'compile.stderr.txt').write_bytes(r.stderr)
        dump(dest / 'compile_receipt.json', {'command': cmd, 'exit_code': r.returncode,
             'source_sha256': sha(HERE / 'policy_replay.cc'),
             'runtime_sources': {str(p): sha(p) for p in [NS3 / 'VERSION', *libs, *sorted((NS3 / 'build/lib').glob('*.dll'))]}})
        if r.returncode:
            raise RuntimeError(r.stderr.decode(errors='replace')[-8000:])
        shutil.copy2(folder / 'policy_replay.exe', dest / 'policy_replay.exe')
    print('P19 ns-3 executable compiled.', flush=True)

def compress(path):
    body = path.read_bytes()
    out = path.with_suffix(path.suffix + '.gz')
    out.write_bytes(gzip.compress(body, compresslevel=6, mtime=0))
    path.unlink()
    return hashlib.sha256(body).hexdigest()

def one(split, sc, run, action='Observe', probe=False, extra=()):
    folder = run_folder(split, sc['scenario_id'], run, action, probe)
    folder.mkdir(parents=True, exist_ok=True)
    source = HERE / sc['offer_path']
    binary = HERE / 'build/policy_replay.exe'
    cmd = [str(binary), f'--action={action}', f'--run={run}', f"--moving={sc['moving']}",
           f"--speed={sc['speed_mps']}", '--input=' + os.path.relpath(source, folder),
           '--output=packets.csv', '--prefix=prefix.csv', '--knobs=knobs.csv', f'--probe={int(probe)}', *extra]
    rp = folder / 'receipt.json'
    if rp.exists():
        rec = read(rp)
        assert rec['command'] == cmd and rec['binary_sha256'] == sha(binary) and rec['input_sha256'] == sha(source)
        assert all(sha(folder / n) == h for n, h in rec['files'].items())
        return rec
    started = time.perf_counter()
    r = subprocess.run(cmd, cwd=folder, env=environment(), capture_output=True, timeout=180)
    (folder / 'stdout.txt').write_bytes(r.stdout)
    (folder / 'stderr.txt').write_bytes(r.stderr)
    if r.returncode:
        raise RuntimeError(f'{folder}: exit {r.returncode}: ' + r.stderr.decode(errors='replace')[-3000:])
    hashes = {name: compress(folder / name) for name in ('packets.csv', 'prefix.csv', 'radio_prefix.csv')}
    rec = {'split': split, 'scenario_id': sc['scenario_id'], 'rng_run': run, 'action': action,
           'probe': probe, 'command': cmd, 'exit_code': 0, 'wall_seconds': round(time.perf_counter() - started, 4),
           'binary_sha256': sha(binary), 'input_sha256': sha(source), 'uncompressed_sha256': hashes,
           'files': {p.name: sha(p) for p in folder.iterdir() if p.is_file() and p.name != 'receipt.json'}}
    dump(rp, rec)
    return rec

def engineering():
    scenarios = scenario_records()
    severe = next(s for s in scenarios if s['scenario_id'] == 'W1S2V1')
    recs = [one('engineering', severe, 9901, a) for a in ACTIONS]
    probe = one('engineering', severe, 9901, probe=True)
    recs.append(probe)
    for name in ('prefix.csv', 'radio_prefix.csv'):
        assert len({r['uncompressed_sha256'][name] for r in recs}) == 1
    assert probe['uncompressed_sha256']['prefix.csv'] == probe['uncompressed_sha256']['packets.csv']
    zero_sc = next(s for s in scenarios if s['scenario_id'] == 'W0S0V0')
    zero = one('engineering_zero', zero_sc, 9902, extra=('--zero=1',))
    low = one('engineering_low', zero_sc, 9902, extra=('--distance=1',))
    recs.extend([zero, low])
    for rec in recs:
        folder = run_folder(rec['split'], rec['scenario_id'], rec['rng_run'], rec['action'], rec['probe'])
        with gzip.open(folder / 'prefix.csv.gz', 'rt') as f:
            rr = list(csv.DictReader(f))
        for row in rr:
            for key in ('offer_ns', 'send_ns', 'receive_ns', 'first_phy_ns', 'last_phy_ns', 'first_ack_ns', 'last_mac_drop_ns'):
                assert int(row[key]) <= 10999999999
            if int(row['first_ack_ns']) >= 0:
                assert int(row['first_ack_ns']) >= int(row['first_phy_ns']) >= int(row['send_ns']) >= int(row['offer_ns'])
        with gzip.open(folder / 'radio_prefix.csv.gz', 'rt') as f:
            for row in csv.DictReader(f):
                assert int(row['time_ns']) <= 10999999999
                assert -200 < float(row['signal_dbm']) < 100
    dump(HERE / 'engineering_gates.json', {'status': 'PASS', 'checks': [
        'five arms and prefix-only probe have identical packet and actual RSSI prefixes',
        'probe stops before intervention and contains no later packet updates',
        'controller ACK timestamp causality and RSSI bounds', 'zero traffic and 1m low-load executions'], 'runs': recs})
    print('P19 engineering gates PASS.', flush=True)

def freeze_design():
    assert read(HERE / 'engineering_gates.json')['status'] == 'PASS'
    names = ['protocol.json', 'protocol.md', 'common.py', 'prepare.py', 'run_study.py', 'policy_model.py',
             'analyze_results.py', 'policy_replay.cc', 'build/policy_replay.exe', 'build/compile_receipt.json', 'engineering_gates.json']
    files = [HERE / x for x in names] + [p for p in (HERE / 'inputs').rglob('*') if p.is_file() and '__pycache__' not in p.parts]
    binding = {'scope': 'Before P19 training, validation and testing',
               'files': {p.relative_to(HERE).as_posix(): sha(p) for p in files}}
    target = HERE / 'design_binding.json'
    if target.exists():
        assert read(target) == binding
    else:
        dump(target, binding)
    print('Design and all scientific inputs frozen.', flush=True)

def phase(split, workers):
    verify_binding(HERE / 'design_binding.json')
    proto = read(HERE / 'protocol.json')
    probe = split in ('validation', 'test_probe')
    key = 'test_runs' if split == 'test_probe' else f'{split}_runs'
    if split in ('test_probe', 'test'):
        verify_binding(HERE / 'model_binding.json')
    if split == 'test':
        decision = verify_binding(HERE / 'decision_binding.json')
        assert decision['model_binding_sha256'] == sha(HERE / 'model_binding.json')
    if split in ('train', 'validation'):
        assert not (HERE / 'model_binding.json').exists(), 'Train/validation artifacts are sealed with the model'
    jobs = [(split, s, r, a, probe) for s in proto['scenarios'] for r in proto[key]
            for a in (['Observe'] if probe else ACTIONS)]
    recs = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(one, *j) for j in jobs]
        for future in as_completed(futures):
            recs.append(future.result())
            if len(recs) % 48 == 0 or len(recs) == len(jobs):
                print(f'{split}: {len(recs)}/{len(jobs)} runs', flush=True)
    for sc in proto['scenarios']:
        for rng in proto[key]:
            group = [r for r in recs if r['scenario_id'] == sc['scenario_id'] and r['rng_run'] == rng]
            for name in ('prefix.csv', 'radio_prefix.csv'):
                assert len({r['uncompressed_sha256'][name] for r in group}) == 1
                if split == 'test':
                    probe_rec = read(run_folder('test_probe', sc['scenario_id'], rng, probe=True) / 'receipt.json')
                    assert group[0]['uncompressed_sha256'][name] == probe_rec['uncompressed_sha256'][name]
    verify_binding(HERE / 'design_binding.json')
    dump(HERE / f'{split}_execution.json', {'status': 'PASS', 'split': split, 'run_count': len(recs),
         'design_binding_sha256': sha(HERE / 'design_binding.json'), 'runs': sorted(recs, key=lambda r: (r['scenario_id'], r['rng_run'], r['action']))})
    print(f'{split} completed and all prefixes verified.', flush=True)

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('stage', choices=['build', 'engineering', 'freeze_design', 'train', 'validation', 'test_probe', 'test'])
    ap.add_argument('--workers', type=int, default=6)
    args = ap.parse_args()
    {'build': build, 'engineering': engineering, 'freeze_design': freeze_design}.get(
        args.stage, lambda: phase(args.stage, args.workers))()
