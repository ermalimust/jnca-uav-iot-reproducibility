"""Verify every P25 file and replay decisions, 118-test statistics and RID sensitivity offline."""
from pathlib import Path
import argparse
import hashlib
import importlib.util
import json
import subprocess
import sys
import tempfile

sys.dont_write_bytecode = True
REPO = Path(__file__).resolve().parents[1]
PACKAGE = 'P25_catalogue_adaptation_supplement_20260916'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=REPO / 'data/v1.1.0' / PACKAGE, help='Extracted supplement directory containing experiment/ and reference_code/')
    parser.add_argument('--output', type=Path, help='New receipt directory outside the supplement')
    parser.add_argument('--check', choices=['all', 'files', 'decisions', 'statistics', 'rid'], default='all')
    args = parser.parse_args()
    root = args.root.resolve()
    if not (root / 'experiment/P25_catalogue_adaptation_20260916/protocol.json').is_file():
        raise SystemExit('P25 is not installed at --root. Run scripts/download_revision.py --part p25 first.')
    output = args.output.resolve() if args.output else Path(tempfile.mkdtemp(prefix='jnca_p25_replay_'))
    if output.is_relative_to(root):
        raise SystemExit('--output must be outside the archived supplement.')
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise SystemExit('--output must be a new or empty directory.')
    records = json.loads((REPO / 'manifests/files_p25_v1.1.0.json').read_text(encoding='utf-8'))['files']
    for record in records:
        path = (root / record['path']).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise ValueError('Missing or unsafe file: ' + record['path'])
        if path.stat().st_size != record['bytes'] or hashlib.sha256(path.read_bytes()).hexdigest() != record['sha256']:
            raise ValueError('File integrity mismatch: ' + record['path'])
    study = root / 'experiment/P25_catalogue_adaptation_20260916'
    completed = ['files']
    if args.check in ('all', 'decisions'):
        subprocess.run([sys.executable, '-B', str(study / 'audit/verify_results.py'), '--study-root', str(study), '--receipt', str(output / 'decisions.json')], check=True)
        completed.append('decisions')
    if args.check in ('all', 'statistics'):
        spec = importlib.util.spec_from_file_location('p25_independent_statistics', study / 'audit/final_statistics_checker.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        module.O = output
        module.main()
        completed.append('statistics')
    if args.check in ('all', 'rid'):
        subprocess.run([sys.executable, '-B', str(root / 'post_analysis/rid_substring_audit_20260916/check_rid_substring.py'), '--output', str(output / 'rid_sensitivity.json')], check=True)
        completed.append('rid')
    # Confirm that replay did not alter any published input, including historical receipts.
    for record in records:
        if hashlib.sha256((root / record['path']).read_bytes()).hexdigest() != record['sha256']:
            raise ValueError('Replay changed an archived input: ' + record['path'])
    receipt = {'status': 'PASS', 'checks': completed, 'files_verified': len(records), 'archived_files_unchanged': True, 'no_api_calls': True, 'output': str(output)}
    (output / 'receipt.json').write_text(json.dumps(receipt, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    main()
