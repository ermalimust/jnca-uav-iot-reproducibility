"""Offline numerical checks against published, unchanged scientific records.

Python + NumPy + pandas only. No simulator execution, API calls or network.
The source tree is read-only. New receipts are written to --output or a new
temporary directory. Original seals are never rewritten. An explicit public
exclusion can represent a historical file hash, but is reported as skipped;
it is not counted as verification of that missing file.
"""
from pathlib import Path
import argparse
import ast
import contextlib
import hashlib
import io
import json
import re
import shutil
import sys
import tempfile
import time
import types

sys.dont_write_bytecode = True


class Checks:
    def __init__(self, root, output, exclusions=None, derivatives=None):
        self.root = root.resolve()
        self.output = output.resolve()
        self.output.mkdir(parents=True, exist_ok=True)
        self.p19 = self.root / 'revision_work/new_experiments/P19_ns3_policy_validation_20260914'
        self.exclusions = {}
        self.skipped = {}
        self.derivatives = {}
        self.derivative_checks = {}
        self.hashed_files = set()
        if exclusions is not None:
            for entry in json.loads(exclusions.read_text(encoding='utf-8'))['files']:
                p = (self.root / entry['path']).resolve()
                assert p.is_relative_to(self.root), entry['path']
                assert re.fullmatch('[0-9a-f]{64}', entry['sha256']), entry
                assert entry['path'] not in self.exclusions, entry['path']
                self.exclusions[p.relative_to(self.root).as_posix()] = entry
        if derivatives is not None:
            for entry in json.loads(derivatives.read_text(encoding='utf-8'))['files']:
                p = (self.root / entry['path']).resolve()
                assert p.is_relative_to(self.root), entry['path']
                for field in ('original_sha256', 'public_sha256'):
                    assert re.fullmatch('[0-9a-f]{64}', entry[field]), entry
                rel = p.relative_to(self.root).as_posix()
                assert rel not in self.derivatives and rel not in self.exclusions, rel
                self.derivatives[rel] = entry

    def sha(self, path):
        path = Path(path).resolve()
        if path.is_file():
            h = hashlib.sha256()
            with path.open('rb') as stream:
                for chunk in iter(lambda: stream.read(1024*1024), b''):
                    h.update(chunk)
            observed = h.hexdigest()
            self.hashed_files.add(str(path))
            rel = path.relative_to(self.root).as_posix() if path.is_relative_to(self.root) else None
            if rel in self.derivatives:
                entry = self.derivatives[rel]
                assert observed == entry['public_sha256'], 'Public derivative hash mismatch: '+rel
                self.derivative_checks[rel] = dict(entry, status='current public hash verified; original historical bytes not rehashed')
                return entry['original_sha256']
            return observed
        assert path.is_relative_to(self.root), 'Missing file outside release root: '+str(path)
        rel = path.relative_to(self.root).as_posix()
        assert rel in self.exclusions, 'Missing non-excluded scientific dependency: '+rel
        entry = self.exclusions[rel]
        self.skipped[rel] = dict(entry, status='not rehashed: explicitly excluded historical artifact')
        # Original verifier still compares this digest with the original seal.
        # The exception is BINARY_HASH initialization, a provenance constant.
        return entry['sha256']

    def verify_artifact(self, root, name, expected):
        path = (root/name).resolve()
        assert path.is_relative_to(root.resolve()), name
        rel = path.relative_to(self.root).as_posix()
        if not path.is_file():
            assert rel in self.exclusions, 'Missing non-excluded artifact: '+rel
            assert self.exclusions[rel]['sha256'] == expected['sha256'], 'Excluded artifact seal mismatch: '+rel
        elif rel in self.derivatives:
            assert self.derivatives[rel]['original_sha256'] == expected['sha256'], 'Derivative original seal mismatch: '+rel
        else:
            assert path.stat().st_size == expected['bytes'], 'Artifact size mismatch: '+rel
        assert self.sha(path) == expected['sha256'], 'Artifact seal mismatch: '+rel

    def load(self, name, path):
        tree = ast.parse(path.read_text(encoding='utf-8'))
        found_sha = False
        for i, node in enumerate(tree.body):
            if isinstance(node, ast.FunctionDef) and node.name == 'sha':
                tree.body[i] = ast.copy_location(ast.Assign(targets=[ast.Name(id='sha', ctx=ast.Store())], value=ast.Name(id='_publication_sha', ctx=ast.Load())), node)
                found_sha = True
        assert found_sha, 'Expected explicit hash helper in '+str(path)
        ast.fix_missing_locations(tree)
        module = types.ModuleType(name)
        module.__file__ = str(path)
        module._publication_sha = self.sha
        sys.modules[name] = module
        exec(compile(tree, str(path), 'exec'), module.__dict__)
        return module

    def main54(self):
        path = self.root / 'revision_work/offline_analysis.py'
        tree = ast.parse(path.read_text(encoding='utf-8'))
        kept = []
        found_verification = False
        for node in tree.body:
            # Output directory already exists as scientific input location;
            # remove its historical mkdir and terminal-encoding mutation.
            if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
                target = ast.unparse(node.value.func)
                if target in ('OUT.mkdir', 'sys.stdout.reconfigure'):
                    continue
            if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'ROOT' for t in node.targets):
                node.value = ast.Name(id='_publication_root', ctx=ast.Load())
            kept.append(node)
            if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'verification' for t in node.targets):
                found_verification = True
                break
        assert found_verification
        body = ast.fix_missing_locations(ast.Module(body=kept, type_ignores=[]))
        scope = {'__file__':str(path), '__name__':'_publication_main54', '_publication_root':self.root}
        exec(compile(body, str(path), 'exec'), scope)
        result = scope['verification']
        assert result['decisions'] == result['matched_actions'] == 54000
        result['scope'] = 'All saved action identities, realized losses, regrets and Boolean violation flags recalculated from archived rounded posteriors, candidates, mission costs and guards. No new model generations or simulation.'
        return result

    def policy(self):
        module = self.load('_publication_p19_policy', self.p19/'independent_design/verify_formal.py')
        module.OUT = self.output / 'policy'
        module.OUT.mkdir(exist_ok=True)
        rows = module.read(self.p19/'decisions/test_prefix_features.json')
        model = module.read(self.p19/'calibration/model.json')
        result = module.verify_decisions(rows, model)
        result['test_probability_rows'] = len(rows)
        assert result['test_probability_rows'] == 384 and result['decisions_recomputed'] == 153216
        result['scope'] = 'Probability vectors and all committed selections, scores, guards and candidate routing reconstructed from saved prefix features and fitted model. Prefix packet/radio logs and model fitting are not rerun.'
        return result

    def statistics(self):
        base = self.p19/'independent_statistics'
        sys.path.insert(0, str(base))
        module = self.load('_publication_p19_statistics', base/'verify_formal_statistics.py')
        module.HERE = self.output/'statistics'
        module.HERE.mkdir(exist_ok=True)
        # This unchanged helper is also hashed by the original checker.
        shutil.copy2(base/'exact_paired_inference.py', module.HERE/'exact_paired_inference.py')
        captured = io.StringIO()
        with contextlib.redirect_stdout(captured):
            module.main()
        (module.HERE/'verification_stdout.txt').write_text(captured.getvalue(), encoding='utf-8')
        result = json.loads((module.HERE/'formal_statistics_audit.json').read_text(encoding='utf-8'))
        result['publication_scope'] = 'All 12 new paired tests and bootstrap intervals recalculated from saved packet-count summaries and committed decisions, then all 113 Holm/BH adjustments recalculated. The prior 101 original raw p-values and historical raw-ledger decoder receipt are preserved/checked, not independently re-derived from their earlier experiments here.'
        return result

    def service(self):
        path = self.root/'revision_work/formal_integration_20260914/verify_measured_service.py'
        tree = ast.parse(path.read_text(encoding='utf-8'))
        replaced_sha = replaced_manifest_loop = 0
        for i, node in enumerate(tree.body):
            if isinstance(node, ast.FunctionDef) and node.name == 'sha':
                tree.body[i] = ast.copy_location(ast.Assign(targets=[ast.Name(id='sha', ctx=ast.Store())], value=ast.Name(id='_publication_sha', ctx=ast.Load())), node)
                replaced_sha += 1
            if isinstance(node, ast.FunctionDef) and node.name == 'check':
                for child in ast.walk(node):
                    if isinstance(child, ast.For) and ast.unparse(child.target) == '(name, x)' and ast.unparse(child.iter) == 'manifest.items()':
                        child.body = ast.parse('_publication_artifact(root, name, x)').body
                        replaced_manifest_loop += 1
        assert replaced_sha == replaced_manifest_loop == 1, 'Unexpected measured-service checker structure'
        ast.fix_missing_locations(tree)
        namespace = {'__file__':str(path), '__name__':'_publication_service',
                     '_publication_sha':self.sha, '_publication_artifact':self.verify_artifact}
        exec(compile(tree, str(path), 'exec'), namespace)
        result = namespace['check'](self.root)
        result['sealed_artifact_manifest_entries_checked'] = result.pop('sealed_artifacts_hash_verified')
        result['publication_scope'] = 'Every original numerical assertion retained: training ranking/selection, 5,868 simulation seconds, 77 full/period/window metrics, 15,000 saved bootstrap records and interval endpoints. Explicit missing author artifacts or binary files are reported separately; current editorial derivatives must match their public hashes and the historical digest in each original seal. Full packet-ID decoding and bootstrap draw metrics are not recomputed.'
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True, type=Path, help='Extracted scientific release root')
    parser.add_argument('--output', type=Path, help='New receipt directory; default: a new temporary directory')
    parser.add_argument('--exclusions', type=Path, help='Explicit exclusions JSON; default: ROOT/public_exclusions.json if present')
    parser.add_argument('--derivatives', type=Path, help='Explicit original/public hash mapping; default: ROOT/public_derivatives.json if present')
    parser.add_argument('--check', choices=['main54','policy','statistics','service','all'], default='all')
    args = parser.parse_args()
    root = args.root.resolve()
    output = args.output or Path(tempfile.mkdtemp(prefix='jnca_replay_check_'))
    exclusions = args.exclusions or (root/'public_exclusions.json' if (root/'public_exclusions.json').is_file() else None)
    derivatives = args.derivatives or (root/'public_derivatives.json' if (root/'public_derivatives.json').is_file() else None)
    runner = Checks(root, output, exclusions, derivatives)
    results = {}
    for kind in (['main54','policy','statistics','service'] if args.check == 'all' else [args.check]):
        start = time.monotonic()
        results[kind] = {'status':'PASS', 'elapsed_seconds':time.monotonic()-start, 'result':getattr(runner,kind)()}
        results[kind]['elapsed_seconds'] = time.monotonic()-start
        print(kind+': PASS', flush=True)
    receipt = {'status':'PASS','checks':results,'historical_hashes_not_reverified':list(runner.skipped.values()),
               'editorial_derivatives':list(runner.derivative_checks.values()), 'current_files_sha256_verified':len(runner.hashed_files),
               'scope':'Offline numerical checks. An excluded file listed here was not rehashed or executed; all numerical inputs required by the selected checks were read.',
               'output_directory':str(runner.output)}
    (runner.output/'verification_receipt.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'status':'PASS','receipt':str(runner.output/'verification_receipt.json'),'historical_files_skipped':len(runner.skipped)},ensure_ascii=False,indent=2))


if __name__ == '__main__':
    main()
