"""Re-run the completed package twice in an isolated directory.

Only refreshed receipts and this audit record are copied back. Scientific CSV
files in the published directory must match both replays byte for byte.
"""
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE=Path(__file__).resolve().parent
def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

science=[p for p in HERE.iterdir() if p.suffix==".csv" or p.name=="verification.json"]
original={p.name:sha(p) for p in science}
source_relative=Path("P1_des_recovery_20260912/recovered/paper3_generator_core/paper3_des/configs/mvp_scenarios_min.json")
receipt_names=["run_manifest.json","action_effect_manifest.json"]
results=[]
temp_parent=Path(tempfile.gettempdir()).resolve()
with tempfile.TemporaryDirectory(prefix="jnca_p6_reentry_",dir=temp_parent) as temp_name:
    temp=Path(temp_name).resolve()
    # Verify the only automatically cleaned-up recursive target before use.
    assert temp.parent==temp_parent and temp.name.startswith("jnca_p6_reentry_")
    stage=temp/HERE.name
    stage.mkdir()
    for p in HERE.iterdir():
        if p.is_file():
            shutil.copy2(p,stage/p.name)
    source_target=temp/source_relative
    source_target.parent.mkdir(parents=True)
    shutil.copy2(HERE.parent/source_relative,source_target)
    commands=[
        [sys.executable,"run_temporal.py","--sequence-policies","sequence_policies.json"],
        [sys.executable,"run_action_effect_sensitivity.py"],
        [sys.executable,"verify_completion.py"],
        [sys.executable,"run_temporal.py","--check-only"]]
    for iteration in [1,2]:
        for cmd in commands:
            completed=subprocess.run(cmd,cwd=stage,text=True,encoding="utf-8",
                                     stdout=subprocess.PIPE,stderr=subprocess.STDOUT,check=True)
            print(json.dumps({"iteration":iteration,"completed":cmd[1],"exit_code":completed.returncode}),flush=True)
        matching={name:sha(stage/name)==expected for name,expected in original.items()}
        assert all(matching.values()),[name for name,ok in matching.items() if not ok]
        # Completion receipt itself is excluded; all claimed members must match.
        final=json.loads((stage/"completion_manifest.json").read_text())
        assert all(sha(stage/name)==digest for name,digest in final["hashes"].items())
        for receipt_name in receipt_names:
            receipt=json.loads((stage/receipt_name).read_text())
            assert not any("manifest" in name or name.startswith("completion_") for name in receipt["hashes"])
            assert all(sha(stage/name)==digest for name,digest in receipt["hashes"].items())
        results.append(dict(iteration=iteration,scientific_files_byte_identical=len(matching),
                            completion_member_hashes=len(final["hashes"]),
                            complete_pipeline_passed=True))
    # Refresh the current receipts from the proven identical isolated execution.
    for name in receipt_names:
        shutil.copy2(stage/name,HERE/name)
    assert all(sha(HERE/name)==digest for name,digest in original.items())
report=dict(passed=True,iterations=results,scientific_sha256=original,
            fix="Explicit stage-owned inputs/outputs eliminate upstream capture of downstream receipts and outputs.",
            current_scientific_files_unchanged=True,
            manifests_refreshed_from_identical_isolated_replay=receipt_names)
(HERE/"reentry_verification.json").write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
subprocess.run([sys.executable,str(HERE/"verify_completion.py")],cwd=HERE,check=True)
print(json.dumps({"passed":True,"iterations":2,"scientific_files_unchanged":len(original)}))
