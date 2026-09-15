"""Read-only integrity verification of saved new experiment inputs and outputs."""
from pathlib import Path
import hashlib,json,subprocess,sys
HERE=Path(__file__).resolve().parent
WORK=HERE.parents[1]
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
total=0
for name in ["P1_des_recovery_20260912","P1b_time_alignment_20260912","P5_calibration_20260912"]:
    folder=HERE/name
    data=json.loads((folder/"run_manifest.json").read_text(encoding="utf-8"))
    for row in data["outputs"]:
        p=folder/row["path"]
        assert p.is_file() and sha(p)==row["sha256"],str(p)
        total+=1
    for row in data.get("inputs",[]):
        assert sha(WORK/row["path"])==row["sha256"],row["path"]
        total+=1
subprocess.run([sys.executable,str(HERE/"P2_provenance_20260912/verify_provenance.py")],check=True)
print(f"PASS: {total} input/output hash checks and the 72-coefficient provenance table.")
