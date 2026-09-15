"""Read-only check of all 72 base coefficients against two packaged sources."""
from pathlib import Path
import ast,csv,hashlib,json
HERE=Path(__file__).resolve().parent
WORK=HERE.parents[2]
P1=HERE.parent/"P1_des_recovery_20260912"
def literal(path,name):
    for node in ast.parse(path.read_text(encoding="utf-8")).body:
        if isinstance(node,ast.AnnAssign) and isinstance(node.target,ast.Name) and node.target.id==name:
            return ast.literal_eval(node.value)
    raise ValueError(name)
current=literal(WORK/"revision_work/analysis/replay_inputs/paper7_agentic_feasibility.py","BASE_COSTS")
precursor=literal(P1/"recovered/paper4_true_des_fleet_stress.py","COST_MATRICES")
note=(P1/"recovered/paper3_generator_core/notes/cost_matrix_rationale.md").read_text(encoding="utf-8").splitlines()
rows=list(csv.DictReader((HERE/"base_coefficient_provenance.csv").open(encoding="utf-8",newline="")))
assert len(rows)==72
seen=set()
for row in rows:
    profile,action,cause=row["profile"],row["action"],row["cause"]
    key=(profile,action,cause); assert key not in seen;seen.add(key)
    j="WBMV".index(cause)
    cells=[x.strip() for x in note[int(row["note_line"])-1].split("|") if x.strip()]
    assert cells[0].strip(chr(96))==action
    assert float(row["current"])==float(row["precursor"])==float(row["design_note"])==current[profile][action][j]==precursor[profile][action][j]==float(cells[j+1])
evidence=json.loads((HERE/"verification.json").read_text(encoding="utf-8"))
for entry in evidence["files"]:
    assert hashlib.sha256((WORK/entry["path"]).read_bytes()).hexdigest()==entry["sha256"]
print("PASS: 72 distinct base coefficients and all three source hashes.")
