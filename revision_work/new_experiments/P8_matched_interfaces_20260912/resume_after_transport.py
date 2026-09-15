"""Resume the pre-frozen experiment after local network sandbox transport failure.
No successful response is discarded; original failed receipts remain untouched.
"""
from pathlib import Path
import hashlib,importlib.util,json,time
HERE=Path(__file__).resolve().parent
ROOT=HERE
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def setup():
    dest=HERE/"connected_run";dest.mkdir(exist_ok=True)
    receipt=json.loads((HERE/"freeze_receipt.json").read_text(encoding="utf-8"))
    for name,value in receipt["files"].items():
        assert sha(HERE/name)==value
        target=dest/name
        if not target.exists():target.write_bytes((HERE/name).read_bytes())
        assert sha(target)==value
    target=dest/"freeze_receipt.json"
    if not target.exists():target.write_bytes((HERE/"freeze_receipt.json").read_bytes())
    assert sha(target)==sha(HERE/"freeze_receipt.json")
    addendum=HERE/"transport_recovery_addendum.json"
    if not addendum.exists():
        old=[json.loads(p.read_text(encoding="utf-8")) for p in (HERE/"raw").glob("*.json")]
        assert all(not c.get("response") for o in old for c in o["calls"])
        assert all(not o["parse_success"] for o in old)
        record={"date":"2026-09-12","frozen_before_connected_run_unix":time.time(),
          "reason":"Initial subprocess ran under network-restricted sandbox; every saved attempt returned immediate URLError and no HTTP/model response. The process was interrupted.",
          "preserved_failed_policies":len(old),"preserved_failed_attempts":sum(len(c["attempts"]) for o in old for c in o["calls"]),
          "successful_model_responses_discarded":0,"action":"Rerun all288 fixed planned policies in connected_run using an authorized network-enabled process. Preserve failed sandbox receipts in original raw. Public inputs, model settings, generator, tool, ranking budget and statistical protocol are byte-identical.",
          "selection":"Environment recovery precedes all successful candidate/output inspection; no outcome-driven replacement.",
          "runner_sha256":sha(HERE/"resume_after_transport.py"),
          "frozen_protocol_sha256":sha(HERE/"protocol.json")}
        addendum.write_text(json.dumps(record,indent=2)+"\n",encoding="utf-8")
    return dest
def main():
    dest=setup()
    spec=importlib.util.spec_from_file_location("p8_frozen_generator",HERE/"generate_matched.py")
    g=importlib.util.module_from_spec(spec);spec.loader.exec_module(g)
    # Work/replay roots were derived from the original root before redirecting outputs.
    g.HERE=dest
    g.main()
if __name__=="__main__":main()

