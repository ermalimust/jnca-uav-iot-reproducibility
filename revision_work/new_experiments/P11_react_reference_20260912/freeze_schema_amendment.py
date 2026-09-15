"""Archive the pre-amendment state and freeze normalization before any outcome evaluation."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,zipfile
import schema_amendment as amendment
HERE=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,o):p.write_text(json.dumps(o,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def main():
    assert not (HERE/'matched_raw.csv').exists() and not (HERE/'primary_contrasts.csv').exists()
    assert not (HERE/'schema_amendment_freeze.json').exists()
    files=[p for p in HERE.rglob('*') if p.is_file() and (p.parent.name in ['raw','calls'] or p.name in ['protocol.json','freeze_receipt.json','react_public.py','generate_react.py','planned_hypotheses.csv','cost_reservations.json','generation_manifest.json'])]
    archive=HERE/'pre_schema_amendment_snapshot.zip'
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED) as z:
        for p in sorted(files):z.write(p,p.relative_to(HERE).as_posix())
    counts={}
    for p in (HERE/'raw').glob('*.json'):
        o=json.loads(p.read_text(encoding='utf-8'));literal=o['policy_literal'];kind=type(literal.get('archetype_actions')).__name__ if isinstance(literal,dict) else 'no_literal_object'
        key=o['variant']+'__'+kind;counts[key]=counts.get(key,0)+1
    receipt={'amended_utc':datetime.now(timezone.utc).isoformat(),'approved_by':'root before performance evaluation',
       'reason':'Shared generation contract did not explicitly specify object versus one-to-one list serialization; ReAct action format carried an object example while some contemporaneous zero responses used an equivalent six-record list. Treating equivalent candidate policies as empty candidates would create a format-artifact comparison.',
       'performance_outcomes_examined_before_amendment':False,'generation_outputs_seen':'Schema-status monitoring and two failing literal-output examples only; no regret, coverage, invalidity, model ranking or inferential result evaluated.',
       'scope':'Apply symmetric equivalent-list normalization ONLY to final policy extraction for offline evaluation. Generation prompts, tools and actual tool observations remain exactly frozen and unchanged. Strict-parser interpretation remains a fully reported sensitivity. This is an outcome-blind amendment, not an original preplanned rule.',
       'accepted_conversion':'Only a list of exactly six objects, each having exactly archetype and actions keys, with unique names equal to the complete six-archetype set and list-valued actions, becomes the corresponding object. Candidate strings, non-strings, order and duplicates are unchanged.',
       'pre_amendment_snapshot':{'file':archive.name,'sha256':sha(archive),'files':{p.relative_to(HERE).as_posix():sha(p) for p in sorted(files)}},
       'snapshot_completed_policy_schema_types':counts,
       'snapshot_saved_policies':sum(counts.values()),
       'snapshot_cost_reservation_usd':json.loads((HERE/'cost_reservations.json').read_text())['reserved_usd'],
       'interrupted_inflight_calls':4,'interrupted_request_handling':'All started-attempt receipts and cost reservations remain in the immutable snapshot. No response had been saved for four interrupted attempts; their service-side completion/charge is unknown. Resume uses the remaining original transport-attempt allowance and retains the unknown attempt records.',
       'selfcheck':amendment.selfcheck(),
       'frozen_files':{n:sha(HERE/n) for n in ['schema_amendment.py','freeze_schema_amendment.py','protocol.json','planned_hypotheses.csv','freeze_receipt.json']}}
    save(HERE/'schema_amendment_freeze.json',receipt)
    print(json.dumps({'snapshot_saved_policies':sum(counts.values()),'schema_types':counts,'normalization_selfcheck':receipt['selfcheck'],'amendment_sha256':sha(HERE/'schema_amendment_freeze.json')},indent=2))
if __name__=='__main__':main()
