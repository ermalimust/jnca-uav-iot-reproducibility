"""Read-only reconstruction of all historical primary policies from saved raw replies."""
from pathlib import Path
import csv,hashlib,json,re
H=Path(__file__).resolve().parent;W=H.parents[2]
sha=lambda b:hashlib.sha256(b).hexdigest()
enc=lambda x:json.dumps(x,ensure_ascii=False,sort_keys=True,separators=(',',':')).encode('utf-8')
ARCH=['low_confidence','wifi_dominant','ble_rid_dominant','mobility_dominant','video_dominant','mixed_high_risk']
inputs={'main':'revision_work/analysis/replay_inputs/llm_runs/qwen_qwen-plus/policies.jsonl',
        'held_out':'revision_work/analysis/replay_inputs/results/ood_mission_semantics/qwen_ood_policies.jsonl'}
rows=[];input_hashes={}
for cohort,name in inputs.items():
 p=W/name;input_hashes[name]=sha(p.read_bytes())
 records=[json.loads(s) for s in p.read_text(encoding='utf-8').splitlines() if s.strip()]
 assert len(records)==({'main':30,'held_out':48}[cohort])
 for r in records:
  raw=r['raw_text'].strip()
  if raw.startswith('```'):
   raw=re.sub(r'^```(?:json)?','',raw,flags=re.I).strip();raw=re.sub(r'```$','',raw).strip()
  try:obj=json.loads(raw)
  except json.JSONDecodeError:obj=json.loads(raw[raw.index('{'):raw.rindex('}')+1])
  actions=obj.get('archetype_actions',{});actions=actions if isinstance(actions,dict) else {}
  normalized={}
  for a in ARCH:
   x=actions.get(a,[]);x=[x] if isinstance(x,str) else x;x=x if isinstance(x,list) else []
   normalized[a]=[str(v) for v in x if str(v).strip()] or ['Observe','FallbackProtect']
  fallback=obj.get('fallback_actions',['FallbackProtect','Observe']);fallback=[fallback] if isinstance(fallback,str) else fallback
  expected={'mission_id':r['mission_id'],'archetype_actions':normalized,'fallback_actions':[str(x) for x in fallback],'notes':str(obj.get('notes','')),'raw':obj}
  assert expected==r['policy'],r['mission_id']
  assert r['model']=='qwen-plus' and r['temperature']==0.2
  assert [m['role'] for m in r['messages']]==['system','user']
  rows.append({'cohort':cohort,'mission_id':r['mission_id'],'record_model':r['model'],'record_temperature':r['temperature'],
               'messages_sha256':sha(enc(r['messages'])),'raw_text_sha256':sha(r['raw_text'].encode('utf-8')),'parsed_policy_sha256':sha(enc(r['policy'])),
               'raw_to_parsed_exact':True,'immutable_snapshot_recorded':any(k in r for k in ['model_snapshot','model_version','system_fingerprint'])})
code=W/'revision_work/analysis/replay_inputs/paper7_llm_candidate_experiment.py'
input_hashes[code.relative_to(W).as_posix()]=sha(code.read_bytes())
assert '"response_format": {"type": "json_object"}' in code.read_text(encoding='utf-8')
with (H/'historical_policy_reconstruction.csv').open('w',encoding='utf-8',newline='') as f:
 c=csv.DictWriter(f,fieldnames=list(rows[0]));c.writeheader();c.writerows(rows)
result={'passed':True,'main_records':30,'held_out_records':48,'all_raw_to_parsed_exact':True,'all_messages_saved':True,
        'explicit_historical_request_fields':['model','messages','temperature','response_format'],
        'response_format_provenance':'Preserved call implementation, not a separately stored field in each policy record.',
        'immutable_snapshot_present':sum(r['immutable_snapshot_recorded'] for r in rows),
        'provider_default_values_recoverable':False,'reconstruction_api_calls':0,'input_sha256':input_hashes,
        'interpretation':'Reconstruction validates exact archived output-to-policy normalization. It neither recreates an unavailable provider snapshot nor guarantees identical output from a fresh request.'}
(H/'verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps({'passed':True,'records_reconstructed':len(rows),'model_calls':0},ensure_ascii=False))
