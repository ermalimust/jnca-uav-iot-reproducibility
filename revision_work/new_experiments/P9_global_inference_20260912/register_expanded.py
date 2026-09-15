"""Register the complete frozen P8/P10 primary inventories for pooled analysis."""
import csv
import hashlib
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
WORK=HERE.parents[2]

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

sources=[
    dict(family='matched_interfaces',path='revision_work/new_experiments/P8_matched_interfaces_20260912/matched_paired_inference.csv',
         count=28,p='p_raw',unit='mission',n=48,
         protocol_path='revision_work/new_experiments/P8_matched_interfaces_20260912/protocol.json'),
    dict(family='matched_temporal',path='revision_work/new_experiments/P10_temporal_matched_20260912/planned_contrasts.csv',
         count=4,p='raw_p',unit='episode',n=96,
         columns={'metric':'outcome','delta':'estimate','inference_unit':'inferential_unit','n':'independent_units'},
         protocol_path='revision_work/new_experiments/P10_temporal_matched_20260912/protocol.json')
]
for spec in sources:
    source=WORK/spec['path'];protocol=WORK/spec['protocol_path']
    if not source.exists():raise SystemExit('Frozen source not yet available: '+spec['path'])
    rows=list(csv.DictReader(source.open(encoding='utf-8-sig')))
    assert len(rows)==spec['count'],(source,len(rows),spec['count'])
    specification=json.loads(protocol.read_text(encoding='utf-8'))
    if spec['family']=='matched_interfaces':
        assert specification['planned_primary_tests']==28
        assert len(specification['comparison_pairs'])*len(specification['comparison_metrics'])+len(specification['component_pairs'])*len(specification['component_metrics'])==28
        expected={(a,b,m) for a,b in specification['comparison_pairs'] for m in specification['comparison_metrics']}
        expected|={(a,b,m) for a,b in specification['component_pairs'] for m in specification['component_metrics']}
        actual={(r['method'],r['reference'],r['metric']) for r in rows}
        assert actual==expected,dict(missing=sorted(expected-actual),unexpected=sorted(actual-expected))
        for col in ['p_holm_28','p_holm','holm_p']:
            if col in rows[0]:spec['holm']=col;break
        for col in ['p_bh_28','p_bh','bh_p']:
            if col in rows[0]:spec['bh']=col;break
    else:
        assert [r['contrast_id'] for r in rows]==[r['id'] for r in specification['planned_contrasts']]
        assert all(r['planned']=='True' for r in rows)
    spec['sha256']=sha(source);spec['protocol_sha256']=sha(protocol)
receipt=dict(date='2026-09-12',frozen_before_aggregate=True,all_planned_primary_rows_included=True,
             expected_total=78,baseline_tests=46,new_primary_tests=32,
             timing='Retrospective expanded sensitivity after all P8/P10 outputs are frozen; no claim that the pooled family was preplanned before the original experiments.',
             inclusion='Every planned primary P8 and P10 contrast, regardless of sign or significance. Alternate24-pair analyses not duplicated.',sources=sources)
(HERE/'expanded_sources.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps(dict(registered=True,expected_total=78,source_counts={s['family']:s['count'] for s in sources})))
