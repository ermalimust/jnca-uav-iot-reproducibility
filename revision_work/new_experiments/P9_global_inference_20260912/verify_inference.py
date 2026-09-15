"""Independent Decimal validation of source identity and multiplicity values."""
import argparse
import csv
import hashlib
import json
from collections import defaultdict
from decimal import Decimal, localcontext
from pathlib import Path

HERE=Path(__file__).resolve().parent
WORK=HERE.parents[2]

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def decimal_adjust(entries):
    # Deliberately direct per-test definitions: no cumulative algorithm from aggregator.
    values=sorted(Decimal(r['p_raw']) for r in entries);n=len(values)
    result=[]
    with localcontext() as ctx:
        ctx.prec=60
        for row in entries:
            p=Decimal(row['p_raw'])
            holm=min(Decimal(1),max(Decimal(n-j)*v for j,v in enumerate(values) if v<=p))
            bh=min(Decimal(1),min(Decimal(n)*v/Decimal(j+1) for j,v in enumerate(values) if v>=p))
            result.append((holm,bh))
    return result

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--tag',default='46');a=ap.parse_args()
    path=HERE/f'pooled_{a.tag}_tests.csv'
    rows=list(csv.DictReader(path.open(encoding='utf-8-sig')))
    manifest=json.loads((HERE/f'input_manifest_{a.tag}.json').read_text(encoding='utf-8'))
    assert manifest['protocol_sha256']==sha(HERE/'protocol.json')
    assert len({r['test_id'] for r in rows})==len(rows)
    checked_sources={}
    for s in manifest['sources']:
        p=(WORK/s['path']).resolve();assert p.is_relative_to(WORK)
        assert sha(p)==s['actual_sha256']
        source=list(csv.DictReader(p.open(encoding='utf-8-sig')));assert len(source)==s['count']
        members=[r for r in rows if r['source_path']==s['path']]
        assert len(members)==len(source)
        for result,original in zip(members,source):
            assert result['p_raw']==original[s['p']],(result['test_id'],'raw p string changed')
        if s.get('protocol_path'): assert sha(WORK/s['protocol_path'])==s['protocol_sha256']
        checked_sources[s['path']]=sha(p)
    if manifest['registration_path']:
        assert sha(HERE/manifest['registration_path'])==manifest['registration_sha256']
    groups=defaultdict(list)
    for r in rows:groups[r['family']].append(r)
    comparisons=0;max_error=Decimal(0)
    def check(actual,expected):
        nonlocal comparisons,max_error
        error=abs(Decimal(actual)-expected);max_error=max(max_error,error)
        assert error<Decimal('2e-14'),(actual,str(expected))
        comparisons+=1
    for members in groups.values():
        for r,(h,b) in zip(members,decimal_adjust(members)):
            check(r['p_holm_family'],h);check(r['p_bh_family'],b)
            if r['p_holm_source']:check(r['p_holm_source'],h)
            if r['p_bh_source']:check(r['p_bh_source'],b)
    for r,(h,b) in zip(rows,decimal_adjust(rows)):
        check(r['p_holm_pooled'],h);check(r['p_bh_pooled'],b)
        assert (r['pooled_holm_reject_005']=='True')==(h<Decimal('.05'))
        assert (r['pooled_bh_reject_005']=='True')==(b<Decimal('.05'))
        assert (r['holm_decision_changed']=='True')==((r['family_holm_reject_005']=='True')!=(h<Decimal('.05')))
    summary=json.loads((HERE/f'summary_{a.tag}.json').read_text(encoding='utf-8'))
    assert summary['tests']==len(rows)
    assert summary['pooled_holm_rejections']==sum(r['pooled_holm_reject_005']=='True' for r in rows)
    result=dict(verified=True,tag=a.tag,rows=len(rows),adjusted_values_checked=comparisons,
                method='60-digit Decimal; independent O(n^2) per-row Holm maximum and BH minimum over ranked raw probabilities.',
                max_absolute_float_decimal_error=str(max_error),complete_raw_p_strings_preserved=True,
                all_sources_unchanged=True,sources=checked_sources,pooled_csv_sha256=sha(path))
    (HERE/f'verification_{a.tag}.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:result[k] for k in ['verified','tag','rows','adjusted_values_checked','max_absolute_float_decimal_error']}))

if __name__=='__main__':main()
