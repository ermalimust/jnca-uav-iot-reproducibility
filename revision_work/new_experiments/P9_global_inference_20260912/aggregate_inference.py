"""Offline all-row multiplicity sensitivity; never writes source experiments."""
import argparse
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
WORK = HERE.parents[2]
BASE = [
    dict(family='heldout', path='revision_work/analysis/ood_mission_cluster_tests.csv', count=9, p='p_raw', holm='p_holm', bh='p_bh', unit='mission', n=48),
    dict(family='triggered', path='revision_work/analysis/followup_stats/triggered_tests.csv', count=6, p='p_raw', holm='p_holm', bh='p_bh', unit='mission', n=12),
    dict(family='main_crossed', path='revision_work/analysis/followup_stats/pairwise_tests.csv', count=14, p='p_raw', holm='p_holm', bh='p_bh', unit='mission'),
    dict(family='prompts', path='revision_work/new_experiments/P4_prompt_completion_20260912/prompt_paired_inference.csv', count=9, p='p_raw', holm='p_holm_9', unit='mission', n=48),
    dict(family='components', path='revision_work/new_experiments/P4_prompt_completion_20260912/component_check_20260912/component_paired_inference.csv', count=8, p='p_raw', holm='p_holm_8', unit='mission', n=48),
]

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def safe_path(relative):
    path=(WORK/relative).resolve()
    if not path.is_relative_to(WORK): raise ValueError('Source path outside current workspace')
    return path

def save_json(path, obj):
    path.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')

def save_csv(path, rows):
    assert rows
    with path.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def adjust(raw):
    n=len(raw); order=sorted(range(n), key=lambda i:raw[i]); holm=[0.]*n;bh=[0.]*n
    high=0.
    for j,i in enumerate(order):
        high=max(high, (n-j)*raw[i]);holm[i]=min(1.,high)
    low=1.
    for j in range(n-1,-1,-1):
        i=order[j];low=min(low,n*raw[i]/(j+1));bh[i]=min(1.,low)
    return holm,bh

def read_source(spec):
    path=safe_path(spec['path'])
    if 'sha256' in spec: assert digest(path)==spec['sha256'], 'Registered source changed: '+spec['path']
    source=list(csv.DictReader(path.open(encoding='utf-8-sig')))
    assert len(source)==spec['count'],(spec['family'],len(source),spec['count'])
    rows=[]
    columns=spec.get('columns',{})
    def take(r,name,*fallback,default=''):
        for key in [columns.get(name,name),*fallback]:
            if key in r and r[key]!='':return r[key]
        return default
    for j,r in enumerate(source,2):
        raw=r[spec['p']];p=float(raw);assert math.isfinite(p) and 0<=p<=1
        method=take(r,'method','method_a','variant',default='guarded')
        reference=take(r,'reference','method_b',default='direct')
        contrast=take(r,'contrast_id','comparison',default=method+' - '+reference)
        if spec['family']=='components': contrast=take(r,'variant')+': guarded - direct'
        metric=take(r,'metric'); assert metric
        test_id=spec['family']+'::'+contrast+'::'+metric
        rows.append(dict(test_id=test_id,family=spec['family'],contrast=contrast,metric=metric,
                         effect=take(r,'delta','difference'),ci_low=take(r,'ci_low'),ci_high=take(r,'ci_high'),
                         inference_unit=take(r,'inference_unit',default=spec.get('unit','')),
                         n=take(r,'n',default=spec.get('n','')),p_raw=raw,
                         p_holm_source=r.get(spec.get('holm',''),'') or '',p_bh_source=r.get(spec.get('bh',''),'') or '',
                         source_path=spec['path'],source_row=j))
    return rows

def tex_escape(s):
    return str(s).replace('\\','\\textbackslash{}').replace('_','\\_').replace('%','\\%').replace('&','\\&')

def pformat(x):
    x=float(x)
    if x<.0001:return f'{x:.3g}'
    return f'{x:.4f}'

def make_tables(rows,tag):
    header=['% Generated from all included raw-p rows. Retrospective sensitivity, not preplanned inference.',
            '\\begin{longtable}{p{0.49\\linewidth}rrrr}',
            '\\caption{Pooled multiplicity sensitivity over '+str(len(rows))+' reported primary tests. All original signs and raw probabilities are retained. $p_{H,F}$ denotes original-family Holm; $p_{H,P}$ and $p_{BH,P}$ denote pooled adjustments.}\\label{tab:pooled-'+tag+'}\\\\',
            '\\toprule Comparison and outcome & $p$ & $p_{H,F}$ & $p_{H,P}$ & $p_{BH,P}$ \\\\ \\midrule',
            '\\endfirsthead','\\toprule Comparison and outcome & $p$ & $p_{H,F}$ & $p_{H,P}$ & $p_{BH,P}$ \\\\ \\midrule','\\endhead']
    for r in rows:
        header.append(tex_escape(r['family']+': '+r['contrast']+'; '+r['metric'])+' & '+' & '.join(pformat(r[k]) for k in ['p_raw','p_holm_family','p_holm_pooled','p_bh_pooled'])+' \\\\')
    header+=['\\bottomrule','\\end{longtable}']
    (HERE/f'pooled_{tag}_all_tests.tex').write_text('\n'.join(header)+'\n',encoding='utf-8')

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--expanded',type=Path);args=ap.parse_args()
    specs=[dict(s) for s in BASE]
    registration=None
    if args.expanded:
        registration=json.loads(args.expanded.read_text(encoding='utf-8'))
        assert registration['frozen_before_aggregate'] is True
        assert registration['all_planned_primary_rows_included'] is True
        for spec in registration['sources']:
            assert spec['count']>0 and spec['sha256'] and spec['protocol_path'] and spec['protocol_sha256']
            assert digest(safe_path(spec['protocol_path']))==spec['protocol_sha256']
        specs+=registration['sources']
    rows=[]
    for spec in specs: rows+=read_source(spec)
    assert len({r['test_id'] for r in rows})==len(rows), 'Duplicate primary test identifiers'
    groups=defaultdict(list)
    for r in rows:groups[r['family']].append(r)
    for members in groups.values():
        h,b=adjust([float(r['p_raw']) for r in members])
        for r,hh,bb in zip(members,h,b):
            r['p_holm_family']=hh;r['p_bh_family']=bb
            for col,value in [('p_holm_source',hh),('p_bh_source',bb)]:
                if r[col]!='':assert math.isclose(float(r[col]),value,rel_tol=2e-12,abs_tol=2e-14),(r['test_id'],col)
    h,b=adjust([float(r['p_raw']) for r in rows])
    for r,hh,bb in zip(rows,h,b):
        r['p_holm_pooled']=hh;r['p_bh_pooled']=bb
        r['family_holm_reject_005']=r['p_holm_family']<.05
        r['pooled_holm_reject_005']=hh<.05
        r['pooled_bh_reject_005']=bb<.05
        r['holm_decision_changed']=r['family_holm_reject_005']!=r['pooled_holm_reject_005']
    tag='expanded_'+str(len(rows)) if args.expanded else '46'
    assert args.expanded or len(rows)==46
    changed=[r for r in rows if r['holm_decision_changed']]
    family_summary=[dict(family=f,tests=len(members),family_holm_reject=sum(r['family_holm_reject_005'] for r in members),
                         pooled_holm_reject=sum(r['pooled_holm_reject_005'] for r in members),
                         pooled_bh_reject=sum(r['pooled_bh_reject_005'] for r in members),
                         holm_decision_changes=sum(r['holm_decision_changed'] for r in members)) for f,members in groups.items()]
    save_csv(HERE/f'pooled_{tag}_tests.csv',rows)
    if changed:save_csv(HERE/f'affected_{tag}_tests.csv',changed)
    save_csv(HERE/f'family_summary_{tag}.csv',family_summary)
    manifest=dict(tag=tag,protocol_sha256=digest(HERE/'protocol.json'),
                  registration_path=str(args.expanded.name) if args.expanded else None,
                  registration_sha256=digest(args.expanded) if args.expanded else None,
                  sources=[dict(**s,actual_sha256=digest(safe_path(s['path']))) for s in specs])
    save_json(HERE/f'input_manifest_{tag}.json',manifest)
    summary=dict(tag=tag,analysis_timing='Retrospective multiplicity sensitivity; not a preplanned confirmatory family.',
                 tests=len(rows),families=family_summary,within_holm_rejections=sum(r['family_holm_reject_005'] for r in rows),
                 pooled_holm_rejections=sum(r['pooled_holm_reject_005'] for r in rows),pooled_bh_rejections=sum(r['pooled_bh_reject_005'] for r in rows),
                 changed_tests=changed,existing_component_tests=[{k:r[k] for k in ['test_id','effect','p_raw','p_holm_family','p_holm_pooled','p_bh_pooled']} for r in rows if r['family']=='components'])
    save_json(HERE/f'summary_{tag}.json',summary)
    make_tables(rows,tag)
    md=[f'# Pooled multiplicity sensitivity: {len(rows)} tests','',summary['analysis_timing'],'',
        'All primary raw-p rows from the registered families are included, regardless of effect direction. Repeated clustering sensitivities are not counted as new hypotheses. Pointwise intervals are unchanged. Holm at 0.05 is primary; BH values are supplied separately.','',
        '| Family | Tests | Within-family Holm | Pooled Holm | Pooled BH |','|---|---:|---:|---:|---:|']
    for f in family_summary:md.append(f"| {f['family']} | {f['tests']} | {f['family_holm_reject']} | {f['pooled_holm_reject']} | {f['pooled_bh_reject']} |")
    md+=['','## Decisions that depend on the adjustment family','','| Contrast | Outcome | Effect | Family Holm | Pooled Holm | Pooled BH |','|---|---|---:|---:|---:|---:|']
    for r in changed:md.append(f"| {r['family']}: {r['contrast']} | {r['metric']} | {r['effect']} | {float(r['p_holm_family']):.8g} | {r['p_holm_pooled']:.8g} | {r['p_bh_pooled']:.8g} |")
    md+=['','## Complete test inventory','','| Family / contrast | Outcome | Raw p | Family Holm | Pooled Holm | Pooled BH |','|---|---|---:|---:|---:|---:|']
    for r in rows:md.append(f"| {r['family']}: {r['contrast']} | {r['metric']} | {r['p_raw']} | {r['p_holm_family']:.8g} | {r['p_holm_pooled']:.8g} | {r['p_bh_pooled']:.8g} |")
    (HERE/f'report_{tag}.md').write_text('\n'.join(md)+'\n',encoding='utf-8')
    print(json.dumps(dict(tag=tag,tests=len(rows),changed=len(changed),family_holm=summary['within_holm_rejections'],pooled_holm=summary['pooled_holm_rejections'],pooled_bh=summary['pooled_bh_rejections'])))

if __name__=='__main__':main()
