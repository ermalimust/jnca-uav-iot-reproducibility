"""Expanded pooled sensitivity: unchanged 78 + all frozen P11 seven + P12 eight.

Writes a NEW analysis only; never overwrites the earlier P9 family or raw p-values.
"""
from pathlib import Path
import csv,hashlib,json,collections
from fractions import Fraction
HERE=Path(__file__).resolve().parent
OUT=HERE/'pooled_inference_93'
OLD=HERE.parent/'P9_global_inference_20260912'
P12=HERE.parent/'P12_context_threshold_followup_20260912'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return list(csv.DictReader(p.open(encoding='utf-8-sig')))
def write(name,rows):
    with (OUT/name).open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def save(name,obj):(OUT/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def adjusted(values,kind):
    n=len(values);order=sorted(range(n),key=lambda i:values[i]);out=[0.]*n
    if kind=='holm':
        acc=0
        for rank,i in enumerate(order):acc=max(acc,min(1.,values[i]*(n-rank)));out[i]=acc
    else:
        acc=1.
        for rank in reversed(range(n)):
            i=order[rank];acc=min(acc,values[i]*n/(rank+1));out[i]=acc
    return out
def independent_fraction(values,kind):
    v=[Fraction(str(x)) for x in values];ordered=sorted(v);m=len(v)
    if kind=='holm':return [float(min(Fraction(1),max((m-j)*p for j,p in enumerate(ordered) if p<=value))) for value in v]
    return [float(min(Fraction(1),min(Fraction(m,j+1)*p for j,p in enumerate(ordered) if p>=value))) for value in v]
def main():
    OUT.mkdir(exist_ok=True)
    oldpath=OLD/'pooled_expanded_78_tests.csv';p11path=HERE/'primary_contrasts.csv';p12path=P12/'results/primary_interactions.csv'
    assert sha(oldpath)=='b0c7bdddf26484f217dd038604db2a16d0e5cba8c39f0137818b1d09ce81128a'
    assert sha(p12path)=='1972672cbb5df3ef12fa6220c5259462a1cecba1186a7397e48b3f84439c6bd8'
    old=read(oldpath);new11=read(p11path);new12=read(p12path);assert (len(old),len(new11),len(new12))==(78,7,8)
    planned={r['contrast_id'] for r in read(HERE/'planned_hypotheses.csv')};assert {r['contrast_id'] for r in new11}==planned
    rows=[]
    for r in old:
        rows.append({k:r[k] for k in ['test_id','family','contrast','metric','effect','ci_low','ci_high','inference_unit','n','p_raw','source_path','source_row']})
        rows[-1].update(p_holm_prior_78=r['p_holm_pooled'],p_bh_prior_78=r['p_bh_pooled'])
    for fam,new,path in [('P11_react_reference',new11,p11path),('P12_context_threshold',new12,p12path)]:
        for i,r in enumerate(new,2):
            rows.append({'test_id':fam+'::'+r['contrast_id'],'family':fam,'contrast':r['contrast_id'],'metric':r['metric'],'effect':r['delta'],'ci_low':r['ci_low'],'ci_high':r['ci_high'],
                'inference_unit':r['inference_unit'],'n':'48','p_raw':r['p_raw'],'source_path':path.relative_to(HERE.parents[2]).as_posix(),'source_row':str(i),'p_holm_prior_78':'','p_bh_prior_78':''})
    assert len(rows)==93 and len({r['test_id'] for r in rows})==93
    for fam in dict.fromkeys(r['family'] for r in rows):
        ii=[i for i,r in enumerate(rows) if r['family']==fam];p=[float(rows[i]['p_raw']) for i in ii]
        for kind in ['holm','bh']:
            got=adjusted(p,kind);exact=independent_fraction(p,kind);assert max(abs(a-b) for a,b in zip(got,exact))<1e-14
            for i,v in zip(ii,got):rows[i]['p_'+kind+'_source_family']=v
    p=[float(r['p_raw']) for r in rows]
    for kind in ['holm','bh']:
        got=adjusted(p,kind);exact=independent_fraction(p,kind);assert max(abs(a-b) for a,b in zip(got,exact))<1e-14
        for r,v in zip(rows,got):r['p_'+kind+'_pooled_93']=v
    for r in rows:
        r['holm_reject_005']=r['p_holm_pooled_93']<.05;r['bh_reject_005']=r['p_bh_pooled_93']<.05
        r['holm_changed_from_source_family']=(r['p_holm_source_family']<.05)!=r['holm_reject_005']
        r['holm_changed_from_prior_78']=bool(r['p_holm_prior_78']) and ((float(r['p_holm_prior_78'])<.05)!=r['holm_reject_005'])
    write('pooled_93_tests.csv',rows)
    byfamily=[]
    for fam in dict.fromkeys(r['family'] for r in rows):
        rr=[r for r in rows if r['family']==fam]
        byfamily.append({'family':fam,'tests':len(rr),'raw_lt_005':sum(float(r['p_raw'])<.05 for r in rr),'source_holm_lt_005':sum(r['p_holm_source_family']<.05 for r in rr),'pooled_holm_lt_005':sum(r['holm_reject_005'] for r in rr),'pooled_bh_lt_005':sum(r['bh_reject_005'] for r in rr)})
    write('family_summary_93.csv',byfamily)
    newrows=[r for r in rows if r['family'] in ['P11_react_reference','P12_context_threshold']];write('new_15_tests_for_integration.csv',newrows)
    five={r['test_id'] for r in read(OLD/'five_original_sensitive_values_expanded.csv')};write('five_original_sensitive_values_93.csv',[r for r in rows if r['test_id'] in five])
    changed=[r for r in rows if r['holm_changed_from_prior_78']]
    if changed:write('changed_from_78.csv',changed)
    # Validate all original string-valued raw p/effect/CI cells, rather than only rounded floats.
    for i,r in enumerate(old):
        for k in ['p_raw','effect','ci_low','ci_high','test_id']:assert rows[i][k]==r[k]
    sources={p.relative_to(HERE.parents[2]).as_posix():sha(p) for p in [oldpath,p11path,p12path,HERE/'protocol.json',HERE/'schema_amendment_freeze.json',P12/'protocol.json']}
    summary={'total_tests':93,'source_families':byfamily,'pooled_holm_rejections':sum(r['holm_reject_005'] for r in rows),'pooled_bh_rejections':sum(r['bh_reject_005'] for r in rows),'original_78_holm_decisions_changed':len(changed),'changed_original_test_ids':[r['test_id'] for r in changed],'primary_new_tests':15,
        'scope':'Retrospective expanded sensitivity to family definition. The earlier 78 are unchanged, P11 seven were specified before generation and its symmetric format amendment before performance evaluation, P12 eight are frozen secondary interactions. This is not a prospectively designed 93-test study. Strict-parser and 24-theme sensitivities are the same hypotheses and are not counted again; eighteen P11 non-generative comparator effects carry descriptive pointwise CIs only.',
        'input_hashes':sources,'independent_fraction_holm_bh_all_93_and_each_family':True,'old_78_rawp_effect_ci_strings_unchanged':True,'all_frozen_new_primary_rows_included':True,'prior_78_file_unchanged':sha(oldpath)=='b0c7bdddf26484f217dd038604db2a16d0e5cba8c39f0137818b1d09ce81128a'}
    save('summary_93.json',summary);save('verification_93.json',{'all_passed':True,**summary,'script_sha256':sha(Path(__file__))})
    lines=['# Expanded pooled inference: 93 tests','',summary['scope'],'',f"Holm rejects {summary['pooled_holm_rejections']}/93 and BH rejects {summary['pooled_bh_rejections']}/93 at 0.05. {len(changed)} original 78-test Holm decisions change.",'','| Family | Tests | Source-family Holm | Pooled Holm | Pooled BH |','|---|---:|---:|---:|---:|']
    lines += [f"| {r['family']} | {r['tests']} | {r['source_holm_lt_005']} | {r['pooled_holm_lt_005']} | {r['pooled_bh_lt_005']} |" for r in byfamily]
    lines += ['','## New frozen comparisons','','| Contrast | Effect | Raw p | Holm 93 | BH 93 |','|---|---:|---:|---:|---:|']
    lines += [f"| {r['contrast']} | {float(r['effect']):.8f} | {float(r['p_raw']):.8g} | {r['p_holm_pooled_93']:.8g} | {r['p_bh_pooled_93']:.8g} |" for r in newrows]
    (OUT/'report_93.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(summary,indent=2),flush=True)
if __name__=='__main__':main()
