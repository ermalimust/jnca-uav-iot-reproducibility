"""Portable, offline exact joint threshold/cost action-radius analysis.

Run with Python + NumPy. Existing inputs are read only. The archived module is
parsed for literal constants and never imported. No model API calls occur.
"""
import argparse,ast,csv,hashlib,json,math,random,time
from pathlib import Path
from decimal import Decimal as D
from fractions import Fraction as F
from datetime import datetime,timezone
import numpy as np
from exact_radius_core import (PATTERNS,THETA,values_bits,admission,choose,solve,
                              boundary_flip_possible,scalar_solve)

FILES={
 'audit':'revision_work/analysis/replay_inputs/results/audit_records/qwen_qwen-plus_guarded_audit_records.jsonl',
 'core':'revision_work/analysis/replay_inputs/paper7_agentic_feasibility.py',
 'margins':'revision_work/analysis/followup_margins/decision_margins.csv'}
ACTIONS=('Observe','WiFiRelief','BLEAvoid','LinkAdapt','VideoShape','FallbackProtect')
CAUSES=('W','B','M','V')

def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for chunk in iter(lambda:f.read(1048576),b''):h.update(chunk)
    return h.hexdigest()

def writejson(p,value):
    p.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')

def writecsv(p,rows):
    with p.open('w',encoding='utf-8',newline='') as f:
        wr=csv.DictWriter(f,fieldnames=list(rows[0]));wr.writeheader();wr.writerows(rows)

def scalar(v):return D(str(v))
def scaled(v,s):
    x=scalar(v)*s
    assert x==x.to_integral_value(),(v,s)
    return int(x)

def root_find(value):
    if value:
        p=Path(value).resolve();assert (p/FILES['audit']).is_file();return p
    for p in Path(__file__).resolve().parents:
        if (p/FILES['audit']).is_file():return p
    raise SystemExit('Specify --workspace-root containing revision_work.')

def constants(p):
    found={}
    for n in ast.parse(p.read_text(encoding='utf-8')).body:
        if isinstance(n,ast.Assign):names=[x.id for x in n.targets if isinstance(x,ast.Name)]
        elif isinstance(n,ast.AnnAssign) and isinstance(n.target,ast.Name):names=[n.target.id]
        else:continue
        for name in names:
            if name in {'BASE_COSTS','ACTION_OVERHEAD','SUPPORTED_ACTIONS','CAUSES'}:found[name]=ast.literal_eval(n.value)
    assert tuple(found['SUPPORTED_ACTIONS'])==ACTIONS and tuple(found['CAUSES'])==CAUSES
    return found

def load_inputs(root):
    co=constants(root/FILES['core']);q=[];flags=[];meta=[];orders=[];costs=[];mission_costs={};archived=[]
    with (root/FILES['audit']).open(encoding='utf-8') as f:
        for rownum,line in enumerate(f,1):
            r=json.loads(line);m=r['mission'];mid=m['mission_id'];g=[bool(m['guards'].get(k,False)) for k in ['safety','rid','video','energy']]
            if mid not in mission_costs:
                c=np.array([[scaled(x,10) for x in co['BASE_COSTS'][m['gold_cost_profile']][a]] for a in ACTIONS],dtype=np.int64)
                if g[1]:c[0,1]+=40;c[2,1]=5
                if g[2]:c[5]+=np.array([10,10,10,25]);c[4,3]=6
                if g[3]:c[3]+=np.array([10,10,0,10])
                mission_costs[mid]=c
            costs.append(mission_costs[mid]);q.append([scaled(r['posterior'][k],1000000) for k in CAUSES]);flags.append(g)
            orders.append(r['candidate_actions'])
            archived.append(ACTIONS.index(r['selected_action']) if r['selected_action'] in ACTIONS else 6)
            meta.append({'mission':mid,'decision_id':r['decision_id'],'seed':r['seed'],'decision_index':r['decision_index'],
                         'source_jsonl_line':rownum,'source_scenario':r['source_window']['scenario_id'],'source_window':r['source_window']['window_id']})
    q=np.array(q,dtype=np.int64);flags=np.array(flags,bool);cm=np.stack(costs)
    n=len(q);exposed=np.zeros((n,6),bool);rank=np.full((n,6),9999,np.int64)
    for i,order in enumerate(orders):
        for j,a in enumerate(order):
            if a in ACTIONS:
                ix=ACTIONS.index(a);exposed[i,ix]=True;rank[i,ix]=min(rank[i,ix],j)
    effects=np.einsum('nm,nam->na',q,cm,dtype=np.int64)
    hs=np.array([scaled(co['ACTION_OVERHEAD'][a],10000000) for a in ACTIONS],dtype=np.int64)
    overhead=np.broadcast_to(hs,effects.shape).copy();scores=effects+overhead
    z,bits=values_bits(q);winner,S=choose(scores,admission(bits,flags),exposed,rank)
    archived=np.array(archived,np.int64)
    assert np.array_equal(winner,archived),('saved-q rational nominal mismatch',int((winner!=archived).sum()))
    assert n==54000 and len(mission_costs)==30
    prior=[]
    with (root/FILES['margins']).open(encoding='utf-8-sig',newline='') as f:
        for i,r in enumerate(csv.DictReader(f)):
            assert r['decision_id']==meta[i]['decision_id']
            prior.append(min(float(r['coefficient_radius']),float(r['guard_relative_radius'])))
    assert len(prior)==n
    return dict(q=q,flags=flags,effects=effects,overhead=overhead,exposed=exposed,rank=rank,winner=winner,meta=meta,prior=np.array(prior),orders=orders)

def arrays(d):return [d[k] for k in ['q','flags','effects','overhead','exposed','rank','winner']]

def synthetic_cases():
    cases=[]
    def add(name,q,flags,order,e,h,expected,att):
        ef=np.zeros(6,np.int64);oh=np.zeros(6,np.int64)
        for a,v in e.items():ef[a]=v
        for a,v in h.items():oh[a]=v
        ex=np.zeros(6,bool);rank=np.full(6,9999,np.int64)
        for i,a in enumerate(order):ex[a]=True;rank[a]=min(rank[a],i)
        qq=np.array([q],np.int64);ff=np.array([flags],bool)
        _,bits=values_bits(qq);winner,_=choose((ef+oh)[None,:],admission(bits,ff),ex[None,:],rank[None,:])
        cases.append(dict(name=name,q=qq[0],flags=ff[0],effects=ef,overhead=oh,exposed=ex,rank=rank,winner=winner[0],expected=expected,att=att))
    off=[False]*4
    add('single_unrejectable_BLE',[1000000,0,0,0],[True]*4,[2],{2:1},{},None,False)
    add('clipping_plateau_later_rival',[1000000,0,0,0],off,[0,1],{1:1},{},None,False)
    add('clipping_plateau_earlier_rival',[1000000,0,0,0],off,[1,0],{1:1},{},F(1),True)
    add('crossing_above_one',[1000000,0,0,0],off,[0,1],{0:1},{1:5},F(4),False)
    add('nominal_tie_strict_later',[1000000,0,0,0],off,[0,1],{0:1,1:1},{},F(0),False)
    add('zero_radius_open_threshold',[420000,0,0,0],[True,False,False,False],[0,1],{1:1},{},F(0),False)
    add('closed_threshold_rejects_winner',[200000,0,0,0],[True,False,False,False],[0,1],{},{1:2},F(11,21),True)
    add('zero_evidence_positive_threshold_unreachable',[0,0,0,0],[True,False,False,False],[0,1],{},{1:1},None,False)
    add('empty_list_fallback_to_escalation',[0,0,0,400000],[False,False,True,False],[],{},{5:1},F(1,7),False)
    add('empty_list_permanent_escalation',[0,0,0,0],[False,False,True,False],[],{},{5:1},None,False)
    add('coincident_open_guard_closed_cost',[840000,0,0,0],[True,False,False,False],[0,2],{0:1,2:1},{0:2},F(1),False)
    add('coincident_closed_guard_strict_cost',[0,0,170000,0],[False,False,False,True],[2,3],{2:1,3:1},{3:1},F(1,2),False)
    add('late_constant_lower_overhead',[1000000,0,0,0],off,[0,1],{},{0:3,1:1},None,False)
    # Initially identical actions in the string list retain first occurrences.
    add('duplicate_names_stable_first_position',[1000000,0,0,0],off,[1,0,1],{1:1},{},F(1),True)
    d={k:np.stack([x[k] for x in cases]) for k in ['q','flags','effects','overhead','exposed','rank']};d['winner']=np.array([x['winner'] for x in cases],np.int64)
    result=solve(*arrays(d));out=[]
    for i,c in enumerate(cases):
        r=None if result['den'][i]==0 else F(int(result['num'][i]),int(result['den'][i]));a=bool(result['attained'][i])
        independent,ia=scalar_solve(*(d[k][i] for k in ['q','flags','effects','overhead','exposed','rank','winner']))
        assert r==c['expected']==independent and a==c['att']==ia,(c['name'],r,c['expected'],independent,a,c['att'],ia)
        out.append({'case':c['name'],'radius':'inf' if r is None else str(r),'attained':a,'independent_fraction_match':True})
    return out

def validate_all(d,r):
    n=len(d['q']);finite=r['den']>0
    boundary,bound=boundary_flip_possible(*arrays(d),r['num'],r['den'])
    assert np.array_equal(boundary[finite],r['attained'][finite]),'boundary/attainment mismatch'
    # Independently check an exact interior point R/2; use radius 1 for infinity.
    inn=np.where(finite,r['num'],1);ind=np.where(finite,2*r['den'],1)
    interior,interior_bound=boundary_flip_possible(*arrays(d),inn,ind)
    assert not interior.any(),'interior flip possible'
    radius=np.divide(r['num'],r['den'],out=np.full(n,np.inf),where=finite)
    # Every finite result gets a robustly interior witness in its achieving
    # pattern, slightly above the mathematical infimum.
    eps=radius+np.maximum(1.,radius)*1e-6
    eps[~finite]=0
    z,bits=values_bits(d['q']);p=np.maximum(r['pattern'],0);patterns=PATTERNS[p]
    theta=np.broadcast_to(THETA,(n,5)).astype(float).copy()
    for j in range(5):
        changed=(patterns[:,j]!=bits[:,j])&finite
        need_le=patterns[:,j] if j<2 else ~patterns[:,j]
        closed=changed&need_le;opened=changed&~need_le
        theta[closed,j]=z[closed,j]
        upper=THETA[j]*(1+eps)
        theta[opened,j]=(z[opened,j]+upper[opened])/2
    assert np.all(theta[finite]>0)
    norm=np.max(np.abs(theta-THETA)/THETA,axis=1)
    assert np.all(norm[finite]<=eps[finite]+1e-12)
    wbits=z>=theta;wbits[:,2:]=z[:,2:]<theta[:,2:]
    assert np.array_equal(wbits[finite],patterns[finite])
    safe=np.minimum(d['winner'],5);ix=np.arange(n)
    factors=np.broadcast_to(np.maximum(0,1-eps)[:,None],(n,6)).copy()
    factors[ix,safe]=1+eps
    ws=d['effects']*factors+d['overhead']
    ok=admission(wbits,d['flags']);S=ok&d['exposed'];has=S.any(axis=1)
    best=np.min(np.where(S,ws,np.inf),axis=1);tie=S&(ws==best[:,None])
    winner=np.argmin(np.where(tie,d['rank'],999999),axis=1)
    winner[~has]=np.where(ok[~has,5],5,6)
    assert np.all(winner[finite]!=d['winner'][finite]),'finite witness failed'
    assert np.all(radius+1e-12>=d['prior']),'exact radius below old sufficient certificate'
    return {'finite':finite,'radius':radius,'witness_epsilon':eps,'witness_theta':theta/2000000.,'witness_winner':winner,
      'boundary_possible':boundary,'checks':{'rows':n,'nominal_matches_archive':n,'exact_boundary_attainment_matches':int(finite.sum()),
      'all_exact_interiors_preserve_selection':True,'finite_strictly_above_radius_witnesses':int(finite.sum()),
      'all_witness_thresholds_positive_and_within_budget':True,'all_witness_patterns_reproduced':True,
      'all_original_sufficient_certificates_respected':True,'max_exact_boundary_score_numerator_bound':bound,
      'max_exact_interior_score_numerator_bound':interior_bound}}

def scalar_spot(d,r):
    n=len(d['q']);chosen=set(random.Random(609120701).sample(range(n),256));seen=set()
    for i,m in enumerate(d['meta']):
        if m['mission'] not in seen:chosen.add(i);seen.add(m['mission'])
    finite=r['den']>0
    groups=[r['num']==0,~finite,finite&(r['num']>r['den']),finite&r['attained'],finite&~r['attained']]
    for mask in groups:chosen.update(map(int,np.flatnonzero(mask)[:8]))
    chosen.update([0,2]);rows=[]
    for i in sorted(chosen):
        radius,att=scalar_solve(*(d[k][i] for k in ['q','flags','effects','overhead','exposed','rank','winner']))
        got=None if not finite[i] else F(int(r['num'][i]),int(r['den'][i]))
        assert radius==got and att==bool(r['attained'][i]),(i,radius,got,att,r['attained'][i])
        rows.append({'row':i,'decision_id':d['meta'][i]['decision_id'],'fraction_radius':'inf' if radius is None else str(radius),
                     'infimum_attained':att,'vector_rational_match':True})
    return rows

def fmt(v):return 'inf' if not math.isfinite(float(v)) else float(v)
def quantile(v,p):
    a=np.sort(v);return fmt(a[max(0,math.ceil(p*len(a))-1)])

def summarize(d,r,v,mask):
    radii=v['radius'][mask];finite=np.isfinite(radii);n=len(radii)
    rn=r['num'][mask];rd=r['den'][mask];att=r['attained'][mask]
    out={'decisions':n,'finite':int(finite.sum()),'infinite':int((~finite).sum()),'zero_infima':int((rn==0).sum()),
         'finite_attained':int((finite&att).sum()),'finite_not_attained':int((finite&~att).sum()),
         'radius_min':quantile(radii,0),'radius_p05':quantile(radii,.05),'radius_median':quantile(radii,.5),'radius_p95':quantile(radii,.95),
         'radius_max':quantile(radii,1)}
    for name,num,den in [('10',1,10),('20',1,5)]:
        greater=(rd==0)|(rn*den>num*rd);same=(rd!=0)&(rn*den==num*rd)
        safe=greater|(same&~att)
        out['closed_box_safe_'+name+'_count']=int(safe.sum());out['closed_box_safe_'+name+'_fraction']=float(safe.mean())
        out['exact_radius_equal_'+name+'_count']=int(same.sum())
        out['prior_strict_certificate_'+name+'_count']=int((d['prior'][mask]>num/den).sum())
        out['prior_strict_certificate_'+name+'_fraction']=float((d['prior'][mask]>num/den).mean())
    prior=d['prior'][mask]
    out['strictly_larger_than_prior_certificate']=int((radii>prior+1e-12).sum())
    return out

def mission_table(rows):
    header=r'''\begin{revision}
\begin{table}[pos=!htbp]
\revisionfloatcolor
\centering
\caption{Exact joint selected-action margins on 1,800 saved decisions per mission. Radius quantiles and stable fractions are percentages. Both nonnegative cost entries and positive guard thresholds vary independently under the same relative bound, with the saved posterior, candidate rule/order, mission flags and overheads fixed. $S_{10}$ and $S_{20}$ are fractions whose selection is unchanged throughout the respective closed 10\% and 20\% boxes, including an equal-radius endpoint only if the flip infimum is not attained.}
\label{tab:mission-exact-joint-margins}
\scriptsize
\setlength{\tabcolsep}{4pt}
\begin{tabular}{lrrrr}
\toprule
Mission & $\rho_{\mathrm{joint}}$: 5th & Median & $S_{10}$ & $S_{20}$ \\
\midrule
'''
    lines=[]
    for row in sorted(rows,key=lambda r:r['mission']):
        name=row['mission'].replace('_',r'\_')
        values=[row[x]*100 for x in ['radius_p05','radius_median','closed_box_safe_10_fraction','closed_box_safe_20_fraction']]
        lines.append(name+' & '+' & '.join(f'{v:.2f}' for v in values)+r' \\')
    return header+'\n'.join(lines)+'\n'+r'''\bottomrule
\end{tabular}
\end{table}
\end{revision}
'''

def write_outputs(root,out,d,r,v,synthetic,scalar,hashes,start):
    count=len(d['q']);rows=[]
    for i,m in enumerate(d['meta']):
        finite=bool(v['finite'][i]);radius=v['radius'][i]
        safe10=not finite or r['num'][i]*10>r['den'][i] or (r['num'][i]*10==r['den'][i] and not r['attained'][i])
        safe20=not finite or r['num'][i]*5>r['den'][i] or (r['num'][i]*5==r['den'][i] and not r['attained'][i])
        row={**m,'selected_action':ACTIONS[d['winner'][i]] if d['winner'][i]<6 else 'EscalateReview',
            'exact_radius_numerator':int(r['num'][i]),'exact_radius_denominator':int(r['den'][i]),'exact_joint_radius':fmt(radius),
            'infimum_attained':bool(r['attained'][i]) if finite else '',
            'achieving_rejection_pattern':''.join('1' if b else '0' for b in PATTERNS[r['pattern'][i]]) if finite else '',
            'pattern_guard_numerator':int(r['gnum'][i]) if finite else '', 'pattern_guard_denominator':int(r['gden'][i]) if finite else '',
            'pattern_cost_numerator':int(r['cnum'][i]) if finite else '', 'pattern_cost_denominator':int(r['cden'][i]) if finite else '',
            'cost_challenger':ACTIONS[r['challenger'][i]] if r['challenger'][i]>=0 else '',
            'prior_separate_common_certificate':fmt(d['prior'][i]),'closed_box_safe_10pct':safe10,'closed_box_safe_20pct':safe20,
            'witness_epsilon':float(v['witness_epsilon'][i]) if finite else '',
            'witness_selected_action':(ACTIONS[v['witness_winner'][i]] if v['witness_winner'][i]<6 else 'EscalateReview') if finite else '',
            'boundary_flip_possible':bool(v['boundary_possible'][i]) if finite else ''}
        for j,key in enumerate(['safety','rid','video','video_risk','energy']):row['witness_threshold_'+key]=float(v['witness_theta'][i,j]) if finite else ''
        rows.append(row)
    writecsv(out/'decision_exact_joint_radii.csv',rows)
    mids=list(dict.fromkeys(m['mission'] for m in d['meta']));midarr=np.array([m['mission'] for m in d['meta']])
    summary=[{'mission':mid,**summarize(d,r,v,midarr==mid)} for mid in mids]
    writecsv(out/'mission_exact_joint_summary.csv',summary)
    (out/'mission_exact_joint_table.tex').write_text(mission_table(summary),encoding='utf-8')
    writecsv(out/'independent_scalar_checks.csv',scalar)
    writecsv(out/'synthetic_edge_cases.csv',synthetic)
    overall=summarize(d,r,v,np.ones(count,bool))
    overall.update({'missions':len(mids),'inference':'Descriptive deterministic properties of archived records; no hypothesis tests.',
       'scope':'Fixed saved q, candidate rule/order, mission flags and overhead; positive independent thresholds and nonnegative independently perturbed cost entries under one relative L-infinity budget.',
       'witness_recipe':'At epsilon strictly above radius, choose the achieving threshold pattern; increase original winner cost row by 1+epsilon and decrease all other rows by max(0,1-epsilon). Empty-list fallback/escalation ignores ranking.'})
    writejson(out/'summary.json',overall)
    verification={**v['checks'],'verified':True,'integer_rational_arithmetic':True,'comparison_integer_bound':r['comparison_integer_bound'],
       'comparison_integer_bound_squared':r['comparison_integer_bound']**2,
       'independent_fraction_rows':len(scalar),'synthetic_edge_cases':len(synthetic),'all_scalar_and_edge_checks_passed':True,
       'input_sources_unchanged':all(sha(root/p)==h for p,h in hashes.items()),'elapsed_seconds':time.perf_counter()-start}
    assert verification['input_sources_unchanged'];writejson(out/'verification.json',verification)
    files=[p for p in out.iterdir() if p.is_file() and p.name not in ['manifest.json']]
    manifest={'created_utc':datetime.now(timezone.utc).isoformat(),'input_sha256':hashes,'files_sha256':{p.name:sha(p) for p in files},
       'no_API_calls':True,'old_scientific_outputs_unchanged':True,'existing_78_test_family_unchanged':True}
    writejson(out/'manifest.json',manifest)
    print(json.dumps({'summary':overall,'verification':verification},ensure_ascii=False),flush=True)

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--workspace-root');ap.add_argument('--output-dir',default=str(Path(__file__).resolve().parent));ap.add_argument('--synthetic-only',action='store_true');args=ap.parse_args()
    root=root_find(args.workspace_root);out=Path(args.output_dir).resolve();out.mkdir(parents=True,exist_ok=True);start=time.perf_counter()
    synthetic=synthetic_cases();print('Synthetic exact edge cases passed: '+str(len(synthetic)),flush=True)
    if args.synthetic_only:return
    hashes={p:sha(root/p) for p in FILES.values()}
    writejson(out/'input_manifest.json',{'frozen_before_radius_computation':True,'protocol_sha256':sha(Path(__file__).resolve().parent/'protocol.json'),'inputs_sha256':hashes})
    d=load_inputs(root);print('Loaded and matched all 54000 nominal selected actions.',flush=True)
    r=solve(*arrays(d));print('Exact rational 32-pattern calculation complete.',flush=True)
    v=validate_all(d,r);print('All-row exact boundary/interior and finite witness checks passed.',flush=True)
    scalar=scalar_spot(d,r);print('Independent Fraction checks passed: '+str(len(scalar)),flush=True)
    write_outputs(root,out,d,r,v,synthetic,scalar,hashes,start)

if __name__=='__main__':main()
