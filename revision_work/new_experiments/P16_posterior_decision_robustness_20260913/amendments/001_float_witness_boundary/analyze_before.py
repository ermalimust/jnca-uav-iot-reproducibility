"""Offline sufficient continuous-box certificates, exact integer comparisons.

Python 3 + NumPy; run from any directory. No API, fitting or source mutation.
"""
import argparse, ast, csv, hashlib, itertools, json, sys
from pathlib import Path
import numpy as np

S = 1_000_000
ACTIONS = ('Observe','WiFiRelief','BLEAvoid','LinkAdapt','VideoShape','FallbackProtect','EscalateReview')
ROUTES = ('low_confidence','wifi_dominant','ble_rid_dominant','mobility_dominant','video_dominant','mixed_high_risk')
REL = 'revision_work/analysis/replay_inputs/'
INPUTS = {'audit':REL+'results/audit_records/qwen_qwen-plus_guarded_audit_records.jsonl',
          'policies':REL+'llm_runs/qwen_qwen-plus/policies.jsonl',
          'core':REL+'paper7_agentic_feasibility.py',
          'router':REL+'paper7_llm_candidate_experiment.py'}
RADII = [10000,25000,50000,100000]

def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,x): p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def table(p,rows):
    with p.open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def load(root):
    literal={}
    for n in ast.parse((root/INPUTS['core']).read_text(encoding='utf-8')).body:
        names=[x.id for x in n.targets if isinstance(x,ast.Name)] if isinstance(n,ast.Assign) else [n.target.id] if isinstance(n,ast.AnnAssign) and isinstance(n.target,ast.Name) else []
        for name in names:
            if name in ('BASE_COSTS','ACTION_OVERHEAD'): literal[name]=ast.literal_eval(n.value)
    policies={r['mission_id']:r['policy'] for r in map(json.loads,(root/INPUTS['policies']).read_text(encoding='utf-8').splitlines())}
    rows=list(map(json.loads,(root/INPUTS['audit']).read_text(encoding='utf-8').splitlines()))
    missions={r['mission']['mission_id']:r['mission'] for r in rows}; mids=sorted(missions); mi={m:i for i,m in enumerate(mids)}
    ranks=np.full((len(mids),6,6),999,dtype=np.int64); costs=[]; flags=[]; lists=[]
    for mid in mids:
        m=missions[mid]; g=[bool(m['guards'].get(k,False)) for k in ('safety','rid','video','energy')]; flags.append(g)
        c=np.array([[round(v*10) for v in literal['BASE_COSTS'][m['gold_cost_profile']][a]] for a in ACTIONS[:6]],dtype=np.int64)
        if g[1]: c[0,1]+=40; c[2,1]=5
        if g[2]: c[5]+=np.array([10,10,10,25]);c[4,3]=6
        if g[3]: c[3]+=np.array([10,10,0,10])
        costs.append(c); ll=[]
        for r,name in enumerate(ROUTES):
            order=policies[mid]['archetype_actions'].get(name,[]) or policies[mid].get('fallback_actions',['FallbackProtect','Observe']); ll.append(order)
            for k,a in enumerate(order):
                if a in ACTIONS[:6]: ranks[mi[mid],r,ACTIONS.index(a)]=min(k,ranks[mi[mid],r,ACTIONS.index(a)])
        lists.append(ll)
    q=np.array([[round(r['posterior'][k]*S) for k in ('W','B','M','V')] for r in rows],dtype=np.int64)
    idx=np.array([mi[r['mission']['mission_id']] for r in rows]); co=np.array(costs)[idx]; fl=np.array(flags,bool)[idx]
    h=np.array([round(literal['ACTION_OVERHEAD'][a]*10*S) for a in ACTIONS[:6]],dtype=np.int64)
    d={'q':q,'cost':co,'flags':fl,'ranks':ranks[idx],'h':h,'idx':idx,'mids':mids,'rows':rows,'lists':lists}
    route,admit,winner=select(d,q)
    assert len(rows)==54000 and len(mids)==30
    assert all(ROUTES[route[i]]==r['posterior_archetype'] for i,r in enumerate(rows))
    assert all(lists[idx[i]][route[i]]==r['candidate_actions'] for i,r in enumerate(rows))
    assert all([a for a in r['candidate_actions'] if a in ACTIONS[:6] and admit[i,ACTIONS.index(a)]]==r['accepted_actions'] for i,r in enumerate(rows))
    assert all(ACTIONS[winner[i]]==r['selected_action'] for i,r in enumerate(rows))
    d.update(winner=winner,route=route,admit=admit)
    return d

def guard_bounds(d,L,U):
    """Always accepted / always rejected over a continuous rectangular box."""
    n=len(L); f=d['flags']; a=np.ones((n,6),bool); r=np.zeros((n,6),bool)
    rl=2*L[:,:3].sum(1)+L[:,3]; ru=2*U[:,:3].sum(1)+U[:,3]
    # Rejection is an OR of active mission constraints. Each condition is
    # monotone; the fallback rejection is a conjunction of two '<' atoms.
    for j in (0,1,3,4):
        a[:,j] &= ~(f[:,1] & (U[:,1]>=280000))
        r[:,j] |= f[:,1] & (L[:,1]>=280000)
    a[:,0] &= ~(f[:,0] & (ru>=840000)); r[:,0] |= f[:,0] & (rl>=840000)
    a[:,3] &= ~(f[:,3] & (L[:,2]<340000)); r[:,3] |= f[:,3] & (U[:,2]<340000)
    a[:,5] = ~f[:,2] | (L[:,3]>=350000) | (rl>=1400000)
    r[:,5] = f[:,2] & (U[:,3]<350000) & (ru<1400000)
    return a,r

def route_points(q):
    r=np.argmax(q,axis=1)+1
    mixed=((q>=340000).sum(1)>=2)|(q.sum(1)>=1050000)
    r[mixed]=5;r[q.max(1)<280000]=0
    return r

def select(d,q):
    route=route_points(q); a,_=guard_bounds(d,q,q); ranks=d['ranks'][np.arange(len(q)),route]
    valid=a&(ranks<999); scores=np.einsum('nam,nm->na',d['cost'],q)+d['h']
    best=np.min(np.where(valid,scores,np.iinfo(np.int64).max),axis=1)
    win=np.argmin(np.where(valid&(scores==best[:,None]),ranks,9999),axis=1)
    none=~valid.any(1);win[none]=np.where(a[none,5],5,6)
    return route,a,win

def possible_routes(L,U):
    p=np.zeros((len(L),6),bool)
    p[:,0]=L.max(1)<280000
    p[:,5]=(U.max(1)>=280000)&(((U>=340000).sum(1)>=2)|(U.sum(1)>=1050000))
    # Exact reachable dominant branches, including the W,B,M,V argmax ties.
    m=np.maximum(280000,L.max(1))
    for j in range(4):
        others=[k for k in range(4) if k!=j]
        earlier_tie=(L[:,:j]==m[:,None]).any(1) if j else np.zeros(len(L),bool)
        p[:,j+1]=(L[:,others].max(1)<340000)&(m+L[:,others].sum(1)<1050000)&(m<=U[:,j])&(~earlier_tie|(m<U[:,j]))
    assert p.any(1).all()
    return p

def certify(d,eps,details=False):
    e=np.broadcast_to(eps,(len(d['q']),))[:,None]
    L=np.maximum(0,d['q']-e);U=np.minimum(S,d['q']+e)
    p=possible_routes(L,U); always,rejected=guard_bounds(d,L,U)
    w=d['winner']; safe=np.minimum(w,5);ix=np.arange(len(w)); wc=d['cost'][ix,safe]
    diff=d['cost']-wc[:,None,:]
    gap=np.sum(np.maximum(diff,0)*L[:,None,:]+np.minimum(diff,0)*U[:,None,:],axis=2)+d['h']-d['h'][safe,None]
    certified=np.ones(len(w),bool); list_same=np.ones(len(w),bool)
    # Integer scores exactly cover all continuous q inside each box.
    for r in range(6):
        rank=d['ranks'][:,r]; wr=rank[ix,safe]; exposure=rank<999
        beaten=(gap>0)|((gap==0)&(rank>=wr[:,None]))
        normal=(w<6)&(wr<999)&always[ix,safe]&np.all(~exposure|rejected|beaten,axis=1)
        fallback=np.all(~exposure|rejected,axis=1)&(((w==5)&always[:,5])|((w==6)&rejected[:,5]))
        certified &= ~p[:,r]|normal|fallback
        # Compare literal candidate sequences, including unsupported names.
        eq=np.array([d['lists'][i][r]==d['lists'][i][rr] for i,rr in zip(d['idx'],d['route'])]) if details else None
        if details:list_same &= ~p[:,r]|eq
    if not details:return certified
    route_same=(p.sum(1)==1)&p[ix,d['route']]
    guard_same=np.all(np.where(d['admit'],always,rejected),axis=1)
    return certified,route_same,list_same,guard_same

def radius_search(d):
    lo=np.zeros(len(d['q']),np.int64);hi=np.full(len(lo),S+1,np.int64)
    assert certify(d,0).all(),'nominal continuous-box certificate failed'
    while (hi-lo>1).any():
        mid=(lo+hi)//2;ok=certify(d,mid);lo=np.where(ok,mid,lo);hi=np.where(ok,hi,mid)
    assert certify(d,lo).all()
    assert not certify(d,np.minimum(lo+1,S))[lo<S].any()
    return lo

def witness_search(d,cert):
    q=d['q'];n=len(q);best=np.full(n,S+1,np.int64);point=q.copy();direction=np.zeros((n,4),np.int64)
    dirs=list(itertools.product((-1,1),repeat=4))+[tuple(s if k==j else 0 for k in range(4)) for j in range(4) for s in (-1,1)]
    for k,dr in enumerate(dirs):
        dr=np.array(dr,np.int64)
        for e in [10000,25000,50000,100000,200000,400000,S]:
            x=np.clip(q+e*dr,0,S);_,_,w=select(d,x);norm=np.abs(x-q).max(1)
            take=(w!=d['winner'])&(norm<best);best[take]=norm[take];point[take]=x[take];direction[take]=dr
        if k%8==7:print('witness directions',k+1,'/ 24',flush=True)
    found=best<=S;lo=cert.copy();hi=best.copy()
    # Each found direction retains a known changed endpoint throughout. There
    # is no monotonicity claim for the actual selector along the direction.
    for _ in range(20):
        mid=(lo+hi)//2;x=np.clip(q+mid[:,None]*direction,0,S);_,_,w=select(d,x)
        change=(w!=d['winner'])&found
        hi=np.where(change,mid,hi);lo=np.where(found&~change,mid,lo)
    x=np.clip(q+hi[:,None]*direction,0,S);_,_,w=select(d,x);norm=np.abs(x-q).max(1)
    assert np.all(w[found]!=d['winner'][found]);assert np.all(norm[found]>cert[found])
    assert np.all(norm[found]<=hi[found]);assert np.all((x>=0)&(x<=S))
    return found,x,w,np.where(found,norm,S+1)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--workspace-root',type=Path);ap.add_argument('--output',type=Path);args=ap.parse_args()
    here=Path(__file__).resolve().parent
    root=args.workspace_root.resolve() if args.workspace_root else next(p for p in here.parents if (p/INPUTS['audit']).is_file())
    out=args.output or here/'results';out.mkdir(parents=True,exist_ok=True)
    binding={'inputs':{k:{'path':v,'sha256':sha(root/v)} for k,v in INPUTS.items()},'analysis_sha256':sha(Path(__file__)),'protocol_sha256':sha(here/'protocol.md')}
    bindpath=out/'input_binding.json'
    if bindpath.exists():assert json.loads(bindpath.read_text(encoding='utf-8'))==binding,'Changed binding: preserve old run and record amendment'
    else:dump(bindpath,binding)
    d=load(root);print('54,000 complete nominal replays passed',flush=True)
    radii=radius_search(d);print('all certificate radii checked',flush=True)
    found,x,w,upper=witness_search(d,radii)
    pooled=[];mission=[];per={}
    for e in RADII:
        checks=certify(d,e,True); per[e]=checks
        for mid in ['ALL']+d['mids']:
            mask=np.ones(len(radii),bool) if mid=='ALL' else d['idx']==d['mids'].index(mid)
            row={'mission':mid,'epsilon':e/S,'decisions':int(mask.sum())}
            for name,v in zip(('action','route','candidate_list','admission'),checks):row[name+'_certified']=int(v[mask].sum());row[name+'_fraction']=float(v[mask].mean())
            row['changed_action_witnesses']=int((found& (upper<=e)&mask).sum())
            row['unresolved']=int(mask.sum()-row['action_certified']-row['changed_action_witnesses'])
            assert row['unresolved']>=0
            (pooled if mid=='ALL' else mission).append(row)
    decisions=[]
    for i,r in enumerate(d['rows']):
        row={'decision_id':r['decision_id'],'mission':r['mission']['mission_id'],'seed':r['seed'],'source_scenario':r['source_window']['scenario_id'],'source_window':r['source_window']['window_id'],
             **{'q_'+k:int(d['q'][i,j])/S for j,k in enumerate('WBMV')},'selected_action':ACTIONS[d['winner'][i]],'certified_radius':int(radii[i])/S,
             'witness_found':bool(found[i]),'witness_radius':int(upper[i])/S if found[i] else '', 'witness_action':ACTIONS[w[i]] if found[i] else '',
             **{'witness_q_'+k:int(x[i,j])/S if found[i] else '' for j,k in enumerate('WBMV')}}
        decisions.append(row)
    table(out/'decision_bounds.csv',decisions);table(out/'pooled_summary.csv',pooled);table(out/'mission_summary.csv',mission)
    quant=lambda a:{str(p):float(np.quantile(a,p,method='inverted_cdf')) for p in [0,.05,.5,.95,1]}
    summary={'scope':'sufficient continuous-box certificate; witnessed upper bound; not an exact minimum flip radius',
      'decisions':len(radii),'missions':len(d['mids']),'unique_mission_q':len({(int(i),*map(int,q)) for i,q in zip(d['idx'],d['q'])}),
      'nominal_replay_checks':{'route':54000,'candidate_list':54000,'admitted_list':54000,'selected_action':54000},
      'certified_radius_quantiles':quant(radii/S),'witnesses_found':int(found.sum()),'no_witness_found':int((~found).sum()),
      'whole_domain_certified':int((radii==S).sum()),'grid_tight_brackets':int((found&(upper==radii+1)).sum()),
      'witness_gap_quantiles':quant((upper[found]-radii[found])/S),'pooled':pooled,'additional_hypothesis_tests':0,
      'checks':{'all_certified_endpoints_pass':True,'all_next_grid_values_fail_certificate_unless_full_domain':True,
                'all_found_witnesses_change_action':True,'all_found_witnesses_outside_certified_box':True}}
    dump(out/'summary.json',summary)
    np.savez_compressed(out/'verification_arrays.npz',q=d['q'],radii=radii,found=found,witness=x,witness_winner=w,upper=upper)
    dump(out/'result_manifest.json',{p.name:sha(p) for p in sorted(out.iterdir()) if p.is_file() and p.name!='result_manifest.json'})
    assert binding['inputs']=={k:{'path':v,'sha256':sha(root/v)} for k,v in INPUTS.items()}
    print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)

if __name__=='__main__':main()
