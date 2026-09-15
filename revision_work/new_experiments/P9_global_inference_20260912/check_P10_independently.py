"""Read-only independent P10 contrasts and scalar prefix/proxy spot verification."""
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent
WORK=HERE.parents[2]
P10=HERE.parent/'P10_temporal_matched_20260912'
P6=HERE.parent/'P6_temporal_completion_20260912'
proto=json.loads((P10/'protocol.json').read_text())
base=json.loads((P6/'protocol.json').read_text());par=base['parameters']
seq=json.loads((P6/'sequence_policies.json').read_text())
actions=base['actions'];modes=base['modes']
bits=[[0,0,0,0],[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[1,0,0,1],[1,0,1,0],[1,1,1,1]]
sources=[P10/'protocol.json',P10/'planned_contrasts.csv',P10/'planned_contrast_episode_differences.csv',P10/'episode_results.csv',P10/'step_logs.csv',P6/'protocol.json',P6/'sequence_policies.json']
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
before={str(p.relative_to(WORK)):sha(p) for p in sources}
def csvread(p):return list(csv.DictReader(p.open(encoding='utf-8-sig')))
def close(a,b,tol=3e-10):assert math.isclose(float(a),float(b),rel_tol=tol,abs_tol=tol),(a,b)

# Rebuild all planned paired differences from the episode table, averaging actual
# generation rows first. No P10 functions are imported.
episodes=csvread(P10/'episode_results.csv');group=defaultdict(list)
for r in episodes:group[(r['interface'],r['score_mode'],int(r['seed']))].append(r)
seeds=sorted({int(r['seed']) for r in episodes});assert len(seeds)==96
def means(interface,score,metric):
    vals=[]
    for s in seeds:
        members=group[(interface,score,s)];assert len(members)==(1 if interface=='full_H2' else 3)
        vals.append(sum(float(r[metric]) for r in members)/len(members))
    return np.array(vals)
prefix=means('prefix_H1','joint','cumulative_cost');sequence=means('sequence_execute2','joint','cumulative_cost')
contrasts=[sequence-prefix,means('prefix_H1','additive','cumulative_cost')-prefix,
           means('sequence_execute2','additive','cumulative_cost')-sequence,
           (means('sequence_execute2','joint','cumulative_additive_cost')-means('prefix_H1','joint','cumulative_additive_cost'))-(sequence-prefix)]
published=csvread(P10/'planned_contrasts.csv');assert [r['contrast_id'] for r in published]==[r['id'] for r in proto['planned_contrasts']]
saved_diff=csvread(P10/'planned_contrast_episode_differences.csv');assert len(saved_diff)==384
draws=np.random.default_rng(proto['statistics']['bootstrap_seed']).integers(0,96,(10000,96))
signs=np.random.default_rng(proto['statistics']['permutation_seed']).choice([-1.,1.],size=(100000,96))
stat_rows=[]
for r,d in zip(published,contrasts):
    selected=[x for x in saved_diff if x['contrast_id']==r['contrast_id']]
    assert [int(x['seed']) for x in selected]==seeds
    for x,v in zip(selected,d):close(x['paired_difference'],v)
    mean=float(sum(d)/96);low,high=np.quantile(np.mean(d[draws],axis=1),[.025,.975])
    # Independent statistic code, using the frozen randomization schedule.
    null=np.einsum('ij,j->i',signs,d)/96
    extreme=int(np.count_nonzero(np.abs(null)>=abs(mean)-1e-12));raw=(extreme+1)/100001
    close(r['estimate'],mean);close(r['ci_low'],low);close(r['ci_high'],high);close(r['raw_p'],raw,1e-14)
    stat_rows.append(dict(contrast_id=r['contrast_id'],estimate=mean,ci_low=float(low),ci_high=float(high),extreme_draws=extreme,p_raw=raw))

def scalar_service(state,mission,a,z):
    qc,qv,*leases=state;leases=list(leases)
    if a>0:leases[a-1]=par['action_duration_steps']
    w,b,m,v=bits[z]
    busy=par['wifi_high_busy_fraction'] if w else par['wifi_low_busy_fraction']
    if w and leases[0]>0:busy=par['wifi_relief_busy_fraction']
    overlap=par['ble_dense_fraction'] if b else par['ble_light_fraction']
    if b and leases[1]>0:overlap=par['ble_avoid_fraction']
    link=par['mobility_service_fraction'] if m else 1.
    if m and leases[2]>0:link=par['adapted_mobility_service_fraction']
    rate=par['nominal_capacity_mbps']*(1-busy)*(1-overlap)*link
    caprate=(par['fallback_video_cap_mbps'] if leases[4]>0 else par['video_shape_cap_mbps'] if leases[3]>0 else float('inf'))
    offered=par['video_burst_mbps'] if v else par['video_normal_mbps'];dt=base['step_ms']/1000
    arrive_c=par['c2_arrivals_mbit_per_step'];dc=qc+arrive_c;dv=qv+min(offered,caprate)*dt
    used=min(dc+dv,rate*dt)
    served_c=min(dc,rate*dt) if leases[4]>0 else used*dc/(dc+dv)
    served_v=min(dv,max(0.,used-served_c));rawc=max(0.,dc-served_c);rawv=max(0.,dv-served_v)
    dropc=max(0.,rawc-par['c2_buffer_mbit']);wait=qc if leases[4]>0 else qc+qv+par['video_packet_mbit']
    delay=par['base_service_ms']+1000*wait/rate
    c_loss=max(max(0,min(1,(delay-par['c2_deadline_ms'])/par['c2_deadline_ms'])),max(0,min(1,dropc/arrive_c)))
    goodput=served_v/dt;v_loss=max(0,min(1,(par['video_target_mbps']-goodput)/par['video_target_mbps']))
    weights=base['missions'][mission]
    cost=10*(weights['c2_weight']*c_loss+weights['video_weight']*v_loss)
    new=[round(min(par['c2_buffer_mbit'],rawc),12),round(min(par['video_buffer_mbit'],rawv),12)]+[max(0,int(x)-1) for x in leases]
    return cost,new

def admitted(belief,mission):
    q=[sum(belief[z]*bits[z][j] for z in range(8)) for j in range(4)]
    risk=q[0]+q[1]+q[2]+.5*q[3];okay=[True]*6;g=base['guards']
    if mission=='emergency':
        okay[0]=risk<g['safety_risk_observe']
        if q[1]>=g['rid_B_probability']:
            for a in [0,1,3,4]:okay[a]=False
    else:okay[5]=not(q[3]<g['video_probability'] and risk<g['video_safety_risk'])
    return okay

spot_seeds=set(seeds[:3]+seeds[-3:]);previous={};spot_count=0;raw_proxy_checks=0
with (P10/'step_logs.csv').open(encoding='utf-8-sig') as stream:
    for r in csv.DictReader(stream):
        if r['interface']!='prefix_H1' or int(r['seed']) not in spot_seeds:continue
        key=(r['policy'],r['score_mode'],r['seed']);prev=previous.get(key)
        state=json.loads(r['state']);belief=json.loads(r['belief']);mission=r['mission'];rep=int(r['generation_replicate']);a=actions.index(r['action'])
        mode=modes[max(range(8),key=lambda z:belief[z])]
        entries=seq['missions'][mission]['by_archetype'][mode];matching=[x for x in entries if x['replicate']==rep];assert len(matching)==1
        submitted={actions.index(s[0]) for s in matching[0]['sequences'] if len(s)==2 and all(x in actions for x in s)}
        allow=admitted(belief,mission)
        physical=[[scalar_service(state,mission,act,z)[0] for z in range(8)] for act in range(6)]
        additive=[[physical[act][0]+sum(bits[z][j]*(physical[act][j+1]-physical[act][0]) for j in range(4)) for z in range(8)] for act in range(6)]
        score=physical if r['score_mode']=='joint' else additive
        expected=[sum(belief[z]*score[act][z] for z in range(8))+base['action_overhead'][actions[act]]+(.2 if prev is not None and prev!=act else 0.) for act in range(6)]
        candidates=sorted(submitted & {act for act in range(6) if allow[act]})
        fallback=not candidates
        if fallback:candidates=[5 if allow[5] else min((act for act in range(6) if allow[act]),key=lambda act:expected[act])]
        assert r['first_candidates']=='|'.join(actions[act] for act in candidates)
        assert int(r['parse_or_guard_fallback'])==fallback
        selected=min(candidates,key=lambda act:expected[act]);assert a==selected
        close(r['expected_objective'],expected[a])
        z=modes.index(r['hidden_mode']);service,new=scalar_service(state,mission,a,z)
        close(r['service_cost'],service)
        for x,y in zip(new,json.loads(r['next_state'])):close(x,y)
        close(r['additive_service_raw'],additive[a][z]);raw_proxy_checks+=48
        previous[key]=a;spot_count+=1
assert spot_count==6*3*2*30
for p in sources:assert sha(p)==before[str(p.relative_to(WORK))]
result=dict(verified=True,scope='Independent saved-data reconstruction; no P6/P10 functions imported and no source files changed.',
            episode_contrasts=stat_rows,paired_episode_rows=384,
            prefix_spot=dict(seeds=sorted(spot_seeds),rows=spot_count,score_modes=['joint','additive'],
                             tests=['literal saved-pair prefixes','current belief guards','minimum expected score','physical service and next state','unclipped additive proxy']),
            scalar_grid_evaluations=raw_proxy_checks,all_input_hashes_unchanged=True,inputs=before)
(HERE/'P10_independent_check.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps(dict(verified=True,paired_episode_rows=384,prefix_spot_rows=spot_count,scalar_grid_evaluations=raw_proxy_checks,contrasts=stat_rows)))
