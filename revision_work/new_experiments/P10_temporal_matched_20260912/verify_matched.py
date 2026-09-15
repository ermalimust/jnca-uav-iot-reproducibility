"""Independent scalar spot checks and full saved-row admission/membership audit."""
import csv,json,hashlib,sys
from pathlib import Path
from collections import defaultdict
from functools import lru_cache
import numpy as np
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent
OLD=HERE.parent/"P6_temporal_completion_20260912"
P=json.loads((OLD/"protocol.json").read_text(encoding="utf-8"))
PLAN=json.loads((HERE/"protocol.json").read_text(encoding="utf-8"))
SEQ=json.loads((OLD/"sequence_policies.json").read_text(encoding="utf-8"))
A=P["actions"];M=P["modes"]
B=np.array([[0,0,0,0],[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[1,0,0,1],[1,0,1,0],[1,1,1,1]])
T=np.eye(8)*.85+.15/8
C=defaultdict(int)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(name):return list(csv.DictReader((HERE/name).open(encoding="utf-8")))
def close(x,y):assert abs(float(x)-float(y))<1e-8,(x,y)
def admitted(b,mission):
    q=np.asarray(b)@B;risk=q[0]+q[1]+q[2]+.5*q[3]
    mask=np.ones(6,dtype=bool)
    if mission=="emergency":
        if risk>=.42:mask[0]=False
        if q[1]>=.28:mask[[0,1,3,4]]=False
    elif q[3]<.35 and risk<.70:mask[5]=False
    return mask
@lru_cache(maxsize=50000)
def scalar(state,a,z,mission):
    # Scalar equations independently transcribed from the service contract.
    qc,qv,*lease=state
    if a:lease[a-1]=3
    w,b,m,v=B[z]
    busy=.055 if not w else .275 if lease[0] else .575
    ble=.0015 if not b else .1 if lease[1] else .4375
    eta=1 if not m else .7 if lease[2] else .45
    rate=12*(1-busy)*(1-ble)*eta
    offered=9 if v else 2
    cap=1 if lease[4] else 3 if lease[3] else float("inf")
    dc=qc+.0048;dv=qv+.1*min(offered,cap)
    total=min(.1*rate,dc+dv)
    sc=min(dc,.1*rate) if lease[4] else total*dc/(dc+dv)
    sv=min(dv,max(0,total-sc))
    rawc=max(0,dc-sc);rawv=max(0,dv-sv)
    nc=min(.05,rawc);nv=min(2,rawv)
    waiting=qc if lease[4] else qc+qv+.012
    delay=2+waiting/rate*1000
    closs=max(min(1,max(0,(delay-10)/10)),min(1,max(0,(rawc-nc)/.0048)))
    goodput=sv/.1
    vloss=min(1,max(0,(3-goodput)/3))
    wc=.35 if mission=="inspection" else .85
    service=10*(wc*closs+(1-wc)*vloss)
    return (round(nc,12),round(nv,12),*[max(0,int(l)-1) for l in lease]),service,delay,goodput
def service(state,a,z,mission,score):
    if score=="joint":return scalar(state,a,z,mission)[1]
    base=scalar(state,a,0,mission)[1]
    return base+sum(B[z,r]*(scalar(state,a,r+1,mission)[1]-base) for r in range(4))
def cost(state,a,z,mission,prev,score):
    return service(state,a,z,mission,score)+P["action_overhead"][A[a]]+.2*(prev>=0 and prev!=a)
def likelihood(a):
    acc=.9 if a==0 else .55
    return np.eye(8)*acc+(1-np.eye(8))*(1-acc)/7
def brute_h2(state,b,t,prev,score):
    mission="inspection" if t<15 else "emergency"
    future="inspection" if t+1<15 else "emergency"
    values=np.full(6,np.inf)
    for a in np.flatnonzero(admitted(b,mission)):
        value=sum(b[z]*cost(state,a,z,mission,prev,score) for z in range(8))
        groups=defaultdict(list)
        for z in range(8):groups[scalar(state,a,z,mission)[0]].append(z)
        if t<29:
            for nxt,zs in groups.items():
                mass=float(np.sum(b[zs]))
                if mass<1e-15:continue
                post=np.zeros(8);post[zs]=b[zs]/mass
                pred=post@T;like=likelihood(a)
                for obs in range(8):
                    joint=pred*like[:,obs];prob=float(joint.sum())
                    posterior=joint/prob
                    child=min(sum(posterior[z]*cost(nxt,a2,z,future,a,score) for z in range(8))
                              for a2 in np.flatnonzero(admitted(posterior,future)))
                    value+=mass*prob*child
        values[a]=value
    return int(np.argmin(values)),float(np.min(values))

rows=read("step_logs.csv")
prior={}
prefix_rows=[]
for r in rows:
    t=int(r["step"]);mission="inspection" if t<15 else "emergency"
    b=np.array(json.loads(r["belief"]));a=A.index(r["action"])
    assert r["mission"]==r["guard_version"]==mission and admitted(b,mission)[a]
    assert abs(b.sum()-1)<1e-10
    key=(r["interface"],r["score_mode"],r["generation_replicate"],r["seed"])
    if t:assert json.loads(r["state"])==json.loads(prior[key]["next_state"])
    if t==15:assert not int(r["executed_continuation"])
    if r["interface"]=="prefix_H1":
        entry=SEQ["missions"][mission]["by_archetype"][M[int(b.argmax())]][int(r["generation_replicate"])]
        candidates={pair[0] for pair in entry["sequences"]}
        assert r["action"] in candidates or int(r["parse_or_guard_fallback"])
        assert int(r["horizon"])==1
        prefix_rows.append(r)
    if int(r["executed_continuation"]):
        prev=prior[key];pb=np.array(json.loads(prev["belief"]))
        pairs=SEQ["missions"][mission]["by_archetype"][M[int(pb.argmax())]][int(r["generation_replicate"])]["sequences"]
        assert [prev["action"],r["action"]] in pairs and prev["mission"]==mission
        C["submitted_second_commands"]+=1
    prior[key]=r
    C["all_row_guard_schedule_and_state_checks"]+=1

# Stratified fixed-index samples use no outcome-based selection.
spot=[]
for interface in ["prefix_H1","sequence_execute2","full_H2"]:
    for score in ["joint","additive"]:
        block=[r for r in rows if r["interface"]==interface and r["score_mode"]==score]
        indices=np.linspace(0,len(block)-1,80,dtype=int)
        spot.extend(block[i] for i in indices)
for r in spot:
    state=tuple(json.loads(r["state"]));a=A.index(r["action"]);z=M.index(r["hidden_mode"])
    nxt,s,delay,good=scalar(state,a,z,r["mission"])
    assert list(nxt)==json.loads(r["next_state"])
    for x,y in [(s,r["service_cost"]),(delay,r["c2_virtual_delay_ms"]),(good,r["video_goodput_mbps"]),
                (service(state,a,z,r["mission"],"additive"),r["additive_service_raw"])]:
        close(x,y)
    C["scalar_physics_and_additive_spot_rows"]+=1

for score in ["joint","additive"]:
    for t in [0,14,15,28]:
        r=next(r for r in rows if r["interface"]=="full_H2" and r["score_mode"]==score and r["seed"]=="60912000" and int(r["step"])==t)
        state=tuple(json.loads(r["state"]));belief=np.array(json.loads(r["belief"]))
        if t:
            previous=next(x for x in rows if x["interface"]=="full_H2" and x["score_mode"]==score and x["seed"]=="60912000" and int(x["step"])==t-1)
            prev=A.index(previous["action"])
        else:prev=-1
        action,value=brute_h2(state,belief,t,prev,score)
        assert A[action]==r["action"];close(value,r["expected_objective"])
        C["independent_full_H2_enumerations"]+=1

differences=read("planned_contrast_episode_differences.csv")
reported=read("planned_contrasts.csv")
signs=np.random.default_rng(PLAN["statistics"]["permutation_seed"]).choice([-1.,1.],size=(100000,96))
draws=np.random.default_rng(PLAN["statistics"]["bootstrap_seed"]).integers(0,96,(10000,96))
for r in reported:
    d=np.array([float(x["paired_difference"]) for x in differences if x["contrast_id"]==r["contrast_id"]])
    assert len(d)==96
    mean=float(d.mean());lo,hi=np.quantile(d[draws].mean(axis=1),[.025,.975])
    p=(1+np.sum(np.abs(signs@d/96)>=abs(mean)-1e-12))/100001
    for x,y in [(mean,r["estimate"]),(lo,r["ci_low"]),(hi,r["ci_high"]),(p,r["raw_p"])]:close(x,y)
    C["planned_contrast_statistics"]+=1

for name in ["run_manifest.json","stability_manifest.json"]:
    manifest=json.loads((HERE/name).read_text(encoding="utf-8"))
    for file,h in manifest["hashes"].items():assert sha(HERE/file)==h;C["local_hash_checks"]+=1
manifest=json.loads((HERE/"run_manifest.json").read_text(encoding="utf-8"))
for file,h in manifest["preserved_P6_file_hashes"].items():assert sha(OLD/file)==h;C["unchanged_P6_hash_checks"]+=1
result=dict(passed=True,checks=dict(C),failures=0,
            scalar_source="Independent equations; does not import run_matched or run_temporal for calculations.",
            statistical_scope="Exactly four planned contrasts; all values reproduced from the saved paired episode differences.")
(HERE/"independent_spot_verification.json").write_text(json.dumps(result,indent=2)+"\n")
print(json.dumps(result))
