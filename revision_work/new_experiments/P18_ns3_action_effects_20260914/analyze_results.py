"""Frozen P18 packet-ledger reduction and paired-block inference.

No input choice, handler setting, exclusion, or hypothesis is selected from outcomes.
All primary denominators are the original offered demand, including shaper drops.
"""
from pathlib import Path
from fractions import Fraction
import csv,gzip,hashlib,itertools,json,math
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent
OUT=HERE/'results'
ACTIONS=['Observe','WiFiRelief','LinkAdapt','VideoShape','FallbackProtect']
METRICS=['c2_deadline_miss_fraction','video_delivery_fraction']
START=11000000000; END=21000000000; HORIZON=23000000000; ACT=11001000000

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(name,obj):(OUT/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def write(name,rows):
    with (OUT/name).open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def adjust(values,method):
    p=np.asarray(values,dtype=float);order=np.argsort(p,kind='stable');n=len(p)
    ranked=np.minimum(1,p[order]*(n-np.arange(n))) if method=='holm' else np.minimum(1,p[order]*n/(np.arange(n)+1))
    ranked=np.maximum.accumulate(ranked) if method=='holm' else np.minimum.accumulate(ranked[::-1])[::-1]
    result=np.empty(n);result[order]=ranked;return result.tolist()

def exact_tail(values):
    scale=math.lcm(*(v.denominator for v in values));weights=[v.numerator*(scale//v.denominator) for v in values]
    assert sum(abs(w) for w in weights)<2**62
    bits=(np.arange(65536,dtype=np.uint32)[:,None]>>np.arange(16,dtype=np.uint32))&1
    sums=(bits.astype(np.int64)*2-1)@np.asarray(weights,dtype=np.int64)
    count=int(np.count_nonzero(np.abs(sums)>=abs(sum(weights))))
    return count,count/65536

def reduce_episode(rec,inputs,cohort_writer):
    folder=HERE/'runs'/f"{rec['scenario_id']}_r{rec['rng_run']:04d}_{rec['action']}"
    assert all(sha(folder/n)==h for n,h in rec['files'].items())
    frame=pd.read_csv(folder/'packets.csv.gz')
    assert list(frame.packet_id)==list(inputs.packet_id)
    assert np.array_equal(frame.flow,inputs.flow) and np.array_equal(frame.payload_bytes,inputs.payload_bytes)
    assert np.array_equal(frame.offer_ns,1000000000+inputs.relative_us*1000)
    sent=frame.send_ns>=0;received=frame.receive_ns>=0;drop=frame.shaper_drop_ns>=0;fail=frame.socket_fail_ns>=0
    assert np.all(sent.astype(int)+drop.astype(int)+fail.astype(int)<=1)
    pending=~(sent|drop|fail)
    assert not np.any(received&~sent)
    assert np.all(frame.loc[sent,'send_ns']>=frame.loc[sent,'offer_ns'])
    assert np.all(frame.loc[received,'receive_ns']>=frame.loc[received,'send_ns'])
    assert np.all(frame.loc[received,'receive_ns']<HORIZON)
    assert np.all(frame.loc[~sent,'phy_attempts']==0)
    phy=frame.phy_attempts>0
    expected_tid=np.where((rec['action']=='FallbackProtect')&(frame.flow==0)&(frame.send_ns>=ACT),64,1)
    assert np.all(frame.loc[phy,'tid_mask']==expected_tid[phy]),'Actual QoS TID mismatch'
    assert np.all(frame.loc[received,'phy_attempts']>=1)
    # Evaluate the entire fixed offer interval, retaining its common 1-ms dispatch prefix.
    eligible=(frame.offer_ns>=START)&(frame.offer_ns<END)
    c2=frame[eligible&(frame.flow==0)];v=frame[eligible&(frame.flow==1)]
    c2rx=c2[c2.receive_ns>=0];delay=(c2rx.receive_ns-c2rx.offer_ns)/1e6
    misses=int(((c2.receive_ns<0)|((c2.receive_ns-c2.offer_ns)>10000000)).sum())
    vd=int(v.loc[v.receive_ns>=0,'payload_bytes'].sum());vo=int(v.payload_bytes.sum())
    vdrop=int(v.loc[v.shaper_drop_ns>=0,'payload_bytes'].sum());vfail=int(v.loc[v.socket_fail_ns>=0,'payload_bytes'].sum())
    vpending=int(v.loc[(v.send_ns<0)&(v.shaper_drop_ns<0)&(v.socket_fail_ns<0),'payload_bytes'].sum())
    vunreceived=int(v.loc[(v.send_ns>=0)&(v.receive_ns<0),'payload_bytes'].sum())
    assert vo==vd+vdrop+vfail+vpending+vunreceived
    assert len(c2)==399 and int(c2.payload_bytes.sum())==24005
    s=rec['scenario_id'];out={'scenario_id':s,'rng_run':rec['rng_run'],'action':rec['action'],'w':int(s[1]),'m':int(s[3]),'v':int(s[5]),
       'ledger_path':(folder/'packets.csv.gz').relative_to(HERE).as_posix(),
       'c2_offered_count':len(c2),'c2_deadline_miss_count':misses,'c2_deadline_miss_fraction':misses/len(c2),
       'c2_received_count':len(c2rx),'c2_received_fraction':len(c2rx)/len(c2),
       'c2_received_delay_mean_ms':float(delay.mean()) if len(delay) else '',
       'c2_received_delay_p95_ms':float(np.quantile(delay,.95,method='linear')) if len(delay) else '',
       'video_offered_bytes':vo,'video_delivered_bytes':vd,'video_delivery_fraction':vd/vo,
       'video_shaper_drop_bytes':vdrop,'video_socket_reject_bytes':vfail,'video_shaper_pending_bytes':vpending,
       'video_unreceived_horizon_bytes':vunreceived,
       'video_receive_window_mbps':float(frame.loc[(frame.flow==1)&(frame.receive_ns>=START)&(frame.receive_ns<END),'payload_bytes'].sum()*8/1e7),
       'evaluation_phy_attempts':int(frame.loc[eligible,'phy_attempts'].sum()),
       'evaluation_mac_acks':int(frame.loc[eligible,'mac_acks'].sum()),
       'evaluation_mac_drops':int(frame.loc[eligible,'mac_drops'].sum()),
       'all_offered_count':len(frame),'all_shaper_pending_count':int(pending.sum()),'all_duplicates':int(frame.duplicates.sum())}
    for i in range(100):
        z=frame[eligible&(frame.offer_ns>=START+i*100000000)&(frame.offer_ns<START+(i+1)*100000000)]
        for flow in (0,1):
            y=z[z.flow==flow];rx=y[y.receive_ns>=0];miss=int(((y.receive_ns<0)|((y.receive_ns-y.offer_ns)>10000000)).sum())
            cohort_writer.writerow([s,rec['rng_run'],rec['action'],i,flow,len(y),int(y.payload_bytes.sum()),len(rx),int(rx.payload_bytes.sum()),miss if flow==0 else ''])
    return out

def main():
    OUT.mkdir(exist_ok=True)
    receipt=json.loads((HERE/'execution_receipt.json').read_text());assert receipt['status']=='PASS' and receipt['formal_runs']==640
    bound=json.loads((HERE/'input_binding.json').read_text())
    assert all(sha(HERE/n)==h for n,h in bound['study_files'].items())
    inputs={p.stem:pd.read_csv(p) for p in (HERE/'inputs').glob('W*.csv')}
    episodes=[]
    with (OUT/'arrival_cohorts_100ms.csv').open('w',newline='',encoding='utf-8') as f:
        cw=csv.writer(f);cw.writerow(['scenario_id','rng_run','action','cohort_index','flow','offered_packets','offered_bytes','received_packets','received_bytes','c2_deadline_miss_packets'])
        for i,rec in enumerate(receipt['runs']):
            episodes.append(reduce_episode(rec,inputs[rec['scenario_id']],cw))
            if (i+1)%80==0:print('Reduced ledgers',i+1,'/640',flush=True)
    write('episode_metrics.csv',episodes)
    index={(r['scenario_id'],r['rng_run'],r['action']):r for r in episodes}
    assert len(index)==640
    scenarios=sorted(inputs);bootstrap=np.random.default_rng(609140718).integers(0,16,(10000,16))
    primary=[];blocks=[]
    for a in ACTIONS[1:]:
        for metric in METRICS:
            num,den=('c2_deadline_miss_count','c2_offered_count') if metric==METRICS[0] else ('video_delivered_bytes','video_offered_bytes')
            diffs=[];ar=[];br=[]
            for run in range(1,17):
                av=sum((Fraction(index[s,run,a][num],index[s,run,a][den]) for s in scenarios),Fraction())/8
                bv=sum((Fraction(index[s,run,'Observe'][num],index[s,run,'Observe'][den]) for s in scenarios),Fraction())/8
                diffs.append(av-bv);ar.append(float(av));br.append(float(bv))
                blocks.append({'action':a,'metric':metric,'rng_run':run,'action_mean':float(av),'reference_mean':float(bv),'delta':float(av-bv),'delta_numerator':(av-bv).numerator,'delta_denominator':(av-bv).denominator})
            values=np.array(list(map(float,diffs)));ci=np.quantile(values[bootstrap].mean(axis=1),[.025,.975],method='linear')
            tail,p=exact_tail(diffs)
            primary.append({'test_id':f'P18_ns3_action_effects::{a}-Observe::{metric}','family':'P18_ns3_action_effects','contrast':a+'-Observe','metric':metric,
              'effect':float(values.mean()),'ci_low':float(ci[0]),'ci_high':float(ci[1]),'inference_unit':'RNG block averaged over eight fixed scenarios','n':16,
              'p_raw':p,'sign_flip_tail_count':tail,'sign_flip_assignments':65536,'action_mean':float(np.mean(ar)),'reference_mean':float(np.mean(br)),'units':'fraction'})
    for method in ('holm','bh'):
        for r,p in zip(primary,adjust([r['p_raw'] for r in primary],method)):r['p_'+method+'_source_family']=p
    write('primary_contrasts.csv',primary);write('paired_run_blocks.csv',blocks)
    df=pd.DataFrame(episodes);numeric=[k for k in episodes[0] if k not in ('scenario_id','rng_run','action','w','m','v','ledger_path')]
    for c in numeric:df[c]=pd.to_numeric(df[c],errors='coerce')
    df.groupby(['scenario_id','action'],sort=True)[numeric].mean().reset_index().to_csv(OUT/'scenario_descriptive_means.csv',index=False)
    df.groupby('action',sort=False)[numeric].mean().reset_index().to_csv(OUT/'action_descriptive_means.csv',index=False)
    save('analysis_receipt.json',{'status':'PASS','episodes':640,'rng_blocks':16,'fixed_scenarios':8,'primary_tests':8,'arrival_cohorts':128000,
       'checks':['all frozen study hashes','all 640 output hashes','original offers and payloads exactly preserved','application partition','video byte conservation','all receive causality','actual expected PHY TID','all 399 evaluation C2 offers'],
       'input_binding_sha256':sha(HERE/'input_binding.json'),'execution_receipt_sha256':sha(HERE/'execution_receipt.json'),
       'shaper_pending_packets':int(df.all_shaper_pending_count.sum()),'duplicate_application_receptions':int(df.all_duplicates.sum()),
       'files':{p.name:sha(p) for p in OUT.glob('*.csv')}})
    print(json.dumps(primary,indent=2),flush=True)
if __name__=='__main__':main()
