"""Independent audit of P18 offered schedules, runtime ledgers, knobs and PCAPs.

Does not import the study reducer or orchestrator. Writes only inside this folder.
"""
from pathlib import Path
from collections import Counter, defaultdict
import argparse, contextlib, csv, gzip, hashlib, io, itertools, json, math, runpy, struct
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent
STUDY=HERE.parent
BEGIN=11_000_000_000
END=21_000_000_000
HORIZON=23_000_000_000
ACTIVE=11_001_000_000
ACTIONS=['Observe','WiFiRelief','LinkAdapt','VideoShape','FallbackProtect']

def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path): return json.loads(Path(path).read_text(encoding='utf-8'))
def out(name,value): (HERE/name).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def require(condition,message):
    if not bool(np.all(condition)): raise AssertionError(message)

def source_inputs():
    with contextlib.redirect_stdout(io.StringIO()):
        source=runpy.run_path(str(HERE/'first20_source_audit.py'))
    c2=[(t,0,n) for t,_,n in source['prefix']]
    raw=pd.read_csv(STUDY/'inputs/c2_source.csv',dtype=np.int64)
    require(raw.shape==(800,3),'c2 source shape')
    require(raw.values==np.asarray([(t,n,i) for t,i,n in source['prefix']]),'source PCAP timing/length/index equality')
    inputs={}
    for w,m,v in itertools.product([0,1],repeat=3):
        expected=list(c2)
        if v:
            expected.extend((cycle*1_000_000+k*800,1,1200) for cycle in range(20) for k in range(625))
        else:
            expected.extend((k*8000,1,1200) for k in range(2500))
        if w:
            expected.extend((k*1200,2,1200) for k in range(16667))
        expected.sort()
        name=f'W{w}M{m}V{v}'
        p=STUDY/'inputs'/f'{name}.csv'
        frame=pd.read_csv(p,dtype=np.int64)
        require(frame.packet_id.to_numpy()==np.arange(len(expected)),'input IDs '+name)
        require(frame[['relative_us','flow','payload_bytes']].to_numpy()==np.asarray(expected),'independent schedule '+name)
        inputs[name]=frame
    return inputs

def check_knobs(folder,rec):
    k=pd.read_csv(folder/'knobs.csv')
    require(list(k.phase)==['before','before','after','after'],'knob phases')
    require(list(k.node)==[0,1,0,1],'knob node IDs')
    distance=1.0 if '--distance=1' in rec['command'] else 15.0
    moving=int(rec['scenario_id'][3])
    expected=[]
    for phase,t in [('before',BEGIN),('after',ACTIVE)]:
        for node in [0,1]:
            active=phase=='after'
            relief=active and rec['action']=='WiFiRelief' and node==1
            fallback=active and rec['action']=='FallbackProtect'
            cap=(1_000_000 if fallback else 4_000_000 if active and rec['action']=='VideoShape' else 0)
            mode='OfdmRate6Mbps' if active and rec['action']=='LinkAdapt' else 'OfdmRate24Mbps'
            xpos=0 if node==0 else distance+((t/1e9-1)*3.5 if moving else 0)
            expected.append(dict(phase=phase,time_ns=t,node=node,mode=mode,cwmin_be=63 if relief else 15,
                                 cwmax_be=1023,aifsn_be=7 if relief else 3,c2_tos=192 if fallback else 0,
                                 video_cap_bps=cap,x_m=xpos,tx_power_dbm=16))
    for i,e in enumerate(expected):
        for key,value in e.items():
            got=k.iloc[i][key]
            require(abs(float(got)-value)<1e-6 if isinstance(value,(float,int)) else got==value,f'knob {folder.name} row{i} {key}')
    return 4

def audit_one(rec,inputs,engineering=False):
    folder=STUDY/('engineering' if engineering else 'runs')/f"{rec['scenario_id']}_r{rec['rng_run']:04d}_{rec['action']}"
    require(rec['input_sha256']==sha(STUDY/'inputs'/f"{rec['scenario_id']}.csv"),'input hash')
    require(rec['binary_sha256']==sha(STUDY/'build/action_effects.exe'),'binary hash')
    require(all(sha(folder/n)==h for n,h in rec['files'].items()),'output hashes')
    raw=gzip.decompress((folder/'packets.csv.gz').read_bytes())
    pre=gzip.decompress((folder/'prefix.csv.gz').read_bytes())
    require(hashlib.sha256(raw).hexdigest()==rec['packets_uncompressed_sha256'],'raw ledger hash')
    require(hashlib.sha256(pre).hexdigest()==rec['prefix_uncompressed_sha256'],'raw prefix hash')
    f=pd.read_csv(io.BytesIO(raw),dtype=np.int64)
    p=pd.read_csv(io.BytesIO(pre),dtype=np.int64)
    zero='--zero=1' in rec['command']
    if zero:
        require(f.empty and p.empty,'zero application offers')
        return {'scenario_id':rec['scenario_id'],'rng_run':rec['rng_run'],'action':rec['action'],'packet_rows':0},None
    source=inputs[rec['scenario_id']]
    require(len(f)==len(source),'ledger/source row count')
    require(f.packet_id.to_numpy()==source.packet_id.to_numpy(),'packet ID sequence')
    require(f.flow.to_numpy()==source.flow.to_numpy(),'flow sequence')
    require(f.payload_bytes.to_numpy()==source.payload_bytes.to_numpy(),'wire payload byte sequence')
    require(f.offer_ns.to_numpy()==source.relative_us.to_numpy()*1000+1_000_000_000,'offer timing')
    sent=f.send_ns>=0;rx=f.receive_ns>=0;drop=f.shaper_drop_ns>=0;fail=f.socket_fail_ns>=0
    require(sent.astype(int)+drop.astype(int)+fail.astype(int)<=1,'mutually exclusive application exits')
    pending=~(sent|drop|fail)
    require(~pending|(f.flow==1),'only video can wait in shaper')
    require(~rx|sent,'receive requires successful socket submission')
    for field in ['send_ns','receive_ns','shaper_drop_ns','socket_fail_ns','first_phy_ns','last_phy_ns','last_mac_drop_ns']:
        x=f[field];valid=x>=0
        require(x>=-1,'invalid absent sentinel '+field)
        require(x[valid]>=f.offer_ns[valid],'event precedes offer '+field)
        require(x[valid]<HORIZON,'event outside horizon '+field)
    require(f.receive_ns[rx]>=f.send_ns[rx],'receive precedes send')
    phy=f.phy_attempts>0
    require(f.phy_attempts>=0,'negative PHY attempt count')
    require(f.phy_attempts<=7,'more than seven frame transmission attempts')
    require(f.first_phy_ns[phy]>=f.send_ns[phy],'PHY before send')
    require(f.last_phy_ns[phy]>=f.first_phy_ns[phy],'reversed PHY timestamps')
    require((f.first_phy_ns[~phy]==-1)&(f.last_phy_ns[~phy]==-1),'zero PHY attempt timestamp')
    require(~phy|sent,'unsent packet had PHY attempt')
    require(~rx|phy,'application receipt with no PHY attempt')
    require(f.duplicates>=0,'negative duplicate count')
    require((f.duplicates==0)|rx,'duplicates without first receive')
    for count in ['mac_acks','mac_drops']:
        require(f[count]>=0,'negative '+count)
    require((f.mac_drops==0)==(f.last_mac_drop_ns==-1),'MAC drop time/count relation')
    require((f.mac_drops==0)==(f.drop_reason_mask==0),'MAC drop reason/count relation')
    tid=np.where((rec['action']=='FallbackProtect')&(f.flow==0)&(f.send_ns>=ACTIVE),64,1)
    require(f.tid_mask[phy]==tid[phy],'actual PHY QoS category')
    classified=f.tid_mask>0
    require(f.tid_mask[classified]==tid[classified],'actual MAC QoS category')
    expected_prefix=f[f.offer_ns<BEGIN]
    require(p.packet_id.to_numpy()==expected_prefix.packet_id.to_numpy(),'prefix offer IDs')
    require(p[['flow','payload_bytes','offer_ns']].to_numpy()==expected_prefix[['flow','payload_bytes','offer_ns']].to_numpy(),'prefix offer identity')
    require((p.loc[:,p.columns.str.endswith('_ns')]<BEGIN).to_numpy(),'post-command event in prefix')
    final=f.set_index('packet_id').loc[p.packet_id]
    for field in ['send_ns','receive_ns','shaper_drop_ns','socket_fail_ns','first_phy_ns']:
        present=p[field].to_numpy()>=0
        require(p[field].to_numpy()[present]==final[field].to_numpy()[present],'immutable prefix event '+field)
    for field in ['duplicates','phy_attempts','mac_acks','mac_drops']:
        require(p[field].to_numpy()<=final[field].to_numpy(),'prefix cumulative count '+field)
    check_knobs(folder,rec)
    selected=(f.offer_ns>=BEGIN)&(f.offer_ns<END)
    c=f[selected&(f.flow==0)];v=f[selected&(f.flow==1)]
    c2_late=(c.receive_ns<0)|(c.receive_ns-c.offer_ns>10_000_000)
    c2_bytes=int(c.payload_bytes.sum())
    require(len(c)==399 and c2_bytes==24005,'C2 evaluation cohort')
    # Independent mutually exclusive five-way video partition, including unsent waiting.
    states=Counter()
    for row in v[['payload_bytes','send_ns','receive_ns','shaper_drop_ns','socket_fail_ns']].itertuples(index=False,name=None):
        n,s,r,d,j=row
        state='received' if r>=0 else 'shaper_drop' if d>=0 else 'socket_reject' if j>=0 else 'network_horizon' if s>=0 else 'shaper_pending'
        states[state]+=n
    require(sum(states.values())==int(v.payload_bytes.sum()),'offered byte conservation')
    summary={'scenario_id':rec['scenario_id'],'rng_run':rec['rng_run'],'action':rec['action'],'packet_rows':len(f),
             'prefix_rows':len(p),'c2_offered_count':len(c),'c2_offered_bytes':c2_bytes,'c2_deadline_miss_count':int(c2_late.sum()),
             'c2_received_count':int((c.receive_ns>=0).sum()),'video_offered_bytes':int(v.payload_bytes.sum()),
             **{'video_'+k+'_bytes':int(states[k]) for k in ['received','shaper_drop','socket_reject','network_horizon','shaper_pending']},
             'video_receive_window_bytes':int(f.loc[(f.flow==1)&(f.receive_ns>=BEGIN)&(f.receive_ns<END),'payload_bytes'].sum()),
             'all_phy_attempts':int(f.phy_attempts.sum()),'max_phy_attempts_per_packet':int(f.phy_attempts.max()),
             'received_with_mac_drop_count':int((rx&(f.mac_drops>0)).sum()),'shaper_pending_count':int(pending.sum()),
             'duplicates':int(f.duplicates.sum())}
    return summary,f

def pcap_frames(path):
    blob=path.read_bytes();magic=blob[:4]
    formats={b'\xd4\xc3\xb2\xa1':('<',1000),b'\xa1\xb2\xc3\xd4':('>',1000),b'\x4d\x3c\xb2\xa1':('<',1),b'\xa1\xb2\x3c\x4d':('>',1)}
    require(magic in formats,'PCAP magic')
    endian,mult=formats[magic]
    link=struct.unpack_from(endian+'I',blob,20)[0]
    require(link in [105,127],'supported 802.11 PCAP link type')
    offset=24;records=[]
    while offset<len(blob):
        sec,sub,cap,original=struct.unpack_from(endian+'IIII',blob,offset);offset+=16
        b=blob[offset:offset+cap];offset+=cap
        require(len(b)==cap and cap==original,'full captured frame')
        if link==127:
            rtlen=struct.unpack_from('<H',b,2)[0];b=b[rtlen:]
        if len(b)<24:continue
        fc=struct.unpack_from('<H',b)[0]
        if (fc>>2)&3!=2:continue
        qos=bool(fc&0x80);h=24+(6 if fc&0x0300==0x0300 else 0)+(2 if qos else 0)+(4 if qos and fc&0x8000 else 0)
        if b[h:h+8]!=b'\xaa\xaa\x03\x00\x00\x00\x08\x00':continue
        ip=b[h+8:]
        if len(ip)<28 or ip[0]>>4!=4 or ip[9]!=17:continue
        ihl=(ip[0]&15)*4;total,ident,frag=struct.unpack_from('>HHH',ip,2)
        require(not(frag&0x3fff),'unfragmented IPv4')
        sport,dport,udp_len=struct.unpack_from('>HHH',ip,ihl)
        require(total==ihl+udp_len,'UDP/IP length conservation')
        payload=ip[ihl+8:total]
        require(len(payload)==udp_len-8,'captured application payload completeness')
        records.append({'src':'.'.join(map(str,ip[12:16])),'dst':'.'.join(map(str,ip[16:20])),
                        'sport':sport,'dport':dport,'ident':ident,'bytes':len(payload),'time_ns':sec*1_000_000_000+sub*mult,
                        'tid':b[24]&15 if qos else None,'retry':bool(fc&0x0800)})
    require(offset==len(blob),'PCAP consumed without trailing bytes')
    return link,records

def audit_pcap(low,frame):
    folder=STUDY/'engineering'/f"{low['scenario_id']}_r{low['rng_run']:04d}_{low['action']}"
    paths=sorted(folder.glob('engineering-*.pcap'))
    require(len(paths)==4,'four device PCAP files')
    result=[]
    for path in paths:
        link,packets=pcap_frames(path)
        node=int(path.name.split('-')[-2])
        if node not in [0,1]:
            result.append({'file':path.name,'node':node,'linktype':link,'udp_frame_records':len(packets),'role':'overhearing node, no application claims'})
            continue
        for flow in [0,1]:
            selected=[r for r in packets if r['dport']==9000+flow]
            unique={}
            for r in selected:
                key=(r['src'],r['dst'],r['sport'],r['dport'],r['ident'])
                if key in unique: require(unique[key]['bytes']==r['bytes'],'retry payload length unchanged')
                unique[key]=r
            own=(node==flow)
            ledger=frame[(frame.flow==flow)&((frame.phy_attempts>0) if own else (frame.receive_ns>=0))]
            require(len(unique)==len(ledger),'unique PCAP vs ledger count '+path.name+str(flow))
            require(sum(r['bytes'] for r in unique.values())==int(ledger.payload_bytes.sum()),'PCAP vs ledger payload byte conservation')
            require(Counter(r['bytes'] for r in unique.values())==Counter(map(int,ledger.payload_bytes)),'PCAP payload size multiset')
            require(all(r['tid']==0 for r in selected),'low-load PCAP BE classification')
            if own:require(len(selected)==int(ledger.phy_attempts.sum()),'PCAP source transmission attempts')
            result.append({'file':path.name,'node':node,'linktype':link,'flow':flow,'role':'source' if own else 'receiver',
                           'udp_frame_records':len(selected),'unique_datagrams':len(unique),'unique_payload_bytes':sum(r['bytes'] for r in unique.values())})
    return result

def main():
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['engineering','formal']);args=parser.parse_args()
    inputs=source_inputs()
    filename='engineering_gates.json' if args.stage=='engineering' else 'execution_receipt.json'
    root=read(STUDY/filename);require(root['status']=='PASS','root execution completed')
    recs=root['runs'];expected=7 if args.stage=='engineering' else 640
    require(len(recs)==expected,'expected run count')
    grouped=defaultdict(list);summaries=[];pcap=[]
    for i,rec in enumerate(recs):
        summary,frame=audit_one(rec,inputs,args.stage=='engineering');summaries.append(summary)
        grouped[(rec['scenario_id'],rec['rng_run'])].append(rec['prefix_uncompressed_sha256'])
        if args.stage=='engineering' and '--pcap=1' in rec['command']:
            pcap=audit_pcap(rec,frame)
        if (i+1)%40==0:print(f'Independent {args.stage} ledger audit {i+1}/{expected}',flush=True)
    for key,hashes in grouped.items(): require(len(set(hashes))==1,'paired observable prefix '+str(key))
    if args.stage=='formal':
        wanted={(s,r,a) for s in inputs for r in range(1,17) for a in ACTIONS}
        require({(r['scenario_id'],r['rng_run'],r['action']) for r in recs}==wanted,'complete formal grid')
        require(len(grouped)==128 and all(len(x)==5 for x in grouped.values()),'128 five-arm prefix comparisons')
        bound=read(STUDY/'input_binding.json')
        require(all(sha(STUDY/n)==h for n,h in bound['study_files'].items()),'frozen study file hashes')
        require(all(sha(Path(n))==h for n,h in bound['runtime_files'].items()),'frozen runtime file hashes')
        with (HERE/'formal_independent_episode_counts.csv').open('w',newline='',encoding='utf-8') as f:
            writer=csv.DictWriter(f,fieldnames=list(summaries[0]));writer.writeheader();writer.writerows(summaries)
    report={'schema_version':1,'status':'PASS','stage':args.stage,'runs':len(recs),'paired_groups':len(grouped),
            'packet_rows_checked':sum(x['packet_rows'] for x in summaries),'source_packets_checked':800,'scenario_schedules_checked':8,
            'source_parser':'independent direct PCAP/IP/UDP header parser, not P14 parser',
            'checks':['all receipt/output/input/binary hashes','schedule generated independently from public prefix and stated cycles',
                      'offer and payload identity','mutually exclusive application exits','all receive/PHY/MAC timestamp causality',
                      'maximum seven frame attempts','all actual TID masks','pre-command prefix times and equal paired snapshots',
                      'all before/after controlled knob readbacks','five-way offered video byte partition'],
            'pcap_checks':pcap,'root_receipt_sha256':sha(STUDY/filename),
            'source_hashes':{n:sha(STUDY/n) for n in ['protocol.json','protocol.md','action_effects.cc','run_study.py','analyze_results.py']},
            'verifier_sha256':sha(Path(__file__)),'episode_counts':summaries if args.stage=='engineering' else 'formal_independent_episode_counts.csv'}
    out(args.stage+'_independent_audit.json',report)
    print(json.dumps({'status':'PASS','stage':args.stage,'runs':len(recs),'packet_rows_checked':report['packet_rows_checked']},indent=2))

if __name__=='__main__':main()
