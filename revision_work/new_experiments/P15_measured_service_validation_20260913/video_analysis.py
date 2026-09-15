"""Direct, deduplicated transport delivery observations from public Parrot video.

P14 is read-only. This script does not fit or run a new diagnostic model.
"""
import sys
sys.dont_write_bytecode = True
from pathlib import Path
import argparse, csv, hashlib, importlib.util, json, struct, zipfile
from datetime import datetime, timezone
import numpy as np

HERE = Path(__file__).resolve().parent
WORK = HERE.parents[2]
P14 = HERE.parent/'P14_public_arrival_validation_20260913'
OUT = HERE/'video/results'
spec = importlib.util.spec_from_file_location('p14_source_parser', P14/'trace_inputs.py')
source = importlib.util.module_from_spec(spec)
spec.loader.exec_module(source)

def digest(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p): return json.loads(Path(p).read_text(encoding='utf-8'))
def save(p,obj): Path(p).write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def write_csv(p,rows):
    with Path(p).open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def insert_interval(ranges,start,end):
    overlap=sum(max(0,min(b,end)-max(a,start)) for a,b in ranges)
    merged=[];a,b=start,end;inserted=False
    for x,y in ranges:
        if y<a: merged.append((x,y))
        elif b<x:
            if not inserted: merged.append((a,b));inserted=True
            merged.append((x,y))
        else: a=min(a,x);b=max(b,y)
    if not inserted: merged.append((a,b))
    ranges[:]=merged
    return end-start-overlap

def parse_video(archive):
    with zipfile.ZipFile(archive) as z:
        records=list(source.pcap_records(z.read('sample_pcaps/parrotar2_modified.pcap')))
    packets=[];rejected=[];target_count=0
    for idx,t,original,p in records:
        if len(p)<34 or p[12:14]!=b'\x08\x00' or p[23]!=6: continue
        src='.'.join(map(str,p[26:30]));dst='.'.join(map(str,p[30:34]))
        ihl=(p[14]&15)*4
        if len(p)<14+ihl+4: continue
        sport,dport=struct.unpack('>HH',p[14+ihl:18+ihl])
        forward=src=='192.168.1.1' and sport==5555 and dst=='192.168.1.3'
        reverse=dst=='192.168.1.1' and dport==5555 and src=='192.168.1.3'
        if not (forward or reverse): continue
        target_count+=1
        try:
            assert p[14]>>4==4 and ihl>=20 and len(p)>=14+ihl+20
            iplen=struct.unpack('>H',p[16:18])[0]
            assert struct.unpack('>H',p[20:22])[0]&0x3fff==0
            tcp=p[14+ihl:];thl=(tcp[12]>>4)*4
            assert thl>=20 and len(tcp)>=thl and iplen>=ihl+thl
            assert original>=14+iplen
            seq,ack=struct.unpack('>II',tcp[4:12])
            flags=tcp[13];length=iplen-ihl-thl
            packets.append(dict(index=idx,timestamp_us=t,forward=forward,client_port=dport if forward else sport,
                                seq=seq,ack=ack,flags=flags,payload_bytes=length,
                                retained_payload_bytes=max(0,len(p)-14-ihl-thl),original_length=original,ip_length=iplen))
        except AssertionError: rejected.append(idx)
    assert not rejected,('Unresolved target packet structure',rejected)
    reversals=sum(b['timestamp_us']<a['timestamp_us'] for a,b in zip(packets,packets[1:]))
    packets.sort(key=lambda p:(p['timestamp_us'],p['index']))
    states={};connections=[];events=[]
    def new_state(p,start):
        q=dict(connection_id=len(connections),client_port=p['client_port'],start_kind=start,
               client_isn=None,server_isn=None,first_timestamp_us=p['timestamp_us'],last_timestamp_us=p['timestamp_us'],
               payload_segments=0,raw_payload_bytes=0,unique_payload_bytes=0,duplicate_payload_bytes=0,
               fully_duplicate_segments=0,partly_duplicate_segments=0,sequence_backsteps=0,
               closed=False,fin_seen=False,rst_seen=False,ranges=[],reference=None)
        connections.append(q);states[p['client_port']]=q;return q
    for p in packets:
        syn=bool(p['flags']&2);ack=bool(p['flags']&16)
        q=states.get(p['client_port'])
        if not p['forward'] and syn and not ack:
            if q is None or q['client_isn']!=p['seq']:
                q=new_state(p,'observed_client_SYN');q['client_isn']=p['seq']
        if q is None: q=new_state(p,'left_censored_no_client_SYN')
        q['last_timestamp_us']=p['timestamp_us']
        if p['forward'] and syn:
            assert q['server_isn'] in (None,p['seq']),'Unresolved server sequence reset'
            q['server_isn']=p['seq']
            if q['reference'] is None: q['reference']=p['seq']+1
        if p['flags']&1: q['fin_seen']=True
        if p['flags']&4: q['rst_seen']=True;q['closed']=True
        if not p['forward'] or not p['payload_bytes']: continue
        assert not syn,'Unexpected data-bearing SYN requires explicit handling'
        raw=p['seq'];ref=q['reference']
        start=raw if ref is None else ref+((raw-ref+(1<<31))%(1<<32)-(1<<31))
        end=start+p['payload_bytes']
        if ref is not None and start<ref: q['sequence_backsteps']+=1
        q['reference']=max(end,ref if ref is not None else end)
        new=insert_interval(q['ranges'],start,end)
        dup=p['payload_bytes']-new
        q['payload_segments']+=1;q['raw_payload_bytes']+=p['payload_bytes']
        q['unique_payload_bytes']+=new;q['duplicate_payload_bytes']+=dup
        q['fully_duplicate_segments']+=int(new==0)
        q['partly_duplicate_segments']+=int(new>0 and dup>0)
        events.append(dict(connection_id=q['connection_id'],packet_index=p['index'],timestamp_us=p['timestamp_us'],
                           client_port=p['client_port'],sequence_raw=raw,sequence_unwrapped=start,
                           raw_payload_bytes=p['payload_bytes'],new_payload_bytes=new,duplicate_payload_bytes=dup,
                           retained_payload_bytes=p['retained_payload_bytes']))
    rows=[]
    for q in connections:
        intervals=q.pop('ranges');q.pop('reference')
        q['observed_union_bytes']=sum(b-a for a,b in intervals)
        q['unobserved_sequence_gap_bytes']=sum(intervals[i+1][0]-intervals[i][1] for i in range(len(intervals)-1))
        assert q['observed_union_bytes']==q['unique_payload_bytes']
        rows.append(q)
    assert sum(r['payload_segments'] for r in rows)==36042
    return events,rows,dict(capture_records=len(records),target_tcp_records=target_count,invalid_target_records=len(rejected),
                            target_capture_order_reversals=reversals,target_tied_timestamps=sum(a['timestamp_us']==b['timestamp_us'] for a,b in zip(packets,packets[1:])),
                            payload_segments=len(events),unique_payload_bytes=sum(e['new_payload_bytes'] for e in events),
                            raw_payload_bytes=sum(e['raw_payload_bytes'] for e in events),
                            duplicate_payload_bytes=sum(e['duplicate_payload_bytes'] for e in events),
                            positive_payload_retained_records=sum(e['retained_payload_bytes']>0 for e in events))

def windows(events,start,end,width=100000):
    n=(end-start)//width
    times=np.array([e['timestamp_us'] for e in events],dtype=np.int64)
    new=np.array([e['new_payload_bytes'] for e in events],dtype=np.int64)
    raw=np.array([e['raw_payload_bytes'] for e in events],dtype=np.int64)
    keep=(times>=start)&(times<start+n*width)
    ids=(times[keep]-start)//width
    count=np.bincount(ids,minlength=n)
    u=np.zeros(n,dtype=np.int64);r=np.zeros(n,dtype=np.int64)
    np.add.at(u,ids,new[keep]);np.add.at(r,ids,raw[keep])
    return [dict(window_index=i,start_us=start+i*width,end_us=start+(i+1)*width,
                 payload_segments=int(count[i]),unique_payload_bytes=int(u[i]),raw_payload_bytes=int(r[i]),
                 rate_mbps=float(u[i]*8/width)) for i in range(n)]

def stats(a):
    a=np.asarray(a,float)
    return dict(n=len(a),mean=float(a.mean()),median=float(np.median(a)),p05=float(np.quantile(a,.05)),
                p95=float(np.quantile(a,.95)),min=float(a.min()),max=float(a.max()),zero_fraction=float(np.mean(a==0)))

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--aviator-zip');ap.add_argument('--download-public-source',action='store_true');args=ap.parse_args()
    archive=source.locate_archive(args.aviator_zip,args.download_public_source)
    OUT.mkdir(parents=True,exist_ok=True)
    index=read(P14/'results/index.json')
    index['runs']=read(P14/'results/run_index.json')
    inputs=[HERE/'protocol.md',Path(__file__),P14/'trace_inputs.py',P14/'results/index.json',P14/'results/run_index.json',P14/'results/native_arrays.npz']
    inputs += [P14/'results'/f'run_{i:03}.npz' for i,r in enumerate(index['runs']) if r['recording_id']=='parrot_ar2']
    hashes={p.relative_to(WORK).as_posix():digest(p) for p in inputs}
    hashes['AVIATOR_public_archive']=digest(archive)
    bind=HERE/'video/input_binding.json'
    if bind.exists(): assert read(bind)['sha256']==hashes,'Input correction requires explicit recorded amendment'
    else: save(bind,dict(frozen_utc=datetime.now(timezone.utc).isoformat(),sha256=hashes,preanalysis='Source flow inventory and prior P14 outputs known; new service statistics not yet computed.'))
    events,connections,audit=parse_video(archive)
    rec=next(r for r in source.extract_emissions(archive) if r['id']=='parrot_ar2')
    origin=rec['metadata']['first_timestamp_us'];end=rec['metadata']['last_timestamp_us']
    all_windows=windows(events,origin,end)
    shifted=windows(events,origin+50000,end)
    selected=[]
    for block in range(4):
        rows=windows(events,origin+block*90000000+20000000,origin+(block+1)*90000000)
        assert len(rows)==700
        for r in rows: r['block_index']=block
        selected.extend(rows)
    feature_names=index['features']
    col=feature_names.index('video_throughput_mbps')
    native=np.load(P14/'results/native_arrays.npz')['features'][:,:,col]
    trace=[];pairs=[];desc=[]
    obs=np.array([r['rate_mbps'] for r in selected]);full=np.array([r['rate_mbps'] for r in all_windows]);phase=np.array([r['rate_mbps'] for r in shifted])
    for i,run in enumerate(index['runs']):
        if run['recording_id']!='parrot_ar2': continue
        a=np.load(P14/'results'/f'run_{i:03}.npz')['features'][:,col];trace.append(a)
        measured=obs[run['block_index']*700:(run['block_index']+1)*700]
        pairs.append(dict(run_index=i,block_index=run['block_index'],scenario_id=run['scenario_id'],
                          W1_mbps=source.weighted_w1(measured,a),observed_mean=float(measured.mean()),model_mean=float(a.mean())))
    trace=np.array(trace);assert trace.shape==(32,700)
    groups={'observed_full':full,'observed_phase50':phase,'observed_P14_intervals':obs,'native_DES_reference':native.ravel(),'Parrot_arrival_DES_reference':trace.ravel()}
    for k,a in groups.items(): desc.append(dict(population=k,**stats(a)))
    for b in range(4): desc.append(dict(population=f'observed_block_{b}',**stats(obs[b*700:(b+1)*700])))
    for i,s in enumerate(index['scenarios']): desc.append(dict(population=s['scenario_id'],**stats(native[i])))
    distances=dict(observed_vs_native_W1_mbps=source.weighted_w1(obs,native.ravel()),
                   observed_vs_arrival_DES_W1_mbps=source.weighted_w1(obs,trace.ravel()),
                   full_vs_phase50_W1_mbps=source.weighted_w1(full,phase))
    audit.update(dict(control_origin_timestamp_us=origin,control_end_timestamp_us=end,
                      complete_primary_windows=len(all_windows),complete_shifted_windows=len(shifted),
                      aligned_observed_windows=len(selected),observed_unique_bytes_in_primary_windows=sum(r['unique_payload_bytes'] for r in all_windows),
                      observed_unique_bytes_in_aligned_windows=sum(r['unique_payload_bytes'] for r in selected),
                      independent_recordings=1,model_native_windows=native.size,model_arrival_windows=trace.size))
    for name,rows in [('payload_events.csv',events),('connections.csv',connections),('windows_100ms.csv',all_windows),('windows_phase50.csv',shifted),('windows_P14_intervals.csv',selected),('descriptives.csv',desc),('reference_distances_by_run.csv',pairs)]:write_csv(OUT/name,rows)
    save(OUT/'source_audit.json',audit)
    save(OUT/'summary.json',dict(source=audit,descriptives=desc,distances=distances,
         scope='Captured unique video-flow TCP payload delivery and unmatched fixed-DES service reference; no full measured diagnostic, calibration, paired-condition error or equivalence claim.',hypothesis_tests_added=0))
    np.savez_compressed(OUT/'comparison_arrays.npz',**groups,native_by_scenario=native,arrival_by_run=trace)
    save(OUT/'result_manifest.json',{p.name:digest(p) for p in sorted(OUT.iterdir()) if p.is_file() and p.name!='result_manifest.json'})
    print(json.dumps(dict(audit=audit,descriptives=desc[:5],distances=distances),indent=2))

if __name__=='__main__':main()
