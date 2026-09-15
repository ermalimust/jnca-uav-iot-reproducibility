"""Strict parsing of author-released, payload-stripped transport captures."""
from pathlib import Path
import hashlib
import ipaddress
import json
import struct
import zipfile
import numpy as np

HERE=Path(__file__).resolve().parent
WORK=HERE.parents[2]
P1=HERE.parent/'P1_des_recovery_20260912'
REPLAY=WORK/'revision_work/analysis/replay_inputs'
PUBLIC_ARCHIVE_SHA256='0567854171e615713960e15e0566bc23534633d3bad8ed0a798b74c51da22beb'
PUBLIC_ARCHIVE_URL='https://raw.githubusercontent.com/aygunbaltaci/AVIATOR/c836ab4b1bab75a85c37803575ae6240d15075f9/uav_datatraces.zip'
FLOW_SPECS=[
 dict(model='DJI Mavic Air',id='mavic_air',member='sample_pcaps/djimavicair_modified.pcap',src='192.168.2.20',sport=10002,dst='192.168.2.1',dport=9003),
 dict(model='DJI Spark',id='spark',member='sample_pcaps/djispark_modified.pcap',src='192.168.2.20',sport=10002,dst='192.168.2.1',dport=9003),
 dict(model='Parrot AR 2.0',id='parrot_ar2',member='sample_pcaps/parrotar2_modified.pcap',src='192.168.1.3',sport=5556,dst='192.168.1.1',dport=5556),
]
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load_json(p): return json.loads(Path(p).read_text(encoding='utf-8'))
def save_json(p,obj): Path(p).write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')

def locate_archive(explicit=None,download=False):
    candidates=[Path(explicit)] if explicit else [HERE/'cache/uav_datatraces.zip',WORK/'output/dataset_search_20260913/aviator_uav_datatraces.zip']
    p=next((p for p in candidates if p.is_file()),None)
    if p is None and download:
        from urllib.request import urlopen
        p=HERE/'cache/uav_datatraces.zip';p.parent.mkdir(parents=True,exist_ok=True)
        with urlopen(PUBLIC_ARCHIVE_URL,timeout=60) as response: data=response.read()
        assert hashlib.sha256(data).hexdigest()==PUBLIC_ARCHIVE_SHA256,'Public archive hash differs'
        p.write_bytes(data)
    if p is None: raise FileNotFoundError('Supply --aviator-zip or use --download-public-source for the pinned public archive.')
    assert sha(p)==PUBLIC_ARCHIVE_SHA256,'Public source bytes differ from the frozen contract'
    return p

def pcap_records(data):
    formats={b'\xd4\xc3\xb2\xa1':'<',b'\xa1\xb2\xc3\xd4':'>'}
    if data[:4] not in formats: raise ValueError('Expected a microsecond-resolution pcap')
    endian=formats[data[:4]]
    _,major,minor,_,_,snaplen,linktype=struct.unpack(endian+'IHHIIII',data[:24])
    assert (major,minor)==(2,4) and linktype==1,'Expected Ethernet transport sample'
    offset=24;index=0
    while offset<len(data):
        assert offset+16<=len(data)
        sec,micro,caplen,original=struct.unpack(endian+'IIII',data[offset:offset+16]);offset+=16
        assert micro<1000000 and caplen<=snaplen and offset+caplen<=len(data)
        packet=data[offset:offset+caplen];offset+=caplen
        yield index,sec*1000000+micro,original,packet
        index+=1

def extract_emissions(archive):
    recordings=[]
    with zipfile.ZipFile(archive) as z:
        for spec in FLOW_SPECS:
            selected=[];total=0;invalid_target=0
            for index,time_us,original,packet in pcap_records(z.read(spec['member'])):
                total+=1
                if len(packet)<34 or packet[12:14]!=b'\x08\x00' or packet[23]!=17: continue
                ihl=(packet[14]&15)*4
                if ihl<20 or len(packet)<14+ihl+8: continue
                src=str(ipaddress.ip_address(packet[26:30]));dst=str(ipaddress.ip_address(packet[30:34]))
                sport,dport,udp_len,_=struct.unpack('>HHHH',packet[14+ihl:22+ihl])
                if (src,sport,dst,dport)!=(spec['src'],spec['sport'],spec['dst'],spec['dport']): continue
                ip_total=struct.unpack('>H',packet[16:18])[0]
                fragment=struct.unpack('>H',packet[20:22])[0]
                if fragment&0x3fff or udp_len<8 or ip_total!=ihl+udp_len or original<14+ip_total:
                    invalid_target+=1;continue
                selected.append((time_us,index,udp_len-8,ip_total,original))
            assert selected and invalid_target==0,'Invalid target flow headers must be resolved before analysis'
            reversals=sum(b[0]<a[0] for a,b in zip(selected,selected[1:]))
            selected.sort(key=lambda r:(r[0],r[1]))
            arr=np.asarray(selected,dtype=np.int64)
            times=arr[:,0]-arr[0,0]
            recordings.append(dict(**spec,times_us=times,packet_indices=arr[:,1],payload_bytes=arr[:,2],
                metadata=dict(total_capture_records=total,selected_datagrams=len(selected),first_timestamp_us=int(arr[0,0]),
                    last_timestamp_us=int(arr[-1,0]),span_us=int(times[-1]),capture_order_reversals=reversals,
                    tied_timestamps=int(np.count_nonzero(np.diff(times)==0)),invalid_target_headers=invalid_target,
                    timestamp_boundary='RC transport capture emission',flight_condition='not identified per sample in the public archive')))
    return recordings

def block_emissions(recording,block_us=90000000):
    times=recording['times_us'];blocks=[]
    for i in range(int(times[-1])//block_us):
        start=i*block_us;end=start+block_us
        mask=(times>=start)&(times<end)
        blocks.append(dict(recording_id=recording['id'],block_index=i,start_us=start,end_us=end,
            relative_us=times[mask]-start,payload_bytes=recording['payload_bytes'][mask],
            source_packet_indices=recording['packet_indices'][mask]))
    return blocks

def complete_counts(times_us,start_us,end_us,phase_us=0,window_us=100000):
    start=int(start_us)+phase_us;n=(int(end_us)-start)//window_us
    assert n>0
    stop=start+n*window_us
    t=np.asarray(times_us);v=t[(t>=start)&(t<stop)]
    idx=((v-start)//window_us).astype(np.int64)
    return np.bincount(idx,minlength=n).astype(np.int64)

def weighted_w1(a,b,wa=None,wb=None):
    a=np.asarray(a,dtype=float);b=np.asarray(b,dtype=float)
    wa=np.full(len(a),1/len(a)) if wa is None else np.asarray(wa,dtype=float)
    wb=np.full(len(b),1/len(b)) if wb is None else np.asarray(wb,dtype=float)
    assert np.isfinite(a).all() and np.isfinite(b).all() and (wa>=0).all() and (wb>=0).all()
    ia=np.argsort(a,kind='stable');ib=np.argsort(b,kind='stable')
    a=a[ia];b=b[ib];ca=np.r_[0,np.cumsum(wa[ia]/wa.sum())];cb=np.r_[0,np.cumsum(wb[ib]/wb.sum())]
    support=np.unique(np.r_[a,b])
    return float(np.sum(np.diff(support)*np.abs(ca[np.searchsorted(a,support[:-1],side='right')]-cb[np.searchsorted(b,support[:-1],side='right')])))

def balanced_pool(groups):
    return np.concatenate(groups),np.concatenate([np.full(len(a),1/(len(groups)*len(a))) for a in groups])
