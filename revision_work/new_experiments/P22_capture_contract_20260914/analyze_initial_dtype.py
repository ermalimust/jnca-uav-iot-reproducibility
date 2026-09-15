"""Observation evidence, fixed-denominator diagnostics and new-source feasibility."""
from pathlib import Path
import datetime, hashlib, json, shutil
import numpy as np
import pandas as pd
R=Path(__file__).resolve().parent;W=R.parents[2];P21=R.parent/'P21_spatial_capture_20260914';P20=R.parent/'P20_observation_adapter_20260914'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,obj):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
def w1(a,b):
 a=np.sort(np.asarray(a));a=a[np.isfinite(a)];b=np.sort(np.asarray(b));b=b[np.isfinite(b)]
 if not len(a) or not len(b):return np.nan
 q=np.unique(np.r_[a,b]);x=q[:-1]
 return float(np.sum(np.diff(q)*abs(np.searchsorted(a,x,side='right')/len(a)-np.searchsorted(b,x,side='right')/len(b))))
assert not (R/'analysis_receipt.json').exists(),'Analysis already sealed'
out=R/'results';out.mkdir(exist_ok=True)
(R/'inputs').mkdir(exist_ok=True)
shutil.copy2(P20/'inputs/vnc20_wifi.csv',R/'inputs/vnc20_wifi.csv')
raw=pd.read_csv(R/'inputs/vnc20_wifi.csv')
binding=[R/'protocol.md',R/'analyze.py',R/'inputs/vnc20_wifi.csv',R/'sources/repository_receipt.json',R/'sources/logs_receipt.json',
         P21/'selection_binding.json',P21/'replay/execution_binding.json',W/'output/spatial_capture_20260914/artifact_manifest.json']
dump(R/'execution_binding.json',dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),files={str(p):sha(p) for p in binding}))
d=raw[raw.wifiType.eq('n')&raw.channelFreq.eq(2437)&raw.traceNr.isin([302,303,304])].copy().sort_values(['traceNr','systime'])
assert len(d)==8710 and not d[['traceNr','systime']].duplicated().any()
d['monitor_data_positive']=d.nBytesReceived.gt(0);d['monitor_beacon_positive']=d.nBeacons.gt(0)
d['consumer_positive']=d.tghptConsumer.gt(0);d['busy_available']=d.channelUtil.between(0,100)
d['any_positive']=d.monitor_data_positive|d.monitor_beacon_positive|d.consumer_positive
d['monitor_empty']=~(d.monitor_data_positive|d.monitor_beacon_positive)
d['evidence_state']=np.select([d.monitor_data_positive,d.monitor_beacon_positive,d.consumer_positive,d.busy_available],
 ['monitor_data','beacon_only','consumer_only','busy_only'],default='unresolved_all')
states=d.groupby(['traceNr','evidence_state']).size().unstack(fill_value=0).reindex(columns=['monitor_data','beacon_only','consumer_only','busy_only','unresolved_all'],fill_value=0)
states['total_seconds']=states.sum(axis=1);states.to_csv(out/'vnc_evidence_states.csv')
for trace,v in d.groupby('traceNr'):
 times=v.systime.to_numpy();positive=set(v.loc[v.consumer_positive,'systime'])
 d.loc[v.index,'consumer_positive_pm1']=[any(t+delta in positive for delta in [-1,0,1]) for t in times]
d.to_csv(out/'vnc_evidence_seconds.csv.gz',index=False,compression={'method':'gzip','mtime':0})
timing=[]
for (trace,radio),v in raw.groupby(['traceNr','wifiType']):
 positives=set(v.loc[v.tghptConsumer.gt(0),'systime']);empty=v.nBytesReceived.eq(0)&v.nBeacons.eq(0)
 timing.append(dict(trace=int(trace),radio=radio,seconds=len(v),monitor_empty_seconds=int(empty.sum()),
   same_second_consumer_positive=int((empty&v.tghptConsumer.gt(0)).sum()),
   consumer_positive_pm1=int(sum(any(t+delta in positives for delta in [-1,0,1]) for t in v.loc[empty,'systime']))))
pd.DataFrame(timing).to_csv(out/'consumer_timing_sensitivity.csv',index=False)

# Keep complete denominators; strata are source-evidence diagnostics, not improved outage accuracy.
rows=[]
for tag,base in [('P20',P20),('P21',P21/'replay')]:
 sim=pd.read_csv(base/'results/simulated_observations.csv');sim=sim[sim.shadow.eq(True)]
 emp=pd.read_csv(base/'results/measured_observations.csv')
 emp=emp.merge(d[['traceNr','systime','consumer_positive','any_positive','busy_available','evidence_state']],on=['traceNr','systime'],validate='one_to_one')
 for group,traces in [('train_trace',[302]),('evaluation',[303,304])]:
  measured=emp[emp.traceNr.isin(traces)]
  masks={'all':np.ones(len(measured),dtype=bool),'any_positive_observation':measured.any_positive,
         'independent_consumer_positive':measured.consumer_positive,'busy_available':measured.busy_available,
         'consumer_positive_and_busy':measured.consumer_positive&measured.busy_available}
  for label,mask in masks.items():
   es=measured[mask];joined=sim.merge(es,left_on=['block','second'],right_on=['block','pilot_second'],validate='many_to_one',suffixes=('','_source'))
   for metric,sc,ec in [('data_rssi','monitor_data_rssi_dbm','data_rssi_dbm'),('beacon_rssi','monitor_beacon_rssi_dbm','beacon_rssi_dbm'),
      ('ap_total_busy','ap_busy_all','busy_fraction'),('ap_rx_cca_busy','ap_busy_sensed','busy_fraction')]:
    rows.append(dict(model=tag,group=group,stratum=label,metric=metric,source_seconds=len(es),source_valid=int(es[ec].notna().sum()),
      simulated_seconds=len(joined),simulated_valid=int(joined[sc].notna().sum()),w1=w1(joined[sc],es[ec]),
      source_monitor_empty_fraction=float(es.no_data_or_beacon.mean()) if len(es) else np.nan,
      simulated_monitor_empty_fraction=float(joined.monitor_no_data_or_beacon.mean()) if len(joined) else np.nan))
pd.DataFrame(rows).to_csv(out/'fixed_replay_evidence_strata.csv',index=False)

# Read all seven public WiChoose logs. Format and byte conversion are traced to source code.
base=R/'sources/wichoose/exp-logs/lustosa5'
files={k:next((base/role).glob(prefix+'-*.csv')) for k,role,prefix in [
 ('tx','tx','wcsender'),('rx','rx','wcreceiver'),('rssi','tx','wcrssiprinter'),
 ('gps_tx','tx','wcgpsprinter'),('gps_rx','rx','wcgpsprinter'),('estimate','tx','wichoicemaker'),('choice','tx','wichoiceprinter')]}
frames={k:pd.read_csv(p,skipinitialspace=True) for k,p in files.items()}
inventory=[]
for name,v in frames.items():
 key=['gpstime']+(['iface'] if 'iface' in v else [])+(['tag'] if 'tag' in v else [])
 inventory.append(dict(log=name,rows=len(v),columns=';'.join(v.columns),first_time=int(v.gpstime.min()),last_time=int(v.gpstime.max()),
   duplicate_key_rows=int(v.duplicated(key,keep=False).sum()),blank_cells=int(v.isna().sum().sum()),key=';'.join(key)))
pd.DataFrame(inventory).to_csv(out/'wichoose_log_inventory.csv',index=False)
for name in ['tx','rx','rssi','gps_tx','gps_rx']:assert not frames[name].duplicated(['gpstime']+(['iface'] if 'iface' in frames[name] else [])).any(),name
tx,rx,rs=frames['tx'],frames['rx'],frames['rssi']
pair=tx.merge(rx,on=['gpstime','iface'],how='outer',indicator='pair_presence',validate='one_to_one')
pair=pair.merge(rs,on=['gpstime','iface'],how='left',indicator='rssi_presence',validate='one_to_one')
gpscols=['gpstime','lat','lon','speed','head','fix','nsats','hdop']
pair=pair.merge(frames['gps_tx'][gpscols],on='gpstime',how='left',validate='many_to_one')
pair['rx_mbps']=pair.nbytesDataRx*8/1e6;pair['socket_tx_mbps']=pair.nbytesDataTx*8/1e6
# Printed previous-GPS-second application counters, not per-packet delivery ratios.
pair['rx_exceeds_tx']=pair.nbytesDataRx.gt(pair.nbytesDataTx)
pair['tx_positive_rx_zero']=pair.nbytesDataTx.gt(0)&pair.nbytesDataRx.eq(0)
pair['rssi_sentinel']=pair.rssi.isin([-100,-256,0])|pair.rssi.isna()
pair.to_csv(out/'wichoose_joined_seconds.csv.gz',index=False,compression={'method':'gzip','mtime':0})
profiles=[]
for iface,v in pair.groupby('iface'):
 matched=v[v.pair_presence.eq('both')]
 core=matched[matched.rssi.notna()&matched['lat'].notna()]
 profiles.append(dict(iface=iface,paired_seconds=len(matched),tx_only=int(v.pair_presence.eq('left_only').sum()),
  rx_only=int(v.pair_presence.eq('right_only').sum()),joint_tx_rx_rssi_gps_seconds=len(core),
  rssi_valid_joint_seconds=int((~core.rssi_sentinel).sum()),
  tx_positive_seconds=int(matched.nbytesDataTx.gt(0).sum()),rx_positive_seconds=int(matched.nbytesDataRx.gt(0).sum()),
  tx_positive_rx_zero_seconds=int(matched.tx_positive_rx_zero.sum()),rx_exceeds_tx_seconds=int(matched.rx_exceeds_tx.sum()),
  rx_mean_mbps=float(matched.rx_mbps.mean()),socket_tx_mean_mbps=float(matched.socket_tx_mbps.mean()),
  tx_non_multiple_1440=int(matched.nbytesDataTx.mod(1440).ne(0).sum()),rx_non_multiple_1440=int(matched.nbytesDataRx.mod(1440).ne(0).sum()),
  first_joint_time=int(core.gpstime.min()),last_joint_time=int(core.gpstime.max())))
pd.DataFrame(profiles).to_csv(out/'wichoose_joint_profile.csv',index=False)
# Fixed source-defined mobility blocks; do not retain just the author's notebook's initial period.
segments=[('traffic_light',1721426790,1721428502),('slow',1721428503,1721428918),('medium',1721428925,1721429236)]
segments_rows=[]
for label,a,b in segments:
 for iface,v in pair[pair.gpstime.between(a,b)&pair.pair_presence.eq('both')].groupby('iface'):
  segments_rows.append(dict(segment=label,iface=iface,source_declared_seconds=b-a+1,paired_seconds=len(v),
   rx_mean_mbps=float(v.rx_mbps.mean()),socket_tx_mean_mbps=float(v.socket_tx_mbps.mean()),
   zero_rx_seconds=int(v.nbytesDataRx.eq(0).sum()),rx_exceeds_tx_seconds=int(v.rx_exceeds_tx.sum())))
pd.DataFrame(segments_rows).to_csv(out/'wichoose_mobility_blocks.csv',index=False)

wiperf=pd.read_csv(R/'sources/wiperf_2022_06_14_data.csv',sep=';')
time=pd.to_datetime(wiperf.gpstime,format='%d/%m/%Y %H:%M.%S.%f')
wiq=dict(rows=len(wiperf),columns=list(wiperf.columns),duplicate_time_interface_rows=int(wiperf.duplicated(['gpstime','rat']).sum()),
 blank_cells=int(wiperf.isna().sum().sum()),times=int(time.nunique()),interface_rows={k:int(v) for k,v in wiperf.groupby('rat').size().items()},
 num_bits_equals_1000_times_throughput=bool(np.allclose(wiperf.num_bits,wiperf.throughput*1000,rtol=0,atol=1e-6)),
 per_packet_send_receive_timestamps_available=False,busy_present=False,capture_health_present=False,
 interface_standard_mapping_assumed=False,unit_scope='Recorded num_bits and throughput differ by factor 1000; use explicit num_bits for further unit tracing, no simulator comparison performed')
dump(out/'wiperf_profile.json',wiq)
# Scan only actual code and Markdown (and notebook source, never embedded notebook JS outputs).
terms=['channelUtil','traceNr','wifi-exp-log-summary','survey','busy','pcap','nbytesAcc','RSSI_DISCONNECTED']
sources=[]
for p in (R/'sources/wichoose').rglob('*'):
 if p.is_file() and p.suffix in ['.cpp','.hpp','.md','.conf','.sh']:
  sources.append((p,p.read_text(encoding='utf-8')))
for p in (R/'sources/notebook_source').glob('*.txt'):sources.append((p,p.read_text(encoding='utf-8')))
hits=[]
for p,text in sources:
 for line_no,line in enumerate(text.splitlines(),1):
  for term in terms:
   if term.lower() in line.lower():hits.append(dict(file=str(p.relative_to(R)).replace('\\','/'),line=line_no,term=term,text=line[:500]))
pd.DataFrame(hits).to_csv(out/'source_definition_hits.csv',index=False)
summary=dict(status='CONDITIONAL_OBSERVATION_CONTRACT_AND_NEW_SOURCE_FOUND',vnc_total_seconds=len(d),
 vnc_evidence_states={k:int(v) for k,v in d.evidence_state.value_counts().items()},
 vnc_tail_unresolved_after_positive_consumer=int(d.evidence_state.eq('unresolved_all').sum()),
 wichoose_public_source_files=read_count if (read_count:=len(json.loads((R/'sources/repository_receipt.json').read_text())['files'])) else 0,
 wichoose_joint_profiles=profiles,wiperf=wiq,original_2019_counter_definition_found=False,
 historical_2024_execution_binary_hash_available=False,original_empirical_posterior_validated=False)
dump(out/'assessment.json',summary)
dump(R/'analysis_receipt.json',dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),files={str(p.relative_to(R)).replace('\\','/'):sha(p) for p in sorted(out.glob('*'))}))
print(states.to_string(),flush=True);print(pd.DataFrame(profiles).to_string(index=False),flush=True)
print(json.dumps(wiq,ensure_ascii=False),flush=True)
