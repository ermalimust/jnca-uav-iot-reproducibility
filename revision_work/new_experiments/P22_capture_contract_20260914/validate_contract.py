"""Independent source checks and corrected source-specific sentinel interpretation."""
from pathlib import Path
import csv, gzip, hashlib, json, math, re
from collections import Counter
import numpy as np
import pandas as pd
R=Path(__file__).resolve().parent;W=R.parents[2];O=R/'results'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,obj):Path(p).write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
assert all(sha(p)==h for p,h in read(R/'execution_binding.json')['files'].items())
assert all(sha(R/p)==h for p,h in read(R/'analysis_receipt.json')['files'].items())
repository=R/'sources/wichoose'
receipt=read(R/'sources/repository_receipt.json')
for entry in receipt['files']:
 assert entry['status']=='retrieved' and sha(repository/entry['path'])==entry['sha256']
logs=read(R/'sources/logs_receipt.json')
for entry in logs['files']:assert sha(repository/entry['path'])==entry['sha256']
# Do not transfer VNC20 sentinels to this distinct source.
header=(repository/'src/wichoice/chaninfo/ChanInfo.hpp').read_text()
sentinel=int(re.search(r'#define\s+RSSI_DISCONNECTED\s+(-?\d+)',header).group(1));assert sentinel==-999
payload=int(re.search(r'#define\s+SND_BUF_LEN\s+(\d+)',(repository/'src/dtransfer/wcsender/DataSender.hpp').read_text()).group(1));assert payload==1440
joined=pd.read_csv(O/'wichoose_joined_seconds.csv.gz')
joined['rssi_sentinel']=joined.rssi.eq(sentinel)|joined.rssi.isna()
joined['rssi_positive_evidence']=joined.rssi.notna()&joined.rssi.ne(sentinel)
joined.to_csv(O/'wichoose_joined_seconds_validated.csv.gz',index=False,compression={'method':'gzip','mtime':0})
profile=pd.read_csv(O/'wichoose_joint_profile.csv')
extra=[]
for iface,v in joined.groupby('iface'):
 core=v[v.pair_presence.eq('both')&v.rssi.notna()&v.lat.notna()]
 actual=int(core.rssi_positive_evidence.sum())
 profile.loc[profile.iface.eq(iface),'rssi_valid_joint_seconds']=actual
 extra.append(dict(iface=iface,core_rows=len(core),valid_rssi_rows=actual,rssi_unavailable=int(core.rssi_sentinel.sum()),
  positive_rx_without_rssi=int((core.nbytesDataRx.gt(0)&core.rssi_sentinel).sum()),
  zero_rx_with_rssi=int((core.nbytesDataRx.eq(0)&core.rssi_positive_evidence).sum())))
profile.to_csv(O/'wichoose_joint_profile_validated.csv',index=False)
pd.DataFrame(extra).to_csv(O/'wichoose_rssi_availability.csv',index=False)
note=dict(kind='Source-specific sentinel correction',original_script='analyze.py',new_source_sentinel=sentinel,
 original_result_files_retained=['wichoose_joint_profile.csv','wichoose_joined_seconds.csv.gz','assessment.json'],
 corrected_files=['wichoose_joint_profile_validated.csv','wichoose_joined_seconds_validated.csv.gz','assessment_validated.json'],
 reason='Initial exploratory profile used VNC20-like sentinels; source header specifies -999 for WiChoose. Corrected before report delivery; original bytes and all VNC/P21 outcomes unchanged.')
dump(R/'source_correction.json',note)

# Independent raw CSV dictionaries reconstruct evidence counts and all Tx/Rx sums.
with (R/'inputs/vnc20_wifi.csv').open(encoding='utf-8',newline='') as f:raw=list(csv.DictReader(f))
counts=Counter();keys=set()
for r in raw:
 if r['wifiType']!='n' or int(r['traceNr']) not in [302,303,304]:continue
 key=(int(r['traceNr']),float(r['systime']));assert key not in keys;keys.add(key)
 state=('monitor_data' if float(r['nBytesReceived'])>0 else 'beacon_only' if float(r['nBeacons'])>0 else
        'consumer_only' if float(r['tghptConsumer'])>0 else 'busy_only' if 0<=float(r['channelUtil'])<=100 else 'unresolved_all')
 counts[(key[0],state)]+=1
states=pd.read_csv(O/'vnc_evidence_states.csv')
for row in states.itertuples(index=False):
 for state in ['monitor_data','beacon_only','consumer_only','busy_only','unresolved_all']:assert counts[(row.traceNr,state)]==getattr(row,state)
base=repository/'exp-logs/lustosa5';raw_dict={}
for tag,role,prefix,column in [('tx','tx','wcsender','nbytesDataTx'),('rx','rx','wcreceiver','nbytesDataRx')]:
 with next((base/role).glob(prefix+'-*.csv')).open(encoding='utf-8',newline='') as f:rows=list(csv.DictReader(f,skipinitialspace=True))
 raw_dict[tag]={(int(r['gpstime']),r['iface']):int(r[column]) for r in rows};assert len(raw_dict[tag])==len(rows)
common=set(raw_dict['tx'])&set(raw_dict['rx'])
assert len(common)==5188
for row in profile.itertuples(index=False):
 selected=[k for k in common if k[1]==row.iface]
 assert len(selected)==row.paired_seconds
 assert sum(raw_dict['rx'][k]>raw_dict['tx'][k] for k in selected)==row.rx_exceeds_tx_seconds
 assert all(raw_dict[tag][k]%payload==0 for tag in ['rx','tx'] for k in selected)
 assert abs(math.fsum(raw_dict['rx'][k] for k in selected)/len(selected)*8/1e6-row.rx_mean_mbps)<1e-10

# Explicit after-results window sensitivity; do not call byte deficit a packet-loss rate.
block_rows=[]
for iface,v in joined[joined.pair_presence.eq('both')].groupby('iface'):
 start=int(v.gpstime.min())
 for window in [1,5,10,30,60]:
  x=v.copy();x['block']=((x.gpstime-start)//window).astype(int)
  for block,z in x.groupby('block'):
   tx=int(z.nbytesDataTx.sum());rx=int(z.nbytesDataRx.sum())
   block_rows.append(dict(iface=iface,window_seconds=window,block=int(block),observed_seconds=len(z),
    tx_bytes=tx,rx_bytes=rx,rx_exceeds_tx=rx>tx,rx_minus_tx_bytes=rx-tx,
    scope='After-results counter-window sensitivity; not matched-packet loss'))
pd.DataFrame(block_rows).to_csv(O/'counter_window_sensitivity.csv',index=False)
bs=pd.DataFrame(block_rows).groupby(['iface','window_seconds']).agg(blocks=('block','size'),rx_exceeds_tx_blocks=('rx_exceeds_tx','sum'),tx_bytes=('tx_bytes','sum'),rx_bytes=('rx_bytes','sum')).reset_index()
bs.to_csv(O/'counter_window_summary.csv',index=False)

# Independently evaluate exact quantile distance for every stored strata metric.
def distance(a,b):
 a=np.sort(np.asarray(a));a=a[np.isfinite(a)];b=np.sort(np.asarray(b));b=b[np.isfinite(b)]
 if not len(a) or not len(b):return np.nan
 p=np.unique(np.r_[np.arange(len(a)+1)/len(a),np.arange(len(b)+1)/len(b)]);q=(p[:-1]+p[1:])/2
 return float(np.sum(np.diff(p)*abs(a[np.minimum((q*len(a)).astype(int),len(a)-1)]-b[np.minimum((q*len(b)).astype(int),len(b)-1)])))
evidence=pd.read_csv(O/'vnc_evidence_seconds.csv.gz')
comp=pd.read_csv(O/'fixed_replay_evidence_strata.csv');errors=[]
mapping={'data_rssi':('monitor_data_rssi_dbm','data_rssi_dbm'),'beacon_rssi':('monitor_beacon_rssi_dbm','beacon_rssi_dbm'),
 'ap_total_busy':('ap_busy_all','busy_fraction'),'ap_rx_cca_busy':('ap_busy_sensed','busy_fraction')}
for tag,base in [('P20',R.parent/'P20_observation_adapter_20260914'),('P21',R.parent/'P21_spatial_capture_20260914/replay')]:
 sim=pd.read_csv(base/'results/simulated_observations.csv');sim=sim[sim.shadow.eq(True)]
 emp=pd.read_csv(base/'results/measured_observations.csv').merge(evidence[['traceNr','systime','any_positive','consumer_positive','busy_available']],on=['traceNr','systime'],validate='one_to_one')
 for row in comp[comp.model.eq(tag)].itertuples(index=False):
  e=emp[emp.traceNr.isin([302] if row.group=='train_trace' else [303,304])]
  if row.stratum=='any_positive_observation':e=e[e.any_positive]
  elif row.stratum=='independent_consumer_positive':e=e[e.consumer_positive]
  elif row.stratum=='busy_available':e=e[e.busy_available]
  elif row.stratum=='consumer_positive_and_busy':e=e[e.consumer_positive&e.busy_available]
  s=sim.merge(e[['block','pilot_second']],left_on=['block','second'],right_on=['block','pilot_second'],validate='many_to_one')
  a,b=mapping[row.metric];actual=distance(s[a],e[b]);assert len(e)==row.source_seconds and len(s)==row.simulated_seconds
  if np.isfinite(actual):errors.append(abs(actual-row.w1))
  else:assert pd.isna(row.w1)
assert max(errors)<1e-10
canonical=read(W/'output/ns3_policy_validation_20260914/editorial_audit.json')['final_document_sha256']
assert all(sha(W/p)==h for p,h in canonical.items())
assessment=read(O/'assessment.json');assessment['wichoose_joint_profiles']=profile.to_dict(orient='records')
assessment.update(source_specific_rssi_sentinel=sentinel,
 wichoose_status='READY_FOR_RECEIVER_THROUGHPUT_FEASIBILITY_NOT_PACKET_LOSS',
 exact_same_second_delivery_ratio_validated=False,all_hardware_counters_present=False,
 historical_source_binary_identity_established=False,source_correction='source_correction.json')
dump(O/'assessment_validated.json',assessment)
verification=dict(status='PASS',vnc_unique_seconds=len(keys),wichoose_tx_rx_pairs=len(common),public_source_files_verified=len(receipt['files']),
 public_logs_verified=len(logs['files']),source_sentinel_verified=sentinel,source_payload_bytes=payload,
 fixed_strata_metric_cells=len(comp),fixed_strata_w1_max_error=max(errors),canonical_p19_documents_unchanged=True,
 source_integrity_verified=True,third_party_code_executed=False,original_results_and_corrected_results_retained=True,
 note='Independent arithmetic within current task; code inspection is not proof of original runtime binary')
dump(R/'verification.json',verification)
print(profile.to_string(index=False),flush=True);print(pd.DataFrame(extra).to_string(index=False),flush=True);print(bs.to_string(index=False),flush=True)
print(json.dumps(verification),flush=True)
