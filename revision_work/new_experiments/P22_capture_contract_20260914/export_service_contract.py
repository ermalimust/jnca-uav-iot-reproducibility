from pathlib import Path
import hashlib, json
import pandas as pd
R=Path(__file__).resolve().parent;O=R/'results';D=R/'service_input';D.mkdir(exist_ok=True)
d=pd.read_csv(O/'wichoose_joined_seconds_validated.csv.gz')
complete=d[d.pair_presence.eq('both')].copy()
# Mapping is source-backed by the experiment notebook; do not transfer it to WiPerf.
complete['radio']=complete.iface.map({'wlan1':'802.11n','wlan2':'802.11ad'})
assert complete.radio.notna().all()
complete['mobility_period']='outside_author_described_blocks'
for name,start,end in [('traffic_light',1721426790,1721428502),('slow',1721428503,1721428918),('medium',1721428925,1721429236)]:
 complete.loc[complete.gpstime.between(start,end),'mobility_period']=name
complete['source']='WiChoose_lustosa5_20240719'
complete['rssi_dbm']=complete.rssi.where(complete.rssi_positive_evidence)
cols=['source','gpstime','iface','radio','mobility_period','nbytesDataTx','nbytesDataRx','socket_tx_mbps','rx_mbps','rssi_dbm','rssi_positive_evidence','lat','lon','speed','head','fix','nsats','hdop']
complete[cols].rename(columns={'gpstime':'second_unix','nbytesDataTx':'socket_accepted_bytes','nbytesDataRx':'received_payload_bytes',
 'socket_tx_mbps':'socket_accepted_mbps','rx_mbps':'received_mbps','speed':'mobile_speed_kmh','head':'mobile_heading_deg'}).to_csv(D/'paired_service_observations.csv',index=False)
receivers=d[d.nbytesDataRx.notna()][['gpstime','iface','nbytesDataRx','pair_presence']]
receivers.to_csv(D/'all_receiver_rows.csv',index=False)
contract={
 'status':'READY_FOR_BOUNDED_RECEIVED_THROUGHPUT_COMPARISON',
 'source':'https://gitlab.com/rui-meireles/wichoose/-/tree/241e2d146aa182cd1f9f91fe7b65414f0298dbb7/exp-logs/lustosa5',
 'commit':'241e2d146aa182cd1f9f91fe7b65414f0298dbb7','seconds':2594,'radio_second_rows':5188,
 'first_second':1721426790,'last_second':1721429383,
 'keys':['source','second_unix','iface'],
 'service_estimator':'received_mbps = received_payload_bytes * 8 / 1e6 over printed previous-GPS-second intervals',
 'socket_counter_estimator':'socket_accepted_mbps = socket_accepted_bytes * 8 / 1e6; socket acceptance is not an on-air transmission count',
 'receiver_scope':'UDP payload bytes delivered to receiving application; no per-packet deadline classes',
 'rssi_scope':'Sender-side /proc/net/wireless value at GPS update; -999 is unavailable; not the VNC20 monitor data-frame mean',
 'rssi_valid_rows':{'wlan1':2594,'wlan2':536},
 'geometry_scope':'Moving sender GPS and heading; receiver coordinates documented as fixed and its GPS log is configured/fixed, not independently measured GNSS',
 'speed_units':'km/h per GPS source and mobility thresholds; convert explicitly if simulator uses m/s',
 'counter_resolution':'Printed previous GPS second; no packet IDs; counter windows and runtime implementation can affect exact ratios',
 'source_code_matches_log_format':True,'historic_executed_binary_hash_provided':False,
 'safe_uses':['Measured received-throughput distribution with all zero-reception rows retained',
              'Known socket-side workload profile and mobility-conditioned service comparison',
              'Source-defined period comparisons with all periods and unclassified tails preserved'],
 'unsupported_uses':['Exact per-second packet-loss rate from 1-rx/tx',
                     'Per-packet C2 deadline miss rate or fixed original 25-feature diagnostic validation',
                     'Treating RSSI sentinel as a measured signal value',
                     'Ground-truth causal action effect from unexecuted alternatives',
                     'Treating generated nbytesEst as measured received data'],
 'missing_for_original_diagnostic':['Hardware busy counter','MAC retry denominator/queue records','C2 deadline/arrival cohort','BLE activity','Separately identified video service'],
 'scope_notes':['One location and one run; not independent missions','No synthetic zero imputation for absent log rows',
                'No VNC20 and WiChoose rows spliced into one observation vector'],
 'source_headers':{'rssi_sentinel':'src/wichoice/chaninfo/ChanInfo.hpp','payload':'src/dtransfer/wcsender/DataSender.hpp',
                   'intervals':'src/dtransfer/DataTransfer.cpp','receiver':'src/dtransfer/wcreceiver/DataReceiver.cpp'},
 'files':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in D.glob('*.csv')}
}
(D/'observation_contract.json').write_text(json.dumps(contract,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(dict(rows=len(complete),times=int(complete.gpstime.nunique()),output=str(D)),ensure_ascii=False),flush=True)
