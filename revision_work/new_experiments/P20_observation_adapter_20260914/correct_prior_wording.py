"""Correct data-frame versus any-frame nonreception in the prior pilot report."""
from pathlib import Path
import datetime, hashlib, json, shutil

R=Path(__file__).resolve().parent
P=R.parents[2]/'output/reduced_observable_feasibility_20260914'
if (P/'correction_reception_states_20260914.md').exists():
    print('Prior wording correction already applied.');raise SystemExit(0)
before={}
for name in ['report.md','artifact_manifest.json','profile_source.py','source_profile.json','build_report.py','sources_receipt.json','feature_mapping.md']:
    before[name]=hashlib.sha256((P/name).read_bytes()).hexdigest()
shutil.copy2(P/'report.md',P/'report_before_reception_correction.md')
shutil.copy2(P/'artifact_manifest.json',P/'artifact_manifest_before_reception_correction.json')
for name in ['report.md','build_report.py']:
    text=(P/name).read_text(encoding='utf-8')
    text=text.replace('這些秒的接收位元組均為 0。這組 8,317 秒是有可用接收觀測的先導分層；不能把無接收秒刪掉後仍稱為涵蓋斷線的完整部署分布。正式設計須保留缺測／無接收機制。',
      '這些秒的資料幀接收位元組均為 0，但其中 178 秒仍有信標。這組 8,317 秒是有可用資料幀 RSSI 的先導分層；不能將「無資料幀」當成「完全無接收」或直接稱為斷線。正式設計須分開保留 data／beacon 缺測狀態。後續核對見 correction_reception_states_20260914.md。')
    text=text.replace('實測 RSSI 來自接收成功的封包，DES 則固定時刻取樣。','實測 rssiMean 與資料幀接收相聯；信標另有欄位。DES 則固定時刻取樣。')
    text=text.replace('另處理無接收的狀態','另處理無資料幀、無信標及两者皆無的狀態')
    text=text.replace('候選分層排除無接收觀測的秒，不能代表包含斷線的完整部署分布。','候選分層排除無資料幀 RSSI 的秒；其中部分仍有信標，不能統稱斷線。')
    (P/name).write_text(text,encoding='utf-8')
name='profile_source.py';text=(P/name).read_text(encoding='utf-8')
text=text.replace('received-frame RSSI requires a received frame; no-frame seconds are not negative signal measurements; removing them conditions this pilot on reception and is not a full deployment output distribution',
 'the rssiMean/bytes relationship concerns data frames; some no-data seconds still have beacons; the pilot conditions on available data-frame RSSI, not all-frame reception')
(P/name).write_text(text,encoding='utf-8')
profile=json.loads((P/'source_profile.json').read_text(encoding='utf-8'))
profile['rssi_boundary_check']['interpretation']='The 363 no-data RSSI seconds include 178 seconds with received beacons; data nonreception is not equivalent to all-frame nonreception or outage.'
profile['rssi_boundary_check']['no_data_seconds_with_beacons']=178
(P/'source_profile.json').write_text(json.dumps(profile,ensure_ascii=False,indent=2),encoding='utf-8')
source=json.loads((P/'sources_receipt.json').read_text(encoding='utf-8'))
for item in source['items']:
    for q in item['queries']:
        caveats=q['source'].get('caveats',[])
        q['source']['caveats']=[c.replace('候選分層排除無接收觀測的秒，不能代表包含斷線的完整部署分布。','候選分層排除無資料幀 RSSI 的秒；其中部分仍有信標，不能統稱斷線。') for c in caveats]
(P/'sources_receipt.json').write_text(json.dumps(source,ensure_ascii=False,indent=2),encoding='utf-8')
mapping=(P/'feature_mapping.md').read_text(encoding='utf-8')
mapping=mapping.replace('接收選擇偏差未解除；須在模擬端採相同接收條件，另處理無接收秒','接收選擇偏差未解除；須區分資料幀與信標，另處理無資料幀、無信標及兩者均無的秒')
(P/'feature_mapping.md').write_text(mapping,encoding='utf-8')
note='''# 2026-09-14 觀測狀態措辭修正

後續 P20 核對了原始 rssiMean、nBytesReceived、meanBeaconRssi、nBeacons 的同期關係：363 個無資料幀 RSSI 的秒中，178 個仍有信標。因此前輪「無接收」應精確區分為「無資料幀」，不能等同所有幀均未收到或斷線。

已修正報告、來源說明、顯示用來源紀錄與再生腳本的相關措辭。原 8,317 秒篩選、全部模型／數值結果、原始公開 CSV 均未改。原報告與原清單保留為 report_before_reception_correction.md 和 artifact_manifest_before_reception_correction.json。

最新三種狀態分層與實際觀測適配結果見 ../../revision_work/new_experiments/P20_observation_adapter_20260914/inputs/missingness_audit.csv。
'''
(P/'correction_reception_states_20260914.md').write_text(note,encoding='utf-8')
files=[]
for f in sorted(P.rglob('*')):
    if f.is_file() and '__pycache__' not in f.parts and f.name!='artifact_manifest.json':
        files.append(dict(path=f.relative_to(P).as_posix(),bytes=f.stat().st_size,sha256=hashlib.sha256(f.read_bytes()).hexdigest()))
(P/'artifact_manifest.json').write_text(json.dumps(dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
   amendment='Data versus any-frame nonreception wording; scientific tables unchanged',files=files),ensure_ascii=False,indent=2),encoding='utf-8')
(R/'prior_wording_correction.json').write_text(json.dumps(dict(before_sha256=before,
 after_sha256={n:hashlib.sha256((P/n).read_bytes()).hexdigest() for n in before},numeric_tables_changed=False),indent=2),encoding='utf-8')
print('Prior wording corrected; source data and numerical tables retained.')
