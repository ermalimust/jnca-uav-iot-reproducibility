"""Independent P14 audit; writes only execution_review.md/json beside this file.
Default: numerical and local manuscript checks. --numerical-only: public science
supplement without manuscript/reviewer-response files. --text-only: refresh local
writing hashes while retaining the earlier numerical receipt.
"""
import sys
sys.dont_write_bytecode=True
import csv, hashlib, importlib.util, json, struct, traceback, zipfile
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
try:
    from scipy.stats import wasserstein_distance as scipy_w1
    import scipy
except ImportError:
    scipy_w1=None
OUT=Path(__file__).resolve().parent
WORK=OUT.parents[1]
P14=WORK/'revision_work/new_experiments/P14_public_arrival_validation_20260913'
RESULT=P14/'results'
P1=P14.parent/'P1_des_recovery_20260912'
REPLAY=WORK/'revision_work/analysis/replay_inputs'
GEN=P1/'recovered/paper3_generator_core/paper3_des/src/generate_scenarios.py'
CONFIG=GEN.parent.parent/'configs/mvp_scenarios_min.json'
REPORT={'audit_date':datetime.now(timezone.utc).isoformat(),'status':'RUNNING','checks':{},
 'write_scope':['independent_execution_check.py','execution_review.md','execution_review.json'],
 'scope':'Execution consistency only; no deployment robustness, calibration or correctness claim.',
 'w1_reference':'Independent quantile-transport integral; SciPy comparison when installed',
 'scipy_available':scipy_w1 is not None}
if scipy_w1 is not None:REPORT['scipy_version']=scipy.__version__
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def find_archive(expected_sha):
    """Accept either supported layout, but only the exact pinned public bytes."""
    candidates=[WORK/'output/dataset_search_20260913/aviator_uav_datatraces.zip',P14/'cache/uav_datatraces.zip']
    for path in candidates:
        if path.is_file() and sha(path)==expected_sha:return path
    raise FileNotFoundError('No matching pinned AVIATOR archive at either supported location: '+', '.join(str(p) for p in candidates))
def set_execution_mode(numerical_only):
    REPORT['execution_mode']='numerical-only' if numerical_only else 'full-local'
    REPORT['numerical_checks_rerun']=True
    if numerical_only:
        REPORT['checks'].pop('manuscript_and_response_claim_alignment',None)
        REPORT['writing_review']={'status':'NOT_CHECKED','reason':'--numerical-only omits manuscript and reviewer-response review; no earlier manual writing claims are retained.'}
def table(name):
    with (RESULT/name).open(encoding='utf-8',newline='') as f:return list(csv.DictReader(f))
def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m
def exact(a,b,where):
    a=np.asarray(a);b=np.asarray(b)
    assert a.shape==b.shape and np.array_equal(a,b),where
def near(a,b,where,atol=2e-11):
    a=np.asarray(a,dtype=float);b=np.asarray(b,dtype=float)
    assert a.shape==b.shape and np.isfinite(a).all() and np.isfinite(b).all(),where
    err=float(np.max(np.abs(a-b))) if a.size else 0.
    assert err<=atol,(where,err)
    return err
def stage(name,**data):
    REPORT['checks'][name]={'status':'PASS',**data};print('PASS',name,flush=True)
def quantile_w1(x,y,wx=None,wy=None):
    """Inverse-CDF integral; runner instead integrates CDFs over data support."""
    x=np.asarray(x,float);y=np.asarray(y,float)
    wx=np.ones(x.size) if wx is None else np.asarray(wx,float)
    wy=np.ones(y.size) if wy is None else np.asarray(wy,float)
    assert len(x) and len(y) and wx.sum()>0 and wy.sum()>0
    ix=np.argsort(x);iy=np.argsort(y)
    x=x[ix];y=y[iy];cx=np.cumsum(wx[ix]/wx.sum());cy=np.cumsum(wy[iy]/wy.sum())
    cx[-1]=cy[-1]=1.
    cuts=np.unique(np.clip(np.r_[0.,cx,cy],0.,1.));mids=(cuts[:-1]+cuts[1:])/2
    xi=np.minimum(np.searchsorted(cx,mids,side='left'),len(x)-1)
    yi=np.minimum(np.searchsorted(cy,mids,side='left'),len(y)-1)
    result=float(np.dot(np.diff(cuts),np.abs(x[xi]-y[yi])))
    if scipy_w1 is not None:near(result,scipy_w1(x,y,wx[ix],wy[iy]),'SciPy inverse-CDF',1e-8)
    return result
def independent_counts(t,start,end,width):
    n=(int(end)-int(start))//int(width)
    edges=int(start)+np.arange(n+1,dtype=np.int64)*int(width)
    return np.diff(np.searchsorted(t,edges,side='left'))
def parse_source():
    audit=read(OUT/'source_semantics_audit.json')
    archive=find_archive(audit['source_binding']['zip_sha256'])
    REPORT['archive_location']=str(archive.relative_to(WORK))
    saved=np.load(RESULT/'selected_emissions.npz');recs=[]
    with zipfile.ZipFile(archive) as z:
        for rid,spec in zip(['mavic_air','spark','parrot_ar2'],audit['main_pcaps']):
            data=z.read(spec['member'])
            assert hashlib.sha256(data).hexdigest()==spec['sha256']
            order='<' if data[:4]==b'\xd4\xc3\xb2\xa1' else '>'
            assert data[:4] in [b'\xd4\xc3\xb2\xa1',b'\xa1\xb2\xc3\xd4']
            assert struct.unpack_from(order+'I',data,20)[0]==1
            selector=spec['control_selector'];off=24;idx=0;chosen=[]
            while off<len(data):
                sec,usec,caplen,orig=struct.unpack_from(order+'IIII',data,off);off+=16
                packet=data[off:off+caplen];off+=caplen
                assert len(packet)==caplen and usec<1000000
                if len(packet)>=34 and packet[12:14]==b'\x08\x00' and packet[14]>>4==4:
                    ihl=(packet[14]&15)*4
                    if packet[23]==17 and ihl>=20 and len(packet)>=14+ihl+8:
                        src='.'.join(str(v) for v in packet[26:30]);dst='.'.join(str(v) for v in packet[30:34])
                        sport,dport,length,_=struct.unpack_from('!HHHH',packet,14+ihl)
                        if (src,dst,sport,dport)==(selector['src_ip'],selector['dst_ip'],selector['src_port'],selector['dst_port']):
                            assert struct.unpack_from('!H',packet,20)[0]&0x3fff==0
                            assert length>=8 and struct.unpack_from('!H',packet,16)[0]==ihl+length
                            chosen.append((sec*1000000+usec,idx,length-8))
                idx+=1
            assert off==len(data) and idx==spec['records']
            chosen.sort(key=lambda a:(a[0],a[1]))
            ar=np.asarray(chosen,dtype=np.int64);origin=int(ar[0,0]);ar[:,0]-=origin
            assert len(ar)==spec['control_records']
            for col,key in enumerate(['times_us','packet_indices','payload_bytes']):exact(ar[:,col],saved[rid+'__'+key],rid+' parser '+key)
            recs.append({'id':rid,'times':ar[:,0],'indices':ar[:,1],'sizes':ar[:,2],'origin':origin})
    stage('independent_binary_source_parser',records=[len(r['times']) for r in recs],all_timestamps_packet_ids_lengths_exact=True)
    return recs
def write_report():
    (OUT/'execution_review.json').write_text(json.dumps(REPORT,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    lines=['# P14 獨立執行核驗','','**狀態：'+REPORT['status']+'**。只核驗固定方案、程式和輸出一致性，不評判實際部署的診斷正確性或穩健性。','',
     '只讀P14輸入和結果；來源另以獨立PCAP解析重讀，W1以inverse-CDF quantile transport積分核對主程式CDF積分。SciPy可用：'+str(REPORT['scipy_available'])+'。','',
     '| 檢查 | 結果 |','|---|---|']
    lines+=['| '+k+' | '+v['status']+' |' for k,v in REPORT['checks'].items()]
    if REPORT.get('failure'):lines+=['','失敗：'+REPORT['failure']]
    if REPORT.get('important_limitations'):lines+=['','解讀界線：']+['- '+s for s in REPORT['important_limitations']]
    if REPORT.get('writing_review',{}).get('status')=='PASS':
        lines+=['','文字核驗：'+REPORT['writing_review']['status']+'。A18、5.21、A12、Discussion、Conclusion及新增response均保留量測輸入與模擬服務的邊界；沒有把q距離或動作變化解讀成部署診斷正確性。R4 M1原本把58,687個來源封包寫得像全部進入模擬，已修正為十個完整blocks的52,402個封包。兩張A18表的全部數字按已核驗CSV逐格核對。']
    elif REPORT.get('execution_mode')=='numerical-only':
        lines+=['','本模式未檢查正文或審稿回覆，且未沿用既有人工文字審核宣稱；此報告只包含公共科學輸入、算法與數值輸出的核驗。']
    if REPORT.get('execution_mode')=='text-only-refresh':
        lines+=['','本次只更新本地文字核驗與文件hash；數值檢查沿用先前通過的完整獨立核驗，沒有重新執行數值算法。']
    if REPORT.get('numerical_summary'):lines+=['','主要復算數值：','']+['    '+l for l in json.dumps(REPORT['numerical_summary'],ensure_ascii=False,indent=2).splitlines()]
    lines+=['','各項數量、誤差、來源hash與精確範圍見 execution_review.json。']
    (OUT/'execution_review.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
def review_writing():
    import re
    paths=[WORK/'source/sections'/f for f in ['A18_public_arrival_validation.tex','05b_completion_evaluation.tex','A12_external_observables.tex','06_discussion.tex','07_conclusion.tex']]
    paths.append(WORK/'response_to_reviewers/response_to_reviewers.tex')
    appendix=paths[0].read_text(encoding='utf-8');response=paths[-1].read_text(encoding='utf-8')
    rows=[]
    for line in appendix.splitlines():
        if re.match(r'^(Mavic Air|Spark|Parrot AR 2\.0|Balanced pool)\s*&',line):
            rows.append([v.strip().replace('\\\\','').replace('\\%','').replace(',','') for v in line.split('&')])
    assert len(rows)==7
    desc=table('arrival_descriptives.csv');dist=table('arrival_distances.csv');qd=table('q_distances_pooled.csv');dc=table('decision_changes_pooled.csv')
    for i,(rid,count) in enumerate([('mavic_air',25790),('spark',18082),('parrot_ar2',14815)]):
        iat=next(r for r in desc if r['recording_or_scenario']==rid and r['metric']=='interarrival_ms')
        counts=next(r for r in desc if r['recording_or_scenario']==rid and r['metric']=='count_per_100ms_phase0')
        iw=next(r for r in dist if r['measured_population']==rid and r['metric']=='interarrival_ms')
        cw=next(r for r in dist if r['measured_population']==rid and r['metric']=='count_per_100ms_phase0')
        expected=[count]+[round(float(v),4) for v in [iat['mean'],iat['p95'],counts['mean'],iw['W1'],cw['W1']]]
        exact([float(v) for v in rows[i][1:]],expected,'A18 measured table '+rid)
    for i,pop in enumerate(['mavic_air','spark','parrot_ar2','equal_recordings_blocks_scenarios']):
        qr=next(r for r in qd if r['population']==pop)
        dr=next(r for r in dc if r['recording_id']==('equal_recordings_equal_missions' if i==3 else pop))
        expected=[round(float(qr[k]),4) for k in ['W1_W','W1_B','W1_M','W1_V','sliced_W1_4D_64']]+[round(100*float(dr['action_change']),2)]
        exact([float(v) for v in rows[3+i][1:]],expected,'A18 propagation table '+pop)
    runs=read(RESULT/'run_index.json');used=sum(r['source_datagrams'] for r in runs if r['scenario_index']==0)
    assert used==52402 and '52,402 datagrams' in response
    assert 'We further substitute 58,687 public AVIATOR' not in response
    assert 'do not establish equality with an operational posterior distribution' in response
    assert 'computes no empirical calibration score from the transport captures' in response
    REPORT['writing_review']={'status':'PASS','files_sha256':{str(p.relative_to(WORK)):sha(p) for p in paths},
      'manual_scope':['A18 source/weighting/metrics and claim boundaries','5.21 measured input versus simulated output distinction','A12 cause labels only required for accuracy/calibration, not D(X) distribution','R1 R2 directly reports finite evidence and does not claim empirical posterior equality','R4 M3 separates logistic marginal model, temporal categorical belief and constructed finite catalogue','Six newly modified responses and discussion/conclusion boundary'],
      'A18_table_numeric_cells_verified':42,
      'corrected_finding':{'location':'Response R4 M1','initial_issue':'58,687-emission corpus wording could imply all emissions entered DES despite discarded partial tails.','resolution':'Now says every complete90-s block extracted from the corpus and explicitly52,402 datagrams.','status':'RESOLVED'},
      'full_corpus_emissions':58687,'complete_block_input_emissions':used,'partial_tail_emissions_excluded':58687-used,
      'reviewer_request_boundary':'R1 R2 receives direct input and conditional propagation evidence; full operational-posterior distribution matching remains unestablished, as explicitly stated.'}
    stage('manuscript_and_response_claim_alignment',A18_numeric_cells=42,corrected_finding_resolved=True)

def main(numerical_only=False):
    set_execution_mode(numerical_only)
    manifest=read(RESULT/'result_manifest.json');binding=read(P14/'input_binding.json')
    assert sha(P14/'input_binding.json')==manifest['input_binding_sha256']
    for rel,h in binding['sha256'].items():
        path=find_archive(h) if rel=='AVIATOR_public_archive' else WORK/rel
        assert sha(path)==h,('input binding',rel)
    for name,h in manifest['files'].items():assert sha(RESULT/name)==h,('result manifest',name)
    REPORT['bindings']={'input_binding_sha256':sha(P14/'input_binding.json'),'result_manifest_sha256':sha(RESULT/'result_manifest.json'),
     'runner_sha256':sha(P14/'run_experiment.py'),'trace_inputs_sha256':sha(P14/'trace_inputs.py'),'protocol_sha256':sha(P14/'protocol.md')}
    stage('frozen_input_and_complete_result_hashes',input_files=len(binding['sha256']),result_files=len(manifest['files']))
    recs=parse_source();index=read(RESULT/'index.json');runs=read(RESULT/'run_index.json')
    native=np.load(RESULT/'native_arrays.npz');trace=np.load(RESULT/'trace_driven_arrays.npz')
    names=read(P1/'feature_columns.json')
    nx,nq,nr,na=[native[k] for k in ['features','q','routes','actions']]
    tx,tq,tr,ta=[trace[k] for k in ['features','q','routes','actions']]
    exact(nx.shape,(8,700,25),'native shape');exact(tx.shape,(80,700,25),'trace shape')
    exact(tq.shape,(80,700,4),'q shape');exact(ta.shape,(80,700,30),'action shape')
    expected=[];blocks={}
    for rec in recs:
        t=rec['times'];n=int(t[-1])//90000000;blocks[rec['id']]=n
        for k in range(n):
            begin=k*90000000;end=begin+90000000
            lo,hi=np.searchsorted(t,[begin,end],side='left');bt=t[lo:hi]-begin
            for s in range(8):expected.append((rec,k,s,begin,end,lo,hi,bt))
    assert list(blocks.values())==[3,3,4] and len(expected)==len(runs)==80
    for i,(rec,k,s,begin,end,lo,hi,bt) in enumerate(expected):
        row=runs[i]
        assert (row['run_index'],row['recording_id'],row['block_index'],row['scenario_index'],row['source_start_us'],row['source_end_us'],row['source_datagrams'])==(i,rec['id'],k,s,begin,end,hi-lo)
        exact(tx[i,:,names.index('c2_arrival_count')],independent_counts(bt,20000000,90000000,100000),'arrival count '+str(i))
        cached=np.load(RESULT/('run_%03d.npz'%i))
        for key,arr in [('features',tx),('q',tq),('routes',tr),('actions',ta)]:exact(cached[key],arr[i],'cached '+str(i)+key)
    stage('all_blocks_and_700_window_arrival_counts',blocks_by_recording=blocks,runs=80,windows=56000,half_open_windows=True)
    lookup={r['scenario_id']:i for i,r in enumerate(index['scenarios'])};n=0
    with (P1/'regenerated/windows/factual_windows.csv').open(encoding='utf-8',newline='') as f:
        for row in csv.DictReader(f):
            if row['scenario_id'] in lookup:
                exact(nx[lookup[row['scenario_id']],int(row['window_id'])],[float(row[k]) for k in names],'native archived feature');n+=1
    assert n==5600
    model=np.load(P1/'diagnostic_model.npz');errors=[]
    for name,x,q in [('native',nx,nq),('trace',tx,tq)]:
        calc=1/(1+np.exp(-np.clip(np.matmul((x-model['mean'])/model['std'],model['weights'])+model['bias'],-40,40)))
        errors.append(near(q,calc,name+' feature to q',2e-14))
        assert np.isfinite(x).all() and np.isfinite(q).all() and ((q>=0)&(q<=1)).all()
    unchanged=['channel_busy_ratio','ble_occupancy_ratio','ble_periodicity_score','short_interruption_count','service_interruption_count','rssi_mean_dbm','rssi_var','distance_m','speed_mps','service_rate_var','telemetry_arrival_count','video_offered_load_mbps','video_burst_ratio']
    for i,row in enumerate(runs):
        for name in unchanged:exact(tx[i,:,names.index(name)],nx[row['scenario_index'],:,names.index(name)],'unchanged '+name)
    stage('all_25d_native_and_all_feature_to_q',native_features_exact=140000,q_values_recomputed=246400,max_q_error=max(errors),unchanged_exogenous_checks=80*700*len(unchanged))
    gen=module('p14_audit_generator',GEN);cfg=read(CONFIG);cfg['global']['duration_s']=90
    scmap={s.scenario_id:s for s in gen.expand_scenarios(cfg,{r['scenario_family'] for r in cfg['scenarios']})}
    scenarios=[scmap[r['scenario_id']] for r in index['scenarios']]
    arrivals=[gen.generate_c2_arrivals(cfg,s,90000.) for s in scenarios]
    samples=np.load(RESULT/'arrival_samples.npz');measured={};des={}
    for rec in recs:
        t=rec['times'];measured[rec['id']]={'interarrival_ms':np.diff(t)/1000.,
         'count_per_100ms_phase0':independent_counts(t,0,t[-1],100000),
         'count_per_100ms_phase50':independent_counts(t,50000,t[-1],100000)}
    for s,arr in zip(scenarios,arrivals):
        t=np.asarray([p['arrival_ms'] for p in arr]);v=t[(t>=20000)&(t<90000)]
        des[s.scenario_id]={'interarrival_ms':np.diff(v),
         'count_per_100ms_phase0':np.diff(np.searchsorted(t,np.arange(20000,90001,100),side='left')),
         'count_per_100ms_phase50':np.diff(np.searchsorted(t,np.arange(20050,89951,100),side='left'))}
    for prefix,groups in [('measured',measured),('native',des)]:
        for rid,vals in groups.items():
            for metric,a in vals.items():exact(a,samples[prefix+'__'+rid+'__'+metric],prefix+rid+metric)
    stage('all_arrival_samples_and_phase50_boundaries',native_phase0_windows=700,native_phase50_windows=699,
     measured_sample_counts={rid:{m:len(a) for m,a in vals.items()} for rid,vals in measured.items()})
    ti=module('p14_audit_trace_utilities',P14/'trace_inputs.py')
    near(quantile_w1([0],[2]),2.,'analytic delta');near(quantile_w1([0,2],[1]),1.,'analytic split')
    near(quantile_w1([0,10],[0,10],[.9,.1],[.1,.9]),8.,'analytic weighted')
    rng=np.random.default_rng(1089);random_err=0.
    for k in range(250):
        a=rng.integers(-4,8,size=int(rng.integers(1,80))).astype(float);b=rng.normal(size=int(rng.integers(1,90)))
        wa=rng.random(len(a));wb=rng.random(len(b))
        random_err=max(random_err,near(ti.weighted_w1(a,b,wa,wb),quantile_w1(a,b,wa,wb),'random inverse-CDF'))
    def pool(groups):return np.concatenate(groups),np.concatenate([np.repeat(1/len(groups)/len(g),len(g)) for g in groups])
    arr_dist={};arrival_w1_error=0.
    REPORT['numerical_tolerances']={'arrival_W1_absolute':1e-8,'q_W1_absolute':2e-11,'feature_to_q_absolute':2e-14,'source_counts_and_native_features':'exact','arrival_tolerance_reason':'CDF cumulative sums and inverse-CDF cumulative sums accumulate floating-point error differently; 1e-8 ms is 1e-5 of the source timestamp resolution.'}
    for row in table('arrival_distances.csv'):
        metric=row['metric']
        a,wa=pool([v[metric] for v in measured.values()]) if row['measured_population']=='equal_recordings' else (measured[row['measured_population']][metric],None)
        b,wb=pool([v[metric] for v in des.values()])
        val=quantile_w1(a,b,wa,wb);arrival_w1_error=max(arrival_w1_error,near(val,float(row['W1']),'arrival W1',1e-8))
        if row['measured_population']=='equal_recordings':arr_dist[metric]=val
    for row in table('arrival_pairwise_distances.csv'):
        arrival_w1_error=max(arrival_w1_error,near(quantile_w1(measured[row['recording_id']][row['metric']],des[row['scenario_id']][row['metric']]),float(row['W1']),'arrival pair',1e-8))
    for row in table('arrival_descriptives.csv'):
        groups=measured if row['group']=='measured' else des;a=groups[row['recording_or_scenario']][row['metric']]
        stats={'n':len(a),'mean':np.mean(a),'median':np.median(a),'p95':np.quantile(a,.95),'min':np.min(a),'max':np.max(a)}
        for k,v in stats.items():near(v,float(row[k]),'descriptive '+k)
    stage('all_arrival_W1_and_descriptives',random_distributions=250,max_random_W1_error=random_err,max_actual_W1_absolute_error=arrival_w1_error,balanced=arr_dist)
    dirs=np.load(RESULT/'sliced_w1_projections.npy')
    rng=np.random.Generator(np.random.PCG64(20260913));expected_dirs=rng.normal(size=(64,4));expected_dirs/=np.sqrt(np.sum(expected_dirs**2,axis=1))[:,None]
    exact(dirs,expected_dirs,'projection seed');near(np.linalg.norm(dirs,axis=1),np.ones(64),'unit norms')
    def qmetric(a,b,wa=None):
        result={f'W1_{k}':quantile_w1(a[:,i],b[:,i],wa) for i,k in enumerate('WBMV')}
        result['sliced_W1_4D_64']=float(np.mean([quantile_w1(a@v,b@v,wa) for v in dirs]))
        return result
    q_by_run=[]
    for i,row in enumerate(table('q_distances_by_run.csv')):
        s=runs[i]['scenario_index'];vals=qmetric(tq[i],nq[s]);q_by_run.append(vals)
        for k,v in vals.items():near(v,float(row[k]),'run q '+str(i)+k)
    qw=np.repeat(np.asarray([1/(3*blocks[r['recording_id']]*8*700) for r in runs]),700);near(qw.sum(),1.,'q mass')
    weights=np.load(RESULT/'pooling_weights.npz');near(qw,weights['q_window_weights'],'q weights',1e-15)
    for rid in blocks:
        mask=np.repeat(np.asarray([r['recording_id']==rid for r in runs]),700);near(qw[mask].sum(),1/3,'record q mass')
    q_pooled=qmetric(tq.reshape(-1,4),nq.reshape(-1,4),qw)
    for row in table('q_distances_pooled.csv'):
        population=row['population']
        if population=='equal_recordings_blocks_scenarios':vals=q_pooled
        elif population=='weighted_mean_of_run_distances_not_pooled':vals={k:sum(v[k]/(3*blocks[runs[i]['recording_id']]*8) for i,v in enumerate(q_by_run)) for k in q_pooled}
        else:
            selected=[i for i,r in enumerate(runs) if r['recording_id']==population]
            vals=qmetric(tq[selected].reshape(-1,4),nq.reshape(-1,4))
        for k,v in vals.items():near(v,float(row[k]),'pooled q '+population+k)
    stage('all_marginal_and_64_projection_W1',per_run=80,pooled_populations=5,pooled=q_pooled,weights_sum=float(qw.sum()))
    sys.path.insert(0,str(REPLAY))
    import paper7_agentic_feasibility as core
    import paper7_llm_candidate_experiment as llm
    mission_map={m.mission_id:m for m in llm.load_missions(REPLAY/'mission_intents.jsonl')}
    missions=[mission_map[m] for m in index['mission_ids']]
    policies=llm.load_replay(REPLAY/'llm_runs/qwen_qwen-plus/policies.jsonl')
    action_names=list(core.SUPPORTED_ACTIONS)+[core.ESCALATION_ACTION];assert action_names==index['actions']
    prepared=[(llm.mission_to_spec(m),policies[m.mission_id]) for m in missions]
    costs=[core.cost_matrix(s) for s,p in prepared];count=0
    for group,q,r,a in [('native',nq,nr,na),('trace',tq,tr,ta)]:
        for run in range(len(q)):
            for w,row in enumerate(q[run]):
                assert llm.archetype_for(row)==index['routes'][int(r[run,w])]
                for j,(spec,policy) in enumerate(prepared):
                    candidates=llm.candidate_actions_for(policy,row)
                    chosen=core.guarded_select(candidates,row,costs[j],spec)[0]
                    assert chosen==action_names[int(a[run,w,j])],(group,run,w,missions[j].mission_id,chosen)
                    count+=1
            if run%10==0:print('Action direct replay',group,run+1,'/',len(q),flush=True)
    assert count==1848000
    stage('all_original_policy_routing_candidates_and_selected_actions',selected_actions=count,trace_selected_actions=1680000,native_selected_actions=168000,full_precision_q=True)
    family=[s.family for s in scenarios];mw=np.zeros((30,8))
    for j,m in enumerate(missions):
        for s,f in enumerate(family):mw[j,s]=m.family_mix.get(f,0.)/family.count(f)
        near(mw[j].sum(),1.,'mission weight')
    near(mw,weights['mission_scenario_weights'],'stored mission weights',1e-15)
    ad=np.asarray([np.mean(ta[i]!=na[r['scenario_index']],axis=0) for i,r in enumerate(runs)])
    rd=np.asarray([np.mean(tr[i]!=nr[r['scenario_index']]) for i,r in enumerate(runs)])
    rec_metrics={};rec_shares={}
    for rid,nb in blocks.items():
        ii=[i for i,r in enumerate(runs) if r['recording_id']==rid]
        mass=np.asarray([mw[:,runs[i]['scenario_index']]/nb for i in ii]);near(mass.sum(axis=0),np.ones(30),'record mission mass')
        rec_metrics[rid]={'action_change':np.sum(ad[ii]*mass,axis=0),'routing_change':np.sum(rd[ii,None]*mass,axis=0),
         'unweighted_scenario_action_change':ad[ii].mean(axis=0),'unweighted_scenario_routing_change':np.repeat(rd[ii].mean(),30)}
        old=[];new=[]
        for ai in range(len(action_names)):
            old.append(np.sum(np.asarray([np.mean(na[runs[i]['scenario_index']]==ai,axis=0) for i in ii])*mass,axis=0))
            new.append(np.sum(np.mean(ta[ii]==ai,axis=1)*mass,axis=0))
        rec_shares[rid]=(np.asarray(old).T,np.asarray(new).T)
    mmap={m.mission_id:i for i,m in enumerate(missions)}
    for row in table('decision_changes_by_run_mission.csv'):
        i=int(row['run_index']);j=mmap[row['mission_id']]
        near(ad[i,j],float(row['action_change']),'run action');near(rd[i],float(row['routing_change']),'run route')
    for row in table('decision_changes_by_recording_mission.csv'):
        rid=row['recording_id'];j=mmap[row['mission_id']]
        for k,a in rec_metrics[rid].items():near(a[j],float(row[k]),'record mission '+k)
    for row in table('action_shares_by_recording_mission.csv'):
        j=mmap[row['mission_id']];k=action_names.index(row['action']);old,new=rec_shares[row['recording_id']]
        near(old[j,k],float(row['native_share']),'native share');near(new[j,k],float(row['trace_driven_share']),'new share')
    global_change={k:float(np.mean([vals[k] for vals in rec_metrics.values()])) for k in next(iter(rec_metrics.values()))}
    for row in table('decision_changes_pooled.csv'):
        vals=global_change if row['recording_id']=='equal_recordings_equal_missions' else {k:a.mean() for k,a in rec_metrics[row['recording_id']].items()}
        for k,v in vals.items():near(v,float(row[k]),'decision pooled '+k)
    pooled_old=np.mean([v[0] for v in rec_shares.values()],axis=(0,1));pooled_new=np.mean([v[1] for v in rec_shares.values()],axis=(0,1))
    for row in table('action_shares_pooled.csv'):
        i=action_names.index(row['action']);near(pooled_old[i],float(row['native_share']),'pooled old share');near(pooled_new[i],float(row['trace_driven_share']),'pooled new share')
    stage('all_mission_weights_and_decision_change_tables',missions=30,recordings=3,per_run_mission_rows=2400,record_mission_rows=90,pooled=global_change)
    selected_runs=[]
    for s in range(8):
        rid=list(blocks)[s%3];block=s%blocks[rid]
        i=next(i for i,r in enumerate(runs) if r['scenario_index']==s and r['recording_id']==rid and r['block_index']==block)
        rec,k,si,begin,end,lo,hi,bt=expected[i]
        supplied=[{'packet_id':j,'arrival_ms':float(t)/1000.,'size_bytes':float(size)} for j,(t,size) in enumerate(zip(bt,rec['sizes'][lo:hi]))]
        old=gen.generate_c2_arrivals
        gen.generate_c2_arrivals=lambda config,scenario,duration_ms,arr=supplied:arr
        try:packets,rows,meta=gen.run_trace(cfg,scenarios[s],'factual')
        finally:gen.generate_c2_arrivals=old
        exact(np.asarray([[float(row[n]) for n in names] for row in rows]),tx[i],'original run_trace '+str(i));selected_runs.append(i)
        print('Original run_trace exact',i,flush=True)
    stage('original_unoptimized_generator_stratified_replay',selected_runs=selected_runs,
     selection_rule='Scenario s: record s mod3, block s mod its record block count; independent of effects.',windows_exact=5600,features_exact=140000)
    am={m:i for i,m in enumerate(index['mission_ids'])};n=0;err=0.
    with (REPLAY/'results/audit_records/qwen_qwen-plus_guarded_audit_records.jsonl').open(encoding='utf-8') as f:
        for line in f:
            row=json.loads(line);i=lookup[row['source_window']['scenario_id']];w=int(row['source_window']['window_id']);j=am[row['mission']['mission_id']]
            ref=np.array([row['posterior'][k] for k in 'WBMV']);new=nq[i,w]
            exact([round(float(v),6) for v in new],ref,'historical 6dp q')
            assert index['routes'][int(nr[i,w])]==row['posterior_archetype']
            assert llm.candidate_actions_for(policies[row['mission']['mission_id']],new)==row['candidate_actions']
            assert action_names[int(na[i,w,j])]==row['selected_action']
            err=max(err,float(np.max(np.abs(new-ref))));n+=1
    assert n==54000
    stage('independent_54000_historical_chain',records=n,max_unrounded_q_error=err)
    summary=read(RESULT/'summary.json')
    for k,v in global_change.items():near(v,summary['decision_change'][k],'summary '+k)
    assert summary['trace_driven_windows']==56000 and summary['added_hypothesis_tests']==0
    REPORT['numerical_summary']={'balanced_arrival_W1':arr_dist,'balanced_q_W1':q_pooled,'decision_changes':global_change}
    REPORT['important_limitations']=[
     'q derives from measured transport arrivals plus simulated service/interference/features; it is not an empirical deployment posterior.',
     'Replacement changes mean rate, burst structure and family-rate association jointly; burstiness is not isolated.',
     'Three records and ten blocks are not IID field campaigns; no field CI, accuracy, regret or safety guarantee is established.',
     'DJI endpoint roles are inferred; datagrams are not decoded unique flight-control commands.',
     'All source/metric/feature-to-q/actions checked; unoptimized feature generation replayed for eight deterministic strata, not all80 runs.'
    ]
    REPORT['status']='PASS'
    if not numerical_only:review_writing()
    write_report()
    print('ALL CHECKS PASS',json.dumps(REPORT['numerical_summary']),flush=True)
if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    mode=parser.add_mutually_exclusive_group()
    mode.add_argument('--numerical-only',action='store_true',help='Audit numerical science artifacts without local manuscript/response files.')
    mode.add_argument('--text-only',action='store_true',help='Refresh local writing hashes without rerunning the numerical checks.')
    args=parser.parse_args()
    try:
        if args.text_only:
            REPORT=read(OUT/'execution_review.json')
            REPORT['execution_mode']='text-only-refresh';REPORT['numerical_checks_rerun']=False
            REPORT['writing_refreshed_utc']=datetime.now(timezone.utc).isoformat()
            review_writing();write_report()
        else:main(numerical_only=args.numerical_only)
    except Exception as exc:
        REPORT['status']='FAIL';REPORT['failure']=repr(exc);REPORT['traceback']=traceback.format_exc()
        write_report();raise
