"""Read-only, relocatable checks of the sealed measured-service release.

Historical final_checks.py additionally checks the then-current manuscript.
This checker separates immutable science from subsequent manuscript edits.
Requires Python, NumPy and pandas; no network, API or ns-3 installation.
"""
from pathlib import Path
import argparse, hashlib, json, math
import numpy as np
import pandas as pd

def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()

def w1(a,b):
    a=np.sort(np.asarray(a).ravel());b=np.sort(np.asarray(b).ravel())
    edges=np.unique(np.r_[a,b]);left=edges[:-1]
    return float(np.dot(np.diff(edges),abs(np.searchsorted(a,left,side='right')/a.size-np.searchsorted(b,left,side='right')/b.size)))

def check(root):
    def read(p):return json.loads(p.read_text(encoding='utf-8'))
    r=root/'revision_work/new_experiments/P24_capacity_replay_20260914'
    p23=r.parent/'P23_wichoose_service_20260914'
    verified={}
    for stage in ['capacity_replay_20260914','wichoose_service_20260914']:
        manifest=read(root/'output'/stage/'artifact_manifest.json')['files']
        for name,x in manifest.items():
            p=(root/name).resolve();assert p.is_relative_to(root.resolve()) and p.is_file(),name
            assert p.stat().st_size==x['bytes'] and sha(p)==x['sha256'],name
        verified[stage]=len(manifest)
    for name in ['training_binding.json','selection_binding.json','evaluation_binding.json','analysis_receipt.json']:
        for rel,h in read(r/name)['files'].items():assert sha(r/rel)==h,(name,rel)
    # The preparation seal records original absolute host paths. Resolve only
    # known descendants by the preserved new_experiments directory structure.
    for name,h in read(r/'preparation_binding.json')['files'].items():
        suffix=name.replace('\\','/').split('/01_目前工作區/',1)
        assert len(suffix)==2,name
        p=(root/suffix[1]).resolve()
        assert p.is_relative_to(root.resolve()) and sha(p)==h,name
    d=pd.read_csv(r/'inputs/observations.csv').set_index('index');truth=d.loc[1616:]
    assert len(truth)==978 and (np.diff(truth.second_unix)==1).all()
    train=pd.read_csv(r/'training_predictions.csv');rank=pd.read_csv(r/'training_ranking.csv').set_index('config_id')
    assert len(train)==1440 and train['index'].max()<1556
    for name,g in train.groupby('config_id'):
        pred=g.groupby('index').received_mbps.mean();y=d.loc[pred.index,'received_mbps']
        assert len(pred)==60 and abs(np.mean(abs(pred-y))-rank.loc[name,'mae_mbps'])<1e-10
        multiplier=math.fsum(float(x*y) for x,y in zip(pred,y))/math.fsum(float(x*x) for x in pred)
        assert abs(multiplier-rank.loc[name,'multiplier'])<1e-12
    near=rank[rank.mae_mbps<=rank.mae_mbps.min()*1.01].reset_index().sort_values(['width','nss','short_gi','config_id'])
    sel=read(r/'selection.json')
    assert sel['near_optimal']==near.config_id.tolist() and sel['selected']==near.iloc[0].config_id
    sim=pd.read_csv(r/'results/continuous_simulation_seconds.csv')
    assert len(sim)==5868 and not sim.duplicated(['case_id','index']).any()
    for c in read(r/'evaluation_cases.json'):
        folder=r/'runs/evaluation'/c['case_id'];seconds=pd.read_csv(folder/'seconds.csv').query('measured == 1')
        expected=sim[sim.case_id==c['case_id']].sort_values('index')
        assert len(seconds)==len(expected)==978
        assert np.allclose(seconds.rx_bytes.to_numpy()*8/1e6,expected.received_mbps.to_numpy(),atol=1e-12,rtol=0)
    preds={name:sim[sim.model.eq(name)].pivot(index='index',columns='run',values='received_mbps').loc[truth.index].to_numpy() for name in ['reference','selected']}
    for label,cid in [('reference','w20_s2_long'),('selected',sel['selected'])]:preds[label+'_calibrated']=preds[label]*rank.loc[cid,'multiplier']
    baseline=pd.read_csv(p23/'results/full_evaluation_predictions.csv').set_index('index')
    for name in ['tx_identity','tx_affine','rssi_tx_ridge_0.1']:preds[name]=baseline.loc[truth.index,name].to_numpy().reshape(-1,1)
    blocks={b['block']:b for b in read(r/'p23_blocks.json')};errors=[];nrows=0
    for table in ['full_metrics.csv','period_metrics.csv','p23_window_metrics.csv']:
        for row in pd.read_csv(r/'results'/table).itertuples(index=False):
            if table=='full_metrics.csv':mask=np.ones(len(truth),dtype=bool)
            elif table=='period_metrics.csv':mask=truth.mobility_period.eq(row.period).to_numpy()
            else:
                b=blocks[row.block];mask=(truth.index>=b['start_index'])&(truth.index<b['end_index_exclusive'])
            p=preds[row.model][mask];y=truth.loc[mask,'received_mbps'].to_numpy()
            dist=w1(p,y);mae=math.fsum(abs(float(a-b)) for a,b in zip(p.mean(axis=1),y))/len(y)
            assert len(y)==row.source_seconds and p.size==row.simulated_or_predicted_values
            errors.extend([abs(dist-row.w1_mbps),abs(mae-row.mae_mbps),abs(dist/y.mean()-row.nw1),abs(mae/y.mean()-row.nmae)])
            nrows+=1
    assert max(errors)<1e-9
    boot=pd.read_csv(r/'results/moving_block_bootstrap.csv');assert len(boot)==15000
    for row in pd.read_csv(r/'results/intervals.csv').itertuples(index=False):
        q=np.quantile(boot[(boot.model==row.model)&(boot.block_length==row.block_length)][row.metric],[.025,.975])
        assert np.allclose(q,[row.low,row.high],rtol=0,atol=1e-10)
    old=read(r/'verification.json');assert old['status']=='PASS'
    return {'status':'PASS','sealed_artifacts_hash_verified':verified,'capacity_candidates':8,'training_runs':72,'continuous_runs':6,
            'unique_measured_seconds':978,'simulation_seconds':5868,'metric_rows_independently_recalculated':nrows,
            'maximum_scalar_error':max(errors),'bootstrap_records':15000,
            'historical_packet_accounting_receipt_sha256':sha(r/'verification.json'),
            'scope':'Relocated hashes, selection, per-second ledgers, distances and interval endpoints checked. Historical full packet-ID audit retained by hash; packet events and bootstrap draw metrics are not re-executed here. No files modified.'}

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[2]);args=parser.parse_args()
    print(json.dumps(check(args.root.resolve()),ensure_ascii=False,indent=2))
