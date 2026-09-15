from pathlib import Path
import json, shutil, datetime
import numpy as np
import pandas as pd
from study import R, P20, sha, dump, predict, dmat, check_binding

assert not (R/'replay').exists(), 'Independent replay folder already initialized'
check_binding('execution_binding.json');check_binding('selection_binding.json')
decision=json.loads((R/'results/spatial_decision.json').read_text())
assert decision['spatial_gate']=='PASS_TO_NS3_PILOT'
S=R/'replay';S.mkdir()
for folder in ['inputs','calibration']:(S/folder).mkdir()
for p in (P20/'inputs').glob('*'):
    if p.is_file():shutil.copy2(p,S/'inputs'/p.name)
for name in ['case_plan.json','observation_adapter.cc','run_study.py','reduce_runs.py','verify_results.py']:
    shutil.copy2(P20/name,S/name)
shutil.copy2(R/'replay_protocol.md',S/'protocol.md')
models=json.loads((R/'calibration/models.json').read_text())
model=models[decision['selected_model']]
t=pd.read_csv(R/'calibration/selected_residuals.csv')
res=t.residual.to_numpy();adj=np.diff(t.systime.to_numpy())==1
rho=float(np.dot(res[:-1][adj],res[1:][adj])/np.dot(res[:-1][adj],res[:-1][adj]))
sigma=float(np.sqrt(np.mean(res**2)))
assert 0<=rho<1
fit=json.loads((P20/'calibration/radio_fit.json').read_text())
fit.update(shadow_std_db=sigma,rho_per_second=rho,training_rmse_db=sigma,
           fit='P21 spatial kernel residual plus OLS distance; AR1 estimated only from full trace302 training residuals',
           spatial_model='kernel_10',selected_model_sha256=sha(R/'calibration/models.json'))
dump(S/'calibration/radio_fit.json',fit)
cases=json.loads((S/'case_plan.json').read_text())
for block,c in {c['block']:c for c in cases}.items():
    path=pd.read_csv(S/c['trajectory'])
    ts=np.round(np.arange(0,20.0001,.1),10)
    x=np.interp(ts,path.time_s,path.x_m);y=np.interp(ts,path.time_s,path.y_m)
    z=pd.DataFrame(dict(receiverX=x,receiverY=y,receiverDist=np.hypot(x,y)))
    correction=predict(model,z)-dmat(z)@np.asarray(model['distance_b'])
    correction[ts<2]=0
    pd.DataFrame(dict(time_s=ts,correction_db=correction)).to_csv(S/'inputs'/f'{block}_spatial.csv',index=False)
for c in cases:c['spatial']=f'inputs/{c["block"]}_spatial.csv'
dump(S/'case_plan.json',cases)
pd.DataFrame(dict(time_s=[0,20],correction_db=[3,3])).to_csv(S/'inputs/engineering_constant.csv',index=False)
pd.DataFrame(dict(time_s=[0,20],correction_db=[0,20])).to_csv(S/'inputs/engineering_ramp.csv',index=False)
code=(S/'observation_adapter.cc').read_text()
code=code.replace('P20 independent observation adapter','P21 spatial extension of P20 independent observation adapter').replace('ns3::P20PilotLoss','ns3::P21SpatialLoss')
needle='  std::vector<double> shadow;'
replacement='''  std::vector<double> shadow;
  std::vector<double> spatialTimes,spatialOffsets;
  void LoadSpatial(const std::string& path){
    if(path.empty())return;
    std::ifstream in(path);NS_ABORT_MSG_IF(!in,"Missing spatial correction");
    std::string line;std::getline(in,line);
    while(std::getline(in,line)){
      std::replace(line.begin(),line.end(),',',' ');std::istringstream ss(line);double t,v;ss>>t>>v;
      NS_ABORT_MSG_IF(!ss||!std::isfinite(t)||!std::isfinite(v),"Invalid spatial correction");
      NS_ABORT_MSG_IF(!spatialTimes.empty()&&t<=spatialTimes.back(),"Spatial times not increasing");
      spatialTimes.push_back(t);spatialOffsets.push_back(v);
    }
    NS_ABORT_MSG_IF(spatialTimes.size()<2||spatialTimes.front()!=0||spatialTimes.back()<endS,"Incomplete spatial timeline");
  }
  double Spatial(double t)const{
    if(spatialTimes.empty()||t<2)return 0;
    auto upper=std::upper_bound(spatialTimes.begin(),spatialTimes.end(),t);
    auto j=std::min(static_cast<size_t>(upper-spatialTimes.begin()),spatialTimes.size()-1);
    auto i=j-1;double f=(t-spatialTimes[i])/(spatialTimes[j]-spatialTimes[i]);
    return spatialOffsets[i]*(1-f)+spatialOffsets[j]*f;
  }'''
assert code.count(needle)==1;code=code.replace(needle,replacement)
needle='    if(target&&!shadow.empty()){';assert code.count(needle)==1
code=code.replace(needle,'    if(target)out+=Spatial(Simulator::Now().GetSeconds());\n'+needle)
code=code.replace('trajectory="",mode="replay"','trajectory="",spatial="",mode="replay"')
needle='  cmd.AddValue("run","RNG run",run);'
code=code.replace(needle,'  cmd.AddValue("spatial","Frozen spatial correction CSV",spatial);\n'+needle)
needle='  auto channel=CreateObject<YansWifiChannel>();'
code=code.replace(needle,'  loss->LoadSpatial(spatial);\n'+needle)
(S/'observation_adapter.cc').write_text(code,encoding='utf-8')
runner=(S/'run_study.py').read_text().replace('p20_compile_','p21_compile_')
needle="    gate=read(R/'engineering_gates.json');assert gate['status']=='PASS'"
runner=runner.replace(needle,needle+"\n    assert read(R/'spatial_engineering.json')['status']=='PASS'")
needle="f'--trajectory={R/c[\"trajectory\"]}',"
assert runner.count(needle)==1
runner=runner.replace(needle,needle+"f'--spatial={R/c[\"spatial\"]}',")
(S/'run_study.py').write_text(runner,encoding='utf-8')
files=[*sorted((S/'inputs').glob('*')),*sorted((S/'calibration').glob('*')),S/'case_plan.json',S/'protocol.md']
dump(S/'pre_evaluation_binding.json',dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    files={str(p.relative_to(S)).replace('\\','/'):sha(p) for p in files},
    source_stage_binding=sha(R/'selection_binding.json'),parent_preparation_sha256=sha(Path(__file__))))
print(json.dumps(dict(shadow_std_db=sigma,rho=rho,cases=len(cases)),ensure_ascii=False),flush=True)
