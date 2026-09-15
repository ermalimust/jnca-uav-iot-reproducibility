"""Recompute two R7 examples from archived saved-q records, without API calls.

Standard library only. No archived file is modified. Core constants are read by
ast.literal_eval, not by importing/executing the archived API-capable module.

Run: python recompute_R7_witness.py --workspace-root <folder containing revision_work>
Optional: --output-dir <new analysis folder>. Defaults to this script's directory.
The default root search also works after moving this folder into a public package.
"""
import argparse, ast, csv, hashlib, json
from pathlib import Path
from decimal import Decimal as D, localcontext

AUDIT='revision_work/analysis/replay_inputs/results/audit_records/qwen_qwen-plus_guarded_audit_records.jsonl'
CORE='revision_work/analysis/replay_inputs/paper7_agentic_feasibility.py'
MARGINS='revision_work/analysis/followup_margins/decision_margins.csv'
IDS=['urban_c2_safety_01:seed0:decision0','urban_c2_safety_01:seed0:decision2']
THETA={'safety_risk':D('.42'),'rid':D('.28'),'video':D('.35'),'video_risk':D('.70'),'energy':D('.34')}

def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for chunk in iter(lambda:f.read(1048576),b''):h.update(chunk)
    return h.hexdigest()

def scalar(v):return D(str(v))
def serial(v):
    if isinstance(v,D):return str(v)
    if isinstance(v,dict):return {k:serial(x) for k,x in v.items()}
    if isinstance(v,(list,tuple)):return [serial(x) for x in v]
    return v

def find_root(explicit):
    if explicit:
        r=Path(explicit).resolve()
        assert (r/AUDIT).is_file(),str(r/AUDIT)
        return r
    for p in [Path(__file__).resolve().parent,*Path(__file__).resolve().parents]:
        if (p/AUDIT).is_file():return p
    raise SystemExit('Use --workspace-root for the folder containing revision_work.')

def constants(path):
    tree=ast.parse(path.read_text(encoding='utf-8'))
    found={}
    required={'CAUSES','SUPPORTED_ACTIONS','ACTION_OVERHEAD','BASE_COSTS'}
    for node in tree.body:
        if isinstance(node,ast.Assign):
            names=[x.id for x in node.targets if isinstance(x,ast.Name)]
        elif isinstance(node,ast.AnnAssign) and isinstance(node.target,ast.Name):names=[node.target.id]
        else:continue
        for name in names:
            if name in required:found[name]=ast.literal_eval(node.value)
    assert set(found)==required
    guard=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='verifier_rejection_reason')
    numbers={n.value for n in ast.walk(guard) if isinstance(n,ast.Constant) and type(n.value) in (float,int)}
    assert {.42,.28,.35,.70,.34}.issubset(numbers)
    return found

def cost_profile(record,base):
    g=record['mission']['guards']
    # Both records deliberately use the same existing safety-only worked example.
    # Its profile has no guard-dependent coefficient adjustments, as defined by
    # cost_matrix in the archived implementation and Appendix A.2.
    assert g=={'safety':True,'rid':False,'video':False,'energy':False}
    assert record['mission']['gold_cost_profile']=='safety_first'
    return {a:list(map(scalar,row)) for a,row in base['safety_first'].items()}

def rejected(action,q,guards,theta,actions):
    if action not in actions:return 'unsupported_action'
    risk=q[0]+q[1]+q[2]+D('.5')*q[3]
    if guards['safety'] and risk>=theta['safety_risk'] and action=='Observe':return 'safety'
    if guards['rid'] and q[1]>=theta['rid'] and action not in ('BLEAvoid','FallbackProtect'):return 'rid'
    if guards['video'] and q[3]<theta['video'] and risk<theta['video_risk'] and action=='FallbackProtect':return 'video'
    if guards['energy'] and q[2]<theta['energy'] and action=='LinkAdapt':return 'energy'
    return ''

def evaluate(record,cm,overhead,theta,causes,actions):
    q=[scalar(record['posterior'][c]) for c in causes]
    rows=[]
    for a in actions:
        reason=rejected(a,q,record['mission']['guards'],theta,actions)
        effect=sum(x*y for x,y in zip(q,cm[a]))
        rows.append({'action':a,'exposed':a in record['candidate_actions'],'admitted':not reason,
                     'rejection_reason':reason,'coefficients':cm[a],
                     'posterior_cause_loss':effect,'overhead':overhead[a],'J':effect+overhead[a]})
    by={r['action']:r for r in rows}
    candidates=[a for a in record['candidate_actions'] if by[a]['admitted']]
    selected=min(candidates,key=lambda a:by[a]['J']) if candidates else ('FallbackProtect' if by['FallbackProtect']['admitted'] else 'EscalateReview')
    return {'candidate_order':record['candidate_actions'],'admitted_candidates':candidates,'selected_action':selected,'all_six_actions':rows}

def run(root,out):
    hashes={p:sha(root/p) for p in [AUDIT,CORE,MARGINS]}
    cc=constants(root/CORE);actions=cc['SUPPORTED_ACTIONS'];causes=cc['CAUSES']
    overhead={a:scalar(v) for a,v in cc['ACTION_OVERHEAD'].items()}
    records={};positions={}
    with (root/AUDIT).open(encoding='utf-8') as f:
        for line_number,line in enumerate(f,1):
            r=json.loads(line)
            if r['decision_id'] in IDS:records[r['decision_id']]=r;positions[r['decision_id']]=line_number
            if len(records)==2:break
    margins={}
    with (root/MARGINS).open(encoding='utf-8-sig',newline='') as f:
        for line_number,r in enumerate(csv.DictReader(f),2):
            if r['decision_id'] in IDS:margins[r['decision_id']]={'csv_line':line_number,'row':r}
            if len(margins)==2:break
    assert set(records)==set(margins)==set(IDS)
    cases=[];output_rows=[]
    for key in IDS:
        r=records[key];cm=cost_profile(r,cc['BASE_COSTS'])
        nominal=evaluate(r,cm,overhead,THETA,causes,actions)
        assert nominal['selected_action']==r['selected_action']
        assert nominal['admitted_candidates']==r['accepted_actions']
        precision=[]
        for x in nominal['all_six_actions']:
            if x['action'] in r['posterior_costs_mlu']:
                old=scalar(r['posterior_costs_mlu'][x['action']]);diff=x['J']-old
                bound=sum(abs(c) for c in x['coefficients'])*D('.0000005')+D('.0000005')
                assert abs(diff)<=bound
                precision.append({'action':x['action'],'archive_score':old,'saved_q_recomputed_score':x['J'],'difference':diff,'rounding_bound':bound})
        scenarios=[]
        if key==IDS[0]:
            by={x['action']:x for x in nominal['all_six_actions']};winner=nominal['selected_action']
            ratios={a:(by[a]['J']-by[winner]['J'])/(by[a]['posterior_cause_loss']+by[winner]['posterior_cause_loss']) for a in nominal['admitted_candidates'] if a!=winner}
            competitor=min(ratios,key=ratios.get);rho=ratios[competitor]
            assert competitor=='FallbackProtect'
            assert abs(rho-scalar(margins[key]['row']['coefficient_radius']))<D('1e-14')
            for epsilon in [D('0'),D('.01'),D('.012')]:
                perturbed={a:list(v) for a,v in cm.items()}
                perturbed[winner]=[v*(1+epsilon) for v in cm[winner]]
                perturbed[competitor]=[v*(1-epsilon) for v in cm[competitor]]
                ev=evaluate(r,perturbed,overhead,THETA,causes,actions)
                assert ev['admitted_candidates']==nominal['admitted_candidates']
                assert ev['selected_action']==('FallbackProtect' if epsilon==D('.012') else 'WiFiRelief')
                scenarios.append({'coefficient_relative_radius':epsilon,'thresholds':THETA,'evaluation':ev})
            claim={'type':'selected_action_flip_witness','coefficient_certificate':rho,'minimizing_competitor':competitor,
                   'all_competitor_ratios':ratios,'perturbation':'WiFiRelief row *=1+epsilon; FallbackProtect row *=1-epsilon; other rows, q, candidate order, guards and all overheads fixed.',
                   'observed_flip_at':D('.012'),'interpretation':'A constructed witness above the existing coefficient certificate, not a new outcome test or a global joint-flip calculation.'}
        else:
            q=[scalar(r['posterior'][c]) for c in causes];risk=q[0]+q[1]+q[2]+D('.5')*q[3]
            rho=abs(risk-THETA['safety_risk'])/THETA['safety_risk']
            assert abs(rho-scalar(margins[key]['row']['guard_relative_radius']))<D('1e-14')
            for theta_s in [D('.42'),D('.37'),D('.36')]:
                theta=dict(THETA);theta['safety_risk']=theta_s
                ev=evaluate(r,cm,overhead,theta,causes,actions)
                assert ev['selected_action']=='WiFiRelief'
                assert ('Observe' in ev['admitted_candidates'])==(theta_s>risk)
                scenarios.append({'safety_threshold':theta_s,'threshold_relative_radius':abs(theta_s-D('.42'))/D('.42'),'thresholds':theta,'evaluation':ev})
            claim={'type':'admission_change_without_selected_action_flip','risk':risk,'guard_admission_certificate':rho,
                   'interpretation':'Only the safety guard is active; its sole affected candidate Observe has higher fixed score than admitted WiFiRelief. Both possible Observe admission states preserve selection. This is an example of conservatism of the admission certificate.'}
        cases.append({'decision_id':key,'source_jsonl_line':positions[key],'archived_margin':margins[key],
                      'source_record':r,'nominal':nominal,'score_precision_check':precision,'claim':claim,'scenarios':scenarios})
        for index,s in enumerate(scenarios):
            for x in s['evaluation']['all_six_actions']:
                output_rows.append({'decision_id':key,'scenario':index,'coefficient_radius':s.get('coefficient_relative_radius','0'),
                    'safety_threshold':s['thresholds']['safety_risk'],'action':x['action'],'exposed':x['exposed'],'admitted':x['admitted'],
                    'posterior_cause_loss':x['posterior_cause_loss'],'overhead':x['overhead'],'J':x['J'],
                    'selected_action':s['evaluation']['selected_action'],'rejection_reason':x['rejection_reason']})
    assert all(sha(root/p)==h for p,h in hashes.items())
    out.mkdir(parents=True,exist_ok=True)
    result={'verified':True,'design':'Two worked examples chosen to illustrate distinct mathematical properties; not a sample-based performance claim.',
      'posterior_precision':'Calculations use the six-decimal q values in the saved audit. Stored cost scores were separately rounded after original computation; their small differences from saved-q recomputation are retained, not hidden. The examples and coefficient/admission radii are recomputed from saved q throughout.',
      'input_manifest_sha256':hashes,'archived_sources_unchanged':True,'core_loading':'ast.literal_eval of constants only; no module import or execution',
      'all_six_actions_compared':True,'all_exposed_admitted_candidates_compared':True,
      'rounding_max_abs_difference':max(abs(x['difference']) for c in cases for x in c['score_precision_check']),
      'cases':cases,'script_sha256':sha(Path(__file__).resolve())}
    (out/'R7_selected_flip_witness.json').write_text(json.dumps(serial(result),ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    with (out/'R7_selected_flip_witness.csv').open('w',encoding='utf-8',newline='') as f:
        wr=csv.DictWriter(f,fieldnames=list(output_rows[0]));wr.writeheader();wr.writerows(serial(output_rows))
    print(json.dumps({'verified':True,'examples':len(cases),'all_action_scenario_rows':len(output_rows),'max_score_rounding_difference':str(result['rounding_max_abs_difference']),'source_hashes_unchanged':True}))

if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--workspace-root');ap.add_argument('--output-dir',default=str(Path(__file__).resolve().parent))
    args=ap.parse_args()
    with localcontext() as ctx:
        ctx.prec=60
        run(find_root(args.workspace_root),Path(args.output_dir).resolve())
