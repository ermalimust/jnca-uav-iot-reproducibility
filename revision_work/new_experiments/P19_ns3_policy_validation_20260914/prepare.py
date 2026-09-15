"""Create the P19 inputs and observable-only instrumentation from frozen P18."""
import itertools
import shutil
from common import *

def main():
    inputs = HERE / 'inputs'
    inputs.mkdir(exist_ok=True)
    dependencies = []
    for name in ['candidates.json', 'contexts.json', 'ood_mission_intents.jsonl',
                 'paper7_agentic_feasibility.py', 'paper7_llm_candidate_experiment.py']:
        src = P12 / 'inputs' / name
        shutil.copy2(src, inputs / name)
        dependencies.append({'source': src.relative_to(W).as_posix(), 'sha256': sha(src), 'copy': 'inputs/' + name})
    missions = [json.loads(s) for s in (inputs / 'ood_mission_intents.jsonl').read_text(encoding='utf-8').splitlines() if s.strip()]
    inclusion = []
    for m in missions:
        why = [name for name in ('rid', 'energy') if m['guards'].get(name, False)]
        assert m['cost_weights']['safety'] + m['cost_weights']['throughput'] > 0
        inclusion.append({'mission_id': m['mission_id'], 'included': not why,
                          'exclusion_reason': ','.join(why), 'intent': m['intent'],
                          'alpha_c2': m['cost_weights']['safety'] / (m['cost_weights']['safety'] + m['cost_weights']['throughput'])})
    assert sum(x['included'] for x in inclusion) == 19
    write_csv(inputs / 'mission_inclusion.csv', inclusion)
    dump(inputs / 'missions_included.json', [m for m, inc in zip(missions, inclusion) if inc['included']])
    selected = {m['mission_id'] for m in missions if not m['guards'].get('rid', False) and not m['guards'].get('energy', False)}
    sources = []
    for c in read(inputs / 'candidates.json'):
        if c['mission_id'] in selected and c.get('source_sha256'):
            src = W / c['source_path']
            assert src.is_file(), src
            if src.is_file():
                assert sha(src) == c['source_sha256']
                dest = inputs / 'archived_generation' / src.name
                dest.parent.mkdir(exist_ok=True)
                shutil.copy2(src, dest)
                sources.append({'source': c['source_path'], 'sha256': sha(src), 'copy': dest.relative_to(HERE).as_posix()})
    for p in (P18 / 'inputs').glob('*'):
        if p.is_file():
            shutil.copy2(p, inputs / p.name)
    shutil.copy2(P18 / 'independent_statistics/pooled_101_tests.csv', inputs / 'prior_101_tests.csv')
    scenarios = []
    for w, motion, v in itertools.product((0, 1), (0, 1, 2), (0, 1)):
        scenarios.append({'scenario_id': f'W{w}S{motion}V{v}', 'w': w, 'motion': motion, 'v': v,
                          'speed_mps': [0, 1.0, 3.5][motion], 'moving': int(motion > 0),
                          'offer_path': f'inputs/W{w}M{int(motion > 0)}V{v}.csv'})
    protocol = {
      'date': '2026-09-14', 'status': 'design fixed before P19 observations and service outcomes',
      'question': 'Independent one-command policy comparison with a trained ns-3 observation/service adapter',
      'origin': 'Designed after inspecting P18; P18 outcomes are development context, not new confirmation data.',
      'actions': ACTIONS, 'scenarios': scenarios, 'rng_seed': 6091419,
      'train_runs': list(range(1001, 1017)), 'validation_runs': list(range(2001, 2009)),
      'test_runs': list(range(3001, 3033)), 'engineering_runs': [9901, 9902],
      'network': 'P18 network, original public Parrot first 20s and all five unchanged handlers; initial range 15m; additional fixed speed 1m/s alongside stationary and 3.5m/s.',
      'timing': {'prefix_snapshot_ns': 10999999999, 'command_ns': 11000000000, 'effective_ns': 11001000000,
                 'offer_end_ns': 21000000000, 'drain_end_ns': 23000000000, 'deadline_ns': 10000000},
      'instrumentation': 'Controller-local C2 offer/transmission/MAC ACK/drop records, controller-received video arrivals and actual video-frame RSSI. No flow2 counters, source video demand, remote C2 receive times, configured state, speed, range or future outcomes enter diagnostic features.',
      'diagnostic': {'model': '12-class multinomial logistic regression, final class reference logit zero',
                     'labels': 'training-only configured external-traffic / motion-regime / video-burst operating conditions',
                     'training': 'one frozen prefix observation per train episode; all classes and RNG blocks equally weighted',
                     'scaling': 'train means and population standard deviations; constant standard deviations replaced by one',
                     'objective': 'mean multiclass cross entropy + sum(nonintercept coefficients squared)/(2*C*n_train)',
                     'C_grid': [0.1, 1.0, 10.0], 'selection': 'minimum validation joint log loss; ascending C tie rule',
                     'probability_use': 'joint12 belief for service expectation; W/M-active/V marginals for unchanged routing/guards; B=0 by explicitly absent BLE model',
                     'scope': 'new ns-3 adapter; not frozen original 25-feature diagnostic or measured operational posterior'},
      'service_calibration': 'Train-bank sample mean C2 miss and original-demand video nondelivery for each of 12 states x 5 actions; every source uses identical table and posterior expectation.',
      'mission_scope': 'All 19 archived OOD missions with rid=false and energy=false; inclusion depends only on metadata, not candidates or outcomes. Same 19 mission weights for every method.',
      'service_loss': 'alpha*C2_miss+(1-alpha)*(1-video_delivery), alpha=archived safety/(safety+throughput); no assigned overhead or mismatch term is presented as packet-measured cost. Unitless fraction, distinct from original MLU.',
      'candidates': 'P12 saved opaque_zero and public_tool_agent, all three replicates, plus broad_first3 and embedding_first3; first3 distinct then capability filter, no refill. Full library uses all five as explicitly broader comparator.',
      'authorization': 'Original routing and nominal guards, wrapped in A5 capability check. Guarded empty accepted list uses verified FallbackProtect, otherwise EscalateReview. Direct chooses first capability-valid entry, otherwise EscalateReview. EscalateReview does not actuate and receives Observe-arm physical outcomes; label/rate retained.',
      'methods': METHODS,
      'numeric_sensitivity': 'Apply unchanged original numerical score to all five guarded candidate sources; descriptive factor, same adapter, packets and guards.',
      'test_order': 'Freeze code, source inputs, engineering gates; train and validate; freeze adapter and train service table; run test prefix-only probes; commit every method decision; only then permit new test action bank.',
      'primary': {'treatment': 'qwen_service', 'comparators': ['qwen_direct', 'embedding_service', 'broad_service', 'full_service'],
                  'endpoints': ['c2_miss', 'video_delivery', 'service_loss'], 'test_count': 12,
                  'unit': '32 independent RNG blocks; average 12 fixed scenarios, 19 fixed missions and all source replicates within each block',
                  'test': 'exact two-sided paired sign-flip using integer-scaled rational block differences and meet-in-the-middle counting; sign-exchangeability null',
                  'intervals': '10000 common paired block-bootstrap samples, percentile pointwise 95%',
                  'bootstrap_seed': 609141919,
                  'multiplicity': 'All 12 tests added to archived 101-test inventory: pooled113 Holm and BH; retain every direction and all original raw tests.'},
      'references': 'Executable full_service uses the same prefix, predicted belief, train table and guards. Hindsight min over all A5 and separately q-admitted A5 uses test outcome bank only after decisions; descriptive finite-bank bounds, not deployable policies or physical-world optima.',
      'scope': 'One-command common-prefix branch replay, conditional on one public input case and 12 protocol scenarios. No field test, BLE/RID coexistence, hardware timing guarantee or unrestricted method superiority.'
    }
    dump(HERE / 'protocol.json', protocol)
    dump(inputs / 'source_binding.json', {'archived_files': dependencies, 'generation_records': sources,
                                         'p18_protocol_sha256': sha(P18 / 'protocol.json'),
                                         'p18_source_sha256': sha(P18 / 'action_effects.cc')})
    src = (P18 / 'action_effects.cc').read_text(encoding='utf-8')
    changes = [
      ('// P18 independent Wi-Fi action-effects experiment.', '// P19 independent observable-prefix policy replay; derived from frozen P18.'),
      ('"ns3::P18LedgerTag"', '"ns3::P19LedgerTag"'),
      ('int64_t first_phy_ns=-1,last_phy_ns=-1,last_mac_drop_ns=-1;', 'int64_t first_phy_ns=-1,last_phy_ns=-1,last_mac_drop_ns=-1,first_ack_ns=-1;'),
      ('static uint32_t mFactor=0;', '''static uint32_t mFactor=0;
static double motionSpeed=3.5;
static bool probe=false;
struct RadioRow {int64_t time_ns; uint32_t packet_id; double signal_dbm,noise_dbm;};
static std::vector<RadioRow> controllerVideoRadio;
static void RadioRx(Ptr<const Packet> p,uint16_t,WifiTxVector,MpduInfo,SignalNoiseDbm sn,uint16_t){
  LedgerTag t;if(!p->PeekPacketTag(t)||rows.at(t.id).flow!=1)return;
  controllerVideoRadio.push_back({Simulator::Now().GetNanoSeconds(),t.id,sn.signal,sn.noise});
}
static void WriteRadio(){
  std::ofstream out("radio_prefix.csv");NS_ABORT_MSG_IF(!out,"Cannot write controller radio observations");
  out<<"time_ns,packet_id,signal_dbm,noise_dbm\\n"<<std::setprecision(17);
  for(const auto& r:controllerVideoRadio)out<<r.time_ns<<','<<r.packet_id<<','<<r.signal_dbm<<','<<r.noise_dbm<<'\\n';
}'''),
      ('auto& r=rows.at(t.id);++r.mac_acks;', 'auto& r=rows.at(t.id);++r.mac_acks;if(r.first_ack_ns<0)r.first_ack_ns=now();'),
      ('drop_reason_mask,tid_mask\\n', 'drop_reason_mask,tid_mask,first_ack_ns\\n'),
      ("<<r.drop_reason_mask<<','<<r.tid_mask<<'\\n';", "<<r.drop_reason_mask<<','<<r.tid_mask<<','<<r.first_ack_ns<<'\\n';"),
      ('cmd.Parse(argc,argv);', 'cmd.AddValue("speed","Motion speed in m/s",motionSpeed);cmd.AddValue("probe","Stop after prefix before any intervention",probe);cmd.Parse(argc,argv);'),
      ('RngSeedManager::SetSeed(6091407)', 'RngSeedManager::SetSeed(6091419)'),
      ('Vector(3.5,0,0)', 'Vector(motionSpeed,0,0)'),
      ('if(pcap)phy.EnablePcapAll("engineering",true);', 'NS_ABORT_MSG_IF(!primary[0]->GetPhy()->TraceConnectWithoutContext("MonitorSnifferRx",MakeCallback(&RadioRx)),"Missing controller RSSI trace");\n  if(pcap)phy.EnablePcapAll("engineering",true);'),
      ('Simulator::Schedule(Seconds(11),&Knobs', 'Simulator::Schedule(NanoSeconds(10999999999ll),&WriteRadio);\n  Simulator::Schedule(Seconds(11),&Knobs'),
      ('Simulator::Stop(Seconds(23));Simulator::Run();', 'Simulator::Stop(probe?NanoSeconds(10999999999ll):Seconds(23));Simulator::Run();')
    ]
    for old, new in changes:
        assert old in src, old
        src = src.replace(old, new)
    (HERE / 'policy_replay.cc').write_text(src, encoding='utf-8')
    shutil.copy2(P18 / 'COPYING', HERE / 'COPYING')
    print('Prepared 19 included missions, 12 conditions and new observable-prefix source.', flush=True)

if __name__ == '__main__':
    main()
