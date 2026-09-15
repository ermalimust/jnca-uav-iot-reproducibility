"""Read-only provenance and pre-outcome mission/action-schedule checks."""
from pathlib import Path
import csv
import hashlib
import json

OUT = Path(__file__).resolve().parent
P = OUT.parent
W = P.parents[2]


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8'))


def csvrows(p):
    with Path(p).open(encoding='utf-8', newline='') as f:
        return list(csv.DictReader(f))


binding = read(P/'inputs/source_binding.json')
sources = binding['archived_files'] + binding['generation_records']
for r in sources:
    assert sha(W/r['source']) == r['sha256'] == sha(P/r['copy'])
missions = [json.loads(s) for s in (P/'inputs/ood_mission_intents.jsonl').read_text(encoding='utf-8').splitlines() if s.strip()]
assert len(missions) == 48
included = [m for m in missions if not m['guards'].get('rid', False) and not m['guards'].get('energy', False)]
assert len(included) == 19 and read(P/'inputs/missions_included.json') == included
records = csvrows(P/'inputs/mission_inclusion.csv')
assert len(records) == 48
for m, r in zip(missions, records):
    assert r['mission_id'] == m['mission_id'] and r['intent'] == m['intent']
    assert r['included'] == str(m in included)
    assert r['exclusion_reason'] == ','.join(n for n in ('rid', 'energy') if m['guards'].get(n, False))
    assert float(r['alpha_c2']) == m['cost_weights']['safety']/(m['cost_weights']['safety']+m['cost_weights']['throughput'])
proto = read(P/'protocol.json')
c2 = csvrows(P/'inputs/c2_source.csv')
assert len(c2) == 800
traffic = {}
for sc in proto['scenarios']:
    offers = csvrows(P/sc['offer_path'])
    assert [int(r['packet_id']) for r in offers] == list(range(len(offers)))
    c = [r for r in offers if r['flow'] == '0']
    assert [(r['relative_us'], r['payload_bytes']) for r in c] == [(r['relative_us'], r['payload_bytes']) for r in c2]
    assert all(0 <= int(r['relative_us']) < 20_000_000 for r in offers)
    assert all(int(r['payload_bytes']) > 0 for r in offers)
    w = [r for r in offers if r['flow'] == '2']
    assert bool(w) == bool(sc['w'])
    video = [r for r in offers if r['flow'] == '1']
    assert sum(int(r['payload_bytes']) for r in video if int(r['relative_us']) >= 10_000_000) == (7_500_000 if sc['v'] else 1_500_000)
    key = sc['w'], sc['v']
    if key in traffic:
        assert offers == traffic[key]
    else:
        traffic[key] = offers
report = {'status': 'PASS', 'archived_file_copies_checked': len(sources), 'original_missions': 48,
          'included_missions': 19, 'inclusion_rule_exact': 'rid=false AND energy=false; whole objects and weights unchanged',
          'public_c2_emissions_per_schedule': 800, 'scenario_schedules_checked': 12,
          'motion_changes_no_traffic_schedule': True, 'same_control_emission_times_and_lengths_all_scenarios': True,
          'method_independent_video_offered_denominators_checked': True,
          'design_binding_sha256': sha(P/'design_binding.json'), 'source_binding_sha256': sha(P/'inputs/source_binding.json')}
(OUT/'source_scope_audit.json').write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
print(json.dumps(report))
