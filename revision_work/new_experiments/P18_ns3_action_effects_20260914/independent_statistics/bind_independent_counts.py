"""Bind the separate protocol-audit packet counts to the frozen primary schema.

This adapter changes only column names and derives ratios from integers. It
does not use the primary analysis or select any episodes.
"""
from pathlib import Path
import csv
import hashlib
import json

HERE = Path(__file__).resolve().parent
P18 = HERE.parent
SOURCE = P18 / 'independent_design/formal_independent_episode_counts.csv'
AUDIT = P18 / 'independent_design/formal_independent_audit.json'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    audit = json.loads(AUDIT.read_text(encoding='utf-8-sig'))
    assert audit['status'] == 'PASS' and audit['runs'] == 640
    assert audit['root_receipt_sha256'] == sha(P18 / 'execution_receipt.json')
    assert all(sha(P18 / name) == value for name, value in audit['source_hashes'].items())
    with SOURCE.open(encoding='utf-8-sig', newline='') as handle:
        source_rows = list(csv.DictReader(handle))
    assert len(source_rows) == 640
    rows = []
    for raw in source_rows:
        scenario = raw['scenario_id']
        record = {key: raw[key] for key in ('scenario_id', 'rng_run', 'action', 'c2_offered_count',
                                           'c2_deadline_miss_count', 'video_offered_bytes')}
        record.update(w=int(scenario[1]), m=int(scenario[3]), v=int(scenario[5]),
                      video_delivered_bytes=int(raw['video_received_bytes']))
        record['c2_deadline_miss_fraction'] = int(raw['c2_deadline_miss_count']) / int(raw['c2_offered_count'])
        record['video_delivery_fraction'] = int(raw['video_received_bytes']) / int(raw['video_offered_bytes'])
        rows.append(record)
    target = HERE / 'independent_episode_metrics.csv'
    with target.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    receipt = {'status': 'PASS', 'episodes': 640, 'source_packet_rows': audit['packet_rows_checked'],
               'scope': 'Primary count schema derived from the independent protocol-audit parser; no main reducer imported; every episode retained.',
               'source_sha256': sha(SOURCE), 'source_audit_sha256': sha(AUDIT),
               'execution_receipt_sha256': sha(P18 / 'execution_receipt.json'),
               'bound_metrics_sha256': sha(target), 'script_sha256': sha(Path(__file__))}
    (HERE / 'independent_count_binding_receipt.json').write_text(json.dumps(receipt, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    main()
