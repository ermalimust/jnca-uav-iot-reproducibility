"""Independent streaming reconstruction of P18 primary counts from packet CSVs.

Index columns: action,scenario_id,rng_run,ledger_path. ledger_path may be
absolute or relative to --base. Episode metrics are read-only. Only this
script's own directory receives outputs.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
COUNT_COLUMNS = ('c2_offered_count', 'c2_deadline_miss_count', 'video_offered_bytes', 'video_delivered_bytes')
START = 11_000_000_000
END = 21_000_000_000
STOP = 23_000_000_000
DEADLINE = 10_000_000


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_csv(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as handle:
        return list(csv.DictReader(handle))


def reconstruct(path):
    summary = dict.fromkeys(COUNT_COLUMNS, 0)
    partitions = Counter()
    seen = set()
    offered = hashlib.sha256()
    count = 0
    opener = gzip.open if path.suffix == '.gz' else Path.open
    with opener(path, 'rt', encoding='utf-8-sig', newline='') as handle:
        for raw in csv.DictReader(handle):
            row = {name: int(value) for name, value in raw.items()}
            identity, flow, size = row['packet_id'], row['flow'], row['payload_bytes']
            offer, send, receive = row['offer_ns'], row['send_ns'], row['receive_ns']
            shape, socket = row['shaper_drop_ns'], row['socket_fail_ns']
            assert identity not in seen and flow in (0, 1, 2) and size > 0
            seen.add(identity)
            assert 1_000_000_000 <= offer < END
            assert (send == -1 or offer <= send <= STOP)
            assert (receive == -1 or send >= 0 and send <= receive <= STOP)
            assert (shape == -1 or offer <= shape <= STOP)
            assert (socket == -1 or offer <= socket <= STOP)
            if shape >= 0:
                assert flow == 1 and send == socket == receive == -1
            if socket >= 0:
                assert send == receive == shape == -1
            offered.update(f'{identity},{flow},{offer},{size}\n'.encode('ascii'))
            count += 1
            if not START <= offer < END:
                continue
            if flow == 0:
                summary['c2_offered_count'] += 1
                summary['c2_deadline_miss_count'] += (receive < 0 or receive - offer > DEADLINE)
            elif flow == 1:
                summary['video_offered_bytes'] += size
                if receive >= 0:
                    summary['video_delivered_bytes'] += size
                    terminal = 'received'
                elif shape >= 0:
                    terminal = 'shaper_drop'
                elif socket >= 0:
                    terminal = 'socket_reject'
                elif send >= 0:
                    terminal = 'network_non_delivery_by_horizon'
                else:
                    terminal = 'shaper_pending_at_horizon'
                partitions[terminal] += size
    assert summary['video_offered_bytes'] == sum(partitions.values())
    assert summary['video_delivered_bytes'] == partitions['received']
    assert summary['c2_offered_count'] > 0 and summary['video_offered_bytes'] > 0
    summary['c2_deadline_miss_fraction'] = summary['c2_deadline_miss_count'] / summary['c2_offered_count']
    summary['video_delivery_fraction'] = summary['video_delivered_bytes'] / summary['video_offered_bytes']
    return summary, partitions, count, offered.hexdigest()


def key(row):
    return row['action'], row['scenario_id'], int(row['rng_run'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--index', type=Path, required=True)
    parser.add_argument('--episodes', type=Path, required=True)
    parser.add_argument('--base', type=Path, default=HERE.parent)
    args = parser.parse_args()
    if args.index.suffix == '.json':
        receipt = json.loads(args.index.read_text(encoding='utf-8-sig'))
        assert receipt['status'] == 'PASS' and receipt['formal_runs'] == 640
        index = []
        for entry in receipt['runs']:
            assert entry['engineering'] is False and entry['exit_code'] == 0
            folder = f"runs/{entry['scenario_id']}_r{int(entry['rng_run']):04d}_{entry['action']}"
            index.append({'action': entry['action'], 'scenario_id': entry['scenario_id'], 'rng_run': entry['rng_run'],
                          'ledger_path': folder + '/packets.csv.gz', 'expected_compressed_sha256': entry['files']['packets.csv.gz']})
    else:
        index = read_csv(args.index)
    episode_rows = read_csv(args.episodes)
    assert len(index) == len(episode_rows) == 640
    metrics = {key(r): r for r in episode_rows}
    assert len(metrics) == 640 and len({key(r) for r in index}) == 640
    assert {key(r) for r in index} == set(metrics)
    output, inputs = [], []
    packet_rows = 0
    offer_hashes = {}
    max_rate_error = 0.0
    for i, entry in enumerate(index):
        path = Path(entry['ledger_path'])
        if not path.is_absolute():
            path = args.base / path
        file_hash = sha(path)
        if entry.get('expected_compressed_sha256'):
            assert file_hash == entry['expected_compressed_sha256']
        reconstructed, partitions, n, offer_hash = reconstruct(path)
        expected = metrics[key(entry)]
        for name in COUNT_COLUMNS:
            assert reconstructed[name] == int(expected[name]), (key(entry), name, reconstructed[name], expected[name])
        for name in ('c2_deadline_miss_fraction', 'video_delivery_fraction'):
            error = abs(reconstructed[name] - float(expected[name]))
            max_rate_error = max(max_rate_error, error)
            assert error <= 1e-12, (key(entry), name, error)
        group = (entry['scenario_id'], int(entry['rng_run']))
        offer_hashes.setdefault(group, set()).add(offer_hash)
        output.append({'action': entry['action'], 'scenario_id': entry['scenario_id'],
                       'rng_run': int(entry['rng_run']), **reconstructed,
                       'video_shaper_drop_bytes': partitions['shaper_drop'],
                       'video_socket_reject_bytes': partitions['socket_reject'],
                       'video_network_horizon_bytes': partitions['network_non_delivery_by_horizon'],
                       'video_shaper_pending_bytes': partitions['shaper_pending_at_horizon'],
                       'all_flow_packet_rows': n, 'offer_schedule_sha256': offer_hash})
        inputs.append({'action': entry['action'], 'scenario_id': entry['scenario_id'],
                       'rng_run': int(entry['rng_run']), 'ledger_path': entry['ledger_path'], 'sha256': file_hash})
        packet_rows += n
        if (i + 1) % 80 == 0:
            print(f'Independently reconstructed {i + 1}/640 packet ledgers', flush=True)
    assert all(len(values) == 1 for values in offer_hashes.values())
    for name, rows in [('packet_reconstructed_metrics.csv', output), ('packet_input_hashes.csv', inputs)]:
        with (HERE / name).open('w', encoding='utf-8', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    result = {'all_passed': True, 'episode_ledgers': 640, 'packet_rows': packet_rows,
              'count_cells_compared': 640 * len(COUNT_COLUMNS), 'rate_cells_compared': 1280,
              'max_rate_abs_error': max_rate_error, 'all_cross_action_offered_schedule_hashes_equal': True,
              'all_video_evaluation_cohort_byte_partitions_conserved': True,
              'primary_cohort_offer_ns': [START, END], 'c2_deadline_ns': DEADLINE,
              'video_receive_horizon_ns': STOP,
              'scope': 'Independent streaming packet-ledger parsing; no import of primary analysis code. The first receive timestamp identifies unique delivery. Original offered denominator includes shaper drops, socket rejection and non-delivery by the horizon.',
              'input_hashes': {'index': sha(args.index), 'episodes': sha(args.episodes)},
              'script_sha256': sha(__file__)}
    (HERE / 'packet_verification_report.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
