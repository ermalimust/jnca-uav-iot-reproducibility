"""Independent full-source P15 check; imports no project parsing/statistics code.

Deduplication uses one byte of occupancy per observed TCP sequence byte.
Windows use cumulative byte counts and bisected timestamp boundaries.
W1 uses an inverse-CDF coupling with exact integer mass breakpoints.
"""
from pathlib import Path
from datetime import datetime, timezone
from bisect import bisect_left
from collections import Counter
import argparse
import csv
import hashlib
import json
import math
import statistics
import struct
import zipfile
import numpy as np

HERE = Path(__file__).resolve().parent
EXP = HERE.parents[1]
WORK = EXP.parents[2]
P14 = EXP.parent / 'P14_public_arrival_validation_20260913'
RESULTS = EXP / 'video/results'
SOURCE_SHA = '0567854171e615713960e15e0566bc23534633d3bad8ed0a798b74c51da22beb'
MOD = 1 << 32


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def js(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def rows(path):
    with Path(path).open(encoding='utf-8', newline='') as f:
        return list(csv.DictReader(f))


def close(a, b, label, tol=2e-10):
    error = abs(float(a) - float(b))
    assert error <= tol, (label, a, b, error)
    return error


def parse_source(archive):
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        data = z.read('sample_pcaps/parrotar2_modified.pcap')
    endian = {b'\xd4\xc3\xb2\xa1': '<', b'\xa1\xb2\xc3\xd4': '>'}[data[:4]]
    major, minor, zone, accuracy, snap, link = struct.unpack_from(endian + 'HHiIII', data, 4)
    assert (major, minor, link) == (2, 4, 1)
    offset = 24
    index = 0
    packets = []
    controls = []
    captured = []
    target_truncated_headers = []
    while offset != len(data):
        assert offset + 16 <= len(data)
        sec, us, size, original = struct.unpack_from(endian + 'IIII', data, offset)
        offset += 16
        assert us < 1000000 and size <= min(snap, original) and offset + size <= len(data)
        p = memoryview(data)[offset:offset + size]
        offset += size
        timestamp = sec * 1000000 + us
        captured.append(timestamp)
        this_index = index
        index += 1
        if len(p) < 14 or p[12:14] != b'\x08\x00':
            continue
        assert len(p) >= 34
        assert p[14] >> 4 == 4
        ih = (p[14] & 15) * 4
        assert ih >= 20 and len(p) >= 14 + ih
        iplen = int.from_bytes(p[16:18], 'big')
        src, dst = bytes(p[26:30]), bytes(p[30:34])
        direction = 1 if (src, dst) == (b'\xc0\xa8\x01\x01', b'\xc0\xa8\x01\x03') else -1 if (src, dst) == (b'\xc0\xa8\x01\x03', b'\xc0\xa8\x01\x01') else 0
        if not direction:
            continue
        protocol = p[23]
        if protocol not in (6, 17):
            continue
        q = p[14 + ih:]
        if len(q) < 4:
            target_truncated_headers.append(this_index)
            continue
        sp, dp = struct.unpack_from('>HH', q)
        fragment = int.from_bytes(p[20:22], 'big')
        if protocol == 17 and direction == -1 and sp == dp == 5556:
            assert len(q) >= 8 and not (fragment & 0x3fff)
            udp_len = int.from_bytes(q[4:6], 'big')
            assert udp_len >= 8 and iplen == ih + udp_len and original >= 14 + iplen
            controls.append((timestamp, this_index))
        if protocol != 6 or (sp if direction == 1 else dp) != 5555:
            continue
        assert len(q) >= 20 and not (fragment & 0x3fff)
        th = (q[12] >> 4) * 4
        assert th >= 20 and len(q) >= th and iplen >= ih + th and original >= 14 + iplen
        seq, ack = struct.unpack_from('>II', q, 4)
        packets.append(dict(index=this_index, time=timestamp, direction=direction,
                            port=dp if direction == 1 else sp, seq=seq, ack=ack,
                            flags=q[13], length=iplen - ih - th,
                            retained=len(q) - th, original=original, iplen=iplen))
    assert not target_truncated_headers, target_truncated_headers
    packet_reversals = sum(b['time'] < a['time'] for a, b in zip(packets, packets[1:]))
    packets.sort(key=lambda x: (x['time'], x['index']))
    controls.sort()
    facts = dict(capture_records=index, target_tcp_records=len(packets),
                 invalid_target_records=0, target_capture_order_reversals=packet_reversals,
                 target_tied_timestamps=sum(a['time'] == b['time'] for a, b in zip(packets, packets[1:])),
                 capture_all_packet_reversals=sum(b < a for a, b in zip(captured, captured[1:])),
                 control_count=len(controls), control_origin_timestamp_us=controls[0][0],
                 control_end_timestamp_us=controls[-1][0], pcap_sha256=hashlib.sha256(data).hexdigest(),
                 snaplen=snap, pcap_bytes=len(data), complete_pcap_record_walk=True)
    return packets, facts


def bitmap_dedup(packets):
    # Epoch boundaries are inferred independently before any byte deduplication.
    epochs = []
    active = {}
    for p in packets:
        c = active.get(p['port'])
        initial_syn = p['direction'] == -1 and p['flags'] & 2 and not p['flags'] & 16
        if c is None or (initial_syn and c['client_isn'] != p['seq']):
            c = dict(id=len(epochs), port=p['port'], client_isn=p['seq'] if initial_syn else None, packets=[])
            epochs.append(c)
            active[p['port']] = c
        c['packets'].append(p)
    events = []
    connections = []
    for c in epochs:
        stream = c['packets']
        syns = [p for p in stream if p['direction'] == 1 and p['flags'] & 2]
        server_isns = {p['seq'] for p in syns}
        assert len(server_isns) <= 1
        server_isn = next(iter(server_isns)) if server_isns else None
        payloads = [p for p in stream if p['direction'] == 1 and p['length'] > 0]
        assert payloads and not any(p['flags'] & 2 for p in payloads)
        anchor = server_isn + 1 if server_isn is not None else payloads[0]['seq']
        for p in payloads:
            # Resolve relative positions against a fixed epoch anchor, not a moving maximum.
            rel = (p['seq'] - anchor) % MOD
            if rel >= MOD // 2:
                rel -= MOD
            p['position'] = rel
        low = min(p['position'] for p in payloads)
        high = max(p['position'] + p['length'] for p in payloads)
        assert high - low < MOD // 2
        bitmap = bytearray(high - low)
        dup_segments = 0
        partial_segments = 0
        backsteps = 0
        raw_total = 0
        unique_total = 0
        previous_max = 0 if server_isn is not None else None
        for p in payloads:
            a = p['position'] - low
            b = a + p['length']
            unseen = bitmap[a:b].count(0)
            p['independent_new_bytes'] = unseen
            bitmap[a:b] = b'\x01' * (b - a)
            duplicate = p['length'] - unseen
            backsteps += int(previous_max is not None and p['position'] < previous_max)
            previous_max = max(previous_max if previous_max is not None else p['position'], p['position'] + p['length'])
            raw_total += p['length']
            unique_total += unseen
            dup_segments += int(unseen == 0)
            partial_segments += int(0 < unseen < p['length'])
            events.append(dict(connection_id=c['id'], packet_index=p['index'], timestamp_us=p['time'],
                               client_port=c['port'], sequence_raw=p['seq'], sequence_unwrapped=anchor + p['position'],
                               raw_payload_bytes=p['length'], new_payload_bytes=unseen,
                               duplicate_payload_bytes=duplicate, retained_payload_bytes=p['retained']))
        fin_by_direction = {d: [p for p in stream if p['direction'] == d and p['flags'] & 1] for d in [-1, 1]}
        fin_acked = {}
        for direction, fins in fin_by_direction.items():
            fin_acked[direction] = bool(fins) and all(any(
                p['direction'] == -direction and p['flags'] & 16 and p['time'] >= f['time']
                and (p['ack'] - (f['seq'] + f['length'] + 1)) % MOD < MOD // 2
                for p in stream) for f in fins)
        graceful = all(fin_acked.values())
        rst = any(p['flags'] & 4 for p in stream)
        first_rst = min((p['time'] for p in stream if p['flags'] & 4), default=math.inf)
        client_syns = [p for p in stream if p['direction'] == -1 and p['flags'] & 2 and not p['flags'] & 16]
        handshake = False
        if c['client_isn'] is not None and server_isn is not None:
            assert any(p['flags'] & 16 and p['ack'] == (c['client_isn'] + 1) % MOD for p in syns)
            handshake = any(p['direction'] == -1 and p['flags'] & 16 and p['ack'] == (server_isn + 1) % MOD for p in stream)
            assert handshake
        conn = dict(connection_id=c['id'], client_port=c['port'],
                    start_kind='observed_client_SYN' if c['client_isn'] is not None else 'left_censored_no_client_SYN',
                    client_isn=c['client_isn'], server_isn=server_isn,
                    first_timestamp_us=stream[0]['time'], last_timestamp_us=stream[-1]['time'],
                    payload_segments=len(payloads), raw_payload_bytes=raw_total, unique_payload_bytes=unique_total,
                    duplicate_payload_bytes=raw_total - unique_total, fully_duplicate_segments=dup_segments,
                    partly_duplicate_segments=partial_segments, sequence_backsteps=backsteps,
                    fin_seen=any(fin_by_direction.values()), rst_seen=rst,
                    observed_union_bytes=bitmap.count(1), unobserved_sequence_gap_bytes=bitmap.count(0),
                    sequence_span_bytes=len(bitmap), anchor=anchor,
                    wrap_events=sum((anchor + p['position']) // MOD != anchor // MOD for p in payloads),
                    leading_missing_bytes_after_observed_SYN=max(0, low) if server_isn is not None else None,
                    observed_client_SYN_packets=len(client_syns), observed_server_SYN_packets=len(syns),
                    complete_opening_handshake=handshake, client_FIN_packets=len(fin_by_direction[-1]),
                    server_FIN_packets=len(fin_by_direction[1]), client_FIN_acknowledged=fin_acked[-1],
                    server_FIN_acknowledged=fin_acked[1], observed_graceful_close=graceful,
                    terminal_observed=bool(rst or graceful),
                    positive_payload_after_RST=sum(p['length'] > 0 and p['direction'] == 1 and first_rst < p['time'] for p in stream),
                    unique_payload_bytes_after_RST=sum(p.get('independent_new_bytes', 0) for p in stream if first_rst < p['time']),
                    reverse_payload_bytes=sum(p['length'] for p in stream if p['direction'] == -1))
        assert conn['observed_union_bytes'] == unique_total
        connections.append(conn)
    events.sort(key=lambda x: (x['timestamp_us'], x['packet_index']))
    return events, connections


def window_records(events, start, end):
    times = [e['timestamp_us'] for e in events]
    accum = {k: [0] for k in ['new_payload_bytes', 'raw_payload_bytes']}
    for e in events:
        for key in accum:
            accum[key].append(accum[key][-1] + e[key])
    result = []
    for i, a in enumerate(range(start, end - 99999, 100000)):
        b = a + 100000
        left, right = bisect_left(times, a), bisect_left(times, b)
        unique = accum['new_payload_bytes'][right] - accum['new_payload_bytes'][left]
        raw = accum['raw_payload_bytes'][right] - accum['raw_payload_bytes'][left]
        result.append(dict(window_index=i, start_us=a, end_us=b, payload_segments=right - left,
                           unique_payload_bytes=unique, raw_payload_bytes=raw, rate_mbps=unique / 12500))
    return result


def describe(data):
    s = sorted(map(float, data))
    def q(p):
        x = (len(s) - 1) * p
        lower = math.floor(x)
        upper = math.ceil(x)
        return s[lower] + (x - lower) * (s[upper] - s[lower])
    return dict(n=len(s), mean=statistics.fmean(s), median=statistics.median(s),
                p05=q(.05), p95=q(.95), min=s[0], max=s[-1], zero_fraction=s.count(0) / len(s))


def w1_inverse_cdf(a, b):
    a, b = sorted(map(float, a)), sorted(map(float, b))
    n, m = len(a), len(b)
    denominator = math.lcm(n, m)
    step_a, step_b = denominator // n, denominator // m
    i = j = left = 0
    terms = []
    while i < n and j < m:
        right = min((i + 1) * step_a, (j + 1) * step_b)
        terms.append((right - left) * abs(a[i] - b[j]) / denominator)
        left = right
        if right == (i + 1) * step_a:
            i += 1
        if right == (j + 1) * step_b:
            j += 1
    assert left == denominator and i == n and j == m
    return math.fsum(terms)


def compare_rows(expected, actual, name):
    assert len(expected) == len(actual), (name, len(expected), len(actual))
    max_error = 0.
    for i, (e, a) in enumerate(zip(expected, actual)):
        for key, value in e.items():
            if key == 'rate_mbps':
                max_error = max(max_error, close(value, a[key], (name, i, key)))
            elif isinstance(value, bool):
                assert str(value) == a[key], (name, i, key, value, a[key])
            elif value is None:
                assert a[key] == '', (name, i, key)
            elif isinstance(value, int):
                assert value == int(a[key]), (name, i, key, value, a[key])
            else:
                assert value == a[key], (name, i, key, value, a[key])
    return max_error


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--aviator-zip')
    args = parser.parse_args()
    candidates = [Path(args.aviator_zip)] if args.aviator_zip else [P14 / 'cache/uav_datatraces.zip', WORK / 'output/dataset_search_20260913/aviator_uav_datatraces.zip']
    archive = next((p for p in candidates if p.is_file()), None)
    assert archive is not None, 'Supply --aviator-zip PATH, or first run the main analysis with --download-public-source.'
    assert sha(archive) == SOURCE_SHA
    binding = js(EXP / 'video/input_binding.json')
    for path, expected in binding['sha256'].items():
        assert sha(archive if path == 'AVIATOR_public_archive' else WORK / path) == expected, path
    manifest = js(RESULTS / 'result_manifest.json')
    for name, expected in manifest.items():
        assert sha(RESULTS / name) == expected, name
    amendment_check = None
    amendment_dir = EXP / 'video/amendments/001_bound_feature_schema'
    if amendment_dir.exists():
        amendment = js(amendment_dir / 'amendment.json')
        old_binding = js(amendment_dir / 'input_binding.json')
        script_key = (EXP / 'video_analysis.py').relative_to(WORK).as_posix()
        assert sha(amendment_dir / 'video_analysis.py') == old_binding['sha256'][script_key]
        for key, value in old_binding['sha256'].items():
            if key != script_key:
                assert binding['sha256'][key] == value, ('amendment unexpectedly changed another input', key)
        previous_script = (amendment_dir / 'video_analysis.py').read_text(encoding='utf-8')
        current_script = (EXP / 'video_analysis.py').read_text(encoding='utf-8')
        assert previous_script.replace("feature_names=read(P14.parent/'P1_des_recovery_20260912/feature_columns.json')", "feature_names=index['features']") == current_script
        for name, expected in amendment['previous_results_sha256'].items():
            assert sha(RESULTS / name) == expected, ('amendment changed result', name)
        amendment_check = dict(status='PASS', sole_script_change='Feature schema read from already-bound P14 index',
                               other_bound_inputs_unchanged=True, result_files_byte_identical=len(amendment['previous_results_sha256']))
    packets, facts = parse_source(archive)
    events, connections = bitmap_dedup(packets)
    compare_rows(events, rows(RESULTS / 'payload_events.csv'), 'payload_events')
    main_connections = rows(RESULTS / 'connections.csv')
    shared_keys = set(main_connections[0]) - {'closed'}
    compare_rows([{k: c[k] for k in shared_keys} for c in connections], main_connections, 'connections')
    start, stop = facts['control_origin_timestamp_us'], facts['control_end_timestamp_us']
    full = window_records(events, start, stop)
    phase = window_records(events, start + 50000, stop)
    aligned = []
    for b in range(4):
        block = window_records(events, start + b * 90000000 + 20000000, start + (b + 1) * 90000000)
        assert len(block) == 700
        for r in block:
            r['block_index'] = b
        aligned.extend(block)
    for name, obs in [('windows_100ms.csv', full), ('windows_phase50.csv', phase), ('windows_P14_intervals.csv', aligned)]:
        compare_rows(obs, rows(RESULTS / name), name)
    facts.update(dict(payload_segments=len(events), unique_payload_bytes=sum(e['new_payload_bytes'] for e in events),
                      raw_payload_bytes=sum(e['raw_payload_bytes'] for e in events),
                      duplicate_payload_bytes=sum(e['duplicate_payload_bytes'] for e in events),
                      positive_payload_retained_records=sum(e['retained_payload_bytes'] > 0 for e in events),
                      complete_primary_windows=len(full), complete_shifted_windows=len(phase), aligned_observed_windows=len(aligned),
                      observed_unique_bytes_in_primary_windows=sum(r['unique_payload_bytes'] for r in full),
                      observed_unique_bytes_in_aligned_windows=sum(r['unique_payload_bytes'] for r in aligned),
                      independent_recordings=1, model_native_windows=5600, model_arrival_windows=22400))
    for key, value in js(RESULTS / 'source_audit.json').items():
        assert value == facts[key], (key, value, facts[key])
    index = js(P14 / 'results/index.json')
    columns = index['features']
    column = columns.index('video_throughput_mbps')
    assert column == 4
    original_columns = P14.parent / 'P1_des_recovery_20260912/feature_columns.json'
    if original_columns.exists():
        assert js(original_columns) == columns
    runs = js(P14 / 'results/run_index.json')
    native = np.load(P14 / 'results/native_arrays.npz', allow_pickle=False)['features'][:, :, column]
    assert native.shape == (8, 700)
    obs = [r['rate_mbps'] for r in aligned]
    expected_distances = rows(RESULTS / 'reference_distances_by_run.csv')
    per_run = []
    arrival = []
    errors = []
    for i, run in enumerate(runs):
        if run['recording_id'] != 'parrot_ar2':
            continue
        values = np.load(P14 / f'results/run_{i:03}.npz', allow_pickle=False)['features'][:, column]
        assert len(values) == 700
        arrival.append(values)
        selected = obs[run['block_index'] * 700:(run['block_index'] + 1) * 700]
        result = dict(run_index=i, block_index=run['block_index'], scenario_id=run['scenario_id'],
                      W1_mbps=w1_inverse_cdf(selected, values), observed_mean=statistics.fmean(selected),
                      model_mean=statistics.fmean(values))
        expected = expected_distances[len(per_run)]
        for k, v in result.items():
            if isinstance(v, str):
                assert expected[k] == v
            else:
                errors.append(close(v, expected[k], ('run', i, k)))
        per_run.append(result)
    arrival = np.asarray(arrival)
    assert arrival.shape == (32, 700)
    groups = dict(observed_full=[r['rate_mbps'] for r in full], observed_phase50=[r['rate_mbps'] for r in phase],
                  observed_P14_intervals=obs, native_DES_reference=native.ravel(), Parrot_arrival_DES_reference=arrival.ravel())
    with np.load(RESULTS / 'comparison_arrays.npz', allow_pickle=False) as stored:
        for key, values in dict(groups, native_by_scenario=native, arrival_by_run=arrival).items():
            np.testing.assert_allclose(stored[key], values, rtol=0, atol=1e-14)
    all_groups = dict(groups)
    all_groups.update({f'observed_block_{b}': obs[b * 700:(b + 1) * 700] for b in range(4)})
    all_groups.update({s['scenario_id']: native[i] for i, s in enumerate(index['scenarios'])})
    summary = js(RESULTS / 'summary.json')
    descriptives = []
    for expected in summary['descriptives']:
        result = dict(population=expected['population'], **describe(all_groups[expected['population']]))
        for key in result:
            if key != 'population':
                errors.append(close(result[key], expected[key], (expected['population'], key)))
        descriptives.append(result)
    for expected, result in zip(rows(RESULTS / 'descriptives.csv'), descriptives):
        assert expected['population'] == result['population']
        for key in result:
            if key != 'population':
                errors.append(close(result[key], expected[key], ('CSV', result['population'], key)))
    distances = dict(observed_vs_native_W1_mbps=w1_inverse_cdf(obs, native.ravel()),
                     observed_vs_arrival_DES_W1_mbps=w1_inverse_cdf(obs, arrival.ravel()),
                     full_vs_phase50_W1_mbps=w1_inverse_cdf(groups['observed_full'], groups['observed_phase50']))
    for key, value in distances.items():
        errors.append(close(value, summary['distances'][key], key))
    # The pooled gap cannot be interpreted as matched-condition prediction error.
    closure_mislabel = [c['client_port'] for c, m in zip(connections, main_connections)
                       if (m['closed'] == 'True') != c['terminal_observed']]
    findings = [dict(id='scope_output_gap', severity='scope', status='PRESERVE',
        detail='The measured TCP delivery population differs from the offered-load-dependent DES proxy. No same-condition prediction error, posterior equivalence, operational calibration or inferred interference cause follows. Keep the raw gap and unknown source offered-load boundary; do not use received output as source load.'),
        dict(id='single_capture', severity='scope', status='PRESERVE',
        detail='Four contiguous blocks are within one recording, not four independent flights. Phase-shift differences are descriptive sensitivity, not uncertainty intervals. Zero windows and sequence gaps do not identify radio outages or packet-loss rates.')]
    if closure_mislabel:
        readme = (EXP / 'README.md').read_text(encoding='utf-8') if (EXP / 'README.md').exists() else ''
        defined_rst = "`closed` in the connection inventory is the parser's RST-seen state" in readme
        findings.append(dict(id='closed_column_semantics', severity='minor',
            status='RESOLVED_BY_EXPLICIT_README_DEFINITION' if defined_rst else 'REQUIRES_LABEL_CORRECTION',
            ports=closure_mislabel, detail='The primary closed flag equals RST-seen only. README explicitly defines this state; the independent report separately records the complete bidirectional FIN/ACK exchange for port 60747. Frozen primary outputs remain unchanged. This does not affect any deduplicated byte/window/distance result.'))
    report = dict(audit_utc=datetime.now(timezone.utc).isoformat(), numerical_status='PASS',
                  scope='Full independent AVIATOR video source parser, byte bitmap, all event/window records, fixed DES arrays and all W1/descriptive results; no P14 simulator rerun and no manuscript build/reference audit.',
                  methods=dict(parser='Independent classic-PCAP/Ethernet/IPv4/TCP decoder',
                               deduplication='Per-epoch bytearray occupancy with fixed SYN/first-payload anchor',
                               windows='Bisected timestamp boundaries and cumulative integer bytes',
                               W1='Inverse-CDF optimal coupling with integer LCM mass breakpoints; math.fsum',
                               descriptives='statistics.fmean/median and explicitly interpolated sorted quantiles'),
                  source=facts, connections=connections, per_run=per_run, descriptives=descriptives, distances=distances,
                  comparisons=dict(events_exact=len(events), window_rows_exact=len(full) + len(phase) + len(aligned),
                                   connection_rows_exact_except_closed=len(connections), DES_reference_values=5600 + 22400,
                                   per_run_distances=32, pooled_and_phase_distances=3,
                                   max_summary_or_distance_abs_error=max(errors),
                                   input_binding_hashes_verified=len(binding['sha256']), result_hashes_verified=len(manifest)),
                  findings=findings, all_payload_bytes_removed=all(e['retained_payload_bytes'] == 0 for e in events),
                  amendment_001=amendment_check,
                  input_hashes=dict(verifier=sha(__file__), input_binding=sha(EXP / 'video/input_binding.json'),
                                    primary_analysis=sha(EXP / 'video_analysis.py'),
                                    result_manifest=sha(RESULTS / 'result_manifest.json'),
                                    feature_columns=sha(original_columns) if original_columns.exists() else None,
                                    feature_contract=sha(WORK / 'output/dataset_search_20260913/feature_contract.md') if (WORK / 'output/dataset_search_20260913/feature_contract.md').exists() else None,
                                    README=sha(EXP / 'README.md') if (EXP / 'README.md').exists() else None))
    HERE.mkdir(parents=True, exist_ok=True)
    (HERE / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(dict(numerical_status='PASS', comparisons=report['comparisons'], source=facts,
                          distances=distances, closure_mislabel_ports=closure_mislabel), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
