"""Independent header parser for the public Parrot controller-emission prefix."""
from pathlib import Path
from collections import Counter
import hashlib
import json
import struct
import zipfile

HERE = Path(__file__).resolve().parent
WORK = HERE.parents[3]
ARCHIVE = WORK / 'output/dataset_search_20260913/aviator_uav_datatraces.zip'
EXPECTED = '0567854171e615713960e15e0566bc23534633d3bad8ed0a798b74c51da22beb'
data = ARCHIVE.read_bytes()
assert hashlib.sha256(data).hexdigest() == EXPECTED
member = 'sample_pcaps/parrotar2_modified.pcap'
with zipfile.ZipFile(ARCHIVE) as z:
    raw = z.read(member)
assert raw[:4] == b'\xd4\xc3\xb2\xa1'
assert struct.unpack_from('<I', raw, 20)[0] == 1
pos, idx, rows = 24, 0, []
while pos < len(raw):
    sec, micros, caplen, wirelen = struct.unpack_from('<4I', raw, pos)
    pos += 16
    pkt = raw[pos:pos+caplen]
    pos += caplen
    assert len(pkt) == caplen
    if len(pkt) >= 42 and pkt[12:14] == b'\x08\x00' and pkt[23] == 17:
        ihl = (pkt[14] & 15) * 4
        src, dst = pkt[26:30], pkt[30:34]
        if len(pkt) >= 14+ihl+8:
            sport, dport, udplen, _ = struct.unpack_from('>4H', pkt, 14+ihl)
            if (src, dst, sport, dport) == (bytes([192,168,1,3]), bytes([192,168,1,1]), 5556, 5556):
                iplen, frag = struct.unpack_from('>H2xH', pkt, 16)
                assert frag & 0x3fff == 0
                assert iplen == ihl + udplen and udplen >= 8 and wirelen >= 14+iplen
                rows.append((sec*1000000+micros, idx, udplen-8))
    idx += 1
assert pos == len(raw)
rows.sort()
origin = rows[0][0]
prefix = [(t-origin, i, n) for t,i,n in rows if 0 <= t-origin < 20_000_000]
assert prefix

def summary(lo, hi):
    chosen = [(t,i,n) for t,i,n in prefix if lo <= t < hi]
    return {'interval_us': [lo,hi], 'packets': len(chosen), 'payload_bytes': sum(n for t,i,n in chosen),
            'payload_histogram': dict(sorted(Counter(n for t,i,n in chosen).items())),
            'first_relative_us': chosen[0][0], 'last_relative_us': chosen[-1][0],
            'zero_payload_packets': sum(n == 0 for t,i,n in chosen)}

report = {'schema_version': 1, 'archive_sha256': EXPECTED, 'archive_member': member,
          'member_sha256': hashlib.sha256(raw).hexdigest(),
          'flow': {'source': '192.168.1.3:5556', 'destination': '192.168.1.1:5556', 'protocol': 'UDP'},
          'capture_origin_us': origin, 'source_control_packet_count': len(rows),
          'first20': summary(0,20_000_000), 'prefix10': summary(0,10_000_000), 'intervention10': summary(10_000_000,20_000_000),
          'packets_exactly_10s': sum(t == 10_000_000 for t,i,n in prefix),
          'packets_exactly_20s': sum(t-origin == 20_000_000 for t,i,n in rows),
          'tied_timestamps_first20': sum(prefix[i][0] == prefix[i-1][0] for i in range(1,len(prefix))),
          'boundary': 'public controller transport-emission capture timing; payload content absent; lengths from preserved UDP headers',
          'no_ns3_effects_examined': True}
(HERE / 'first20_source_audit.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n',encoding='utf-8')
print(json.dumps(report, ensure_ascii=False, indent=2))
