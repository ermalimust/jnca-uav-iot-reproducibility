"""Authorized DeepSeek calls; persist exact non-secret payloads and responses."""
from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import time
import urllib.error
import urllib.request

HERE = Path(__file__).resolve().parent
DEST = HERE / "blind_annotation_run"

def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p, obj): p.write_text(json.dumps(obj, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
def utc(): return datetime.now(timezone.utc).isoformat()

def request_batch(batchno, batch, rubric, protocol, credential):
    prefix = DEST / f"batch_{batchno:02d}"
    payload = {"model": protocol["requested_model"], "temperature": protocol["temperature"],
               "thinking": protocol["thinking"], "response_format": protocol["response_format"],
               "max_tokens": protocol["max_tokens"], "stream": False,
               "messages": [
                   {"role": "system", "content": "You are an independent reader of operational mission requirements. Annotate each mission independently using the rubric. Do not infer an unseen reference answer. Return a JSON object with one field, annotations, containing one annotation per input. The following fixed rubric defines the labels and annotation JSON schema.\n\n"+rubric},
                   {"role": "user", "content": "Annotate every record in this JSON input list. Preserve each audit_id exactly. Return {\"annotations\": [annotation_objects]} only.\n"+json.dumps(batch, ensure_ascii=False)}]}
    raw_path = prefix.with_suffix(".response.json")
    request_path = prefix.with_suffix(".request.json")
    if raw_path.exists():
        assert json.loads(request_path.read_text(encoding="utf-8")) == payload
        response = json.loads(raw_path.read_text(encoding="utf-8"))
    else:
        save(request_path, payload)
        for attempt in range(2):
            start = time.monotonic(); started = utc()
            req = urllib.request.Request(protocol["endpoint"], data=json.dumps(payload).encode("utf-8"),
                    headers={"Authorization": "Bearer "+credential, "Content-Type": "application/json"}, method="POST")
            try:
                with urllib.request.urlopen(req, timeout=180) as r: response = json.loads(r.read().decode("utf-8"))
                save(raw_path, response)
                save(prefix.with_suffix(".receipt.json"), {"started_utc":started, "completed_utc":utc(), "elapsed_seconds":time.monotonic()-start,
                    "request_sha256":sha(request_path), "response_sha256":sha(raw_path), "request_model":protocol["requested_model"],
                    "response_model":response.get("model"), "usage":response.get("usage"), "attempt":attempt+1})
                break
            except urllib.error.HTTPError as exc:
                body = exc.read().decode("utf-8", errors="replace").replace(credential, "[REDACTED]")
                save(prefix.with_suffix(f".error_{attempt+1}.json"), {"http_status":exc.code,"body":body,"time_utc":utc()})
                if exc.code not in (429,500,502,503,504) or attempt: raise RuntimeError(f"DeepSeek request returned HTTP {exc.code}") from None
                time.sleep(5)
            except (urllib.error.URLError, TimeoutError):
                save(prefix.with_suffix(f".error_{attempt+1}.json"), {"error":"transport_or_timeout","time_utc":utc()})
                if attempt: raise RuntimeError("DeepSeek transport/timeout after one retry") from None
                time.sleep(5)
    choice = response["choices"][0]
    assert choice.get("finish_reason") == "stop", (batchno, choice.get("finish_reason"))
    content = choice["message"]["content"]
    annotations = json.loads(content)["annotations"]
    expected = {r["audit_id"]:r for r in batch}
    assert len(annotations) == len(expected) and {a["audit_id"] for a in annotations} == set(expected)
    for a in annotations:
        assert a["profile"] in ("balanced","safety_first","throughput_preserving")
        assert set(a["guards"]) == {"safety","rid","video","energy"}
        assert all(type(v) is bool for v in a["guards"].values())
        assert a["confidence"] in ("high","medium","low")
        assert isinstance(a["plausible_alternatives"], list)
        assert isinstance(a["other_constraints"], list)
    print(json.dumps({"batch":batchno,"records":len(annotations),"model":response.get("model"),"finish_reason":choice.get("finish_reason")}), flush=True)
    return annotations

def main():
    protocol = json.loads((HERE/"blind_annotation_protocol.json").read_text(encoding="utf-8"))
    records = [json.loads(l) for l in (HERE/"blind_annotation_inputs.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    rubric = (HERE/"blind_annotation_rubric.md").read_text(encoding="utf-8").split("Downstream protocol:")[0]
    credential = os.environ.get("DEEPSEEK_API_KEY", "").strip()
    if not credential: raise RuntimeError("Configured DeepSeek credential is unavailable")
    assert len(records) == 48 and all(set(r)=={"audit_id","intent"} for r in records)
    DEST.mkdir(exist_ok=True)
    with ThreadPoolExecutor(max_workers=protocol["http_concurrency"]) as ex:
        jobs = [ex.submit(request_batch, i//12+1, records[i:i+12], rubric, protocol, credential) for i in range(0,48,12)]
        annotations = [a for job in as_completed(jobs) for a in job.result()]
    annotations.sort(key=lambda a:a["audit_id"])
    save(DEST/"annotations_frozen.json", annotations)
    save(DEST/"freeze_receipt.json", {"frozen_utc":utc(), "records":len(annotations), "annotations_sha256":sha(DEST/"annotations_frozen.json"),
         "protocol_sha256":sha(HERE/"blind_annotation_protocol.json"), "input_sha256":sha(HERE/"blind_annotation_inputs.jsonl"),
         "rubric_sha256":sha(HERE/"blind_annotation_rubric.md"), "gold_read_by_generation_script":False,
         "payload_contains_only_rubric_and_opaque_id_text_records":True})
    print(json.dumps({"frozen_records":len(annotations),"annotation_sha256":sha(DEST/"annotations_frozen.json")}), flush=True)

if __name__ == "__main__": main()
