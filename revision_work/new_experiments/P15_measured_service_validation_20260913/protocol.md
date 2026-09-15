# P15: Direct public service measurements

Protocol written before calculating the new video service statistics, 2026-09-13. The already known source inventory contains 36,042 payload-bearing Parrot video TCP segments. P14 outputs already exist and are treated as fixed reference data.

## AVIATOR video observation

Use the same pinned AVIATOR archive and Parrot member as P14. Select server 192.168.1.1:5555 to controller 192.168.1.3 TCP, retaining client ports as separate connections. Parse both directions to identify connection epochs. Compute TCP payload length from retained IPv4 length and IP/TCP header lengths, excluding SYN/FIN sequence-space bytes and zero-payload acknowledgements. Require complete headers, unfragmented IP and consistent original record length. Stop on unresolved structural errors. Sort by timestamp and original record index; report ties and reversals.

Unwrap TCP sequence numbers per connection. At the first observed arrival of each sequence-byte range, count only bytes absent from earlier observed intervals. Deduplicate over the entire capture before choosing evaluation windows. Keep connection start/censoring flags, sequence gaps and duplicate-byte totals. These are capture observations, not claims about application playback, on-path loss or capture completeness.

For a width of 100,000 microseconds, rate in Mbps is unique captured payload bytes multiplied by 8 / 100,000. Primary descriptive windows use the P14 Parrot control-emission origin and all complete windows ending by the last selected control emission; retain zero-video windows and omit only the incomplete tail. A fixed 50-ms grid shift is a sensitivity check, not an opportunity to select the better result. Report mean, median, p05/p95, min/max and zero-window fraction, plus connection summaries and total reconciliation.

## Fixed DES reference

For a descriptive service-output reference, retain P14's four complete Parrot blocks and each [20,90)-s interval after warm-up, giving 2,800 distinct observed windows. Compare their captured video-flow delivery-rate distribution with (a) all eight fixed native DES scenarios (5,600 windows, equal scenario mass) and (b) all 32 fixed Parrot-arrival DES runs (22,400 windows, equal block/scenario mass). Release every block/scenario result and Wasserstein-1 distance in Mbps. No fitting, rate rescaling, favorable scenario selection or hypothesis tests.

The observed service output and DES throughput proxy share a rate unit but differ in the population and service accounting boundary. Public source offered video load, codec settings and interference conditions are not matched to the DES scenarios. The distances therefore quantify a declared output-reference gap, not prediction error for paired conditions, equivalence, causal effects or full empirical diagnostic validation. Observed throughput is not reused as offered load. No measured/simulated input mixture is passed off as a fully observed 25-dimensional diagnostic vector.

## DUCC assessment

Obtain the official v1.0.1 public release, verify author-published checksums, and inspect actual schema and data. Establish units and direction before computing service statistics. Preserve flight/location/link strata, missingness and timestamps. Its cellular test traffic and 2-Hz aggregates require a separate measurement contract; no automatic mapping to 100-ms Wi-Fi/BLE C2 deadline, backoff or queue features. Inclusion in the manuscript depends on direct relevance and compatibility, not on the numerical direction of a result. Maintain the assessment even if the source is unsuitable for the fixed-diagnostic claim.

## Verification and claim handling

Independently recompute the video sequence-range union and every reported window count/rate; verify all source and result hashes. Preserve pre-existing experimental results and adverse outcomes. No cause accuracy, calibration, action correctness or operational posterior equivalence will be inferred. Any implementation correction after the input binding is created must be recorded with its reason and previous binding rather than silently changing a frozen protocol.
