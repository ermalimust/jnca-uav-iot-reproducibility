# Public measured transport arrivals and fixed-model sensitivity

This protocol addresses the public-data-only constraint and the external-validity concern in R1 C6/R2, R2 M3, R3 C1 and R4 M1. It specifies a measured-input comparison and a measurement-driven simulation. It does not designate generated service features or generated diagnostic probabilities as empirical deployment observations.

## Sources and eligibility

Use the three real-device samples in the authors' AVIATOR `uav_datatraces.zip`, not the outputs of their traffic generator. The archive does not identify each of these samples as indoor or outdoor; call them real-device recordings, not three flight campaigns. The separately released random-flight PPI capture has no inner packet headers and cannot identify the target traffic, so it is retained in the source audit but excluded from the transport-flow study.

Selected IPv4/UDP tuples are fixed before comparison:

- DJI Mavic Air: `192.168.2.20:10002 -> 192.168.2.1:9003`.
- DJI Spark: `192.168.2.20:10002 -> 192.168.2.1:9003`.
- Parrot AR 2.0: `192.168.1.3:5556 -> 192.168.1.1:5556`.

They select controller-originating, control-related UDP streams. DJI endpoint roles are inferred from the paper's capture setup and traffic asymmetry; the Parrot destination port is also supported by its documented AT-command interface. Payload-stripped records do not establish that each datagram contains one flight-control command. Exclude unrelated discovery/background flows and the Parrot TCP video/ACK traffic. UDP/IP headers and original lengths are inspected even though payloads were removed. Do not infer video labels from packet size or use recorded-length bytes as original packet size.

Use integer microsecond timestamps. Keep every qualifying datagram and tied timestamp, with stable record-order tie breaking. Log any capture-order reversal; sort by timestamp to recover the recorded chronology. Do not deduplicate similar headers, trim interarrival tails, smooth, fit, rate-normalize, loop, interpolate, or synthesize the recorded emissions. Do not add a first interarrival of zero. These are RC transport capture times, not UAV receipt times, internal command-generation timestamps, network delay, MAC enqueue times, or measured service-start events.

## A. Directly measured input comparison

For each of the three selected recordings, describe all adjacent transport interarrival intervals and all complete 100 ms count windows anchored at the first qualifying emission. A complete window ends no later than the final qualifying emission; a remaining partial tail is excluded by that fixed rule. Include zero-count windows. Report original/selected record counts, time extent, complete-window count, mean, median and 95th percentile. Linear-interpolated quantiles are descriptive.

The unchanged reference is the arrival process of all eight original held-out DES scenarios at their archived 90-second runtime. Its evaluation interval is [20,90) seconds, preserving the original warm-up. For reference interarrivals, use adjacent arrivals within this interval. Use its complete 100 ms count windows. The DES period is 20, 35 or 50 ms depending on the scenario family, with its original jitter; no fit is made to the measurements.

Compute exact one-dimensional Wasserstein-1 distances for interarrival milliseconds and counts per 100 ms. Give equal probability mass to each measured recording and, separately, each held-out DES scenario; distribute that mass uniformly over its observations. Report the balanced pooled comparison, every recording-versus-balanced-DES comparison, and all recording/scenario pairs. These weights describe a declared comparison population; they do not establish matching flight conditions. Also report count-window results at a predeclared 50 ms grid-phase offset for both populations, with only complete windows retained. No phase is selected according to the result.

There is one released sample per UAV model. Do not treat packets, adjacent windows or extracted blocks as independent field campaigns. Do not report a p-value, field-population confidence interval or distribution-equivalence claim from these three samples. No result enters the existing 93-test inventory because no new hypothesis test is conducted.

## B. Measured-arrival-driven DES sensitivity

Anchor each recording at its first qualifying emission and partition it into every complete, contiguous, nonoverlapping 90-second block. All blocks and windows are half-open [start,end); a complete block ends no later than the last qualifying emission. Preserve all observed relative emission times within each block, without rate changes or gap filling. Disclose the discarded partial tail and the per-block 20-second simulation warm-up. Each block starts a new simulated queue at the original initial state; this is not reconstruction of a real queue at the corresponding recording time.

Cross every block with all eight held-out DES scenarios. Retain each scenario identifier, seed, radio and interference parameters, video process, service/retry/backoff functions, 100 ms feature definitions and original family-dependent queue-proxy period. Replace the C2 arrival sequence only. Each simulated packet has a new consecutive local ID; the service draws use the original deterministic seed/packet/attempt scheme. The trace's UDP payload length can be recorded in the arrival metadata, but the original service routine does not use packet size, so no size-dependent physical effect is claimed.

Keep all 700 post-warm-up windows from each run, including empty-arrival windows and windows that the historical generator labels ambiguous. Do not use outcome labels to select the new or reference distribution. No cause labels, calibration scores, realized invalidity or regret are computed for this experiment. The new service observations and q values are model outputs under recorded transport arrivals. Replacing the source process changes average rate, burst structure and the original relationship between cause family and arrival rate; it does not isolate burstiness alone.

Load `diagnostic_model.npz` directly and retain its feature order, training mean/std, coefficients and biases; clip each logit to [-40,40] before applying its independent sigmoid, in W/B/M/V order. Do not invoke a helper that trains a model; do not calibrate or normalize q to sum to one. Compare its outputs to the unchanged native DES baseline on the same scenario and evaluation-window keys.

Report each cause's marginal W1 and a joint 4D sliced-W1 diagnostic using 64 unit-L2 Gaussian projection directions generated once with NumPy PCG64 seed 20260913 and saved in the release. Joint sliced-W1 is the mean of projected exact W1 distances, not a claim about full 4D Wasserstein equality. Report all per-record/scenario/block results. For pooled outputs, each recording has mass 1/3, each of its blocks equal mass, each scenario equal mass, and every window equal conditional mass. The native baseline gives each scenario equal mass. A weighted average of stratum distances is labeled separately from a distance between pooled distributions.

To locate the diagnostic shift in the decision interface, use the saved 30 mission policies, their original costs and guards, and full-precision q. For every paired native/trace-driven scenario-window, recompute q-based routing, candidate exposure, guard admission and selection using the unchanged functions. Report selected-action change and routing-change frequencies, plus selected-action shares. This is action sensitivity, not correctness or gain. Within each mission preserve its original family weights, with all original positive-weight families required to be present; give equal weight to the 30 missions, equal weight to the three recordings, and equal weight to blocks within each recording. Export per-mission/per-record values and the unweighted crossed-scenario results for inspection.

## Verification before and after running

Bind the public archive, source descriptions, frozen generator/config/model/feature list and saved policies by SHA256 before the first distribution or sensitivity calculation. Preserve the earlier experiments. First regenerate the eight native factual traces and require exact equality of every one of their 25 feature values with the archived factual windows. Recompute the fixed q values and verify the original recorded 54,000 q/routing/candidate/selected-action records. Archived q values are rounded to six decimals: require equality after the same rounding and report the maximum unrounded discrepancy. Routing and selection use the full-precision model outputs. Stop on a mismatch.

The optimized summarizer may pass only the exact arrival cohort into the unchanged source function; verify that this reproduces the native reference. After running, independently check source-flow extraction and window counts, native reproduction, every new feature-to-q calculation, pooling weights, W1 computations, changed-action summaries, and the absence of empirical-posterior/accuracy claims. Preserve all outcomes, including large shifts. Freeze results and export complete input bindings, arrays, tables and verification receipts.

## Writing boundary

Use the direct comparison to quantify the relationship between the declared DES source process and the public recordings. Use the second experiment to quantify how the original model's diagnostic and decision interface respond to those measured source timings. Retain the distinction between measured transport inputs, simulated service processes and empirical deployment posteriors. R1 Recommendation 2 gains measured-input and propagation evidence but its requested full operational-posterior match is not labeled established.
