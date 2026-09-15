# P16: posterior-error robustness of the frozen decision pipeline

Protocol written before computing P16 results, 2026-09-13.

## Question and fixed inputs

How much simultaneous absolute error in the four marginal probabilities can be certified to preserve a recorded final action, including posterior routing, candidate exposure, admission and cost ranking? The four coordinates are independent domain coordinates in [0,1]^4; no simplex normalization is imposed.

Use every one of the 54,000 original guarded audit records, all 30 saved Qwen policies, the archived routing/selection implementation and its mission costs, guard flags and overheads. Replay the saved six-decimal probabilities, not regenerated predictions. No model API, new diagnostic fit, field claim or changes to the original method. Bind all input and analysis files with SHA256 before the primary computation. Preserve any subsequent amendment and its reason.

## Primary analysis

For each saved q and absolute radius epsilon, consider the entire continuous clipped box [max(0,q-epsilon),min(1,q+epsilon)]. Integer-scaled endpoints and linear forms make comparisons exact for epsilon on the 10^-6 grid.

Conservatively enumerate every router branch that may intersect the box. For each such branch, prove either that the original action is always admissible and beats every possibly admissible competitor, with the saved first-candidate tie rule; or that every candidate is always rejected and the fallback/escalation logic always selects the original action. A box is certified only when every enumerated branch passes. Router overapproximation and independent interval bounds can reject a box even when the action is in fact stable. Therefore this is a sufficient certificate, not an exact minimum action-change distance.

Report counts/fractions at epsilon = 0.01, 0.025, 0.05 and 0.10 (absolute probability units); these settings are fixed before results. Report the largest radius supported by this monotone certificate on the 10^-6 grid, found by integer bisection. Also report sufficient certificates for unchanged routing, candidate list and all six admission outcomes, making internal changes distinct from final-action changes.

## Counterexample search and checks

Use all 16 sign corners and eight signed coordinate directions; check radii 0.01,0.025,0.05,0.10,0.20,0.40,1.0 and retain the nearest observed changed action. Refine a found directional bracket by integer bisection. A found point gives an upper bound on the true nearest change. Failure to find a change gives no safety guarantee. Every reported witness must be replayed through the complete policy and stay within its reported error norm. Search outcomes and certificates are descriptive, not additional significance tests.

Verify all nominal records, all reported certificate endpoints, all witnesses, deterministic boundary/duplicate/fallback examples, and independently check a deterministic sample with a separately implemented scalar interval certificate. Random probes supplement the analytic proof and cannot establish it. Preserve outputs for all missions and decisions, including fragile cases. A decision-stability certificate does not establish posterior calibration, diagnostic truth or physical safety; distribution-level Wasserstein distances are not used as pointwise error bounds.

## Reviewer mapping

- R1 R2 / R1 C6: quantify decision consequences of bounded posterior error; original request for operational posterior distributions remains separately assessed.
- R1 C1 / R1 R1: four marginal coordinates and sensitivity of the defined decision rule.
- R3 C1: complementary robustness evidence within the specified model; an independent protocol environment is a separate feasibility task.

No acceptance probability, population prevalence or exact-radius claim will be inferred from these archived decision frequencies.
