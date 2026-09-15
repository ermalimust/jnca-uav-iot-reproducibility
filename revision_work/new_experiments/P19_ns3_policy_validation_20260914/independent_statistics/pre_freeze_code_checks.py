"""Read-only main-module checks on synthetic inputs; no study outcomes loaded."""
import sys
sys.dont_write_bytecode = True
from pathlib import Path
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
import hashlib
import json
from fractions import Fraction
import numpy as np
import policy_model as main_model
import analyze_results as main_analysis
from exact_paired_inference import exact_sign_flip, holm_bh


def objective_derivatives(X, y, beta, C):
    """Independent loops/Kronecker Hessian and log-sum-exp objective."""
    n, d = X.shape
    k = len(beta)
    logits = np.column_stack((X @ beta.T, np.zeros(n)))
    maxima = logits.max(axis=1)
    lognormalizer = maxima + np.log(np.exp(logits - maxima[:, None]).sum(axis=1))
    probs = np.exp(logits - lognormalizer[:, None])
    penalty = beta.copy()
    penalty[:, -1] = 0
    loss = float(np.mean(lognormalizer - logits[np.arange(n), y]) + np.sum(penalty**2) / (2*C*n))
    gradient = penalty / (C*n)
    hessian = np.zeros((k*d, k*d))
    for index in range(n):
        residual = probs[index, :k].copy()
        if y[index] < k:
            residual[y[index]] -= 1
        gradient += np.outer(residual, X[index]) / n
        covariance = np.diag(probs[index, :k]) - np.outer(probs[index, :k], probs[index, :k])
        hessian += np.kron(covariance, np.outer(X[index], X[index])) / n
    ridge = np.ones_like(beta) / (C*n)
    ridge[:, -1] = 0
    hessian += np.diag(ridge.ravel())
    return loss, gradient, hessian, probs


def main():
    rng = np.random.default_rng(609141921)
    # Derivative fixture is distinct from the main engineering fixture.
    X = np.column_stack((rng.normal(size=(36, 3)), np.ones(36)))
    y = np.arange(36) % 12
    beta = rng.normal(scale=.2, size=(11, 4))
    loss, gradient, hessian, probs = objective_derivatives(X, y, beta, .7)
    eps = 1e-5
    numerical_gradient = np.empty(beta.size)
    numerical_hessian = np.empty_like(hessian)
    for index in range(beta.size):
        offset = np.zeros(beta.size)
        offset[index] = eps
        upper = objective_derivatives(X, y, beta + offset.reshape(beta.shape), .7)
        lower = objective_derivatives(X, y, beta - offset.reshape(beta.shape), .7)
        numerical_gradient[index] = (upper[0] - lower[0]) / (2*eps)
        numerical_hessian[:, index] = (upper[1].ravel() - lower[1].ravel()) / (2*eps)
    gradient_error = float(np.max(np.abs(gradient.ravel() - numerical_gradient)))
    hessian_error = float(np.max(np.abs(hessian - numerical_hessian)))
    assert gradient_error < 1e-8 and hessian_error < 1e-8
    covariance = -probs[:, :11, None] * probs[:, None, :11]
    ix = np.arange(11)
    covariance[:, ix, ix] += probs[:, :11]
    root_hessian = np.einsum('ni,nj,nkl->kilj', X, X, covariance, optimize=True).reshape(44, 44) / len(X)
    ridge = np.ones((11, 4)) / (.7*len(X)); ridge[:, -1] = 0
    root_hessian.flat[::45] += ridge.ravel()
    assert np.max(np.abs(root_hessian-hessian)) < 1e-13
    fixtures = []
    for setting in ('random', 'separable', 'constant'):
        values = rng.normal(size=(192, 22))
        labels = np.arange(192) % 12
        if setting == 'separable':
            values[:, :12] += 5*np.eye(12)[labels]
        if setting == 'constant':
            values[:] = 0
        values = np.column_stack((values, np.ones(192)))
        for C in (.1, 1., 10.):
            coefficients, history = main_model.fit_softmax(values, labels, C)
            last_loss, last_gradient, _, final_probs = objective_derivatives(values, labels, coefficients[:-1], C)
            residual = float(np.max(np.abs(last_gradient)))
            assert residual < 1e-9
            assert np.allclose(coefficients[-1], 0, atol=0, rtol=0)
            assert np.max(np.abs(final_probs-main_model.predict_prob(values, coefficients))) < 1e-14
            assert all(b['objective'] <= a['objective'] + 1e-14 for a,b in zip(history,history[1:]))
            fixtures.append({'fixture': setting, 'C': C, 'iterations': len(history), 'independent_gradient_max': residual, 'objective': last_loss})
    rows = [{f: float(i+j) for j,f in enumerate(main_model.FEATURES)} for i in range(4)]
    for r in rows:
        r[main_model.FEATURES[0]] = None
        r[main_model.FEATURES[1]] = 5.
    rows[0][main_model.FEATURES[2]] = None
    rows[1][main_model.FEATURES[3]] = float('inf')
    standardized, means, scales = main_model.imputed_matrix(rows)
    assert np.isfinite(standardized).all() and means[0] == 0 and scales[0] == scales[1] == 1
    assert np.max(np.abs(standardized[:,:-1].mean(axis=0))) < 1e-13
    altered = [{f: 1000. for f in main_model.FEATURES}]
    transformed, held_means, held_scales = main_model.imputed_matrix(altered, means.copy(), scales.copy())
    assert np.array_equal(held_means, means) and np.array_equal(held_scales, scales)
    assert np.allclose(transformed[0,:-1], (1000.-means)/scales)
    exact_cases = [[Fraction(0)]*32, [Fraction(1)]*32, [Fraction((-1)**i) for i in range(32)], [Fraction(int(v), 399*19) for v in rng.integers(-100, 101, 32)]]
    exact_checks = []
    for values in exact_cases:
        independent = exact_sign_flip(values)
        tail, assignments, p = main_analysis.exact_tail(values)
        assert (tail, assignments, p) == (independent['tail_count'], independent['assignments'], independent['p_raw'])
        exact_checks.append({'tail_count': tail, 'assignments': assignments, 'p_raw': p})
    probabilities = [Fraction(int(v), 2**32) for v in rng.integers(0, 2**32+1, 113)]
    independent_adjusted = holm_bh(probabilities)
    for name in ('holm','bh'):
        reference = np.array([float(x) for x in independent_adjusted[name]])
        calculated = main_analysis.adjustment([float(x) for x in probabilities], name)
        assert np.max(np.abs(reference-calculated)) < 1e-14
    missions = json.loads((ROOT/'inputs/missions_included.json').read_text(encoding='utf-8'))
    alpha = []
    for m in missions:
        safety, throughput = [Fraction(str(m['cost_weights'][k])) for k in ('safety','throughput')]
        assert safety >= 0 and throughput >= 0 and safety+throughput > 0
        assert m['guards']['rid'] is False and m['guards']['energy'] is False
        alpha.append({'mission_id':m['mission_id'],'alpha_c2':str(safety/(safety+throughput))})
    assert len(missions) == len({m['mission_id'] for m in missions}) == 19
    result = {'status':'PASS', 'scope':'Read-only source review and synthetic arithmetic fixtures; no TRAIN/VAL/TEST observations or service outcomes read.',
              'blockers':[], 'source_sha256':{name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in ('policy_model.py','analyze_results.py','common.py','protocol.json')},
              'derivatives':{'finite_difference_gradient_error':gradient_error,'finite_difference_hessian_error':hessian_error,'hessian_flattening':'matches independent Kronecker sum'},
              'softmax_fixtures':fixtures,'imputation':'PASS, TRAIN parameters reused without validation/test recomputation',
              'exact32':exact_checks,'multiplicity_113':'PASS versus exact rational Holm/BH','mission_weights':alpha,
              'source_review':['12 classes use final-class reference parameterization, unpenalized intercept, stated ridge/(C*n).',
               'One TRAIN Observe prefix per state/run; five TRAIN arms enter service means only.',
               'VAL selects minimum joint log loss across frozen C grid; ties ascending C.',
               'C2 miss, video delivery and mission-specific weighted service loss averaged over repetitions, 19 missions and 12 scenarios inside each of 32 RNG blocks.',
               'All 12 primary tests retained in 113 pool; old 101 row cells preserved verbatim.',
               'Bootstrap common 10000 x 32 index matrix, seed 609141919, linear percentiles.',
               'Hindsight finite-bank minima are descriptive and separate from deployable full_service.'],
              'nonblocking_notes':['Feature receive totals/IAT use [1,10.99), while 99 complete 100 ms bin metrics use [1,10.9); both are pre-command observations.',
               'Formal independent verifier will enforce complete key grids and rederive service alpha from archived metadata; the primary main reducer presently asserts counts but not every grid key.']}
    (HERE/'pre_freeze_code_audit.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'status':result['status'],'softmax_fixtures':len(fixtures),'maximum_gradient_residual':max(r['independent_gradient_max'] for r in fixtures),'exact32_cases':len(exact_checks),'source_sha256':result['source_sha256']},indent=2))


if __name__ == '__main__':
    main()
