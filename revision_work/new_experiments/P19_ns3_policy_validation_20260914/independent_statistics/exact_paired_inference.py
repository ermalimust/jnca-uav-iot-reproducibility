"""Independent exact paired inference for at most 32 complete RNG blocks.

Rational input avoids floating-point ambiguity at sign-flip equality. Every
2**n sign assignment is counted through meet-in-the-middle. The module does
not read P18/P19 outcomes or choose hypotheses, samples or endpoints.
"""
from __future__ import annotations

from bisect import bisect_left, bisect_right
from fractions import Fraction
from functools import reduce
from itertools import product
from math import gcd, lcm
from pathlib import Path
import hashlib
import json
import random


def as_fraction(value):
    return value if isinstance(value, Fraction) else Fraction(str(value))


def integer_weights(values):
    rational = [as_fraction(value) for value in values]
    assert 1 <= len(rational) <= 32
    common = lcm(*(value.denominator for value in rational))
    weights = [value.numerator * (common // value.denominator) for value in rational]
    divisor = reduce(gcd, (abs(value) for value in weights), 0)
    if divisor > 1:
        weights = [value // divisor for value in weights]
    return weights


def signed_sums(weights):
    sums = [0]
    for weight in weights:
        sums = [value - weight for value in sums] + [value + weight for value in sums]
    return sums


def monotone_tail_counts(left, right, threshold):
    """A second tail count with monotone pointers, not binary search."""
    first = sorted(left)
    second = sorted(right)
    upper = 0
    index = len(second)
    for value in first:
        while index > 0 and value + second[index - 1] >= threshold:
            index -= 1
        upper += len(second) - index
    lower = 0
    index = len(second)
    for value in first:
        while index > 0 and value + second[index - 1] > -threshold:
            index -= 1
        lower += index
    return lower, upper


def exact_sign_flip(values):
    weights = integer_weights(values)
    n = len(weights)
    total = 1 << n
    threshold = abs(sum(weights))
    if threshold == 0:
        return {'n': n, 'tail_count': total, 'assignments': total,
                'p_numerator': 1, 'p_denominator': 1, 'p_raw': 1.0,
                'two_exact_count_implementations_agree': True,
                'zero_observed_sum': True}
    split = n // 2
    left = signed_sums(weights[:split])
    right = sorted(signed_sums(weights[split:]))
    upper = sum(len(right) - bisect_left(right, threshold - value) for value in left)
    lower = sum(bisect_right(right, -threshold - value) for value in left)
    independent_lower, independent_upper = monotone_tail_counts(left, right, threshold)
    assert (lower, upper) == (independent_lower, independent_upper)
    assert lower == upper, 'Every sign assignment has its opposite'
    count = lower + upper
    assert 2 <= count <= total
    probability = Fraction(count, total)
    return {'n': n, 'tail_count': count, 'assignments': total,
            'p_numerator': probability.numerator, 'p_denominator': probability.denominator,
            'p_raw': float(probability), 'two_exact_count_implementations_agree': True,
            'zero_observed_sum': False}


def exact_bootstrap_interval(values, sample_indices, probabilities=(Fraction(1, 40), Fraction(39, 40))):
    """Rational linear percentiles of common complete-block bootstrap draws."""
    values = [as_fraction(value) for value in values]
    denominator = lcm(*(value.denominator for value in values))
    weights = [value.numerator * (denominator // value.denominator) for value in values]
    n = len(values)
    assert 1 <= n <= 32 and len(sample_indices) == 10000
    sums = []
    for sample in sample_indices:
        assert len(sample) == n
        assert all(0 <= int(index) < n for index in sample)
        sums.append(sum(weights[int(index)] for index in sample))
    sums.sort()
    quantiles = []
    for probability in probabilities:
        position = (len(sums) - 1) * probability
        lower = position.numerator // position.denominator
        remainder = position - lower
        first = Fraction(sums[lower], denominator * n)
        if remainder:
            second = Fraction(sums[lower + 1], denominator * n)
            first += remainder * (second - first)
        quantiles.append(first)
    return quantiles


def holm_bh(values):
    """Exact decimal/rational adjustment; stable identity order is retained."""
    p = [as_fraction(value) for value in values]
    assert all(0 <= value <= 1 for value in p)
    order = sorted(range(len(p)), key=p.__getitem__)
    holm = [Fraction()] * len(p)
    running = Fraction()
    for rank, index in enumerate(order):
        running = max(running, min(Fraction(1), (len(p) - rank) * p[index]))
        holm[index] = running
    bh = [Fraction()] * len(p)
    running = Fraction(1)
    for rank in range(len(p) - 1, -1, -1):
        index = order[rank]
        running = min(running, len(p) * p[index] / (rank + 1))
        bh[index] = running
    return {'holm': list(map(float, holm)), 'bh': list(map(float, bh))}


def brute_force(values):
    weights = integer_weights(values)
    target = abs(sum(weights))
    return sum(abs(sum(sign * weight for sign, weight in zip(signs, weights))) >= target
               for signs in product((-1, 1), repeat=len(weights)))


def algorithm_selfcheck():
    checks = []
    fixed = [('all_positive_32', [1] * 32, Fraction(2, 2**32)),
             ('all_zero_32', [0] * 32, Fraction(1)),
             ('balanced_32', [1, -1] * 16, Fraction(1)),
             ('one_active_dimension_32', [1] + [0] * 31, Fraction(1)),
             ('all_negative_32', [-1] * 32, Fraction(2, 2**32)),
             ('two_active_dimensions_32', [1, 1] + [0] * 30, Fraction(1, 2))]
    for name, values, expected in fixed:
        result = exact_sign_flip(values)
        assert Fraction(result['p_numerator'], result['p_denominator']) == expected
        checks.append({'case': name, 'status': 'PASS', **result})
    rng = random.Random(6091419)
    for n in range(1, 17):
        for replicate in range(3):
            values = [Fraction(rng.randint(-8, 8), rng.choice((1, 2, 3, 7))) for _ in range(n)]
            result = exact_sign_flip(values)
            assert result['tail_count'] == brute_force(values)
    checks.append({'case': '48_small_rational_vectors_against_full_enumeration', 'status': 'PASS'})
    assert holm_bh(['0.01', '0.04', '0.03']) == {'holm': [0.03, 0.06, 0.06], 'bh': [0.03, 0.04, 0.04]}
    checks.append({'case': 'known_holm_bh', 'status': 'PASS'})
    samples = [[0, 0]] * 250 + [[0, 1]] * 9500 + [[1, 1]] * 250
    assert exact_bootstrap_interval([Fraction(), Fraction(1)], samples) == [Fraction(39, 80), Fraction(41, 80)]
    checks.append({'case': 'exact_percentile_interpolation_at_two_tie_boundaries', 'status': 'PASS'})
    import numpy as np
    generator = np.random.default_rng(6091419)
    sample = generator.integers(0, 32, (10000, 32))
    values = [Fraction(i - 9, 7 + i % 3) for i in range(32)]
    exact = exact_bootstrap_interval(values, sample)
    array = np.array(list(map(float, values)))
    numerical = np.quantile(array[sample].mean(axis=1), [.025, .975], method='linear')
    assert max(abs(float(a) - float(b)) for a, b in zip(exact, numerical)) < 1e-14
    checks.append({'case': 'exact_32_block_bootstrap_vs_numpy_linear_quantiles', 'status': 'PASS'})
    result = {'status': 'PASS', 'scope': 'Synthetic arithmetic checks only, before P19 outcomes.', 'checks': checks,
              'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    target = Path(__file__).resolve().parent / 'preoutcome_exact_inference_checks.json'
    target.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    algorithm_selfcheck()
