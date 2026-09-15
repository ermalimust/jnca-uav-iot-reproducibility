"""Format all Table 50 percentages from exact block fractions and quantiles.

Only a replacement table body and its formatting receipt are written here.
The manuscript remains read-only. Decimal ROUND_HALF_UP is explicit.
"""
from pathlib import Path
from fractions import Fraction
from decimal import Decimal, ROUND_HALF_UP, localcontext
import csv
import hashlib
import json
import math
import numpy as np

HERE = Path(__file__).resolve().parent


def read(name):
    with (HERE / name).open(encoding='utf-8-sig', newline='') as handle:
        return list(csv.DictReader(handle))


def decimal_string(value, digits, signed=False):
    if not isinstance(value, Fraction):
        value = Fraction(str(value))
    with localcontext() as context:
        context.prec = 50
        decimal = Decimal(value.numerator) / Decimal(value.denominator)
        result = str(decimal.quantize(Decimal(1).scaleb(-digits), rounding=ROUND_HALF_UP))
    return ('+' if signed and value > 0 else '') + result


def exact_percentile(values, sample, probability):
    denominator = math.lcm(*(v.denominator for v in values))
    weights = np.asarray([v.numerator * (denominator // v.denominator) for v in values], dtype=np.int64)
    assert sum(abs(int(v)) for v in weights) < 2**62
    sums = np.sort(weights[sample].sum(axis=1))
    position = Fraction(len(sums) - 1) * probability
    lower = position.numerator // position.denominator
    fraction = position - lower
    first = Fraction(int(sums[lower]), denominator * 16)
    if not fraction:
        return first
    second = Fraction(int(sums[lower + 1]), denominator * 16)
    return first + fraction * (second - first)


def main():
    block_rows = read('run_block_values.csv')
    blocks = {(r['action'], r['metric'], int(r['rng_run'])):
              Fraction(int(r['exact_numerator']), int(r['exact_denominator'])) for r in block_rows}
    primary = read('primary_contrasts.csv')
    pooled = {r['test_id']: r for r in read('pooled_101_tests.csv')}
    sample = np.load(HERE / 'bootstrap_run_indices.npy', allow_pickle=False)
    assert sample.shape == (10000, 16)
    lines, formatted = [], []
    for row in primary:
        action = row['contrast'].split('-Observe')[0]
        metric = row['metric']
        means = [blocks[(action, metric, run)] for run in range(1, 17)]
        diffs = [v - blocks[('Observe', metric, run)] for run, v in zip(range(1, 17), means)]
        mean = sum(means, Fraction()) / 16
        effect = sum(diffs, Fraction()) / 16
        low = exact_percentile(diffs, sample, Fraction(1, 40))
        high = exact_percentile(diffs, sample, Fraction(39, 40))
        assert abs(float(low) - float(row['ci_low'])) < 1e-14
        assert abs(float(high) - float(row['ci_high'])) < 1e-14
        outcome = 'C2 miss' if metric == 'c2_deadline_miss_fraction' else 'Video delivery'
        cells = [decimal_string(mean * 100, 4), decimal_string(effect * 100, 4, True),
                 decimal_string(low * 100, 4), decimal_string(high * 100, 4),
                 decimal_string(pooled[row['test_id']]['p_holm_pooled_101'], 5)]
        lines.append(f'{action} & {outcome} & {cells[0]} & {cells[1]} & $[{cells[2]},{cells[3]}]$ & {cells[4]} ' + r'\\')
        formatted.append({'test_id': row['test_id'], 'mean_percent': cells[0], 'difference_pp': cells[1],
                          'ci_low_pp': cells[2], 'ci_high_pp': cells[3], 'pooled_holm': cells[4],
                          'exact_ci_low': str(low), 'exact_ci_high': str(high)})
    (HERE / 'table50_body_4decimal.tex').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    receipt = {'status': 'PASS', 'rounding': 'decimal ROUND_HALF_UP, 4 places for percent/pp, 5 for Holm',
               'means_and_differences': 'exact rational block values',
               'intervals': 'exact integer bootstrap sums and rational linear percentile interpolation at 1/40 and 39/40',
               'all_exact_intervals_match_primary_float_within_1e_14': True, 'rows': formatted,
               'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (HERE / 'table50_exact_formatting_receipt.json').write_text(json.dumps(receipt, indent=2) + '\n', encoding='utf-8')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
