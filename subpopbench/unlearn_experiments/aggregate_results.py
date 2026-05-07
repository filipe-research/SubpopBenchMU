"""
Agrega JSONs de experimentos de unlearning em tabela markdown.
Uso: python -m subpopbench.unlearn_experiments.aggregate_results --input_dir ./output/phase1
Opcional: --include_per_group para detalhar accuracy por grupo
"""
import json
import argparse
import statistics
from pathlib import Path
from collections import defaultdict


def fmt(vals):
    clean = [v for v in vals if v is not None]
    if not clean:
        return '-'
    m = statistics.mean(clean)
    if len(clean) > 1:
        s = statistics.stdev(clean)
        return f'{m*100:.1f}±{s*100:.1f}'
    return f'{m*100:.1f}'


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--input_dir', required=True)
    p.add_argument('--include_per_group', action='store_true')
    args = p.parse_args()

    files = sorted(Path(args.input_dir).glob('*.json'))
    groups = defaultdict(list)

    for f in files:
        data = json.loads(f.read_text())
        a, r = data['args'], data['results']
        regime = a.get('regime', f"class_{a.get('target_class', '?')}")
        ratio = a.get('forget_ratio')

        # Sub-chave para diferenciar variantes do mesmo regime
        sub = ''
        if regime == 'ts_fbc':
            sub = f" (M={a.get('pool_multiplier', '?')})"
        elif regime == 'fbc_band':
            sub = f" (band={a.get('fbc_band_low', '?')}-{a.get('fbc_band_high', '?')})"
        elif regime == 'fbc':
            cw = a.get('fbc_classwise', True)
            fc = a.get('fbc_filter_correct', True)
            if not cw or not fc:
                sub = f" (cw={cw},fc={fc})"

        regime_label = regime + sub
        groups[(regime_label, ratio)].append(r)

    print(f"\n# {len(files)} JSONs, {len(groups)} configurações\n")

    cols = ['Regime', 'Ratio', 'N', 'avg_acc', 'WGA', 'FA', 'RA',
            'fa_BA', 'fa_BC', 'ΔGap']
    if args.include_per_group:
        cols += ['g0', 'g1', 'g2', 'g3']

    print('| ' + ' | '.join(cols) + ' |')
    print('|' + '|'.join(['---'] * len(cols)) + '|')

    for key in sorted(groups.keys(), key=lambda k: (k[0], k[1] or 0)):
        regime, ratio = key
        runs = groups[key]
        ratio_str = f'{int(ratio*100)}%' if ratio else '-'
        row = [
            regime, ratio_str, str(len(runs)),
            fmt([r['avg_acc'] for r in runs]),
            fmt([r['wga'] for r in runs]),
            fmt([r['fa'] for r in runs]),
            fmt([r['ra'] for r in runs]),
            fmt([r.get('fa_bias_aligned') for r in runs]),
            fmt([r.get('fa_bias_conflicting') for r in runs]),
            fmt([r.get('delta_gap') for r in runs]),
        ]
        if args.include_per_group:
            for gid in ['0', '1', '2', '3']:
                row.append(fmt([r['per_group_acc'].get(gid) for r in runs]))
        print('| ' + ' | '.join(row) + ' |')


if __name__ == '__main__':
    main()
