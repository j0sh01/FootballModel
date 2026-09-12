#!/usr/bin/env python
"""
Rebuild the multi-algorithm ensemble members for every league.

A market is only scored as an ensemble when all of its members were trained on the
same features and the same train/test split. Retraining a single algorithm on its
own, or training one before the feature cache changed, silently breaks that: the
members then have different holdout sets and the API reports no accuracy for the
market.

This script retrains every member of every ensemble market from
`data/cached_features`, overwriting the `.pkl` files in place, so all members of a
market come from one identical run. Single-model markets are left alone.

Run:
    python retrain_ensembles.py                       # every league, every ensemble market
    python retrain_ensembles.py --leagues serieb ligue2
    python retrain_ensembles.py --markets match_result btts
    python retrain_ensembles.py --dry-run             # list the work, train nothing
"""

import argparse
import os
import sys
import time

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data_loader import LEAGUE_CONFIG
from src.market_models import EnsembleModel, MarketModel, discover_market_models

CACHE_DIR = os.path.join('data', 'cached_features')
MODELS_DIR = 'models'


def load_cached(league: str):
    """Return the engineered features for a league, or None if not cached."""
    path = os.path.join(CACHE_DIR, f'{league}_features.pkl')
    if not os.path.exists(path):
        return None
    return pd.read_pickle(path)


def find_targets(leagues, markets=None):
    """(league, market, [model_type, ...]) for every market with several algorithms."""
    targets = []
    for league in leagues:
        found = discover_market_models(os.path.join(MODELS_DIR, league))
        for market, paths in sorted(found.items()):
            if markets and market not in markets:
                continue
            if len(paths) > 1:
                targets.append((league, market, sorted(paths)))
    return targets


def snapshot(league: str, market: str):
    """Holdout size per member, plus the combined accuracy when it is scoreable."""
    paths = discover_market_models(os.path.join(MODELS_DIR, league))[market]
    combined = EnsembleModel(market, league, model_types=sorted(paths))
    sizes = {}
    for model_type, path in sorted(paths.items()):
        member = MarketModel.load(path)
        sizes[model_type] = len(member.results.get('y_test') or [])
        combined.add_model(model_type, member)
    return sizes, (combined.results or {}).get('accuracy')


def format_members(sizes: dict) -> str:
    """'xgb:486 rf:478 gb:478' — a compact per-member holdout readout."""
    if not sizes:
        return 'n/a'
    short = {'xgboost': 'xgb', 'random_forest': 'rf', 'gradient_boosting': 'gb'}
    return ' '.join(f'{short.get(mt, mt)}:{n}' for mt, n in sorted(sizes.items()))


def format_acc(acc) -> str:
    return 'n/a' if acc is None else f'{acc * 100:.1f}%'


def main():
    parser = argparse.ArgumentParser(
        description='Retrain every ensemble member from cached features.'
    )
    parser.add_argument('--leagues', nargs='+', help='league keys (default: all)')
    parser.add_argument('--markets', nargs='+',
                        help='markets to refresh (default: every ensemble market)')
    parser.add_argument('--dry-run', action='store_true',
                        help='list the work without training anything')
    args = parser.parse_args()

    leagues = args.leagues or list(LEAGUE_CONFIG)
    unknown = [lg for lg in leagues if lg not in LEAGUE_CONFIG]
    if unknown:
        parser.error(f"unknown league(s): {', '.join(unknown)}")

    targets = find_targets(leagues, set(args.markets) if args.markets else None)
    if not targets:
        print('Nothing to do: no market has more than one algorithm on disk.')
        return

    n_models = sum(len(types) for _, _, types in targets)
    print(f'{len(targets)} ensemble markets across '
          f'{len({lg for lg, _, _ in targets})} leagues -> {n_models} models to retrain')

    if args.dry_run:
        for league, market, model_types in targets:
            print(f'  {league:<14} {market:<16} {", ".join(model_types)}')
        return

    # What each market looks like before we touch it
    before = {(lg, m): snapshot(lg, m) for lg, m, _ in targets}

    started = time.time()
    trained = 0
    for league, market, model_types in targets:
        cached = load_cached(league)
        if cached is None:
            print(f'\n{league} / {market}: SKIPPED, no cached features')
            continue

        print(f'\n{"=" * 64}')
        print(f'{league} / {market}  ({len(cached)} cached matches)')
        print(f'{"=" * 64}')

        for model_type in model_types:
            path = os.path.join(MODELS_DIR, league, f'{market}_{model_type}.pkl')
            t0 = time.time()
            model = MarketModel(market, league, model_type)
            result = model.train(cached, save_path=path)
            trained += 1
            acc = result.get('accuracy')
            print(f'  -> {model_type}: {time.time() - t0:.1f}s'
                  f'{f", accuracy {acc * 100:.1f}%" if acc is not None else ""}')

    elapsed = time.time() - started
    print(f'\nRetrained {trained} models in {elapsed:.0f}s ({elapsed / max(trained, 1):.1f}s/model)')

    # --- report ---------------------------------------------------------
    print(f'\n{"league":<14} {"market":<16} {"before":<28} {"after":<28} {"ensemble accuracy":<20}')
    print('-' * 110)
    fixed, scoreable, after_acc = 0, 0, {}
    for league, market, _ in targets:
        sizes_before, acc_before = before[(league, market)]
        sizes_after, acc_after = snapshot(league, market)
        after_acc[(league, market)] = acc_after
        if acc_after is not None:
            scoreable += 1
        if acc_before is None and acc_after is not None:
            fixed += 1

        aligned = len(set(sizes_after.values())) == 1 and len(sizes_after) > 1
        note = 'OK' if aligned else 'MISALIGNED'
        if acc_before is None and acc_after is not None:
            note += ' (newly scoreable)'
        print(f'{league:<14} {market:<16} {format_members(sizes_before):<28} '
              f'{format_members(sizes_after):<28} '
              f'{format_acc(acc_before):>5} -> {format_acc(acc_after):<5}  {note}')

    print('-' * 110)
    print(f'{scoreable}/{len(targets)} markets scoreable as an ensemble '
          f'({fixed} fixed by this run)')


if __name__ == '__main__':
    main()
