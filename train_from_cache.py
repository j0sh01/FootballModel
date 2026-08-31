#!/usr/bin/env python
"""
Train remaining leagues from cached features + run walk-forward backtests.
Run: python train_from_cache.py
"""
import sys, os, json, time
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data_loader import LEAGUE_CONFIG
from src.market_models import MarketModel, walk_forward_backtest

CACHE_DIR = 'data/cached_features'
MODELS_DIR = 'models'
OUTPUTS_DIR = 'outputs'

# Markets
NEW_MARKETS = [
    'home_clean_sheet', 'away_clean_sheet',
    'home_win_to_nil', 'away_win_to_nil',
    'over_under_15', 'over_under_35',
    'ht_double_chance_1x'
]
ENSEMBLE_MARKETS = ['match_result', 'over_under_25', 'btts']
MODEL_TYPES = ['xgboost', 'random_forest', 'gradient_boosting']
BACKTEST_MARKETS = ['match_result', 'over_under_25', 'btts', 'double_chance_1x']

def load_cached(league):
    path = os.path.join(CACHE_DIR, f'{league}_features.pkl')
    if os.path.exists(path):
        return pd.read_pickle(path)
    return None

def count_models(league):
    d = os.path.join(MODELS_DIR, league)
    if not os.path.exists(d):
        return 0
    return len([f for f in os.listdir(d) if f.endswith('.pkl')])

def main():
    print("=" * 60)
    print("TRAINING FROM CACHE + WALK-FORWARD BACKTEST")
    print("=" * 60)
    
    all_results = {}
    backtest_results = {}
    
    for league in LEAGUE_CONFIG:
        n = count_models(league)
        name = LEAGUE_CONFIG[league]['name']
        print(f"\n{'#'*60}")
        print(f"# {name} ({n} models)")
        print(f"{'#'*60}")
        
        df = load_cached(league)
        if df is None:
            print("  No cached features, skipping")
            continue
        
        print(f"  Loaded {len(df)} matches from cache")
        league_dir = os.path.join(MODELS_DIR, league)
        os.makedirs(league_dir, exist_ok=True)
        
        league_results = {}
        
        # 1. Train new markets
        for market in NEW_MARKETS:
            path = os.path.join(league_dir, f"{market}_xgboost.pkl")
            if os.path.exists(path):
                print(f"  ✓ {market} (exists)")
                continue
            print(f"  → Training {market}...")
            model = MarketModel(market, league, 'xgboost')
            res = model.train(df, save_path=path)
            if res and 'accuracy' in res:
                league_results[market] = res['accuracy']
        
        # 2. Train ensemble for key markets
        for market in ENSEMBLE_MARKETS:
            best_acc = 0
            best_model = None
            for mt in MODEL_TYPES:
                path = os.path.join(league_dir, f"{market}_{mt}.pkl")
                if os.path.exists(path):
                    print(f"  ✓ {market}_{mt} (exists)")
                    # Load accuracy from saved model
                    import joblib
                    try:
                        data = joblib.load(path)
                        acc = data.get('results', {}).get('accuracy', 0)
                        if acc > best_acc:
                            best_acc = acc
                            best_model = mt
                    except:
                        pass
                    continue
                print(f"  → Training {market} ({mt})...")
                model = MarketModel(market, league, mt)
                res = model.train(df, save_path=path)
                if res and 'accuracy' in res:
                    if res['accuracy'] > best_acc:
                        best_acc = res['accuracy']
                        best_model = mt
            
            if best_acc > 0:
                league_results[f"{market}_ensemble"] = {
                    'accuracy': best_acc,
                    'best_model': best_model
                }
                print(f"  → Best {market}: {best_model} ({best_acc:.4f})")
        
        # 3. Walk-forward backtest
        print(f"\n  --- Walk-Forward Backtests ---")
        bt_league = {}
        for market in BACKTEST_MARKETS:
            print(f"  Backtesting {market}...")
            results = walk_forward_backtest(
                df, market, league, 'xgboost',
                train_size=min(1500, len(df) - 200),
                test_size=min(200, max(50, len(df) // 5)),
                step=100
            )
            if not results.empty and 'accuracy' in results.columns:
                bt_league[market] = {
                    'mean_accuracy': float(results['accuracy'].mean()),
                    'std_accuracy': float(results['accuracy'].std()),
                    'min_accuracy': float(results['accuracy'].min()),
                    'max_accuracy': float(results['accuracy'].max()),
                    'n_windows': len(results)
                }
                print(f"    {market}: {results['accuracy'].mean():.4f} ± {results['accuracy'].std():.4f}")
        
        backtest_results[league] = bt_league
        all_results[league] = league_results
    
    # Save everything
    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    
    with open(os.path.join(OUTPUTS_DIR, 'training_new_markets.json'), 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    
    with open(os.path.join(OUTPUTS_DIR, 'backtest_results.json'), 'w') as f:
        json.dump(backtest_results, f, indent=2, default=str)
    
    # Final summary
    print(f"\n{'='*60}")
    print("COMPLETE SUMMARY")
    print(f"{'='*60}")
    
    print("\n📊 NEW MARKET TRAINING:")
    for league, results in all_results.items():
        print(f"\n  {LEAGUE_CONFIG[league]['name']}:")
        for market, acc in results.items():
            if isinstance(acc, dict):
                print(f"    {market:25s}: {acc.get('accuracy', 'N/A'):.4f} ({acc.get('best_model', '')})")
            elif isinstance(acc, (int, float)):
                print(f"    {market:25s}: {acc:.4f}")
    
    print("\n📊 WALK-FORWARD BACKTEST:")
    for league, bt in backtest_results.items():
        print(f"\n  {LEAGUE_CONFIG[league]['name']}:")
        for market, metrics in bt.items():
            print(f"    {market:25s}: {metrics['mean_accuracy']:.4f} ± {metrics['std_accuracy']:.4f} ({metrics['n_windows']} windows)")
    
    print(f"\nAll results saved to {OUTPUTS_DIR}/")
    print(f"All models saved to {MODELS_DIR}/")

if __name__ == '__main__':
    main()
