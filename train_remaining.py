#!/usr/bin/env python
"""
Train remaining leagues (those with only 7 models) and run walk-forward backtest.
Run: python train_remaining.py
"""

import sys
import os
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data_loader import load_league_data, LEAGUE_CONFIG
from src.feature_engineering import engineer_features
from src.market_models import MarketModel, walk_forward_backtest
import time

# New markets that need training for incomplete leagues
NEW_MARKETS = [
    'home_clean_sheet', 'away_clean_sheet',
    'home_win_to_nil', 'away_win_to_nil',
    'over_under_15', 'over_under_35',
    'ht_double_chance_1x'
]

# All markets
ALL_MARKETS = [
    'match_result', 'over_under_25', 'btts',
    'double_chance_1x', 'double_chance_x2',
    'goals_bucket', 'ht_result'
] + NEW_MARKETS

# Ensemble markets
ENSEMBLE_MARKETS = ['match_result', 'over_under_25', 'btts']
MODEL_TYPES = ['xgboost', 'random_forest', 'gradient_boosting']

MODELS_DIR = 'models'
OUTPUTS_DIR = 'outputs'

def check_league_status(league):
    """Check how many models a league has."""
    league_dir = os.path.join(MODELS_DIR, league)
    if not os.path.exists(league_dir):
        return 0
    pkl_files = [f for f in os.listdir(league_dir) if f.endswith('.pkl')]
    return len(pkl_files)

def train_new_markets(league, df):
    """Train only the new markets for a league."""
    league_dir = os.path.join(MODELS_DIR, league)
    os.makedirs(league_dir, exist_ok=True)
    
    results = {}
    
    for market in NEW_MARKETS:
        # Check if already trained
        model_path = os.path.join(league_dir, f"{market}_xgboost.pkl")
        if os.path.exists(model_path):
            print(f"  ✓ {market} (already exists)")
            continue
        
        print(f"\n--- {market} ---")
        model = MarketModel(market, league, 'xgboost')
        save_path = os.path.join(league_dir, f"{market}_xgboost.pkl")
        res = model.train(df, save_path=save_path)
        if res and 'accuracy' in res:
            results[market] = res['accuracy']
    
    return results

def train_ensemble_for_market(league, market, df):
    """Train ensemble models for a market."""
    league_dir = os.path.join(MODELS_DIR, league)
    results = {}
    
    for model_type in MODEL_TYPES:
        model_path = os.path.join(league_dir, f"{market}_{model_type}.pkl")
        if os.path.exists(model_path):
            print(f"  ✓ {market}_{model_type} (already exists)")
            continue
        
        print(f"\n--- {market} ({model_type}) ---")
        model = MarketModel(market, league, model_type)
        save_path = os.path.join(league_dir, f"{market}_{model_type}.pkl")
        res = model.train(df, save_path=save_path)
        if res and 'accuracy' in res:
            results[model_type] = res['accuracy']
    
    return results

def run_backtest_for_league(league, df):
    """Run walk-forward backtest for key markets."""
    print(f"\n{'='*60}")
    print(f"WALK-FORWARD BACKTEST: {LEAGUE_CONFIG[league]['name']}")
    print(f"{'='*60}")
    
    backtest_markets = ['match_result', 'over_under_25', 'btts', 'double_chance_1x']
    backtest_results = {}
    
    for market in backtest_markets:
        print(f"\n--- Backtesting {market} ---")
        results = walk_forward_backtest(
            df, market, league,
            model_type='xgboost',
            train_size=min(1500, len(df) - 200),
            test_size=min(200, len(df) // 5),
            step=100
        )
        if not results.empty:
            if 'accuracy' in results.columns:
                mean_acc = results['accuracy'].mean()
                std_acc = results['accuracy'].std()
                print(f"  Mean Accuracy: {mean_acc:.4f} (+/- {std_acc:.4f})")
                backtest_results[market] = {
                    'mean_accuracy': float(mean_acc),
                    'std_accuracy': float(std_acc),
                    'min_accuracy': float(results['accuracy'].min()),
                    'max_accuracy': float(results['accuracy'].max()),
                    'n_windows': len(results)
                }
            elif 'mae' in results.columns:
                mean_mae = results['mae'].mean()
                print(f"  Mean MAE: {mean_mae:.4f}")
                backtest_results[market] = {
                    'mean_mae': float(mean_mae),
                    'n_windows': len(results)
                }
    
    return backtest_results

def main():
    print("=" * 60)
    print("FOOTBALL PREDICTION - TRAINING REMAINING LEAGUES + BACKTEST")
    print("=" * 60)
    
    # Check which leagues need work
    all_results = {}
    backtest_all = {}
    
    for league in LEAGUE_CONFIG:
        n_models = check_league_status(league)
        print(f"\n{LEAGUE_CONFIG[league]['name']}: {n_models} models")
        
        if n_models < 20:
            print(f"  → Needs training ({n_models}/20 models)")
            
            # Load data
            t0 = time.time()
            df_raw = load_league_data(league)
            if df_raw.empty:
                print(f"  No data, skipping")
                continue
            
            df, _ = engineer_features(df_raw)
            t1 = time.time()
            print(f"  Feature engineering: {t1-t0:.1f}s")
            
            # Train new markets
            new_results = train_new_markets(league, df)
            
            # Train ensemble for key markets
            for market in ENSEMBLE_MARKETS:
                ens_results = train_ensemble_for_market(league, market, df)
                new_results[f"{market}_ensemble"] = ens_results
            
            all_results[league] = new_results
            
            # Final model count
            final_count = check_league_status(league)
            print(f"\n  Final: {final_count} models")
        else:
            print(f"  ✓ Already complete ({n_models} models)")
            # Still load data for backtest
            df_raw = load_league_data(league)
            if not df_raw.empty:
                df, _ = engineer_features(df_raw)
        
        # Run walk-forward backtest
        bt_results = run_backtest_for_league(league, df)
        backtest_all[league] = bt_results
    
    # Save results
    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    
    # Save training results
    training_path = os.path.join(OUTPUTS_DIR, 'training_new_markets.json')
    with open(training_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    # Save backtest results
    backtest_path = os.path.join(OUTPUTS_DIR, 'backtest_results.json')
    with open(backtest_path, 'w') as f:
        json.dump(backtest_all, f, indent=2)
    
    print(f"\n{'='*60}")
    print("ALL COMPLETE!")
    print(f"{'='*60}")
    
    # Summary
    print("\n📊 BACKTEST SUMMARY:")
    for league, bt in backtest_all.items():
        print(f"\n{LEAGUE_CONFIG[league]['name']}:")
        for market, metrics in bt.items():
            if 'mean_accuracy' in metrics:
                print(f"  {market:25s}: {metrics['mean_accuracy']:.4f} ± {metrics['std_accuracy']:.4f}")
            elif 'mean_mae' in metrics:
                print(f"  {market:25s}: MAE={metrics['mean_mae']:.4f}")
    
    print(f"\nResults saved to {OUTPUTS_DIR}/")

if __name__ == '__main__':
    main()
