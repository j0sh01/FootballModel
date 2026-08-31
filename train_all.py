#!/usr/bin/env python
"""
Quick Start: Train all models for all leagues
Run: python train_all.py
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data_loader import load_league_data, LEAGUE_CONFIG
from src.feature_engineering import engineer_features
from src.market_models import MarketModel
import json

# Configuration
LEAGUES = ['epl', 'championship', 'bundesliga1', 'bundesliga2', 
           'seriea', 'serieb', 'laliga', 'laligab', 'ligue1', 'ligue2']

# All markets (low-risk + new)
MARKETS = [
    # Core markets
    'match_result', 'over_under_25', 'btts',
    'double_chance_1x', 'double_chance_x2',
    'goals_bucket', 'ht_result',
    # New markets
    'home_clean_sheet', 'away_clean_sheet',
    'home_win_to_nil', 'away_win_to_nil',
    'over_under_15', 'over_under_35',
    'ht_double_chance_1x'
]

# Ensemble models (combine multiple algorithms)
ENSEMBLE_MARKETS = ['match_result', 'over_under_25', 'btts']
MODEL_TYPES = ['xgboost', 'random_forest', 'gradient_boosting']

# Single model markets
SINGLE_MODEL_TYPE = 'xgboost'

MODELS_DIR = 'models'
OUTPUTS_DIR = 'outputs'

def main():
    print("=" * 60)
    print("FOOTBALL PREDICTION SYSTEM - TRAINING")
    print("=" * 60)
    
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    
    all_results = {}
    
    for league in LEAGUES:
        print(f"\n{'#'*60}")
        print(f"# {LEAGUE_CONFIG[league]['name']}")
        print(f"{'#'*60}")
        
        # Load data
        df_raw = load_league_data(league)
        if df_raw.empty:
            print(f"  No data, skipping")
            continue
        
        # Feature engineering
        df, elos = engineer_features(df_raw)
        
        # Create league directory
        league_dir = os.path.join(MODELS_DIR, league)
        os.makedirs(league_dir, exist_ok=True)
        
        league_results = {}
        
        for market in MARKETS:
            print(f"\n--- {market} ---")
            
            # Use ensemble for key markets, single model for others
            if market in ENSEMBLE_MARKETS:
                # Ensemble: train multiple models and combine
                ensemble_results = {}
                for model_type in MODEL_TYPES:
                    model = MarketModel(market, league, model_type)
                    save_path = os.path.join(league_dir, f"{market}_{model_type}.pkl")
                    results = model.train(df, save_path=save_path)
                    if results and 'accuracy' in results:
                        ensemble_results[model_type] = results['accuracy']
                
                # Use best model's results
                if ensemble_results:
                    best_model_type = max(ensemble_results, key=ensemble_results.get)
                    best_accuracy = ensemble_results[best_model_type]
                    print(f"\n  Ensemble best: {best_model_type} ({best_accuracy:.4f})")
                    league_results[market] = {
                        'accuracy': best_accuracy,
                        'best_model': best_model_type,
                        'ensemble_accuracies': ensemble_results
                    }
            else:
                # Single model
                model = MarketModel(market, league, SINGLE_MODEL_TYPE)
                save_path = os.path.join(league_dir, f"{market}_{SINGLE_MODEL_TYPE}.pkl")
                
                results = model.train(df, save_path=save_path)
                
                if results:
                    # Save only scalar values
                    league_results[market] = {
                        k: v for k, v in results.items()
                        if isinstance(v, (int, float, str, bool))
                    }
        
        all_results[league] = league_results
    
    # Save summary
    summary_path = os.path.join(OUTPUTS_DIR, 'training_results.json')
    with open(summary_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"TRAINING COMPLETE")
    print(f"{'='*60}")
    print(f"Models saved to: {MODELS_DIR}/")
    print(f"Results saved to: {summary_path}")
    
    # Print summary table
    print(f"\n{'='*60}")
    print("PERFORMANCE SUMMARY")
    print(f"{'='*60}")
    
    for league, results in all_results.items():
        print(f"\n{LEAGUE_CONFIG[league]['name']}:")
        for market, metrics in results.items():
            if 'accuracy' in metrics:
                print(f"  {market:25s}: {metrics['accuracy']:.4f}")

if __name__ == '__main__':
    main()
