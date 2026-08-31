#!/usr/bin/env python
"""
Football Prediction System - Full Setup Script
Run this after cloning to download data, engineer features, and train all models.

Usage:
    python setup.py                    # Full setup (download + features + train)
    python setup.py --download-only    # Only download data
    python setup.py --train-only       # Only train (assumes data exists)
"""

import sys
import os
import argparse
import urllib.request
import time

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


# =============================================================================
# CONFIGURATION
# =============================================================================

SEASONS = list(range(19, 27))  # 2019/20 to 2026/27

LEAGUES = {
    'epl': {'code': 'E0', 'name': 'English Premier League', 'dir': 'data/epl'},
    'championship': {'code': 'E1', 'name': 'English Championship', 'dir': 'data/championship'},
    'bundesliga1': {'code': 'D1', 'name': 'Bundesliga 1', 'dir': 'data/bundesliga1'},
    'bundesliga2': {'code': 'D2', 'name': 'Bundesliga 2', 'dir': 'data/bundesliga2'},
    'seriea': {'code': 'I1', 'name': 'Serie A', 'dir': 'data/seriea'},
    'serieb': {'code': 'I2', 'name': 'Serie B', 'dir': 'data/serieb'},
    'laliga': {'code': 'SP1', 'name': 'La Liga', 'dir': 'data/laliga'},
    'laligab': {'code': 'SP2', 'name': 'La Liga 2', 'dir': 'data/laligab'},
    'ligue1': {'code': 'F1', 'name': 'Ligue 1', 'dir': 'data/ligue1'},
    'ligue2': {'code': 'F2', 'name': 'Ligue 2', 'dir': 'data/ligue2'},
}

BASE_URL = "https://www.football-data.co.uk/mmz4281/{season_code}/{code}.csv"


# =============================================================================
# DOWNLOAD
# =============================================================================

def download_data():
    """Download all league data from football-data.co.uk."""
    print("\n" + "=" * 60)
    print("STEP 1: DOWNLOADING DATA")
    print("=" * 60)
    
    total_files = 0
    total_matches = 0
    
    for league_key, config in LEAGUES.items():
        league_dir = config['dir']
        os.makedirs(league_dir, exist_ok=True)
        
        print(f"\n{config['name']}:")
        league_matches = 0
        
        for season in SEASONS:
            season_code = f"{season:02d}{(season + 1) % 100:02d}"
            url = BASE_URL.format(season_code=season_code, code=config['code'])
            filepath = os.path.join(league_dir, f"{config['code']}_{season_code}.csv")
            
            if os.path.exists(filepath):
                # Count lines
                with open(filepath, 'r') as f:
                    lines = sum(1 for _ in f) - 1  # subtract header
                league_matches += lines
                print(f"  ✓ {config['code']}_{season_code}.csv ({lines} matches) [cached]")
                continue
            
            try:
                urllib.request.urlretrieve(url, filepath)
                with open(filepath, 'r') as f:
                    lines = sum(1 for _ in f) - 1
                league_matches += lines
                total_files += 1
                print(f"  ✓ {config['code']}_{season_code}.csv ({lines} matches)")
                time.sleep(0.2)  # Be nice to the server
            except Exception as e:
                print(f"  ✗ {config['code']}_{season_code}.csv: {e}")
                # Remove empty/broken file
                if os.path.exists(filepath):
                    os.remove(filepath)
        
        total_matches += league_matches
        print(f"  Total: {league_matches} matches")
    
    print(f"\n{'=' * 60}")
    print(f"Download complete: {total_files} new files, {total_matches} total matches")
    print(f"{'=' * 60}")
    
    return total_files > 0


# =============================================================================
# FEATURE ENGINEERING
# =============================================================================

def engineer_features():
    """Cache engineered features for all leagues."""
    print("\n" + "=" * 60)
    print("STEP 2: FEATURE ENGINEERING")
    print("=" * 60)
    
    from src.data_loader import load_league_data
    from src.feature_engineering import engineer_features as _engineer
    
    cache_dir = 'data/cached_features'
    os.makedirs(cache_dir, exist_ok=True)
    
    for league_key, config in LEAGUES.items():
        cache_path = os.path.join(cache_dir, f'{league_key}_features.pkl')
        
        if os.path.exists(cache_path):
            print(f"  ✓ {config['name']} [cached]")
            continue
        
        print(f"\nProcessing {config['name']}...")
        
        df_raw = load_league_data(league_key)
        if df_raw.empty:
            print(f"  ✗ No data available")
            continue
        
        df, elos = _engineer(df_raw)
        df.to_pickle(cache_path)
        print(f"  ✓ Cached: {len(df)} matches, {len(df.columns)} columns")
    
    print(f"\n{'=' * 60}")
    print("Feature engineering complete!")
    print(f"{'=' * 60}")


# =============================================================================
# TRAINING
# =============================================================================

def train_models():
    """Train all models using cached features."""
    print("\n" + "=" * 60)
    print("STEP 3: TRAINING MODELS")
    print("=" * 60)
    
    import pandas as pd
    from src.data_loader import LEAGUE_CONFIG
    from src.market_models import MarketModel
    
    cache_dir = 'data/cached_features'
    models_dir = 'models'
    os.makedirs(models_dir, exist_ok=True)
    
    # Markets to train
    core_markets = ['match_result', 'over_under_25', 'btts', 'double_chance_1x',
                    'double_chance_x2', 'goals_bucket', 'ht_result']
    new_markets = ['home_clean_sheet', 'away_clean_sheet', 'home_win_to_nil',
                   'away_win_to_nil', 'over_under_15', 'over_under_35', 'ht_double_chance_1x']
    all_markets = core_markets + new_markets
    
    # Ensemble markets (train 3 model types)
    ensemble_markets = ['match_result', 'over_under_25', 'btts']
    model_types = ['xgboost', 'random_forest', 'gradient_boosting']
    
    total_trained = 0
    
    for league in LEAGUES:
        cache_path = os.path.join(cache_dir, f'{league}_features.pkl')
        if not os.path.exists(cache_path):
            print(f"\n  ✗ {LEAGUES[league]['name']}: No cached features")
            continue
        
        df = pd.read_pickle(cache_path)
        league_dir = os.path.join(models_dir, league)
        os.makedirs(league_dir, exist_ok=True)
        
        print(f"\n{LEAGUES[league]['name']} ({len(df)} matches):")
        
        for market in all_markets:
            if market in ensemble_markets:
                for mt in model_types:
                    path = os.path.join(league_dir, f"{market}_{mt}.pkl")
                    if os.path.exists(path):
                        continue
                    
                    try:
                        model = MarketModel(market, league, mt)
                        model.train(df, save_path=path)
                        total_trained += 1
                    except Exception as e:
                        print(f"    ✗ {market} ({mt}): {e}")
            else:
                path = os.path.join(league_dir, f"{market}_xgboost.pkl")
                if os.path.exists(path):
                    continue
                
                try:
                    model = MarketModel(market, league, 'xgboost')
                    model.train(df, save_path=path)
                    total_trained += 1
                except Exception as e:
                    print(f"    ✗ {market}: {e}")
    
    print(f"\n{'=' * 60}")
    print(f"Training complete: {total_trained} new models trained")
    print(f"{'=' * 60}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Football Prediction System Setup')
    parser.add_argument('--download-only', action='store_true', help='Only download data')
    parser.add_argument('--train-only', action='store_true', help='Only train models')
    parser.add_argument('--skip-download', action='store_true', help='Skip data download')
    parser.add_argument('--skip-features', action='store_true', help='Skip feature engineering')
    parser.add_argument('--skip-train', action='store_true', help='Skip model training')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("FOOTBALL PREDICTION SYSTEM - SETUP")
    print("=" * 60)
    
    start_time = time.time()
    
    if args.download_only:
        download_data()
        return
    
    if args.train_only:
        train_models()
        return
    
    # Full setup
    if not args.skip_download:
        download_data()
    
    if not args.skip_features:
        engineer_features()
    
    if not args.skip_train:
        train_models()
    
    elapsed = time.time() - start_time
    minutes = int(elapsed // 60)
    seconds = int(elapsed % 60)
    
    print(f"\n{'=' * 60}")
    print(f"SETUP COMPLETE! (took {minutes}m {seconds}s)")
    print(f"{'=' * 60}")
    print(f"\nTo start the API:")
    print(f"  python api/app.py")
    print(f"\nTo retrain models:")
    print(f"  python train_from_cache.py")
    print(f"\nAPI docs:")
    print(f"  http://localhost:8000/docs")


if __name__ == '__main__':
    main()
