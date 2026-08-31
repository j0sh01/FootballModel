#!/usr/bin/env python
"""
Cache engineered features for all leagues.
Run once to avoid slow feature engineering on every training run.
"""

import sys
import os
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data_loader import load_league_data, LEAGUE_CONFIG
from src.feature_engineering import engineer_features

CACHE_DIR = 'data/cached_features'

def main():
    os.makedirs(CACHE_DIR, exist_ok=True)
    
    for league in LEAGUE_CONFIG:
        cache_path = os.path.join(CACHE_DIR, f'{league}_features.pkl')
        
        if os.path.exists(cache_path):
            print(f"✓ {league} (cached)")
            continue
        
        print(f"\nProcessing {LEAGUE_CONFIG[league]['name']}...")
        df_raw = load_league_data(league)
        if df_raw.empty:
            print(f"  No data, skipping")
            continue
        
        df, elos = engineer_features(df_raw)
        
        # Save to cache
        df.to_pickle(cache_path)
        print(f"  ✓ Cached: {len(df)} matches, {len(df.columns)} columns")
    
    print(f"\nAll features cached to {CACHE_DIR}/")

def load_cached_features(league):
    """Load cached features for a league."""
    cache_path = os.path.join(CACHE_DIR, f'{league}_features.pkl')
    if os.path.exists(cache_path):
        return pd.read_pickle(cache_path)
    return None

if __name__ == '__main__':
    main()
