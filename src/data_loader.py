"""
Data Loader Module
Loads and manages football data from multiple leagues and seasons.
"""

import pandas as pd
import numpy as np
import os
import glob
from pathlib import Path
from typing import List, Dict, Optional, Tuple


# League configuration
LEAGUE_CONFIG = {
    'epl': {
        'code': 'E0',
        'name': 'English Premier League',
        'country': 'England',
        'tier': 1,
        'dir': 'data/epl'
    },
    'championship': {
        'code': 'E1',
        'name': 'English Championship',
        'country': 'England',
        'tier': 2,
        'dir': 'data/championship'
    },
    'bundesliga1': {
        'code': 'D1',
        'name': 'Bundesliga 1',
        'country': 'Germany',
        'tier': 1,
        'dir': 'data/bundesliga1'
    },
    'bundesliga2': {
        'code': 'D2',
        'name': 'Bundesliga 2',
        'country': 'Germany',
        'tier': 2,
        'dir': 'data/bundesliga2'
    },
    'seriea': {
        'code': 'I1',
        'name': 'Serie A',
        'country': 'Italy',
        'tier': 1,
        'dir': 'data/seriea'
    },
    'serieb': {
        'code': 'I2',
        'name': 'Serie B',
        'country': 'Italy',
        'tier': 2,
        'dir': 'data/serieb'
    },
    'laliga': {
        'code': 'SP1',
        'name': 'La Liga',
        'country': 'Spain',
        'tier': 1,
        'dir': 'data/laliga'
    },
    'laligab': {
        'code': 'SP2',
        'name': 'La Liga 2',
        'country': 'Spain',
        'tier': 2,
        'dir': 'data/laligab'
    },
    'ligue1': {
        'code': 'F1',
        'name': 'Ligue 1',
        'country': 'France',
        'tier': 1,
        'dir': 'data/ligue1'
    },
    'ligue2': {
        'code': 'F2',
        'name': 'Ligue 2',
        'country': 'France',
        'tier': 2,
        'dir': 'data/ligue2'
    }
}

# Season range
SEASON_START = 19  # 2019/20
SEASON_END = 26    # 2026/27


def get_season_code(season_start: int) -> str:
    """Convert season start year to URL code (e.g., 19 -> '1920')."""
    return f"{season_start:02d}{(season_start + 1) % 100:02d}"


def get_season_label(season_start: int) -> str:
    """Convert season start year to display label (e.g., 19 -> '2019/20')."""
    return f"20{season_start:02d}/20{(season_start + 1) % 100:02d}"


def load_league_data(league: str, seasons: Optional[List[int]] = None) -> pd.DataFrame:
    """
    Load data for a specific league across multiple seasons.
    
    Args:
        league: League key from LEAGUE_CONFIG
        seasons: List of season start years (default: SEASON_START to SEASON_END)
    
    Returns:
        Combined DataFrame with all seasons
    """
    if league not in LEAGUE_CONFIG:
        raise ValueError(f"Unknown league: {league}. Available: {list(LEAGUE_CONFIG.keys())}")
    
    config = LEAGUE_CONFIG[league]
    data_dir = config['dir']
    code = config['code']
    
    if seasons is None:
        seasons = list(range(SEASON_START, SEASON_END + 1))
    
    dfs = []
    for season in seasons:
        season_code = get_season_code(season)
        filepath = os.path.join(data_dir, f"{code}_{season_code}.csv")
        
        if os.path.exists(filepath):
            try:
                df = pd.read_csv(filepath)
                df['Season'] = get_season_label(season)
                df['SeasonStart'] = season
                df['League'] = league
                df['LeagueName'] = config['name']
                dfs.append(df)
                print(f"  ✓ {code}_{season_code}.csv ({len(df)} matches)")
            except Exception as e:
                print(f"  ✗ Error loading {filepath}: {e}")
        else:
            print(f"  ✗ {filepath} not found")
    
    if not dfs:
        return pd.DataFrame()
    
    combined = pd.concat(dfs, ignore_index=True)
    print(f"\n  Total: {len(combined)} matches for {config['name']}")
    return combined


def load_all_leagues(seasons: Optional[List[int]] = None) -> Dict[str, pd.DataFrame]:
    """
    Load data for all leagues.
    
    Returns:
        Dictionary mapping league key to DataFrame
    """
    data = {}
    for league in LEAGUE_CONFIG:
        print(f"\nLoading {LEAGUE_CONFIG[league]['name']}...")
        data[league] = load_league_data(league, seasons)
    return data


def load_combined_leagues(league_keys: Optional[List[str]] = None,
                         seasons: Optional[List[int]] = None) -> pd.DataFrame:
    """
    Load and combine data from multiple leagues.
    
    Args:
        league_keys: List of league keys (default: all)
        seasons: List of season start years
    
    Returns:
        Combined DataFrame
    """
    if league_keys is None:
        league_keys = list(LEAGUE_CONFIG.keys())
    
    dfs = []
    for league in league_keys:
        df = load_league_data(league, seasons)
        if not df.empty:
            dfs.append(df)
    
    if not dfs:
        return pd.DataFrame()
    
    return pd.concat(dfs, ignore_index=True)


def get_league_summary(data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Generate a summary of all loaded data."""
    summary = []
    total_matches = 0
    
    for league, df in data.items():
        if not df.empty:
            n_matches = len(df)
            n_seasons = df['Season'].nunique()
            n_teams = df['HomeTeam'].nunique()
            min_date = df['Date'].min()
            max_date = df['Date'].max()
            
            summary.append({
                'League': LEAGUE_CONFIG[league]['name'],
                'Country': LEAGUE_CONFIG[league]['country'],
                'Tier': LEAGUE_CONFIG[league]['tier'],
                'Seasons': n_seasons,
                'Matches': n_matches,
                'Teams': n_teams,
                'First Match': min_date,
                'Last Match': max_date
            })
            total_matches += n_matches
    
    summary_df = pd.DataFrame(summary)
    
    # Add totals row
    totals = pd.DataFrame([{
        'League': 'TOTAL',
        'Country': '',
        'Tier': '',
        'Seasons': '',
        'Matches': total_matches,
        'Teams': '',
        'First Match': '',
        'Last Match': ''
    }])
    summary_df = pd.concat([summary_df, totals], ignore_index=True)
    
    return summary_df


def get_teams_by_league(data: Dict[str, pd.DataFrame]) -> Dict[str, List[str]]:
    """Get sorted list of teams for each league."""
    teams = {}
    for league, df in data.items():
        if not df.empty:
            home_teams = set(df['HomeTeam'].unique())
            away_teams = set(df['AwayTeam'].unique())
            teams[league] = sorted(home_teams | away_teams)
    return teams


# Quick access functions
def load_epl(seasons=None):
    """Shortcut to load EPL data."""
    return load_league_data('epl', seasons)

def load_championship(seasons=None):
    """Shortcut to load Championship data."""
    return load_league_data('championship', seasons)

def load_bundesliga1(seasons=None):
    """Shortcut to load Bundesliga 1 data."""
    return load_league_data('bundesliga1', seasons)

def load_seriea(seasons=None):
    """Shortcut to load Serie A data."""
    return load_league_data('seriea', seasons)

def load_laliga(seasons=None):
    """Shortcut to load La Liga data."""
    return load_league_data('laliga', seasons)

def load_ligue1(seasons=None):
    """Shortcut to load Ligue 1 data."""
    return load_league_data('ligue1', seasons)
