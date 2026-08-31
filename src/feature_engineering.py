"""
Advanced Feature Engineering Module
Creates predictive features from raw match data for multiple betting markets.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple
from functools import lru_cache


# =============================================================================
# CORE FEATURE FUNCTIONS
# =============================================================================

def calculate_form(df: pd.DataFrame, team: str, date: pd.Timestamp, 
                   n_matches: int = 5, metric: str = 'points') -> float:
    """
    Calculate team form based on recent matches.
    
    Args:
        df: Full dataframe with match history
        team: Team name
        date: Reference date (only consider matches before this)
        n_matches: Number of recent matches to consider
        metric: 'points', 'goals_scored', 'goals_conceded', 'wins'
    
    Returns:
        Form value (0-1 for points, raw for others)
    """
    team_matches = df[
        ((df['HomeTeam'] == team) | (df['AwayTeam'] == team)) & 
        (df['Date'] < date)
    ].sort_values('Date', ascending=False).head(n_matches)
    
    if len(team_matches) == 0:
        return 0.5 if metric == 'points' else 0
    
    if metric == 'points':
        points = 0
        for _, row in team_matches.iterrows():
            if row['HomeTeam'] == team:
                if row['FTR'] == 'H': points += 3
                elif row['FTR'] == 'D': points += 1
            else:
                if row['FTR'] == 'A': points += 3
                elif row['FTR'] == 'D': points += 1
        return points / (3 * len(team_matches))
    
    elif metric == 'goals_scored':
        scored = 0
        for _, row in team_matches.iterrows():
            if row['HomeTeam'] == team:
                scored += row.get('FTHG', 0)
            else:
                scored += row.get('FTAG', 0)
        return scored / len(team_matches)
    
    elif metric == 'goals_conceded':
        conceded = 0
        for _, row in team_matches.iterrows():
            if row['HomeTeam'] == team:
                conceded += row.get('FTAG', 0)
            else:
                conceded += row.get('FTHG', 0)
        return conceded / len(team_matches)
    
    elif metric == 'wins':
        wins = 0
        for _, row in team_matches.iterrows():
            if row['HomeTeam'] == team:
                if row['FTR'] == 'H': wins += 1
            else:
                if row['FTR'] == 'A': wins += 1
        return wins / len(team_matches)
    
    return 0


def calculate_h2h(df: pd.DataFrame, home_team: str, away_team: str, 
                  date: pd.Timestamp, n_matches: int = 10) -> float:
    """
    Calculate head-to-head advantage for home team.
    
    Returns:
        Advantage score (0-1, where 1 = home team dominates)
    """
    h2h = df[
        ((df['HomeTeam'] == home_team) & (df['AwayTeam'] == away_team)) | 
        ((df['HomeTeam'] == away_team) & (df['AwayTeam'] == home_team))
    ][df['Date'] < date].sort_values('Date', ascending=False).head(n_matches)
    
    if len(h2h) == 0:
        return 0.5
    
    home_wins = 0
    away_wins = 0
    draws = 0
    
    for _, row in h2h.iterrows():
        if row['FTR'] == 'H':
            if row['HomeTeam'] == home_team:
                home_wins += 1
            else:
                away_wins += 1
        elif row['FTR'] == 'A':
            if row['AwayTeam'] == home_team:
                home_wins += 1
            else:
                away_wins += 1
        else:
            draws += 1
    
    total = home_wins + away_wins + draws
    return (home_wins + 0.5 * draws) / total


def calculate_rolling_stats(df: pd.DataFrame, team: str, date: pd.Timestamp,
                           stat_cols: List[str], n_matches: int = 5,
                           decay_factor: float = 0.95) -> Dict[str, float]:
    """
    Calculate time-decayed rolling averages for specified stats.
    
    Args:
        df: Full dataframe
        team: Team name
        date: Reference date
        stat_cols: List of statistic columns to average
        n_matches: Number of recent matches
        decay_factor: Decay factor for time weighting
    
    Returns:
        Dictionary of rolling averages
    """
    team_matches = df[
        ((df['HomeTeam'] == team) | (df['AwayTeam'] == team)) & 
        (df['Date'] < date)
    ].sort_values('Date', ascending=False).head(n_matches)
    
    if len(team_matches) < n_matches:
        return {col: 0 for col in stat_cols}
    
    rolling_stats = {}
    for col in stat_cols:
        stat_values = []
        weights = []
        for i, (_, row) in enumerate(team_matches.iterrows()):
            # Adjust column name for away stats
            if row['HomeTeam'] != team:
                stat_col_name = col.replace('H', 'A') if 'H' in col else col.replace('A', 'H')
            else:
                stat_col_name = col
            
            if stat_col_name in row and pd.notna(row[stat_col_name]):
                stat_values.append(row[stat_col_name])
                weights.append(decay_factor ** i)
        
        if sum(weights) > 0:
            rolling_stats[col] = np.average(stat_values, weights=weights)
        else:
            rolling_stats[col] = 0
    
    return rolling_stats


def calculate_elo(df: pd.DataFrame, k_factor: int = 20, initial_elo: int = 1500) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """
    Calculate Elo ratings for each team over time.
    
    Returns:
        Tuple of (DataFrame with Elo columns, final Elo ratings dict)
    """
    elos = {}
    df = df.sort_values('Date').reset_index(drop=True)
    home_elos, away_elos = [], []
    
    for i, row in df.iterrows():
        home_team = row['HomeTeam']
        away_team = row['AwayTeam']
        
        home_elo = elos.get(home_team, initial_elo)
        away_elo = elos.get(away_team, initial_elo)
        
        home_elos.append(home_elo)
        away_elos.append(away_elo)
        
        # Expected scores
        expected_home = 1 / (1 + 10 ** ((away_elo - home_elo) / 400))
        expected_away = 1 - expected_home
        
        # Actual scores
        if row['FTR'] == 'H':
            actual_home, actual_away = 1, 0
        elif row['FTR'] == 'A':
            actual_home, actual_away = 0, 1
        else:
            actual_home, actual_away = 0.5, 0.5
        
        # Update Elo
        elos[home_team] = home_elo + k_factor * (actual_home - expected_home)
        elos[away_team] = away_elo + k_factor * (actual_away - expected_away)
    
    df['HomeElo'] = home_elos
    df['AwayElo'] = away_elos
    
    return df, elos


# =============================================================================
# ODDS FEATURES
# =============================================================================

def create_odds_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create normalized implied probabilities from betting odds."""
    df = df.copy()
    
    # 3-way outcome odds
    for col in ['AvgH', 'AvgD', 'AvgA']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            df[col] = df[col].replace(0, np.nan)
    
    # Implied probabilities for 3-way outcome
    if all(col in df.columns for col in ['AvgH', 'AvgD', 'AvgA']):
        df['ImpliedProb_H'] = 1 / df['AvgH']
        df['ImpliedProb_D'] = 1 / df['AvgD']
        df['ImpliedProb_A'] = 1 / df['AvgA']
        
        total_prob_3way = df['ImpliedProb_H'] + df['ImpliedProb_D'] + df['ImpliedProb_A']
        df['NormProb_H'] = df['ImpliedProb_H'] / total_prob_3way
        df['NormProb_D'] = df['ImpliedProb_D'] / total_prob_3way
        df['NormProb_A'] = df['ImpliedProb_A'] / total_prob_3way
        
        # Bookmaker margin (overround)
        df['Overround'] = total_prob_3way - 1
    
    # Over/Under 2.5 odds
    for col in ['Avg>2.5', 'Avg<2.5']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            df[col] = df[col].replace(0, np.nan)
    
    if all(col in df.columns for col in ['Avg>2.5', 'Avg<2.5']):
        df['ImpliedProb_O2.5'] = 1 / df['Avg>2.5']
        df['ImpliedProb_U2.5'] = 1 / df['Avg<2.5']
        total_prob_ou = df['ImpliedProb_O2.5'] + df['ImpliedProb_U2.5']
        df['NormProb_O2.5'] = df['ImpliedProb_O2.5'] / total_prob_ou
        df['NormProb_U2.5'] = df['ImpliedProb_U2.5'] / total_prob_ou
    
    # Asian Handicap odds
    for col in ['AvgAHH', 'AvgAHA']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            df[col] = df[col].replace(0, np.nan)
    
    if all(col in df.columns for col in ['AvgAHH', 'AvgAHA']):
        df['ImpliedProb_AHH'] = 1 / df['AvgAHH']
        df['ImpliedProb_AHA'] = 1 / df['AvgAHA']
        total_prob_ah = df['ImpliedProb_AHH'] + df['ImpliedProb_AHA']
        df['NormProb_AHH'] = df['ImpliedProb_AHH'] / total_prob_ah
        df['NormProb_AHA'] = df['ImpliedProb_AHA'] / total_prob_ah
    
    return df


# =============================================================================
# MARKET-SPECIFIC TARGET VARIABLES
# =============================================================================

def create_market_targets(df: pd.DataFrame) -> pd.DataFrame:
    """Create target variables for all supported markets."""
    df = df.copy()
    
    # Match Result (1X2)
    outcome_map = {'A': 0, 'D': 1, 'H': 2}
    df['Target_MatchResult'] = df['FTR'].map(outcome_map)
    
    # Half-Time Result
    df['Target_HTResult'] = df['HTR'].map(outcome_map)
    
    # Over/Under 2.5
    df['TotalGoals'] = df['FTHG'] + df['FTAG']
    df['Target_OverUnder25'] = (df['TotalGoals'] > 2.5).astype(int)
    
    # Both Teams to Score
    df['Target_BTTS'] = ((df['FTHG'] > 0) & (df['FTAG'] > 0)).astype(int)
    
    # Double Chance: Home or Draw (1X)
    df['Target_DoubleChance_1X'] = df['FTR'].isin(['H', 'D']).astype(int)
    
    # Double Chance: Home or Away (12)
    df['Target_DoubleChance_12'] = df['FTR'].isin(['H', 'A']).astype(int)
    
    # Double Chance: Draw or Away (X2)
    df['Target_DoubleChance_X2'] = df['FTR'].isin(['D', 'A']).astype(int)
    
    # Asian Handicap
    if 'AHh' in df.columns:
        df['AHh'] = pd.to_numeric(df['AHh'], errors='coerce')
        effective_score = (df['FTHG'] - df['FTAG']) + df['AHh']
        df['Target_AsianHandicap'] = np.nan
        df.loc[effective_score > 0, 'Target_AsianHandicap'] = 1
        df.loc[effective_score < 0, 'Target_AsianHandicap'] = 0
    
    # Total Goals Buckets
    def goals_bucket(goals):
        if goals <= 1: return '0-1'
        elif goals <= 3: return '2-3'
        else: return '4+'
    df['Target_GoalsBucket'] = df['TotalGoals'].apply(goals_bucket)
    
    # Total Corners
    df['Target_TotalCorners'] = df['HC'] + df['AC']
    
    # Total Yellow Cards
    df['Target_TotalYellowCards'] = df['HY'] + df['AY']
    
    # Total Shots
    df['Target_TotalShots'] = df['HS'] + df['AS']
    
    # Total Shots on Target
    df['Target_TotalShotsOnTarget'] = df['HST'] + df['AST']
    
    # Half-Time Home Goals
    df['Target_HTHomeGoals'] = df['HTHG']
    
    # Half-Time Away Goals
    df['Target_HTAwayGoals'] = df['HTAG']
    
    # Goal Difference
    df['Target_GoalDiff'] = df['FTHG'] - df['FTAG']
    
    # Clean Sheet - Home
    df['Target_HomeCleanSheet'] = (df['FTAG'] == 0).astype(int)
    
    # Clean Sheet - Away
    df['Target_AwayCleanSheet'] = (df['FTHG'] == 0).astype(int)
    
    # Win to Nil - Home
    df['Target_HomeWinToNil'] = ((df['FTR'] == 'H') & (df['FTAG'] == 0)).astype(int)
    
    # Win to Nil - Away
    df['Target_AwayWinToNil'] = ((df['FTR'] == 'A') & (df['FTHG'] == 0)).astype(int)
    
    # Over/Under 1.5
    df['Target_OverUnder15'] = (df['TotalGoals'] > 1.5).astype(int)
    
    # Over/Under 3.5
    df['Target_OverUnder35'] = (df['TotalGoals'] > 3.5).astype(int)
    
    # Double Chance HT: Home or Draw
    df['Target_HTDoubleChance_1X'] = df['HTR'].isin(['H', 'D']).astype(int)
    
    # Half-Time Away Goals
    df['Target_HTAwayGoals'] = df['HTAG']
    
    # Correct Score (for ensemble, not primary)
    df['Target_CorrectScore'] = df['FTHG'].astype(str) + '-' + df['FTAG'].astype(str)
    
    return df


# =============================================================================
# ADVANCED FEATURE FUNCTIONS
# =============================================================================

def calculate_streak(df: pd.DataFrame, team: str, date: pd.Timestamp, 
                     n_matches: int = 10) -> Dict[str, float]:
    """
    Calculate winning/losing streaks and momentum.
    
    Returns:
        Dictionary with streak features
    """
    team_matches = df[
        ((df['HomeTeam'] == team) | (df['AwayTeam'] == team)) & 
        (df['Date'] < date)
    ].sort_values('Date', ascending=False).head(n_matches)
    
    if len(team_matches) == 0:
        return {'current_streak': 0, 'max_win_streak': 0, 'max_loss_streak': 0, 'momentum': 0}
    
    results = []
    for _, row in team_matches.iterrows():
        if row['HomeTeam'] == team:
            if row['FTR'] == 'H': results.append(1)
            elif row['FTR'] == 'D': results.append(0)
            else: results.append(-1)
        else:
            if row['FTR'] == 'A': results.append(1)
            elif row['FTR'] == 'D': results.append(0)
            else: results.append(-1)
    
    # Current streak (positive = wins, negative = losses)
    current_streak = 0
    if results[0] > 0:
        for r in results:
            if r > 0: current_streak += 1
            else: break
    elif results[0] < 0:
        for r in results:
            if r < 0: current_streak -= 1
            else: break
    
    # Max win/loss streaks
    max_win = max_loss = current = 0
    for r in results:
        if r > 0:
            current = max(current + 1, 0)
            max_win = max(max_win, current)
        elif r < 0:
            current = min(current - 1, 0)
            max_loss = max(max_loss, abs(current))
        else:
            current = 0
    
    # Momentum (weighted recent results)
    momentum = sum(r * (0.8 ** i) for i, r in enumerate(results))
    
    return {
        'current_streak': current_streak,
        'max_win_streak': max_win,
        'max_loss_streak': max_loss,
        'momentum': momentum
    }


def calculate_home_away_split(df: pd.DataFrame, team: str, date: pd.Timestamp,
                             n_matches: int = 5) -> Dict[str, float]:
    """
    Calculate separate home and away form.
    
    Returns:
        Dictionary with home/away split features
    """
    # Home matches
    home_matches = df[
        (df['HomeTeam'] == team) & (df['Date'] < date)
    ].sort_values('Date', ascending=False).head(n_matches)
    
    # Away matches
    away_matches = df[
        (df['AwayTeam'] == team) & (df['Date'] < date)
    ].sort_values('Date', ascending=False).head(n_matches)
    
    # Home form
    if len(home_matches) == 0:
        home_form = 0.5
        home_goals_scored = 0
        home_goals_conceded = 0
    else:
        points = 0
        scored = 0
        conceded = 0
        for _, row in home_matches.iterrows():
            if row['FTR'] == 'H': points += 3
            elif row['FTR'] == 'D': points += 1
            scored += row.get('FTHG', 0)
            conceded += row.get('FTAG', 0)
        home_form = points / (3 * len(home_matches))
        home_goals_scored = scored / len(home_matches)
        home_goals_conceded = conceded / len(home_matches)
    
    # Away form
    if len(away_matches) == 0:
        away_form = 0.5
        away_goals_scored = 0
        away_goals_conceded = 0
    else:
        points = 0
        scored = 0
        conceded = 0
        for _, row in away_matches.iterrows():
            if row['FTR'] == 'A': points += 3
            elif row['FTR'] == 'D': points += 1
            scored += row.get('FTAG', 0)
            conceded += row.get('FTHG', 0)
        away_form = points / (3 * len(away_matches))
        away_goals_scored = scored / len(away_matches)
        away_goals_conceded = conceded / len(away_matches)
    
    return {
        'home_form_split': home_form,
        'away_form_split': away_form,
        'home_goals_scored_split': home_goals_scored,
        'home_goals_conceded_split': home_goals_conceded,
        'away_goals_scored_split': away_goals_scored,
        'away_goals_conceded_split': away_goals_conceded,
        'home_advantage': home_form - away_form
    }


def calculate_goal_timing(df: pd.DataFrame, team: str, date: pd.Timestamp,
                          n_matches: int = 5) -> Dict[str, float]:
    """
    Calculate goal timing patterns (HT vs FT goals).
    
    Returns:
        Dictionary with goal timing features
    """
    team_matches = df[
        ((df['HomeTeam'] == team) | (df['AwayTeam'] == team)) & 
        (df['Date'] < date)
    ].sort_values('Date', ascending=False).head(n_matches)
    
    if len(team_matches) == 0:
        return {'ht_goals_ratio': 0.5, 'second_half_goals_ratio': 0.5}
    
    ht_goals = 0
    ft_goals = 0
    second_half_goals = 0
    
    for _, row in team_matches.iterrows():
        if row['HomeTeam'] == team:
            ht_goals += row.get('HTHG', 0)
            ft_goals += row.get('FTHG', 0)
        else:
            ht_goals += row.get('HTAG', 0)
            ft_goals += row.get('FTAG', 0)
        
        if ft_goals > 0:
            second_half_goals = ft_goals - ht_goals
    
    ht_ratio = ht_goals / (ft_goals + 1e-6)
    second_half_ratio = second_half_goals / (ft_goals + 1e-6)
    
    return {
        'ht_goals_ratio': ht_ratio,
        'second_half_goals_ratio': second_half_ratio
    }


def calculate_league_position_proxy(df: pd.DataFrame, team: str, date: pd.Timestamp,
                                    n_matches: int = 10) -> float:
    """
    Calculate a proxy for league position based on points per game.
    
    Returns:
        Points per game (proxy for league position)
    """
    team_matches = df[
        ((df['HomeTeam'] == team) | (df['AwayTeam'] == team)) & 
        (df['Date'] < date)
    ].sort_values('Date', ascending=False).head(n_matches)
    
    if len(team_matches) == 0:
        return 1.0  # Neutral
    
    points = 0
    for _, row in team_matches.iterrows():
        if row['HomeTeam'] == team:
            if row['FTR'] == 'H': points += 3
            elif row['FTR'] == 'D': points += 1
        else:
            if row['FTR'] == 'A': points += 3
            elif row['FTR'] == 'D': points += 1
    
    ppg = points / len(team_matches)
    return ppg / 3  # Normalize to 0-1


def calculate_clean_sheet_prob(df: pd.DataFrame, team: str, date: pd.Timestamp,
                               n_matches: int = 10) -> float:
    """
    Calculate clean sheet probability.
    
    Returns:
        Clean sheet rate (0-1)
    """
    team_matches = df[
        ((df['HomeTeam'] == team) | (df['AwayTeam'] == team)) & 
        (df['Date'] < date)
    ].sort_values('Date', ascending=False).head(n_matches)
    
    if len(team_matches) == 0:
        return 0.25  # Default
    
    clean_sheets = 0
    for _, row in team_matches.iterrows():
        if row['HomeTeam'] == team:
            if row.get('FTAG', 0) == 0: clean_sheets += 1
        else:
            if row.get('FTHG', 0) == 0: clean_sheets += 1
    
    return clean_sheets / len(team_matches)


def calculate_btts_prob(df: pd.DataFrame, home_team: str, away_team: str,
                        date: pd.Timestamp, n_matches: int = 10) -> float:
    """
    Calculate BTTS probability based on team histories.
    
    Returns:
        BTTS rate (0-1)
    """
    # Home team BTTS when playing at home
    home_btts_matches = df[
        (df['HomeTeam'] == home_team) & (df['Date'] < date)
    ].sort_values('Date', ascending=False).head(n_matches)
    
    home_btts_rate = 0.5
    if len(home_btts_matches) >= 5:
        home_btts = ((home_btts_matches['FTHG'] > 0) & 
                     (home_btts_matches['FTAG'] > 0)).sum()
        home_btts_rate = home_btts / len(home_btts_matches)
    
    # Away team BTTS when playing away
    away_btts_matches = df[
        (df['AwayTeam'] == away_team) & (df['Date'] < date)
    ].sort_values('Date', ascending=False).head(n_matches)
    
    away_btts_rate = 0.5
    if len(away_btts_matches) >= 5:
        away_btts = ((away_btts_matches['FTHG'] > 0) & 
                     (away_btts_matches['FTAG'] > 0)).sum()
        away_btts_rate = away_btts / len(away_btts_matches)
    
    return (home_btts_rate + away_btts_rate) / 2


def calculate_attack_defense_strength(df: pd.DataFrame, team: str, date: pd.Timestamp,
                                     n_matches: int = 10) -> Dict[str, float]:
    """
    Calculate attack and defense strength relative to league average.
    
    Returns:
        Dictionary with attack/defense strength
    """
    # Get league averages
    recent_matches = df[df['Date'] < date].tail(n_matches * 20)
    if len(recent_matches) == 0:
        return {'attack_strength': 1.0, 'defense_strength': 1.0}
    
    avg_goals_per_match = (recent_matches['FTHG'].sum() + recent_matches['FTAG'].sum()) / len(recent_matches)
    avg_home_goals = recent_matches['FTHG'].mean()
    avg_away_goals = recent_matches['FTAG'].mean()
    
    # Team's goals scored and conceded
    team_matches = df[
        ((df['HomeTeam'] == team) | (df['AwayTeam'] == team)) & 
        (df['Date'] < date)
    ].sort_values('Date', ascending=False).head(n_matches)
    
    if len(team_matches) == 0:
        return {'attack_strength': 1.0, 'defense_strength': 1.0}
    
    goals_scored = 0
    goals_conceded = 0
    home_games = 0
    away_games = 0
    
    for _, row in team_matches.iterrows():
        if row['HomeTeam'] == team:
            goals_scored += row.get('FTHG', 0)
            goals_conceded += row.get('FTAG', 0)
            home_games += 1
        else:
            goals_scored += row.get('FTAG', 0)
            goals_conceded += row.get('FTHG', 0)
            away_games += 1
    
    total_games = home_games + away_games
    if total_games == 0:
        return {'attack_strength': 1.0, 'defense_strength': 1.0}
    
    attack_strength = (goals_scored / total_games) / (avg_goals_per_match / 2)
    defense_strength = (goals_conceded / total_games) / (avg_goals_per_match / 2)
    
    return {
        'attack_strength': attack_strength,
        'defense_strength': defense_strength
    }


# =============================================================================
# MAIN FEATURE ENGINEERING PIPELINE
# =============================================================================
def engineer_features(df: pd.DataFrame, include_elo: bool = True,
                     include_rolling: bool = True, include_odds: bool = True,
                     include_targets: bool = True) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """
    Full feature engineering pipeline.
    
    Args:
        df: Raw match data
        include_elo: Whether to calculate Elo ratings
        include_rolling: Whether to calculate rolling stats
        include_odds: Whether to calculate odds features
        include_targets: Whether to create target variables
    
    Returns:
        Tuple of (DataFrame with features, final Elo ratings)
    """
    print("Feature Engineering Pipeline")
    print("=" * 50)
    
    # Parse dates
    df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y', errors='coerce')
    df = df.dropna(subset=['Date'])
    df = df.sort_values('Date').reset_index(drop=True)
    
    print("1. Calculating form features...")
    # Form (last 5 matches)
    df['HomeTeamForm'] = df.apply(
        lambda x: calculate_form(df, x['HomeTeam'], x['Date'], 5, 'points'), axis=1
    )
    df['AwayTeamForm'] = df.apply(
        lambda x: calculate_form(df, x['AwayTeam'], x['Date'], 5, 'points'), axis=1
    )
    
    # Goals scored/conceded form
    df['HomeGoalsScoredForm'] = df.apply(
        lambda x: calculate_form(df, x['HomeTeam'], x['Date'], 5, 'goals_scored'), axis=1
    )
    df['AwayGoalsScoredForm'] = df.apply(
        lambda x: calculate_form(df, x['AwayTeam'], x['Date'], 5, 'goals_scored'), axis=1
    )
    df['HomeGoalsConcededForm'] = df.apply(
        lambda x: calculate_form(df, x['HomeTeam'], x['Date'], 5, 'goals_conceded'), axis=1
    )
    df['AwayGoalsConcededForm'] = df.apply(
        lambda x: calculate_form(df, x['AwayTeam'], x['Date'], 5, 'goals_conceded'), axis=1
    )
    
    # Win rate form
    df['HomeWinForm'] = df.apply(
        lambda x: calculate_form(df, x['HomeTeam'], x['Date'], 5, 'wins'), axis=1
    )
    df['AwayWinForm'] = df.apply(
        lambda x: calculate_form(df, x['AwayTeam'], x['Date'], 5, 'wins'), axis=1
    )
    
    print("2. Calculating H2H features...")
    df['H2H_Advantage'] = df.apply(
        lambda x: calculate_h2h(df, x['HomeTeam'], x['AwayTeam'], x['Date']), axis=1
    )
    
    if include_rolling:
        print("3. Calculating rolling statistics...")
        # Performance stats
        home_perf = df.apply(
            lambda row: calculate_rolling_stats(df, row['HomeTeam'], row['Date'], 
                                               ['FTHG', 'HS', 'HST', 'HC']),
            axis=1, result_type='expand'
        )
        home_perf.columns = ['HomeAvgGoals', 'HomeAvgShots', 'HomeAvgShotsTarget', 'HomeAvgCorners']
        
        away_perf = df.apply(
            lambda row: calculate_rolling_stats(df, row['AwayTeam'], row['Date'], 
                                               ['FTAG', 'AS', 'AST', 'AC']),
            axis=1, result_type='expand'
        )
        away_perf.columns = ['AwayAvgGoals', 'AwayAvgShots', 'AwayAvgShotsTarget', 'AwayAvgCorners']
        
        df = pd.concat([df, home_perf, away_perf], axis=1)
        
        # Discipline stats
        home_disc = df.apply(
            lambda row: calculate_rolling_stats(df, row['HomeTeam'], row['Date'], ['HY', 'HR']),
            axis=1, result_type='expand'
        )
        home_disc.columns = ['HomeAvgYellows', 'HomeAvgReds']
        
        away_disc = df.apply(
            lambda row: calculate_rolling_stats(df, row['AwayTeam'], row['Date'], ['AY', 'AR']),
            axis=1, result_type='expand'
        )
        away_disc.columns = ['AwayAvgYellows', 'AwayAvgReds']
        
        df = pd.concat([df, home_disc, away_disc], axis=1)
    
    if include_elo:
        print("4. Calculating Elo ratings...")
        df, elos = calculate_elo(df)
    else:
        elos = {}
    
    if include_odds:
        print("5. Creating odds features...")
        df = create_odds_features(df)
    
    if include_targets:
        print("6. Creating market targets...")
        df = create_market_targets(df)
    
    # Advanced features
    print("7. Calculating advanced features...")
    
    # Streaks and momentum
    home_streaks = df.apply(
        lambda row: calculate_streak(df, row['HomeTeam'], row['Date']),
        axis=1, result_type='expand'
    )
    home_streaks.columns = ['HomeStreak', 'HomeMaxWinStreak', 'HomeMaxLossStreak', 'HomeMomentum']
    
    away_streaks = df.apply(
        lambda row: calculate_streak(df, row['AwayTeam'], row['Date']),
        axis=1, result_type='expand'
    )
    away_streaks.columns = ['AwayStreak', 'AwayMaxWinStreak', 'AwayMaxLossStreak', 'AwayMomentum']
    
    df = pd.concat([df, home_streaks, away_streaks], axis=1)
    
    # Home/Away split form
    home_split = df.apply(
        lambda row: calculate_home_away_split(df, row['HomeTeam'], row['Date']),
        axis=1, result_type='expand'
    )
    home_split.columns = ['HomeHomeForm', 'HomeAwayForm', 'HomeHomeGoalsScored',
                          'HomeHomeGoalsConceded', 'HomeAwayGoalsScored',
                          'HomeAwayGoalsConceded', 'HomeHomeAdvantage']
    
    away_split = df.apply(
        lambda row: calculate_home_away_split(df, row['AwayTeam'], row['Date']),
        axis=1, result_type='expand'
    )
    away_split.columns = ['AwayHomeForm', 'AwayAwayForm', 'AwayHomeGoalsScored',
                          'AwayHomeGoalsConceded', 'AwayAwayGoalsScored',
                          'AwayAwayGoalsConceded', 'AwayHomeAdvantage']
    
    df = pd.concat([df, home_split, away_split], axis=1)
    
    # Goal timing
    home_timing = df.apply(
        lambda row: calculate_goal_timing(df, row['HomeTeam'], row['Date']),
        axis=1, result_type='expand'
    )
    home_timing.columns = ['HomeHTGoalsRatio', 'Home2ndHalfGoalsRatio']
    
    away_timing = df.apply(
        lambda row: calculate_goal_timing(df, row['AwayTeam'], row['Date']),
        axis=1, result_type='expand'
    )
    away_timing.columns = ['AwayHTGoalsRatio', 'Away2ndHalfGoalsRatio']
    
    df = pd.concat([df, home_timing, away_timing], axis=1)
    
    # League position proxy
    df['HomeLeaguePosition'] = df.apply(
        lambda row: calculate_league_position_proxy(df, row['HomeTeam'], row['Date']),
        axis=1
    )
    df['AwayLeaguePosition'] = df.apply(
        lambda row: calculate_league_position_proxy(df, row['AwayTeam'], row['Date']),
        axis=1
    )
    
    # Clean sheet probability
    df['HomeCleanSheetProb'] = df.apply(
        lambda row: calculate_clean_sheet_prob(df, row['HomeTeam'], row['Date']),
        axis=1
    )
    df['AwayCleanSheetProb'] = df.apply(
        lambda row: calculate_clean_sheet_prob(df, row['AwayTeam'], row['Date']),
        axis=1
    )
    
    # Attack/Defense strength
    home_strength = df.apply(
        lambda row: calculate_attack_defense_strength(df, row['HomeTeam'], row['Date']),
        axis=1, result_type='expand'
    )
    home_strength.columns = ['HomeAttackStrength', 'HomeDefenseStrength']
    
    away_strength = df.apply(
        lambda row: calculate_attack_defense_strength(df, row['AwayTeam'], row['Date']),
        axis=1, result_type='expand'
    )
    away_strength.columns = ['AwayAttackStrength', 'AwayDefenseStrength']
    
    df = pd.concat([df, home_strength, away_strength], axis=1)
    
    # Derived features
    print("8. Creating derived features...")
    df['FormDiff'] = df['HomeTeamForm'] - df['AwayTeamForm']
    df['GoalsScoredDiff'] = df['HomeGoalsScoredForm'] - df['AwayGoalsScoredForm']
    df['GoalsConcededDiff'] = df['HomeGoalsConcededForm'] - df['AwayGoalsConcededForm']
    
    if 'HomeElo' in df.columns and 'AwayElo' in df.columns:
        df['EloDiff'] = df['HomeElo'] - df['AwayElo']
    
    # Shots efficiency
    if 'HomeAvgShots' in df.columns:
        df['HomeShotEfficiency'] = df['HomeAvgGoals'] / (df['HomeAvgShots'] + 1e-6)
        df['AwayShotEfficiency'] = df['AwayAvgGoals'] / (df['AwayAvgShots'] + 1e-6)
    
    # Momentum diff
    df['MomentumDiff'] = df['HomeMomentum'] - df['AwayMomentum']
    df['StreakDiff'] = df['HomeStreak'] - df['AwayStreak']
    
    # Attack vs Defense matchup
    df['AttackVsDefense'] = df['HomeAttackStrength'] - df['AwayDefenseStrength']
    df['DefenseVsAttack'] = df['HomeDefenseStrength'] - df['AwayAttackStrength']
    
    print(f"\n✓ Feature engineering complete: {len(df)} matches, {len(df.columns)} columns")
    
    return df, elos


# =============================================================================
# FEATURE LISTS FOR MODELS
# =============================================================================

# Base features used across all markets
BASE_FEATURES = [
    # Form
    'HomeTeamForm', 'AwayTeamForm', 'FormDiff',
    'HomeGoalsScoredForm', 'AwayGoalsScoredForm', 'GoalsScoredDiff',
    'HomeGoalsConcededForm', 'AwayGoalsConcededForm', 'GoalsConcededDiff',
    'HomeWinForm', 'AwayWinForm',
    # H2H
    'H2H_Advantage',
    # Rolling Stats
    'HomeAvgGoals', 'HomeAvgShots', 'HomeAvgShotsTarget', 'HomeAvgCorners',
    'AwayAvgGoals', 'AwayAvgShots', 'AwayAvgShotsTarget', 'AwayAvgCorners',
    'HomeAvgYellows', 'HomeAvgReds', 'AwayAvgYellows', 'AwayAvgReds',
    # Elo
    'HomeElo', 'AwayElo', 'EloDiff',
    # Efficiency
    'HomeShotEfficiency', 'AwayShotEfficiency',
    # Streaks & Momentum
    'HomeStreak', 'AwayStreak', 'StreakDiff',
    'HomeMomentum', 'AwayMomentum', 'MomentumDiff',
    # Home/Away Split
    'HomeHomeForm', 'HomeAwayForm',
    'AwayHomeForm', 'AwayAwayForm',
    'HomeHomeAdvantage', 'AwayHomeAdvantage',
    # Goal Timing
    'HomeHTGoalsRatio', 'Home2ndHalfGoalsRatio',
    'AwayHTGoalsRatio', 'Away2ndHalfGoalsRatio',
    # League Position
    'HomeLeaguePosition', 'AwayLeaguePosition',
    # Clean Sheet
    'HomeCleanSheetProb', 'AwayCleanSheetProb',
    # Attack/Defense Strength
    'HomeAttackStrength', 'HomeDefenseStrength',
    'AwayAttackStrength', 'AwayDefenseStrength',
    'AttackVsDefense', 'DefenseVsAttack'
]

# Market-specific feature sets
MARKET_FEATURES = {
    'match_result': BASE_FEATURES + ['NormProb_H', 'NormProb_D', 'NormProb_A'],
    'ht_result': BASE_FEATURES + ['NormProb_H', 'NormProb_D', 'NormProb_A'],
    'over_under_25': BASE_FEATURES + ['NormProb_O2.5', 'NormProb_U2.5'],
    'btts': BASE_FEATURES + ['NormProb_O2.5', 'NormProb_U2.5',
                              'HomeCleanSheetProb', 'AwayCleanSheetProb'],
    'double_chance_1x': BASE_FEATURES + ['NormProb_H', 'NormProb_D', 'NormProb_A'],
    'double_chance_12': BASE_FEATURES + ['NormProb_H', 'NormProb_D', 'NormProb_A'],
    'double_chance_x2': BASE_FEATURES + ['NormProb_H', 'NormProb_D', 'NormProb_A'],
    'asian_handicap': BASE_FEATURES + ['NormProb_AHH', 'NormProb_AHA'],
    'goals_bucket': BASE_FEATURES + ['NormProb_H', 'NormProb_D', 'NormProb_A', 
                                      'NormProb_O2.5', 'NormProb_U2.5'],
    'total_corners': BASE_FEATURES,
    'total_yellow_cards': BASE_FEATURES,
    'total_shots': BASE_FEATURES,
    'total_shots_on_target': BASE_FEATURES,
    'ht_home_goals': BASE_FEATURES,
    'ht_away_goals': BASE_FEATURES,
    'goal_difference': BASE_FEATURES + ['NormProb_H', 'NormProb_D', 'NormProb_A'],
    # New markets
    'home_clean_sheet': BASE_FEATURES + ['HomeCleanSheetProb', 'AwayAttackStrength'],
    'away_clean_sheet': BASE_FEATURES + ['AwayCleanSheetProb', 'HomeAttackStrength'],
    'home_win_to_nil': BASE_FEATURES + ['NormProb_H', 'HomeCleanSheetProb'],
    'away_win_to_nil': BASE_FEATURES + ['NormProb_A', 'AwayCleanSheetProb'],
    'over_under_15': BASE_FEATURES + ['NormProb_O2.5', 'NormProb_U2.5'],
    'over_under_35': BASE_FEATURES + ['NormProb_O2.5', 'NormProb_U2.5'],
    'ht_double_chance_1x': BASE_FEATURES + ['NormProb_H', 'NormProb_D', 'NormProb_A']
}

# Target variable mapping
MARKET_TARGETS = {
    'match_result': 'Target_MatchResult',
    'ht_result': 'Target_HTResult',
    'over_under_25': 'Target_OverUnder25',
    'btts': 'Target_BTTS',
    'double_chance_1x': 'Target_DoubleChance_1X',
    'double_chance_12': 'Target_DoubleChance_12',
    'double_chance_x2': 'Target_DoubleChance_X2',
    'asian_handicap': 'Target_AsianHandicap',
    'goals_bucket': 'Target_GoalsBucket',
    'total_corners': 'Target_TotalCorners',
    'total_yellow_cards': 'Target_TotalYellowCards',
    'total_shots': 'Target_TotalShots',
    'total_shots_on_target': 'Target_TotalShotsOnTarget',
    'ht_home_goals': 'Target_HTHomeGoals',
    'ht_away_goals': 'Target_HTAwayGoals',
    'goal_difference': 'Target_GoalDiff',
    # New markets
    'home_clean_sheet': 'Target_HomeCleanSheet',
    'away_clean_sheet': 'Target_AwayCleanSheet',
    'home_win_to_nil': 'Target_HomeWinToNil',
    'away_win_to_nil': 'Target_AwayWinToNil',
    'over_under_15': 'Target_OverUnder15',
    'over_under_35': 'Target_OverUnder35',
    'ht_double_chance_1x': 'Target_HTDoubleChance_1X'
}

# Market types (classification vs regression)
MARKET_TYPES = {
    'match_result': 'classification',
    'ht_result': 'classification',
    'over_under_25': 'classification',
    'btts': 'classification',
    'double_chance_1x': 'classification',
    'double_chance_12': 'classification',
    'double_chance_x2': 'classification',
    'asian_handicap': 'classification',
    'goals_bucket': 'classification',
    'total_corners': 'regression',
    'total_yellow_cards': 'regression',
    'total_shots': 'regression',
    'total_shots_on_target': 'regression',
    'ht_home_goals': 'regression',
    'ht_away_goals': 'regression',
    'goal_difference': 'regression',
    # New markets
    'home_clean_sheet': 'classification',
    'away_clean_sheet': 'classification',
    'home_win_to_nil': 'classification',
    'away_win_to_nil': 'classification',
    'over_under_15': 'classification',
    'over_under_35': 'classification',
    'ht_double_chance_1x': 'classification'
}
