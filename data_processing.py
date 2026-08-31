import pandas as pd
import numpy as np

def load_data(files):
    """Loads and concatenates data from a list of CSV files."""
    """Loads and concatenates data from a list of CSV files."""
    df_list = [pd.read_csv(file) for file in files]
    return pd.concat(df_list, ignore_index=True)

def preprocess_data(df):
    """Preprocesses the raw dataframe."""
    df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y', errors='coerce')
    # Create a 3-way outcome target: 0=Away, 1=Draw, 2=Home
    outcome_map = {'A': 0, 'D': 1, 'H': 2}
    df['MatchOutcome'] = df['FTR'].map(outcome_map)
    df['TotalGoals'] = df['FTHG'] + df['FTAG']
    df['Over2_5'] = (df['TotalGoals'] > 2.5).astype(int)
    # Drop the old HomeWin column if it exists to avoid confusion
    if 'HomeWin' in df.columns:
        df = df.drop(columns=['HomeWin'])
    return df

def calculate_form(df, team, date, n_matches=5):
    """Calculates a team's form based on the last n matches."""
    team_matches = df[((df['HomeTeam'] == team) | (df['AwayTeam'] == team)) & 
                     (df['Date'] < date)].sort_values('Date', ascending=False).head(n_matches)
    
    if len(team_matches) < n_matches:
        return 0.5  # Neutral form if not enough matches
    
    points = 0
    for _, row in team_matches.iterrows():
        if row['HomeTeam'] == team:
            if row['FTR'] == 'H':
                points += 3
            elif row['FTR'] == 'D':
                points += 1
        else:  # Away team
            if row['FTR'] == 'A':
                points += 3
            elif row['FTR'] == 'D':
                points += 1
                
    return points / (3 * n_matches)  # Normalize to 0-1

def calculate_h2h(df, home_team, away_team, date):
    """Calculates the head-to-head advantage."""
    h2h = df[((df['HomeTeam'] == home_team) & (df['AwayTeam'] == away_team) | 
             (df['HomeTeam'] == away_team) & (df['AwayTeam'] == home_team)) & 
            (df['Date'] < date)].sort_values('Date', ascending=False)
    
    if len(h2h) == 0:
        return 0.5  # No previous matches
    
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

def calculate_rolling_stats(df, team, date, stat_cols, n_matches=5, decay_factor=0.95):
    """Calculates time-decayed rolling averages for specified stats."""
    team_matches = df[((df['HomeTeam'] == team) | (df['AwayTeam'] == team)) & 
                     (df['Date'] < date)].sort_values('Date', ascending=False).head(n_matches)
    
    if len(team_matches) < n_matches:
        return {col: 0 for col in stat_cols}  # Return zeros if not enough matches

    rolling_stats = {}
    for col in stat_cols:
        stat_values = []
        weights = []
        for i, (_, row) in enumerate(team_matches.iterrows()):
            # Adjust column name for away stats (e.g., FTHG -> FTAG)
            if row['HomeTeam'] != team:
                stat_col_name = col.replace('H', 'A') if 'H' in col else col.replace('A', 'H')
            else:
                stat_col_name = col
            
            if stat_col_name in row:
                stat_values.append(row[stat_col_name])
                weights.append(decay_factor**i)
        
        if sum(weights) > 0:
            rolling_stats[col] = np.average(stat_values, weights=weights)
        else:
            rolling_stats[col] = 0

    return rolling_stats

def calculate_elo(df):
    """Calculates Elo ratings for each team over time."""
    elos = {}
    elo_k = 20 # Elo K-factor
    initial_elo = 1500

    df = df.sort_values('Date').reset_index(drop=True)
    home_elos, away_elos = [], []

    for i, row in df.iterrows():
        home_team = row['HomeTeam']
        away_team = row['AwayTeam']

        # Initialize Elo ratings if not present
        home_elo = elos.get(home_team, initial_elo)
        away_elo = elos.get(away_team, initial_elo)

        home_elos.append(home_elo)
        away_elos.append(away_elo)

        # Elo calculation
        expected_home = 1 / (1 + 10**((away_elo - home_elo) / 400))
        expected_away = 1 - expected_home

        if row['FTR'] == 'H':
            actual_home, actual_away = 1, 0
        elif row['FTR'] == 'A':
            actual_home, actual_away = 0, 1
        else: # Draw
            actual_home, actual_away = 0.5, 0.5

        new_home_elo = home_elo + elo_k * (actual_home - expected_home)
        new_away_elo = away_elo + elo_k * (actual_away - expected_away)

        elos[home_team] = new_home_elo
        elos[away_team] = new_away_elo

    df['HomeElo'] = home_elos
    df['AwayElo'] = away_elos
    return df, elos

def create_odds_features(df):
    """Calculates normalized implied probabilities from betting odds."""
    # Use average odds for robustness
    odds_cols_3way = ['AvgH', 'AvgD', 'AvgA']
    odds_cols_ou = ['Avg>2.5', 'Avg<2.5']

    # Handle potential zero or NaN odds to avoid division errors
    for col in odds_cols_3way + odds_cols_ou:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        df[col] = df[col].replace(0, np.nan)

    # Calculate implied probabilities for 3-way outcome
    df['ImpliedProb_H'] = 1 / df['AvgH']
    df['ImpliedProb_D'] = 1 / df['AvgD']
    df['ImpliedProb_A'] = 1 / df['AvgA']
    
    # Normalize 3-way probabilities
    total_prob_3way = df['ImpliedProb_H'] + df['ImpliedProb_D'] + df['ImpliedProb_A']
    df['NormProb_H'] = df['ImpliedProb_H'] / total_prob_3way
    df['NormProb_D'] = df['ImpliedProb_D'] / total_prob_3way
    df['NormProb_A'] = df['ImpliedProb_A'] / total_prob_3way

    # Calculate and normalize Over/Under 2.5 probabilities
    df['ImpliedProb_O2.5'] = 1 / df['Avg>2.5']
    df['ImpliedProb_U2.5'] = 1 / df['Avg<2.5']
    total_prob_ou = df['ImpliedProb_O2.5'] + df['ImpliedProb_U2.5']
    df['NormProb_O2.5'] = df['ImpliedProb_O2.5'] / total_prob_ou
    df['NormProb_U2.5'] = df['ImpliedProb_U2.5'] / total_prob_ou

    # Calculate and normalize implied probabilities for Asian Handicap
    ah_odds_cols = ['AvgAHH', 'AvgAHA']
    for col in ah_odds_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        df[col] = df[col].replace(0, np.nan)

    df['ImpliedProb_AHH'] = 1 / df['AvgAHH']
    df['ImpliedProb_AHA'] = 1 / df['AvgAHA']
    total_prob_ah = df['ImpliedProb_AHH'] + df['ImpliedProb_AHA']
    df['NormProb_AHH'] = df['ImpliedProb_AHH'] / total_prob_ah
    df['NormProb_AHA'] = df['ImpliedProb_AHA'] / total_prob_ah

    return df

def create_shots_on_target_target(df):
    """Creates a target variable for the total shots on target market."""
    df['TotalShotsOnTarget'] = df['HST'] + df['AST']
    return df

def create_shots_target(df):
    """Creates a target variable for the total shots market."""
    df['TotalShots'] = df['HS'] + df['AS']
    return df

def create_yellow_cards_target(df):
    """Creates a target variable for the total yellow cards market."""
    df['TotalYellowCards'] = df['HY'] + df['AY']
    return df

def create_corners_target(df):
    """Creates a target variable for the total corners market."""
    df['TotalCorners'] = df['HC'] + df['AC']
    return df

def create_asian_handicap_target(df):
    """Creates a target variable for the Asian Handicap market."""
    # Ensure necessary columns are numeric, coercing errors
    df['AHh'] = pd.to_numeric(df['AHh'], errors='coerce')

    # Calculate effective score
    # Effective Score = (Home Goals - Away Goals) + Handicap
    effective_score = (df['FTHG'] - df['FTAG']) + df['AHh']

    # Create target variable: 1 for Home Win (on AH), 0 for Away Win (on AH)
    # Pushes (effective_score == 0) will be marked as NaN and can be dropped later.
    df['AsianHandicapOutcome'] = np.nan
    df.loc[effective_score > 0, 'AsianHandicapOutcome'] = 1  # Home covers
    df.loc[effective_score < 0, 'AsianHandicapOutcome'] = 0  # Away covers

    return df

def feature_engineer(df):
    """Engineers features for the model."""
    df['HomeTeamForm'] = df.apply(lambda x: calculate_form(df, x['HomeTeam'], x['Date']), axis=1)
    df['AwayTeamForm'] = df.apply(lambda x: calculate_form(df, x['AwayTeam'], x['Date']), axis=1)
    df['H2H_Advantage'] = df.apply(lambda x: calculate_h2h(df, x['HomeTeam'], x['AwayTeam'], x['Date']), axis=1)

    # Advanced feature engineering: Rolling stats
    stat_cols = ['FTHG', 'FTAG', 'HS', 'AS', 'HST', 'AST', 'HC', 'AC']
    
    home_stats = df.apply(lambda row: calculate_rolling_stats(df, row['HomeTeam'], row['Date'], ['FTHG', 'HS', 'HST', 'HC']), axis=1, result_type='expand')
    home_stats.columns = ['HomeAvgGoals', 'HomeAvgShots', 'HomeAvgShotsTarget', 'HomeAvgCorners']

    away_stats = df.apply(lambda row: calculate_rolling_stats(df, row['AwayTeam'], row['Date'], ['FTAG', 'AS', 'AST', 'AC']), axis=1, result_type='expand')
    away_stats.columns = ['AwayAvgGoals', 'AwayAvgShots', 'AwayAvgShotsTarget', 'AwayAvgCorners']

    df = pd.concat([df, home_stats, away_stats], axis=1)

    # --- Advanced Feature Engineering ---
    # 1. Implied Probabilities from Betting Odds
    df = create_odds_features(df)

    # 2. Rolling averages for discipline
    discipline_home_stats = df.apply(lambda row: calculate_rolling_stats(df, row['HomeTeam'], row['Date'], ['HY', 'HR']), axis=1, result_type='expand')
    discipline_home_stats.columns = ['HomeAvgYellows', 'HomeAvgReds']

    discipline_away_stats = df.apply(lambda row: calculate_rolling_stats(df, row['AwayTeam'], row['Date'], ['AY', 'AR']), axis=1, result_type='expand')
    discipline_away_stats.columns = ['AwayAvgYellows', 'AwayAvgReds']

    df = pd.concat([df, discipline_home_stats, discipline_away_stats], axis=1)

    # Calculate Elo ratings
    df, elos = calculate_elo(df)

    # Create Asian Handicap target
    df = create_asian_handicap_target(df)

    # Create Total Corners target
    df = create_corners_target(df)

    # Create Total Yellow Cards target
    df = create_yellow_cards_target(df)

    # Create Total Shots target
    df = create_shots_target(df)

    # Create Total Shots on Target target
    df = create_shots_on_target_target(df)

    return df, elos
