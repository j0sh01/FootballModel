import pandas as pd
import numpy as np
import joblib
from data_processing import load_data, preprocess_data, feature_engineer, calculate_form, calculate_h2h, calculate_rolling_stats

# Define file paths
files = ['E0 .csv', 'E01.csv', 'E02.csv']
OUTCOME_MODEL_PATH = 'epl_3way_model.pkl'
GOALS_MODEL_PATH = 'epl_goals_model.pkl'
OVER_UNDER_MODEL_PATH = 'epl_over_under_model.pkl'
AH_MODEL_PATH = 'epl_asian_handicap_model.pkl'
CORNERS_MODEL_PATH = 'epl_corners_model.pkl'
YELLOW_CARDS_MODEL_PATH = 'epl_yellow_cards_model.pkl'
SHOTS_MODEL_PATH = 'epl_shots_model.pkl'
SHOTS_ON_TARGET_MODEL_PATH = 'epl_shots_on_target_model.pkl'
HTR_MODEL_PATH = 'epl_htr_model.pkl'
HTHG_MODEL_PATH = 'epl_hthg_model.pkl'
HTAG_MODEL_PATH = 'epl_htag_model.pkl'

# Load all models and data
outcome_model = joblib.load(OUTCOME_MODEL_PATH)
goals_model_data = joblib.load(GOALS_MODEL_PATH)
goals_model = goals_model_data['model']
goals_encoder = goals_model_data['encoder']
over_under_model = joblib.load(OVER_UNDER_MODEL_PATH)
ah_model = joblib.load(AH_MODEL_PATH)
corners_model = joblib.load(CORNERS_MODEL_PATH)
yellow_cards_model = joblib.load(YELLOW_CARDS_MODEL_PATH)
shots_model = joblib.load(SHOTS_MODEL_PATH)
shots_on_target_model = joblib.load(SHOTS_ON_TARGET_MODEL_PATH)
htr_model_data = joblib.load(HTR_MODEL_PATH)
htr_model = htr_model_data['model']
htr_encoder = htr_model_data['encoder']
hthg_model = joblib.load(HTHG_MODEL_PATH)
htag_model = joblib.load(HTAG_MODEL_PATH)
df = load_data(files)
df = preprocess_data(df)
df, elos = feature_engineer(df)
initial_elo = 1500 # Should be consistent with data_processing

def predict_match(home_team, away_team, odds_home=None, odds_draw=None, odds_away=None, odds_ahh=None, odds_aha=None, ah_line=None):
    """Predicts the outcome of a single match."""
    current_date = pd.Timestamp.now()
    
    # Calculate features
    home_form = calculate_form(df, home_team, current_date)
    away_form = calculate_form(df, away_team, current_date)
    h2h = calculate_h2h(df, home_team, away_team, current_date)

    # Calculate rolling stats for performance and discipline
    home_perf_stats = calculate_rolling_stats(df, home_team, current_date, ['FTHG', 'HS', 'HST', 'HC'])
    away_perf_stats = calculate_rolling_stats(df, away_team, current_date, ['FTAG', 'AS', 'AST', 'AC'])
    home_disc_stats = calculate_rolling_stats(df, home_team, current_date, ['HY', 'HR'])
    away_disc_stats = calculate_rolling_stats(df, away_team, current_date, ['AY', 'AR'])

    # Get latest Elo ratings
    home_elo = elos.get(home_team, initial_elo)
    away_elo = elos.get(away_team, initial_elo)

    # Use provided odds or default values
    odds_h = odds_home if odds_home is not None else df['B365H'].mean()
    odds_d = odds_draw if odds_draw is not None else df['B365D'].mean()
    odds_a = odds_away if odds_away is not None else df['B365A'].mean()

    # Calculate and normalize implied probabilities
    implied_prob_h = 1 / odds_h
    implied_prob_d = 1 / odds_d
    implied_prob_a = 1 / odds_a
    total_prob_3way = implied_prob_h + implied_prob_d + implied_prob_a
    norm_prob_h = implied_prob_h / total_prob_3way
    norm_prob_d = implied_prob_d / total_prob_3way
    norm_prob_a = implied_prob_a / total_prob_3way

    # --- Base Features --- 
    base_features = {
        'HomeTeamForm': home_form, 'AwayTeamForm': away_form, 'H2H_Advantage': h2h,
        'HomeAvgGoals': home_perf_stats['FTHG'], 'HomeAvgShots': home_perf_stats['HS'], 
        'HomeAvgShotsTarget': home_perf_stats['HST'], 'HomeAvgCorners': home_perf_stats['HC'],
        'AwayAvgGoals': away_perf_stats['FTAG'], 'AwayAvgShots': away_perf_stats['AS'],
        'AwayAvgShotsTarget': away_perf_stats['AST'], 'AwayAvgCorners': away_perf_stats['AC'],
        'HomeAvgYellows': home_disc_stats['HY'], 'HomeAvgReds': home_disc_stats['HR'],
        'AwayAvgYellows': away_disc_stats['AY'], 'AwayAvgReds': away_disc_stats['AR'],
        'HomeElo': home_elo, 'AwayElo': away_elo
    }

    # --- Market 1: Match Outcome (Home/Draw/Away) ---
    outcome_features = base_features.copy()
    outcome_features.update({'NormProb_H': norm_prob_h, 'NormProb_D': norm_prob_d, 'NormProb_A': norm_prob_a})
    df_outcome = pd.DataFrame([outcome_features])
    df_outcome = df_outcome[outcome_model.feature_names_in_]

    outcome_prob = outcome_model.predict_proba(df_outcome)[0]
    outcome_pred = np.argmax(outcome_prob)
    outcome_map = {0: 'Away Win', 1: 'Draw', 2: 'Home Win'}
    outcome_prediction = {
        'Market': 'Match Outcome',
        'Prediction': outcome_map[outcome_pred],
        'Confidence': f"{np.max(outcome_prob):.2%}",
        'Probabilities': {
            'Home Win': f"{outcome_prob[2]:.2%}",
            'Draw': f"{outcome_prob[1]:.2%}",
            'Away Win': f"{outcome_prob[0]:.2%}"
        }
    }

    # --- Market 2: Half-Time Result ---
    htr_features = outcome_features.copy()
    df_htr = pd.DataFrame([htr_features])
    df_htr = df_htr[htr_model.feature_names_in_]

    htr_pred_proba = htr_model.predict_proba(df_htr)[0]
    htr_prediction_label = htr_encoder.inverse_transform([htr_pred_proba.argmax()])[0]
    htr_confidence = htr_pred_proba.max()

    htr_prediction_map = {
        'H': 'Home Win',
        'D': 'Draw',
        'A': 'Away Win'
    }

    htr_prediction = {
        "Market": "Half-Time Result",
        "Prediction": htr_prediction_map[htr_prediction_label],
        "Confidence": f"{htr_confidence:.2%}",
        "Probabilities": {
            htr_prediction_map[label]: f"{prob:.2%}" for label, prob in zip(htr_encoder.classes_, htr_pred_proba)
        }
    }

    # --- Market 3: Over/Under 2.5 Goals ---
    # Calculate Over/Under probabilities
    # Default to 50/50 if odds are not available or invalid
    try:
        odds_o25 = df['Avg>2.5'].mean() # Default value
        odds_u25 = df['Avg<2.5'].mean() # Default value
        implied_prob_o25 = 1 / odds_o25
        implied_prob_u25 = 1 / odds_u25
        total_prob_ou = implied_prob_o25 + implied_prob_u25
        norm_prob_o25 = implied_prob_o25 / total_prob_ou
        norm_prob_u25 = implied_prob_u25 / total_prob_ou
    except (ZeroDivisionError, KeyError):
        norm_prob_o25, norm_prob_u25 = 0.5, 0.5

    over_under_features = base_features.copy()
    over_under_features.update({'NormProb_O2.5': norm_prob_o25, 'NormProb_U2.5': norm_prob_u25})
    df_over_under = pd.DataFrame([over_under_features])
    df_over_under = df_over_under[over_under_model.feature_names_in_]

    over_under_prob = over_under_model.predict_proba(df_over_under)[0]
    over_under_pred = np.argmax(over_under_prob)
    over_under_map = {0: 'Under 2.5', 1: 'Over 2.5'}
    over_under_prediction = {
        'Market': 'Over/Under 2.5',
        'Prediction': over_under_map[over_under_pred],
        'Confidence': f"{max(over_under_prob)*100:.2f}%",
        'Probabilities': {
            'Over 2.5': f"{over_under_prob[1]:.2%}",
            'Under 2.5': f"{over_under_prob[0]:.2%}"
        }
    }

    # --- Market 3: Total Goals ---
    goals_features = base_features.copy()
    goals_features.update({
        'NormProb_H': norm_prob_h, 'NormProb_D': norm_prob_d, 'NormProb_A': norm_prob_a,
        'NormProb_O2.5': norm_prob_o25, 'NormProb_U2.5': norm_prob_u25
    })
    df_goals = pd.DataFrame([goals_features])
    df_goals = df_goals[goals_model.feature_names_in_]

    goals_prediction_proba = goals_model.predict_proba(df_goals)[0]
    goals_confidence = goals_prediction_proba.max()
    goals_prediction_encoded = goals_prediction_proba.argmax()
    goals_prediction_label = goals_encoder.inverse_transform([goals_prediction_encoded])[0]

    goals_prediction = {
        'Market': 'Total Goals',
        'Prediction': goals_prediction_label,
        'Confidence': f'{goals_confidence:.2%}',
        'Probabilities': {label: f'{prob:.2%}' for label, prob in zip(goals_encoder.classes_, goals_prediction_proba)}
    }

    # Combine all predictions
    # --- Market 4: Asian Handicap ---
    # Calculate and normalize implied probabilities for Asian Handicap
    try:
        odds_ahh_val = odds_ahh if odds_ahh is not None else df['AvgAHH'].mean()
        odds_aha_val = odds_aha if odds_aha is not None else df['AvgAHA'].mean()
        implied_prob_ahh = 1 / odds_ahh_val
        implied_prob_aha = 1 / odds_aha_val
        total_prob_ah = implied_prob_ahh + implied_prob_aha
        norm_prob_ahh = implied_prob_ahh / total_prob_ah
        norm_prob_aha = implied_prob_aha / total_prob_ah
    except (ZeroDivisionError, KeyError):
        norm_prob_ahh, norm_prob_aha = 0.5, 0.5

    ah_features = base_features.copy()
    ah_features.update({'NormProb_AHH': norm_prob_ahh, 'NormProb_AHA': norm_prob_aha})
    df_ah = pd.DataFrame([ah_features])
    df_ah = df_ah[ah_model.feature_names_in_]

    ah_prob = ah_model.predict_proba(df_ah)[0]
    ah_pred = np.argmax(ah_prob)
    ah_map = {0: f"{away_team} {ah_line:+.2f}", 1: f"{home_team} {ah_line:+.2f}"}
    ah_prediction = {
        'Market': 'Asian Handicap',
        'Prediction': ah_map[ah_pred],
        'Confidence': f"{max(ah_prob)*100:.2f}%",
        'Probabilities': {
            f"{home_team} {ah_line:+.2f}": f"{ah_prob[1]:.2%}",
            f"{away_team} {ah_line:+.2f}": f"{ah_prob[0]:.2%}"
        }
    }

    # --- Market 5: Total Corners ---
    corners_features = base_features.copy()
    df_corners = pd.DataFrame([corners_features])
    df_corners = df_corners[corners_model.feature_names_in_]
    
    corners_pred = corners_model.predict(df_corners)[0]
    corners_prediction = {
        'Market': 'Total Corners',
        'Prediction': f"{corners_pred:.2f}"
    }

    # --- Market 6: Total Yellow Cards ---
    yellow_cards_features = base_features.copy()
    df_yellow_cards = pd.DataFrame([yellow_cards_features])
    df_yellow_cards = df_yellow_cards[yellow_cards_model.feature_names_in_]
    
    yellow_cards_pred = yellow_cards_model.predict(df_yellow_cards)[0]
    yellow_cards_prediction = {
        'Market': 'Total Yellow Cards',
        'Prediction': f"{yellow_cards_pred:.2f}"
    }

    # --- Market 7: Total Shots ---
    shots_features = base_features.copy()
    df_shots = pd.DataFrame([shots_features])
    df_shots = df_shots[shots_model.feature_names_in_]
    
    shots_pred = shots_model.predict(df_shots)[0]
    shots_prediction = {
        'Market': 'Total Shots',
        'Prediction': f"{shots_pred:.2f}"
    }

    # --- Market 8: Total Shots on Target ---
    shots_on_target_features = base_features.copy()
    df_shots_on_target = pd.DataFrame([shots_on_target_features])
    df_shots_on_target = df_shots_on_target[shots_on_target_model.feature_names_in_]
    
    shots_on_target_pred = shots_on_target_model.predict(df_shots_on_target)[0]
    shots_on_target_prediction = {
        'Market': 'Total Shots on Target',
        'Prediction': f"{shots_on_target_pred:.2f}"
    }

    # --- Market 9: Half-Time Home Goals ---
    hthg_features = base_features.copy()
    df_hthg = pd.DataFrame([hthg_features])
    df_hthg = df_hthg[hthg_model.feature_names_in_]
    
    hthg_pred = hthg_model.predict(df_hthg)[0]
    hthg_prediction = {
        'Market': 'Half-Time Home Goals',
        'Prediction': f"{hthg_pred:.2f}"
    }

    # --- Market 10: Half-Time Away Goals ---
    htag_features = base_features.copy()
    df_htag = pd.DataFrame([htag_features])
    df_htag = df_htag[htag_model.feature_names_in_]
    
    htag_pred = htag_model.predict(df_htag)[0]
    htag_prediction = {
        'Market': 'Half-Time Away Goals',
        'Prediction': f"{htag_pred:.2f}"
    }

    # Combine all predictions
    full_prediction = {
        'Match': f"{home_team} vs {away_team}",
        'Match Outcome': outcome_prediction,
        'Half-Time Result': htr_prediction,
        'Over/Under 2.5': over_under_prediction,
        'Total Goals': goals_prediction,
        'Asian Handicap': ah_prediction,
        'Total Corners': corners_prediction,
        'Total Yellow Cards': yellow_cards_prediction,
        'Total Shots': shots_prediction,
        'Total Shots on Target': shots_on_target_prediction,
        'Half-Time Home Goals': hthg_prediction,
        'Half-Time Away Goals': htag_prediction
    }
    
    return full_prediction

if __name__ == '__main__':
    # Example usage
    home = 'West Ham'
    away = 'Brentford'
    odds_h = 2.51
    odds_d = 3.40
    odds_a = 2.88
    # Asian Handicap odds for the match
    odds_ahh = 1.95  # Home team handicap odds
    odds_aha = 1.95  # Away team handicap odds
    ah_line = -0.25   # Handicap line for the home team

    prediction = predict_match(home, away, odds_h, odds_d, odds_a, odds_ahh, odds_aha, ah_line)
    # Pretty print the prediction
    import json
    print(json.dumps(prediction, indent=4))
