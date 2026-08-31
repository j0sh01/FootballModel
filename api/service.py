"""
Football Prediction Service
Loads trained models and provides prediction capabilities.
"""

import os
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
from functools import lru_cache
import joblib

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_loader import LEAGUE_CONFIG, load_league_data
from src.feature_engineering import (
    engineer_features, MARKET_FEATURES, MARKET_TARGETS, MARKET_TYPES,
    calculate_form, calculate_h2h, calculate_rolling_stats, calculate_elo,
    calculate_streak, calculate_home_away_split, calculate_goal_timing,
    calculate_clean_sheet_prob, calculate_attack_defense_strength,
    calculate_league_position_proxy
)
from src.market_models import MarketModel, compare_teams as _compare_teams, print_team_comparison


MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models')
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')


class PredictionService:
    """Singleton service for making predictions."""
    
    _instance = None
    _initialized = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        
        self.models: Dict[str, Dict[str, MarketModel]] = {}  # league -> market -> model
        self.league_data: Dict[str, pd.DataFrame] = {}  # league -> engineered df
        self.elos: Dict[str, Dict[str, int]] = {}  # league -> team -> elo
        self.teams: Dict[str, List[str]] = {}  # league -> team names
        
        self._load_all_models()
    
    def _load_all_models(self):
        """Load all trained models from disk."""
        print("Loading trained models...")
        
        for league in LEAGUE_CONFIG:
            league_dir = os.path.join(MODELS_DIR, league)
            if not os.path.exists(league_dir):
                continue
            
            self.models[league] = {}
            
            for f in os.listdir(league_dir):
                if not f.endswith('.pkl'):
                    continue
                
                basename = f.replace('.pkl', '')
                model_type = None
                market = None
                
                # Match longest model type first
                for mt in ['gradient_boosting', 'random_forest', 'xgboost']:
                    if basename.endswith('_' + mt):
                        model_type = mt
                        market = basename[:-len(mt)-1]
                        break
                
                if model_type is None:
                    continue
                
                # Prefer xgboost, then random_forest, then gradient_boosting
                if market not in self.models[league] or model_type == 'xgboost':
                    try:
                        path = os.path.join(league_dir, f)
                        model = MarketModel.load(path)
                        self.models[league][market] = model
                    except Exception as e:
                        print(f"  Error loading {f}: {e}")
        
        # Print summary
        total_models = sum(len(markets) for markets in self.models.values())
        print(f"Loaded {total_models} models across {len(self.models)} leagues")
    
    def _ensure_league_data(self, league: str):
        """Load and engineer features for a league if not already done."""
        if league in self.league_data:
            return
        
        print(f"Loading data for {LEAGUE_CONFIG[league]['name']}...")
        
        # Check for cached features first
        cache_path = os.path.join(DATA_DIR, 'cached_features', f'{league}_features.pkl')
        if os.path.exists(cache_path):
            df = pd.read_pickle(cache_path)
        else:
            df_raw = load_league_data(league)
            if df_raw.empty:
                raise ValueError(f"No data available for {league}")
            df, elos = engineer_features(df_raw)
            self.elos[league] = elos
        
        self.league_data[league] = df
        
        # Extract team names
        home_teams = set(df['HomeTeam'].unique())
        away_teams = set(df['AwayTeam'].unique())
        self.teams[league] = sorted(home_teams | away_teams)
    
    def get_leagues(self) -> List[Dict[str, Any]]:
        """Get all available leagues."""
        leagues = []
        for key, config in LEAGUE_CONFIG.items():
            has_models = key in self.models and len(self.models[key]) > 0
            leagues.append({
                'key': key,
                'name': config['name'],
                'country': config['country'],
                'tier': config['tier'],
                'has_models': has_models,
                'n_models': len(self.models.get(key, {})),
                'markets': list(self.models.get(key, {}).keys())
            })
        return leagues
    
    def get_teams(self, league: str) -> List[str]:
        """Get teams in a league."""
        self._ensure_league_data(league)
        return self.teams.get(league, [])
    
    def get_models_info(self, league: str) -> List[Dict[str, Any]]:
        """Get information about trained models for a league."""
        if league not in self.models:
            return []
        
        models_info = []
        for market, model in self.models[league].items():
            results = model.results or {}
            models_info.append({
                'market': market,
                'model_type': model.model_type,
                'market_type': model.market_type,
                'accuracy': results.get('accuracy'),
                'mae': results.get('mae'),
                'n_features': len(model.feature_names) if model.feature_names else 0
            })
        
        return models_info
    
    def predict_match(
        self,
        league: str,
        home_team: str,
        away_team: str,
        markets: Optional[List[str]] = None,
        odds: Optional[Dict[str, float]] = None
    ) -> Dict[str, Any]:
        """
        Predict the outcome of a match.
        
        Args:
            league: League key
            home_team: Home team name
            away_team: Away team name
            markets: List of markets to predict (default: all available)
            odds: Optional bookmaker odds
        
        Returns:
            Dictionary with predictions for each market
        """
        self._ensure_league_data(league)
        
        if league not in self.models:
            raise ValueError(f"No models trained for {league}")
        
        df = self.league_data[league]
        
        # Calculate features
        current_date = pd.Timestamp.now()
        
        features = self._calculate_match_features(df, home_team, away_team, current_date, odds)
        
        # Make predictions for each market
        predictions = {}
        available_markets = list(self.models[league].keys())
        
        if markets is None:
            markets = available_markets
        
        for market in markets:
            if market not in self.models[league]:
                continue
            
            model = self.models[league][market]
            
            try:
                feature_df = pd.DataFrame([features])
                pred = model.predict(feature_df)
                
                result = {
                    'prediction': str(pred[0]) if hasattr(pred, '__getitem__') else str(pred),
                    'model_type': model.model_type,
                    'market_type': model.market_type
                }
                
                # Get probabilities for classification
                if model.market_type == 'classification':
                    try:
                        proba_result = model.predict_proba(feature_df)
                        proba = proba_result[0] if isinstance(proba_result, tuple) else proba_result
                        class_labels = proba_result[1] if isinstance(proba_result, tuple) and len(proba_result) > 1 else None
                        
                        if class_labels is None:
                            if model.label_encoder is not None:
                                class_labels = [str(c) for c in model.label_encoder.classes_]
                            else:
                                class_labels = [str(i) for i in range(proba.shape[1])]
                        
                        result['confidence'] = float(np.max(proba))
                        result['probabilities'] = {
                            str(label): round(float(prob), 4)
                            for label, prob in zip(class_labels, proba[0])
                        }
                    except Exception:
                        pass
                
                # Get regression value
                if model.market_type == 'regression':
                    result['predicted_value'] = float(pred[0]) if hasattr(pred, '__getitem__') else float(pred)
                
                predictions[market] = result
                
            except Exception as e:
                predictions[market] = {'error': str(e)}
        
        return {
            'league': LEAGUE_CONFIG[league]['name'],
            'league_key': league,
            'home_team': home_team,
            'away_team': away_team,
            'predictions': predictions
        }
    
    def compare_teams(self, league: str, home_team: str, away_team: str) -> Dict[str, Any]:
        """Compare two teams."""
        self._ensure_league_data(league)
        df = self.league_data[league]
        
        comparison = _compare_teams(home_team, away_team, df)
        return comparison
    
    def _calculate_match_features(
        self,
        df: pd.DataFrame,
        home_team: str,
        away_team: str,
        date: pd.Timestamp,
        odds: Optional[Dict[str, float]] = None
    ) -> Dict[str, Any]:
        """Calculate all features for a match prediction."""
        # Form
        home_form = calculate_form(df, home_team, date, 5, 'points')
        away_form = calculate_form(df, away_team, date, 5, 'points')
        home_goals_scored = calculate_form(df, home_team, date, 5, 'goals_scored')
        away_goals_scored = calculate_form(df, away_team, date, 5, 'goals_scored')
        home_goals_conceded = calculate_form(df, home_team, date, 5, 'goals_conceded')
        away_goals_conceded = calculate_form(df, away_team, date, 5, 'goals_conceded')
        home_wins = calculate_form(df, home_team, date, 5, 'wins')
        away_wins = calculate_form(df, away_team, date, 5, 'wins')
        
        # H2H
        h2h = calculate_h2h(df, home_team, away_team, date)
        
        # Rolling stats
        home_perf = calculate_rolling_stats(df, home_team, date, ['FTHG', 'HS', 'HST', 'HC'])
        away_perf = calculate_rolling_stats(df, away_team, date, ['FTAG', 'AS', 'AST', 'AC'])
        home_disc = calculate_rolling_stats(df, home_team, date, ['HY', 'HR'])
        away_disc = calculate_rolling_stats(df, away_team, date, ['AY', 'AR'])
        
        # Elo
        _, elos = calculate_elo(df)
        home_elo = elos.get(home_team, 1500)
        away_elo = elos.get(away_team, 1500)
        
        # Streaks
        home_streak = calculate_streak(df, home_team, date)
        away_streak = calculate_streak(df, away_team, date)
        
        # Home/Away split
        home_split = calculate_home_away_split(df, home_team, date)
        away_split = calculate_home_away_split(df, away_team, date)
        
        # Goal timing
        home_timing = calculate_goal_timing(df, home_team, date)
        away_timing = calculate_goal_timing(df, away_team, date)
        
        # Clean sheet
        home_cs = calculate_clean_sheet_prob(df, home_team, date)
        away_cs = calculate_clean_sheet_prob(df, away_team, date)
        
        # Attack/Defense
        home_strength = calculate_attack_defense_strength(df, home_team, date)
        away_strength = calculate_attack_defense_strength(df, away_team, date)
        
        # League position
        home_pos = calculate_league_position_proxy(df, home_team, date)
        away_pos = calculate_league_position_proxy(df, away_team, date)
        
        # Build feature dict
        features = {
            'HomeTeamForm': home_form,
            'AwayTeamForm': away_form,
            'FormDiff': home_form - away_form,
            'HomeGoalsScoredForm': home_goals_scored,
            'AwayGoalsScoredForm': away_goals_scored,
            'GoalsScoredDiff': home_goals_scored - away_goals_scored,
            'HomeGoalsConcededForm': home_goals_conceded,
            'AwayGoalsConcededForm': away_goals_conceded,
            'GoalsConcededDiff': home_goals_conceded - away_goals_conceded,
            'HomeWinForm': home_wins,
            'AwayWinForm': away_wins,
            'H2H_Advantage': h2h,
            'HomeAvgGoals': home_perf['FTHG'],
            'HomeAvgShots': home_perf['HS'],
            'HomeAvgShotsTarget': home_perf['HST'],
            'HomeAvgCorners': home_perf['HC'],
            'AwayAvgGoals': away_perf['FTAG'],
            'AwayAvgShots': away_perf['AS'],
            'AwayAvgShotsTarget': away_perf['AST'],
            'AwayAvgCorners': away_perf['AC'],
            'HomeAvgYellows': home_disc['HY'],
            'HomeAvgReds': home_disc['HR'],
            'AwayAvgYellows': away_disc['AY'],
            'AwayAvgReds': away_disc['AR'],
            'HomeElo': home_elo,
            'AwayElo': away_elo,
            'EloDiff': home_elo - away_elo,
            'HomeShotEfficiency': home_perf['FTHG'] / (home_perf['HS'] + 1e-6),
            'AwayShotEfficiency': away_perf['FTAG'] / (away_perf['AS'] + 1e-6),
            'HomeStreak': home_streak['current_streak'],
            'AwayStreak': away_streak['current_streak'],
            'StreakDiff': home_streak['current_streak'] - away_streak['current_streak'],
            'HomeMomentum': home_streak['momentum'],
            'AwayMomentum': away_streak['momentum'],
            'MomentumDiff': home_streak['momentum'] - away_streak['momentum'],
            'HomeHomeForm': home_split['home_form_split'],
            'HomeAwayForm': home_split['away_form_split'],
            'AwayHomeForm': away_split['home_form_split'],
            'AwayAwayForm': away_split['away_form_split'],
            'HomeHomeAdvantage': home_split['home_advantage'],
            'AwayHomeAdvantage': away_split['home_advantage'],
            'HomeHTGoalsRatio': home_timing['ht_goals_ratio'],
            'Home2ndHalfGoalsRatio': home_timing['second_half_goals_ratio'],
            'AwayHTGoalsRatio': away_timing['ht_goals_ratio'],
            'Away2ndHalfGoalsRatio': away_timing['second_half_goals_ratio'],
            'HomeLeaguePosition': home_pos,
            'AwayLeaguePosition': away_pos,
            'HomeCleanSheetProb': home_cs,
            'AwayCleanSheetProb': away_cs,
            'HomeAttackStrength': home_strength['attack_strength'],
            'HomeDefenseStrength': home_strength['defense_strength'],
            'AwayAttackStrength': away_strength['attack_strength'],
            'AwayDefenseStrength': away_strength['defense_strength'],
            'AttackVsDefense': home_strength['attack_strength'] - away_strength['defense_strength'],
            'DefenseVsAttack': home_strength['defense_strength'] - away_strength['attack_strength'],
        }
        
        # Add odds features
        if odds and 'home' in odds and 'draw' in odds and 'away' in odds:
            total = 1/odds['home'] + 1/odds['draw'] + 1/odds['away']
            features['NormProb_H'] = (1/odds['home']) / total
            features['NormProb_D'] = (1/odds['draw']) / total
            features['NormProb_A'] = (1/odds['away']) / total
        else:
            # Estimate implied probabilities from features (Elo-based)
            elo_diff = features.get('EloDiff', 0)
            # Simple Elo-based probability estimate
            home_win_prob = 1 / (1 + 10 ** (-elo_diff / 400))
            draw_prob = 0.25  # Average draw probability
            away_win_prob = 1 - home_win_prob - draw_prob
            if away_win_prob < 0.05:
                away_win_prob = 0.05
                draw_prob = 1 - home_win_prob - away_win_prob
            total_prob = home_win_prob + draw_prob + away_win_prob
            features['NormProb_H'] = home_win_prob / total_prob
            features['NormProb_D'] = draw_prob / total_prob
            features['NormProb_A'] = away_win_prob / total_prob
        
        if odds and 'over_25' in odds and 'under_25' in odds:
            total_ou = 1/odds['over_25'] + 1/odds['under_25']
            features['NormProb_O2.5'] = (1/odds['over_25']) / total_ou
            features['NormProb_U2.5'] = (1/odds['under_25']) / total_ou
        else:
            # Estimate from home/away scoring rates
            home_goals = features.get('HomeGoalsScoredForm', 1.3)
            away_goals = features.get('AwayGoalsScoredForm', 1.1)
            expected_total = home_goals + away_goals
            # Rough mapping: >2.5 prob based on expected total
            over_prob = min(0.8, max(0.2, (expected_total - 2.0) / 2.0))
            under_prob = 1 - over_prob
            features['NormProb_O2.5'] = over_prob
            features['NormProb_U2.5'] = under_prob
        
        return features


def get_service() -> PredictionService:
    """Get the singleton prediction service."""
    return PredictionService()
