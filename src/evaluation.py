"""
Evaluation & Backtesting Framework
Comprehensive model evaluation with profit simulation and market analysis.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, log_loss, brier_score_loss,
    mean_absolute_error, mean_squared_error, r2_score
)
from sklearn.calibration import calibration_curve
import warnings
warnings.filterwarnings('ignore')


# =============================================================================
# CLASSIFICATION METRICS
# =============================================================================

def evaluate_classification(y_true, y_pred, y_prob=None, 
                           target_names=None, average='weighted'):
    """
    Comprehensive classification evaluation.
    
    Returns:
        Dictionary of metrics
    """
    results = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average=average, zero_division=0),
        'recall': recall_score(y_true, y_pred, average=average, zero_division=0),
        'f1': f1_score(y_true, y_pred, average=average, zero_division=0),
    }
    
    if y_prob is not None:
        try:
            results['log_loss'] = log_loss(y_true, y_prob)
        except:
            pass
        
        # Brier score for binary
        if y_prob.shape[1] == 2:
            try:
                results['brier_score'] = brier_score_loss(y_true, y_prob[:, 1])
            except:
                pass
    
    # Per-class metrics
    if target_names is not None:
        report = classification_report(y_true, y_pred, target_names=target_names, 
                                      output_dict=True, zero_division=0)
        results['classification_report'] = report
        results['per_class'] = {}
        for name in target_names:
            if name in report:
                results['per_class'][name] = {
                    'precision': report[name]['precision'],
                    'recall': report[name]['recall'],
                    'f1': report[name]['f1-score'],
                    'support': report[name]['support']
                }
    
    # Confusion matrix
    results['confusion_matrix'] = confusion_matrix(y_true, y_pred)
    
    return results


# =============================================================================
# REGRESSION METRICS
# =============================================================================

def evaluate_regression(y_true, y_pred):
    """
    Comprehensive regression evaluation.
    
    Returns:
        Dictionary of metrics
    """
    results = {
        'mae': mean_absolute_error(y_true, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
        'r2': r2_score(y_true, y_pred),
        'mape': np.mean(np.abs((y_true - y_pred) / (y_true + 1e-6))) * 100
    }
    
    return results


# =============================================================================
# PROFIT SIMULATION
# =============================================================================

def simulate_profit(y_true, y_pred, odds, stake=10, 
                   model_confidence=None, min_confidence=0.0,
                   market_type='match_result'):
    """
    Simulate betting profit/loss.
    
    Args:
        y_true: Actual outcomes
        y_pred: Predicted outcomes
        odds: Odds for predicted outcomes
        stake: Stake per bet
        model_confidence: Model confidence scores
        min_confidence: Minimum confidence to place bet
        market_type: Type of market
    
    Returns:
        Dictionary with profit metrics
    """
    results = {
        'total_bets': 0,
        'winning_bets': 0,
        'losing_bets': 0,
        'total_staked': 0,
        'total_returns': 0,
        'profit': 0,
        'roi': 0,
        'win_rate': 0,
        'avg_odds': 0,
        'yield_per_bet': 0
    }
    
    if model_confidence is not None:
        mask = model_confidence >= min_confidence
        y_true = np.array(y_true)[mask]
        y_pred = np.array(y_pred)[mask]
        odds = np.array(odds)[mask]
    
    total_bets = len(y_true)
    if total_bets == 0:
        return results
    
    winning_bets = 0
    total_returns = 0
    total_staked = total_bets * stake
    
    for i in range(total_bets):
        if y_pred[i] == y_true[i]:
            winning_bets += 1
            total_returns += odds[i] * stake
    
    profit = total_returns - total_staked
    roi = (profit / total_staked) * 100 if total_staked > 0 else 0
    win_rate = (winning_bets / total_bets) * 100
    
    results.update({
        'total_bets': total_bets,
        'winning_bets': winning_bets,
        'losing_bets': total_bets - winning_bets,
        'total_staked': total_staked,
        'total_returns': total_returns,
        'profit': profit,
        'roi': roi,
        'win_rate': win_rate,
        'avg_odds': np.mean(odds),
        'yield_per_bet': profit / total_bets if total_bets > 0 else 0
    })
    
    return results


def simulate_value_betting(y_true, y_pred, y_prob, odds, 
                          edge_threshold=0.05, stake=10):
    """
    Simulate value betting (betting when model sees value vs bookmaker odds).
    
    Args:
        y_true: Actual outcomes
        y_pred: Predicted outcomes
        y_prob: Model probabilities
        odds: Bookmaker odds
        edge_threshold: Minimum edge to place bet
        stake: Stake per bet
    
    Returns:
        Dictionary with value betting metrics
    """
    results = {
        'total_bets': 0,
        'winning_bets': 0,
        'total_staked': 0,
        'total_returns': 0,
        'profit': 0,
        'roi': 0,
        'avg_edge': 0,
        'avg_odds': 0
    }
    
    # Calculate implied probabilities
    implied_probs = 1 / odds
    
    # Calculate edge (model prob - implied prob)
    edges = y_prob - implied_probs
    
    # Filter for value bets
    value_mask = edges >= edge_threshold
    
    if value_mask.sum() == 0:
        return results
    
    y_true_value = np.array(y_true)[value_mask]
    y_pred_value = np.array(y_pred)[value_mask]
    odds_value = np.array(odds)[value_mask]
    edges_value = edges[value_mask]
    
    total_bets = len(y_true_value)
    winning_bets = (y_true_value == y_pred_value).sum()
    total_staked = total_bets * stake
    total_returns = winning_bets * odds_value * stake
    profit = total_returns - total_staked
    
    results.update({
        'total_bets': total_bets,
        'winning_bets': winning_bets,
        'losing_bets': total_bets - winning_bets,
        'total_staked': total_staked,
        'total_returns': total_returns,
        'profit': profit,
        'roi': (profit / total_staked) * 100 if total_staked > 0 else 0,
        'avg_edge': float(np.mean(edges_value)),
        'avg_odds': float(np.mean(odds_value))
    })
    
    return results


# =============================================================================
# CALIBRATION ANALYSIS
# =============================================================================

def analyze_calibration(y_true, y_prob, n_bins=10):
    """
    Analyze probability calibration.
    
    Returns:
        Dictionary with calibration metrics
    """
    try:
        prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins)
        
        # Expected Calibration Error
        ece = np.mean(np.abs(prob_true - prob_pred))
        
        return {
            'ece': ece,
            'prob_true': prob_true,
            'prob_pred': prob_pred
        }
    except:
        return {'ece': None}


# =============================================================================
# BACKTESTING
# =============================================================================

def walk_forward_backtest(df, features, target, model_class, model_params,
                         train_size=500, test_size=100, step=50,
                         odds_col='AvgH', market_type='classification'):
    """
    Walk-forward backtesting for time-series data.
    
    Args:
        df: DataFrame with features and target
        features: List of feature columns
        target: Target column name
        model_class: Model class to use
        model_params: Model parameters
        train_size: Number of training samples
        test_size: Number of test samples
        step: Step size for rolling window
        odds_col: Column name for odds
        market_type: 'classification' or 'regression'
    
    Returns:
        DataFrame with backtest results
    """
    from sklearn.model_selection import train_test_split
    
    results = []
    n = len(df)
    
    for start in range(0, n - train_size - test_size + 1, step):
        end = start + train_size + test_size
        
        if end > n:
            break
        
        train = df.iloc[start:start + train_size]
        test = df.iloc[start + train_size:end]
        
        # Prepare data
        X_train = train[features].dropna()
        y_train = train.loc[X_train.index, target]
        X_test = test[features].dropna()
        y_test = test.loc[X_test.index, target]
        
        if len(X_train) == 0 or len(X_test) == 0:
            continue
        
        # Handle NaN in target
        valid_mask = y_train.notna() & y_test.notna()
        X_train = X_train[valid_mask[:len(X_train)]]
        y_train = y_train[valid_mask[:len(X_train)]]
        X_test = X_test[valid_mask[len(X_train):]]
        y_test = y_test[valid_mask[len(X_train):]]
        
        if len(X_train) < 50 or len(X_test) < 10:
            continue
        
        # Train model
        try:
            model = model_class(**model_params)
            model.fit(X_train, y_train)
            
            # Predict
            y_pred = model.predict(X_test)
            
            # Calculate metrics
            if market_type == 'classification':
                accuracy = accuracy_score(y_test, y_pred)
                results.append({
                    'start_idx': start,
                    'end_idx': end,
                    'train_size': len(X_train),
                    'test_size': len(X_test),
                    'accuracy': accuracy
                })
            else:
                mae = mean_absolute_error(y_test, y_pred)
                results.append({
                    'start_idx': start,
                    'end_idx': end,
                    'train_size': len(X_train),
                    'test_size': len(X_test),
                    'mae': mae
                })
        except Exception as e:
            continue
    
    return pd.DataFrame(results)


# =============================================================================
# MARKET ANALYSIS
# =============================================================================

def analyze_market_efficiency(df, odds_col='AvgH', result_col='FTR'):
    """
    Analyze market efficiency for a betting market.
    
    Returns:
        Dictionary with market efficiency metrics
    """
    # Implied probabilities
    implied = 1 / df[odds_col]
    
    # Average implied probability vs actual frequency
    avg_implied = implied.mean()
    
    # For 3-way market
    if 'AvgH' in df.columns and 'AvgD' in df.columns and 'AvgA' in df.columns:
        implied_h = 1 / df['AvgH']
        implied_d = 1 / df['AvgD']
        implied_a = 1 / df['AvgA']
        
        # Actual frequencies
        actual_h = (df['FTR'] == 'H').mean()
        actual_d = (df['FTR'] == 'D').mean()
        actual_a = (df['FTR'] == 'A').mean()
        
        return {
            'avg_overround': (implied_h + implied_d + implied_a).mean() - 1,
            'home_implied_vs_actual': implied_h.mean() - actual_h,
            'draw_implied_vs_actual': implied_d.mean() - actual_d,
            'away_implied_vs_actual': implied_a.mean() - actual_a,
            'home_actual_freq': actual_h,
            'draw_actual_freq': actual_d,
            'away_actual_freq': actual_a
        }
    
    return {'avg_implied': avg_implied}


def analyze_model_edge(y_true, y_pred, y_prob, odds, threshold=0.05):
    """
    Analyze where the model has an edge over bookmakers.
    
    Returns:
        DataFrame with edge analysis
    """
    implied_probs = 1 / odds
    model_probs = y_prob
    edges = model_probs - implied_probs
    
    # Create analysis dataframe
    analysis = pd.DataFrame({
        'y_true': y_true,
        'y_pred': y_pred,
        'y_prob': model_probs,
        'implied_prob': implied_probs,
        'edge': edges,
        'has_edge': edges >= threshold
    })
    
    # Summary
    edge_summary = {
        'total_matches': len(analysis),
        'matches_with_edge': analysis['has_edge'].sum(),
        'edge_pct': analysis['has_edge'].mean() * 100,
        'avg_edge_when_present': analysis[analysis['has_edge']]['edge'].mean(),
        'accuracy_with_edge': analysis[analysis['has_edge']]['y_pred'].eq(
            analysis[analysis['has_edge']]['y_true']
        ).mean() if analysis['has_edge'].sum() > 0 else 0,
        'accuracy_without_edge': analysis[~analysis['has_edge']]['y_pred'].eq(
            analysis[~analysis['has_edge']]['y_true']
        ).mean() if (~analysis['has_edge']).sum() > 0 else 0
    }
    
    return analysis, edge_summary


# =============================================================================
# REPORTING
# =============================================================================

def generate_evaluation_report(results: Dict[str, Any], 
                              market_name: str,
                              league: str = None) -> str:
    """Generate a formatted evaluation report."""
    report = []
    report.append(f"\n{'='*60}")
    report.append(f"EVALUATION REPORT: {market_name.upper()}")
    if league:
        report.append(f"League: {league}")
    report.append(f"{'='*60}\n")
    
    if 'accuracy' in results:
        report.append(f"Accuracy:  {results['accuracy']:.4f}")
    if 'precision' in results:
        report.append(f"Precision: {results['precision']:.4f}")
    if 'recall' in results:
        report.append(f"Recall:    {results['recall']:.4f}")
    if 'f1' in results:
        report.append(f"F1 Score:  {results['f1']:.4f}")
    if 'log_loss' in results:
        report.append(f"Log Loss:  {results['log_loss']:.4f}")
    if 'mae' in results:
        report.append(f"MAE:       {results['mae']:.4f}")
    if 'rmse' in results:
        report.append(f"RMSE:      {results['rmse']:.4f}")
    if 'r2' in results:
        report.append(f"R²:        {results['r2']:.4f}")
    
    # Profit simulation
    if 'profit_sim' in results:
        ps = results['profit_sim']
        report.append(f"\n--- Profit Simulation ---")
        report.append(f"Total Bets:     {ps['total_bets']}")
        report.append(f"Winning Bets:   {ps['winning_bets']}")
        report.append(f"Win Rate:       {ps['win_rate']:.1f}%")
        report.append(f"Profit:         ${ps['profit']:.2f}")
        report.append(f"ROI:            {ps['roi']:.1f}%")
        report.append(f"Avg Odds:       {ps['avg_odds']:.2f}")
    
    report.append(f"\n{'='*60}\n")
    
    return '\n'.join(report)


def compare_leagues(league_results: Dict[str, Dict], market: str) -> pd.DataFrame:
    """Compare model performance across leagues."""
    rows = []
    for league, results in league_results.items():
        if market in results:
            r = results[market]
            row = {'League': league}
            for key in ['accuracy', 'precision', 'recall', 'f1', 'log_loss', 'mae']:
                if key in r:
                    row[key] = r[key]
            rows.append(row)
    
    return pd.DataFrame(rows).sort_values('accuracy' if 'accuracy' in rows[0] else 'mae', 
                                          ascending='mae' in rows[0])
