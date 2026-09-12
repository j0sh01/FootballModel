"""
Market Models Module
Trains and manages prediction models for different football betting markets.
"""

import pandas as pd
import numpy as np
import joblib
import os
from typing import Dict, List, Optional, Tuple, Any
from sklearn.model_selection import train_test_split, GridSearchCV, TimeSeriesSplit
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import accuracy_score, classification_report
import xgboost as xgb
from sklearn.ensemble import (
    RandomForestClassifier, RandomForestRegressor,
    GradientBoostingClassifier, GradientBoostingRegressor,
    VotingClassifier, StackingClassifier
)
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.svm import SVC, SVR
from sklearn.neural_network import MLPClassifier, MLPRegressor
import warnings
warnings.filterwarnings('ignore')

from src.feature_engineering import MARKET_FEATURES, MARKET_TARGETS, MARKET_TYPES


# =============================================================================
# MODEL REGISTRY
# =============================================================================

CLASSIFICATION_MODELS = {
    'xgboost': {
        'class': xgb.XGBClassifier,
        'params': {
            'n_estimators': [100, 200],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.05, 0.1],
            'subsample': [0.8, 1.0],
            'colsample_bytree': [0.8, 1.0],
            'use_label_encoder': False,
            'eval_metric': 'mlogloss',
            'random_state': 42
        },
        'default': {
            'n_estimators': 200,
            'max_depth': 5,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'use_label_encoder': False,
            'eval_metric': 'mlogloss',
            'random_state': 42
        }
    },
    'random_forest': {
        'class': RandomForestClassifier,
        'params': {
            'n_estimators': [100, 200],
            'max_depth': [5, 10, 15],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2],
            'random_state': 42,
            'n_jobs': -1
        },
        'default': {
            'n_estimators': 200,
            'max_depth': 10,
            'min_samples_split': 5,
            'min_samples_leaf': 2,
            'random_state': 42,
            'n_jobs': -1
        }
    },
    'gradient_boosting': {
        'class': GradientBoostingClassifier,
        'params': {
            'n_estimators': [100, 200],
            'max_depth': [3, 5],
            'learning_rate': [0.05, 0.1],
            'subsample': [0.8, 1.0],
            'random_state': 42
        },
        'default': {
            'n_estimators': 200,
            'max_depth': 5,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'random_state': 42
        }
    },
    'logistic_regression': {
        'class': LogisticRegression,
        'params': {
            'C': [0.1, 1.0, 10.0],
            'max_iter': [1000],
            'random_state': 42
        },
        'default': {
            'C': 1.0,
            'max_iter': 1000,
            'random_state': 42
        }
    },
    'mlp': {
        'class': MLPClassifier,
        'params': {
            'hidden_layer_sizes': [(64, 32), (128, 64)],
            'activation': ['relu'],
            'max_iter': [500],
            'random_state': 42
        },
        'default': {
            'hidden_layer_sizes': (128, 64),
            'activation': 'relu',
            'max_iter': 500,
            'random_state': 42
        }
    }
}

REGRESSION_MODELS = {
    'xgboost': {
        'class': xgb.XGBRegressor,
        'params': {
            'n_estimators': [100, 200],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.05, 0.1],
            'subsample': [0.8, 1.0],
            'colsample_bytree': [0.8, 1.0],
            'random_state': 42
        },
        'default': {
            'n_estimators': 200,
            'max_depth': 5,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42
        }
    },
    'random_forest': {
        'class': RandomForestRegressor,
        'params': {
            'n_estimators': [100, 200],
            'max_depth': [5, 10, 15],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2],
            'random_state': 42,
            'n_jobs': -1
        },
        'default': {
            'n_estimators': 200,
            'max_depth': 10,
            'min_samples_split': 5,
            'min_samples_leaf': 2,
            'random_state': 42,
            'n_jobs': -1
        }
    },
    'gradient_boosting': {
        'class': GradientBoostingRegressor,
        'params': {
            'n_estimators': [100, 200],
            'max_depth': [3, 5],
            'learning_rate': [0.05, 0.1],
            'subsample': [0.8, 1.0],
            'random_state': 42
        },
        'default': {
            'n_estimators': 200,
            'max_depth': 5,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'random_state': 42
        }
    },
    'ridge': {
        'class': Ridge,
        'params': {
            'alpha': [0.1, 1.0, 10.0]
        },
        'default': {
            'alpha': 1.0
        }
    },
    'mlp': {
        'class': MLPRegressor,
        'params': {
            'hidden_layer_sizes': [(64, 32), (128, 64)],
            'activation': ['relu'],
            'max_iter': [500],
            'random_state': 42
        },
        'default': {
            'hidden_layer_sizes': (128, 64),
            'activation': 'relu',
            'max_iter': 500,
            'random_state': 42
        }
    }
}


# =============================================================================
# MODEL TRAINING
# =============================================================================

class MarketModel:
    """Wrapper for training and managing a single market model."""
    
    def __init__(self, market_name: str, league: str, model_type: str = 'xgboost',
                 use_grid_search: bool = False, test_size: float = 0.2):
        """
        Args:
            market_name: Market identifier (e.g., 'match_result', 'over_under_25')
            league: League identifier (e.g., 'epl', 'bundesliga1')
            model_type: Model algorithm (e.g., 'xgboost', 'random_forest')
            use_grid_search: Whether to use GridSearchCV
            test_size: Test split proportion
        """
        self.market_name = market_name
        self.league = league
        self.model_type = model_type
        self.use_grid_search = use_grid_search
        self.test_size = test_size
        
        self.market_type = MARKET_TYPES.get(market_name, 'classification')
        self.features = MARKET_FEATURES.get(market_name, [])
        self.target = MARKET_TARGETS.get(market_name)
        
        self.model = None
        self.scaler = None
        self.label_encoder = None
        self.feature_names = None
        self.results = {}
    
    def _get_model_config(self):
        """Get model configuration based on type and market type."""
        registry = CLASSIFICATION_MODELS if self.market_type == 'classification' else REGRESSION_MODELS
        
        if self.model_type not in registry:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
        return registry[self.model_type]
    
    def prepare_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare features and target from dataframe."""
        available_features = [f for f in self.features if f in df.columns]
        self.feature_names = available_features
        
        df_clean = df.dropna(subset=available_features + [self.target])
        
        X = df_clean[available_features].copy()
        y = df_clean[self.target].copy()
        
        # Handle any remaining inf/nan
        X = X.replace([np.inf, -np.inf], np.nan)
        X = X.fillna(0)
        
        return X, y
    
    def train(self, df: pd.DataFrame, save_path: str = None) -> Dict:
        """
        Train the model on the given data.
        
        Returns:
            Dictionary with training results
        """
        print(f"\nTraining {self.market_name} model for {self.league}")
        print("=" * 50)
        
        X, y = self.prepare_data(df)
        
        if len(X) < 50:
            print("Not enough data to train")
            return {}
        
        print(f"Samples: {len(X)}, Features: {len(self.feature_names)}")
        
        # Handle label encoding for classification
        if self.market_type == 'classification':
            le = LabelEncoder()
            y_encoded = le.fit_transform(y)
            self.label_encoder = le
            # Ensure target names are strings
            target_names = [str(c) for c in le.classes_]
        else:
            y_encoded = y.values
            target_names = None
        
        # Time-series split
        split_idx = int(len(X) * (1 - self.test_size))
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y_encoded[:split_idx], y_encoded[split_idx:]
        
        print(f"Train: {len(X_train)}, Test: {len(X_test)}")
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        self.scaler = scaler
        
        # Get model config
        config = self._get_model_config()
        
        # Train model
        if self.use_grid_search:
            print("Running GridSearchCV...")
            tscv = TimeSeriesSplit(n_splits=3)
            model = config['class'](**{k: v for k, v in config['params'].items() 
                                       if not isinstance(v, list)})
            
            # Simple parameter grid
            param_grid = {k: v for k, v in config['params'].items() 
                         if isinstance(v, list)}
            
            if param_grid:
                grid_search = GridSearchCV(
                    model, param_grid, cv=tscv, 
                    scoring='accuracy' if self.market_type == 'classification' else 'neg_mean_absolute_error',
                    n_jobs=-1, verbose=0
                )
                
                if self.market_type == 'classification':
                    sample_weights = compute_sample_weight('balanced', y_train)
                    grid_search.fit(X_train_scaled, y_train, sample_weight=sample_weights)
                else:
                    grid_search.fit(X_train_scaled, y_train)
                
                self.model = grid_search.best_estimator_
                print(f"Best params: {grid_search.best_params_}")
            else:
                self.model = config['class'](**config['default'])
                if self.market_type == 'classification':
                    sample_weights = compute_sample_weight('balanced', y_train)
                    self.model.fit(X_train_scaled, y_train, sample_weight=sample_weights)
                else:
                    self.model.fit(X_train_scaled, y_train)
        else:
            print(f"Training with {self.model_type} (default params)...")
            self.model = config['class'](**config['default'])
            
            if self.market_type == 'classification':
                sample_weights = compute_sample_weight('balanced', y_train)
                self.model.fit(X_train_scaled, y_train, sample_weight=sample_weights)
            else:
                self.model.fit(X_train_scaled, y_train)
        
        # Evaluate
        y_pred = self.model.predict(X_test_scaled)
        
        if self.market_type == 'classification':
            accuracy = accuracy_score(y_test, y_pred)
            print(f"\nAccuracy: {accuracy:.4f}")
            
            if target_names is not None and len(target_names) == len(np.unique(y_test)):
                try:
                    report = classification_report(
                        y_test, y_pred, 
                        target_names=target_names,
                        zero_division=0
                    )
                    print(f"\nClassification Report:\n{report}")
                except Exception as e:
                    print(f"Could not generate classification report: {e}")
            
            # Get probabilities if available
            y_prob = None
            if hasattr(self.model, 'predict_proba'):
                try:
                    y_prob = self.model.predict_proba(X_test_scaled)
                except:
                    pass
            
            self.results = {
                'accuracy': float(accuracy),
                'y_test': y_test.tolist() if hasattr(y_test, 'tolist') else y_test,
                'y_pred': y_pred.tolist() if hasattr(y_pred, 'tolist') else y_pred,
                'y_prob': y_prob.tolist() if y_prob is not None and hasattr(y_prob, 'tolist') else y_prob,
                'target_names': target_names
            }
        else:
            from sklearn.metrics import mean_absolute_error, r2_score
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            print(f"\nMAE: {mae:.4f}, R²: {r2:.4f}")
            
            self.results = {
                'mae': float(mae),
                'r2': float(r2),
                'y_test': y_test.tolist() if hasattr(y_test, 'tolist') else y_test,
                'y_pred': y_pred.tolist() if hasattr(y_pred, 'tolist') else y_pred
            }
        
        # Save model
        if save_path:
            self.save(save_path)
        
        return self.results
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions on new data."""
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        available_features = [f for f in self.feature_names if f in X.columns]
        X_scaled = self.scaler.transform(X[available_features])
        
        # Get predictions
        predictions = self.model.predict(X_scaled)
        
        # Decode labels if we have a label encoder
        if self.label_encoder is not None and self.market_type == 'classification':
            try:
                predictions = self.label_encoder.inverse_transform(predictions.astype(int))
            except:
                pass
        
        return predictions
    
    def predict_proba(self, X: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
        """Get prediction probabilities (classification only).
        
        Returns:
            Tuple of (probabilities array, class labels)
        """
        if self.market_type != 'classification':
            raise ValueError("predict_proba only available for classification models")
        
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        available_features = [f for f in self.feature_names if f in X.columns]
        X_scaled = self.scaler.transform(X[available_features])
        proba = self.model.predict_proba(X_scaled)
        
        # Get class labels
        if self.label_encoder is not None:
            class_labels = [str(c) for c in self.label_encoder.classes_]
        else:
            class_labels = [str(i) for i in range(proba.shape[1])]
        
        return proba, class_labels
    
    def save(self, path: str):
        """Save model to disk."""
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'label_encoder': self.label_encoder,
            'feature_names': self.feature_names,
            'market_name': self.market_name,
            'league': self.league,
            'model_type': self.model_type,
            'market_type': self.market_type,
            'results': self.results
        }
        
        joblib.dump(model_data, path)
        print(f"Model saved to {path}")
    
    @classmethod
    def load(cls, path: str) -> 'MarketModel':
        """Load model from disk."""
        model_data = joblib.load(path)
        
        instance = cls(
            market_name=model_data['market_name'],
            league=model_data['league'],
            model_type=model_data['model_type']
        )
        instance.model = model_data['model']
        instance.scaler = model_data['scaler']
        instance.label_encoder = model_data['label_encoder']
        instance.feature_names = model_data['feature_names']
        instance.results = model_data['results']
        
        return instance


# =============================================================================
# ENSEMBLE MODEL
# =============================================================================

class EnsembleModel:
    """Ensemble of learners for a single market.

    Exposes the same interface as `MarketModel` (`predict`, `predict_proba`,
    `feature_names`, `label_encoder`, `market_type`, `results`), so callers can
    treat a single model and a combined one interchangeably. Soft voting
    averages member probabilities over a shared class-label order.
    """

    #: Preference order. The first available member supplies the reference
    #: label encoder and feature order.
    DEFAULT_MEMBERS = ['xgboost', 'random_forest', 'gradient_boosting']
    
    def __init__(self, market_name: str, league: str,
                 model_types: List[str] = None, method: str = 'soft'):
        self.market_name = market_name
        self.league = league
        self.model_types = list(model_types) if model_types else list(self.DEFAULT_MEMBERS)
        self.method = method or 'soft'
        self.models: Dict[str, MarketModel] = {}
        self.results = {}

        # MarketModel-compatible surface
        self.model_type = 'ensemble'
        self.market_type = MARKET_TYPES.get(market_name, 'classification')
        self.features = MARKET_FEATURES.get(market_name, [])
        self.target = MARKET_TARGETS.get(market_name)
        self.feature_names = None
        self.label_encoder = None
        self.scaler = None
        self.model = None
    
    def train(self, df: pd.DataFrame, save_dir: str = None) -> Dict:
        """Train all models in the ensemble."""
        print(f"\nTraining ensemble for {self.market_name} ({self.league})")
        print("=" * 50)
        
        for model_type in self.model_types:
            print(f"\n--- Training {model_type} ---")
            model = MarketModel(
                market_name=self.market_name,
                league=self.league,
                model_type=model_type
            )
            
            save_path = None
            if save_dir:
                save_path = os.path.join(save_dir, f"{self.market_name}_{model_type}.pkl")
            
            model.train(df, save_path=save_path)
            self.add_model(model_type, model)
        
        return self.models
    
    def predict(self, X: pd.DataFrame, method: str = 'soft') -> np.ndarray:
        """
        Make ensemble predictions.
        
        Args:
            X: Features
            method: 'hard' (majority vote) or 'soft' (average probabilities)
        """
        method = (method or self.method or 'soft').lower()
        members = self._ordered_members()
        if not members:
            raise ValueError("Ensemble has no members")

        if method == 'hard':
            votes = np.concatenate([np.asarray(m.predict(X)).ravel() for _, m in members])
            labels, counts = np.unique(votes, return_counts=True)
            return np.array([labels[int(np.argmax(counts))]])
        
        else:  # soft
            voters = [mt for mt, m in self._ordered_members() if hasattr(m.model, 'predict_proba')]
            
            if self.market_type == 'classification' and voters:
                proba, labels = self._aligned_probabilities(X)
                return np.array([labels[int(np.argmax(proba[0]))]])

            # Regression, or no member exposes probabilities: average the values
            values = np.array([np.asarray(m.predict(X), dtype=float) for _, m in members])
            return values.mean(axis=0)
    
    def save(self, save_dir: str):
        """Save all models in the ensemble."""
        os.makedirs(save_dir, exist_ok=True)
        
        for model_type, model in self.models.items():
            path = os.path.join(save_dir, f"{self.market_name}_{model_type}.pkl")
            model.save(path)
    
    # -- members & MarketModel-compatible surface ---------------------------

    def add_model(self, model_type: str, model: MarketModel) -> 'EnsembleModel':
        """Register a member and refresh the derived attributes."""
        self.models[model_type] = model
        if model_type not in self.model_types:
            self.model_types.append(model_type)
        self._refresh()
        return self

    def _ordered_members(self) -> List[Tuple[str, MarketModel]]:
        """Members sorted by preference, so xgboost leads when present."""
        rank = {mt: i for i, mt in enumerate(self.DEFAULT_MEMBERS)}
        return sorted(self.models.items(), key=lambda kv: rank.get(kv[0], len(rank)))

    def _refresh(self):
        """Recompute the MarketModel-compatible attributes from the members."""
        members = [m for _, m in self._ordered_members()]
        if not members:
            return

        ref = members[0]
        self.label_encoder = ref.label_encoder
        self.scaler = ref.scaler
        self.model = ref.model
        self.market_type = ref.market_type or self.market_type

        # Union of member features, keeping the reference member's order
        feats = list(ref.feature_names or [])
        for member in members[1:]:
            for name in (member.feature_names or []):
                if name not in feats:
                    feats.append(name)
        self.feature_names = feats

        self.results = self._aggregate_results()

    def _aggregate_results(self) -> Dict:
        """Score the ensemble on the holdout probabilities its members stored.

        Members trained together by `train_league_models` were fitted on the
        same split, so their saved `y_test`/`y_prob` line up and the ensemble's
        own accuracy can be measured rather than guessed.
        """
        scored = [m for m in self.models.values() if m.results]
        if not scored:
            return {}

        best = dict(max(scored, key=lambda m: m.results.get('accuracy', 0) or 0).results)

        members = [m for m in self.models.values() if m.market_type == 'classification']
        if len(members) < 2:
            return best

        unscored = {
            'members': list(self.models.keys()),
            'accuracy_note': 'members were trained on different data splits; retrain to score the ensemble',
        }

        y_tests = [m.results.get('y_test') for m in members]
        y_probs = [m.results.get('y_prob') for m in members]
        if any(y is None for y in y_tests) or any(p is None for p in y_probs):
            return unscored

        y_test = np.asarray(y_tests[0])
        if any(np.asarray(y).shape != y_test.shape for y in y_tests):
            # Members came from different data snapshots, so there is no single
            # holdout to score on. Report it as unknown rather than passing off
            # one member's accuracy as the ensemble's.
            return unscored

        probas = [np.asarray(p, dtype=float) for p in y_probs]
        if any(p.ndim != 2 or p.shape != probas[0].shape for p in probas):
            return unscored

        avg_prob = np.mean(probas, axis=0)
        y_pred = np.argmax(avg_prob, axis=1)
        return {
            'accuracy': float(accuracy_score(y_test, y_pred)),
            'y_test': y_test.tolist(),
            'y_pred': y_pred.tolist(),
            'y_prob': avg_prob.tolist(),
            'target_names': members[0].results.get('target_names'),
            'members': list(self.models.keys()),
        }

    def _class_labels(self) -> List[str]:
        """Class labels shared by the members, in a stable order."""
        ref = self._ordered_members()[0][1]
        if ref.label_encoder is not None:
            return [str(c) for c in ref.label_encoder.classes_]
        if ref.results.get('target_names'):
            return [str(c) for c in ref.results['target_names']]
        probs = ref.results.get('y_prob')
        width = len(probs[0]) if probs else len(self.models)
        return [str(i) for i in range(width)]

    def _aligned_probabilities(self, X: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
        """Average member probabilities over a common class-label order."""
        labels = self._class_labels()
        rows = []
        for _, model in self._ordered_members():
            proba, member_labels = model.predict_proba(X)
            member_labels = [str(label) for label in member_labels]
            row = np.asarray(proba, dtype=float)[0]
            if member_labels != labels:
                index = {label: i for i, label in enumerate(member_labels)}
                if any(label not in index for label in labels):
                    continue  # member has a different class space
                row = np.asarray([row[index[label]] for label in labels])
            rows.append(row)

        if not rows:
            proba, member_labels = self._ordered_members()[0][1].predict_proba(X)
            return np.asarray(proba, dtype=float), [str(label) for label in member_labels]

        avg = np.mean(rows, axis=0)
        total = avg.sum()
        return (avg / total if total > 0 else avg).reshape(1, -1), labels

    def predict_proba(self, X: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
        """Averaged class probabilities, shaped exactly like `MarketModel`'s."""
        if self.market_type != 'classification':
            raise ValueError("predict_proba only available for classification models")
        return self._aligned_probabilities(X)

    @classmethod
    def load(cls, save_dir: str, market_name: str, league: str,
             model_types: List[str] = None, method: str = 'soft') -> 'EnsembleModel':
        """Load ensemble from disk."""
        instance = cls(market_name, league, model_types, method=method)
        
        for model_type in instance.model_types:
            path = os.path.join(save_dir, f"{market_name}_{model_type}.pkl")
            if os.path.exists(path):
                instance.add_model(model_type, MarketModel.load(path))
        
        return instance


# =============================================================================
# MODEL DISCOVERY & LOADING
# =============================================================================

#: Longest suffix first, so 'gradient_boosting' is not read as 'boosting'.
MODEL_TYPE_SUFFIXES = [
    'gradient_boosting', 'random_forest', 'logistic_regression',
    'xgboost', 'ridge', 'mlp',
]


def parse_model_filename(filename: str) -> Optional[Tuple[str, str]]:
    """Split `<market>_<model_type>.pkl` into `(market, model_type)`.

    Returns None for files that do not follow the convention.
    """
    if not filename.endswith('.pkl'):
        return None

    basename = filename[: -len('.pkl')]
    for model_type in MODEL_TYPE_SUFFIXES:
        if basename.endswith('_' + model_type):
            return basename[: -len(model_type) - 1], model_type
    return None


def discover_market_models(league_dir: str) -> Dict[str, Dict[str, str]]:
    """Map `market -> {model_type: path}` for every `.pkl` in a league dir."""
    found: Dict[str, Dict[str, str]] = {}
    if not os.path.isdir(league_dir):
        return found

    for filename in sorted(os.listdir(league_dir)):
        parsed = parse_model_filename(filename)
        if parsed is None:
            continue
        market, model_type = parsed
        found.setdefault(market, {})[model_type] = os.path.join(league_dir, filename)

    return found


def load_market_model(market: str, league: str, paths: Dict[str, str],
                      prefer: str = 'xgboost', ensemble: str = 'soft') -> Any:
    """Load one market, combining algorithms into an ensemble when available.

    Args:
        market: Market identifier (e.g. 'match_result')
        league: League identifier (e.g. 'epl')
        paths: model_type -> path, as returned by `discover_market_models`
        prefer: algorithm to use when combining is off or unavailable
        ensemble: 'soft'/'hard' to combine members, 'off' for a single model

    Returns:
        An `EnsembleModel` when several algorithms exist and combining is on,
        otherwise a single `MarketModel`.
    """
    available = list(paths)
    if not available:
        raise ValueError(f"No model files for '{market}' ({league})")

    # Stable preference order, so the member list and the single-model fallback
    # do not depend on the order the files happened to be listed in.
    rank = {mt: i for i, mt in enumerate(EnsembleModel.DEFAULT_MEMBERS)}
    available.sort(key=lambda mt: rank.get(mt, len(rank)))

    mode = (ensemble or 'off').strip().lower()
    if mode in ('soft', 'hard') and len(available) > 1:
        combined = EnsembleModel(market, league, model_types=available, method=mode)
        for model_type in available:
            combined.add_model(model_type, MarketModel.load(paths[model_type]))
        return combined

    chosen = prefer if prefer in paths else available[0]
    return MarketModel.load(paths[chosen])


# =============================================================================
# LEAGUE TRAINING PIPELINE
# =============================================================================

def train_league_models(df: pd.DataFrame, league: str, 
                       markets: List[str] = None,
                       model_types: List[str] = None,
                       save_dir: str = 'models',
                       use_grid_search: bool = False) -> Dict:
    """
    Train models for all markets for a specific league.
    
    Args:
        df: Preprocessed DataFrame with features
        league: League identifier
        markets: List of market names (default: low-risk markets)
        model_types: List of model types
        save_dir: Directory to save models
    
    Returns:
        Dictionary of trained models
    """
    if markets is None:
        # Low-risk markets
        markets = [
            'match_result', 'over_under_25', 'btts',
            'double_chance_1x', 'double_chance_x2',
            'goals_bucket', 'ht_result'
        ]
    
    if model_types is None:
        model_types = ['xgboost']
    
    league_dir = os.path.join(save_dir, league)
    os.makedirs(league_dir, exist_ok=True)
    
    trained_models = {}
    
    for market in markets:
        print(f"\n{'='*60}")
        print(f"Training {market} for {league}")
        print(f"{'='*60}")
        
        if len(model_types) > 1:
            # Ensemble
            ensemble = EnsembleModel(market, league, model_types)
            ensemble.train(df, league_dir)
            trained_models[market] = ensemble
        else:
            # Single model
            model = MarketModel(market, league, model_types[0], use_grid_search)
            save_path = os.path.join(league_dir, f"{market}_{model_types[0]}.pkl")
            model.train(df, save_path=save_path)
            trained_models[market] = model
    
    return trained_models


# =============================================================================
# PREDICTION
# =============================================================================

def predict_match(home_team: str, away_team: str, 
                 models: Dict[str, Any],
                 df: pd.DataFrame,
                 odds: Dict[str, float] = None) -> Dict:
    """
    Make predictions for a match using trained models.
    
    Args:
        home_team: Home team name
        away_team: Away team name
        models: Dictionary of trained models
        df: Historical data for feature calculation
        odds: Optional dict of odds
    
    Returns:
        Dictionary of predictions
    """
    from src.feature_engineering import (
        calculate_form, calculate_h2h, calculate_rolling_stats, calculate_elo
    )
    
    current_date = pd.Timestamp.now()
    
    # Calculate features
    home_form = calculate_form(df, home_team, current_date, 5, 'points')
    away_form = calculate_form(df, away_team, current_date, 5, 'points')
    h2h = calculate_h2h(df, home_team, away_team, current_date)
    
    home_perf = calculate_rolling_stats(df, home_team, current_date, ['FTHG', 'HS', 'HST', 'HC'])
    away_perf = calculate_rolling_stats(df, away_team, current_date, ['FTAG', 'AS', 'AST', 'AC'])
    home_disc = calculate_rolling_stats(df, home_team, current_date, ['HY', 'HR'])
    away_disc = calculate_rolling_stats(df, away_team, current_date, ['AY', 'AR'])
    
    home_goals_form = calculate_form(df, home_team, current_date, 5, 'goals_scored')
    away_goals_form = calculate_form(df, away_team, current_date, 5, 'goals_scored')
    home_conceded_form = calculate_form(df, home_team, current_date, 5, 'goals_conceded')
    away_conceded_form = calculate_form(df, away_team, current_date, 5, 'goals_conceded')
    home_win_form = calculate_form(df, home_team, current_date, 5, 'wins')
    away_win_form = calculate_form(df, away_team, current_date, 5, 'wins')
    
    # Elo
    _, elos = calculate_elo(df)
    home_elo = elos.get(home_team, 1500)
    away_elo = elos.get(away_team, 1500)
    
    # Build feature dict
    features = {
        'HomeTeamForm': home_form,
        'AwayTeamForm': away_form,
        'FormDiff': home_form - away_form,
        'HomeGoalsScoredForm': home_goals_form,
        'AwayGoalsScoredForm': away_goals_form,
        'GoalsScoredDiff': home_goals_form - away_goals_form,
        'HomeGoalsConcededForm': home_conceded_form,
        'AwayGoalsConcededForm': away_conceded_form,
        'GoalsConcededDiff': home_conceded_form - away_conceded_form,
        'HomeWinForm': home_win_form,
        'AwayWinForm': away_win_form,
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
        'AwayShotEfficiency': away_perf['FTAG'] / (away_perf['AS'] + 1e-6)
    }
    
    # Add odds features if available
    if odds:
        if 'home' in odds and 'draw' in odds and 'away' in odds:
            total = 1/odds['home'] + 1/odds['draw'] + 1/odds['away']
            features['NormProb_H'] = (1/odds['home']) / total
            features['NormProb_D'] = (1/odds['draw']) / total
            features['NormProb_A'] = (1/odds['away']) / total
            features['Overround'] = total - 1
        
        if 'over_25' in odds and 'under_25' in odds:
            total_ou = 1/odds['over_25'] + 1/odds['under_25']
            features['NormProb_O2.5'] = (1/odds['over_25']) / total_ou
            features['NormProb_U2.5'] = (1/odds['under_25']) / total_ou
        
        if 'ah_home' in odds and 'ah_away' in odds:
            total_ah = 1/odds['ah_home'] + 1/odds['ah_away']
            features['NormProb_AHH'] = (1/odds['ah_home']) / total_ah
            features['NormProb_AHA'] = (1/odds['ah_away']) / total_ah
    
    # Make predictions
    predictions = {}
    
    for market_name, model in models.items():
        try:
            feature_df = pd.DataFrame([features])
            
            # EnsembleModel.predict() mirrors MarketModel.predict() and returns
            # decoded label strings for classification markets.
            pred = model.predict(feature_df)

            if model.market_type == 'classification':
                proba_result = model.predict_proba(feature_df) if hasattr(model, 'predict_proba') else None
                proba = proba_result[0] if proba_result is not None else None
                class_labels = proba_result[1] if proba_result is not None else None

                pred_label = pred[0] if hasattr(pred, '__getitem__') else pred
                
                pred_dict = {
                    'prediction': str(pred_label),
                    'confidence': float(np.max(proba)) if proba is not None else None
                }
                
                if proba is not None and class_labels is not None:
                    pred_dict['probabilities'] = {
                        str(label): float(prob) 
                        for label, prob in zip(class_labels, proba[0])
                    }
                
                predictions[market_name] = pred_dict
            else:
                predictions[market_name] = {
                    'prediction': float(pred[0])
                }
        
        except Exception as e:
            predictions[market_name] = {'error': str(e)}
    
    return {
        'match': f"{home_team} vs {away_team}",
        'predictions': predictions
    }


# =============================================================================
# TEAM COMPARISON
# =============================================================================

def compare_teams(home_team: str, away_team: str, df: pd.DataFrame) -> Dict:
    """
    Comprehensive team comparison for pre-match analysis.
    
    Args:
        home_team: Home team name
        away_team: Away team name
        df: Historical data
    
    Returns:
        Dictionary with team comparison metrics
    """
    from src.feature_engineering import (
        calculate_form, calculate_h2h, calculate_rolling_stats,
        calculate_streak, calculate_home_away_split, calculate_goal_timing,
        calculate_clean_sheet_prob, calculate_attack_defense_strength,
        calculate_league_position_proxy
    )
    
    current_date = pd.Timestamp.now()
    
    # Basic form
    home_form = calculate_form(df, home_team, current_date, 10, 'points')
    away_form = calculate_form(df, away_team, current_date, 10, 'points')
    
    # Goals
    home_goals_scored = calculate_form(df, home_team, current_date, 10, 'goals_scored')
    away_goals_scored = calculate_form(df, away_team, current_date, 10, 'goals_scored')
    home_goals_conceded = calculate_form(df, home_team, current_date, 10, 'goals_conceded')
    away_goals_conceded = calculate_form(df, away_team, current_date, 10, 'goals_conceded')
    
    # Win rate
    home_wins = calculate_form(df, home_team, current_date, 10, 'wins')
    away_wins = calculate_form(df, away_team, current_date, 10, 'wins')
    
    # H2H
    h2h = calculate_h2h(df, home_team, away_team, current_date, 20)
    
    # Streaks
    home_streak = calculate_streak(df, home_team, current_date, 10)
    away_streak = calculate_streak(df, away_team, current_date, 10)
    
    # Home/Away split
    home_split = calculate_home_away_split(df, home_team, current_date, 10)
    away_split = calculate_home_away_split(df, away_team, current_date, 10)
    
    # Goal timing
    home_timing = calculate_goal_timing(df, home_team, current_date, 10)
    away_timing = calculate_goal_timing(df, away_team, current_date, 10)
    
    # Clean sheet
    home_cs = calculate_clean_sheet_prob(df, home_team, current_date, 15)
    away_cs = calculate_clean_sheet_prob(df, away_team, current_date, 15)
    
    # Attack/Defense strength
    home_strength = calculate_attack_defense_strength(df, home_team, current_date, 15)
    away_strength = calculate_attack_defense_strength(df, away_team, current_date, 15)
    
    # League position
    home_pos = calculate_league_position_proxy(df, home_team, current_date, 15)
    away_pos = calculate_league_position_proxy(df, away_team, current_date, 15)
    
    # Rolling stats
    home_perf = calculate_rolling_stats(df, home_team, current_date, ['FTHG', 'HS', 'HST', 'HC'], 10)
    away_perf = calculate_rolling_stats(df, away_team, current_date, ['FTAG', 'AS', 'AST', 'AC'], 10)
    
    # Recent matches
    home_recent = df[
        (df['HomeTeam'] == home_team) | (df['AwayTeam'] == home_team)
    ].sort_values('Date', ascending=False).head(10)
    
    away_recent = df[
        (df['HomeTeam'] == away_team) | (df['AwayTeam'] == away_team)
    ].sort_values('Date', ascending=False).head(10)
    
    comparison = {
        'teams': {'home': home_team, 'away': away_team},
        'form': {
            'home': {
                'overall': round(home_form, 3),
                'wins': round(home_wins, 3),
                'goals_scored': round(home_goals_scored, 2),
                'goals_conceded': round(home_goals_conceded, 2),
                'home_form': round(home_split['home_form_split'], 3),
                'away_form': round(home_split['away_form_split'], 3),
                'home_advantage': round(home_split['home_advantage'], 3)
            },
            'away': {
                'overall': round(away_form, 3),
                'wins': round(away_wins, 3),
                'goals_scored': round(away_goals_scored, 2),
                'goals_conceded': round(away_goals_conceded, 2),
                'home_form': round(away_split['home_form_split'], 3),
                'away_form': round(away_split['away_form_split'], 3),
                'home_advantage': round(away_split['home_advantage'], 3)
            }
        },
        'head_to_head': {
            'advantage': round(h2h, 3),
            'description': f"{home_team} dominates" if h2h > 0.6 else \
                          f"{away_team} dominates" if h2h < 0.4 else "Balanced"
        },
        'streaks': {
            'home': {
                'current': home_streak['current_streak'],
                'momentum': round(home_streak['momentum'], 2),
                'max_win_streak': home_streak['max_win_streak'],
                'max_loss_streak': home_streak['max_loss_streak']
            },
            'away': {
                'current': away_streak['current_streak'],
                'momentum': round(away_streak['momentum'], 2),
                'max_win_streak': away_streak['max_win_streak'],
                'max_loss_streak': away_streak['max_loss_streak']
            }
        },
        'attacking': {
            'home': {
                'strength': round(home_strength['attack_strength'], 3),
                'avg_goals': round(home_perf['FTHG'], 2),
                'avg_shots': round(home_perf['HS'], 2),
                'avg_shots_on_target': round(home_perf['HST'], 2)
            },
            'away': {
                'strength': round(away_strength['attack_strength'], 3),
                'avg_goals': round(away_perf['FTAG'], 2),
                'avg_shots': round(away_perf['AS'], 2),
                'avg_shots_on_target': round(away_perf['AST'], 2)
            }
        },
        'defensive': {
            'home': {
                'strength': round(home_strength['defense_strength'], 3),
                'clean_sheet_prob': round(home_cs, 3),
                'avg_corners_conceded': round(home_perf['HC'], 2)
            },
            'away': {
                'strength': round(away_strength['defense_strength'], 3),
                'clean_sheet_prob': round(away_cs, 3),
                'avg_corners_conceded': round(away_perf['AC'], 2)
            }
        },
        'goal_timing': {
            'home': {
                'ht_goals_ratio': round(home_timing['ht_goals_ratio'], 3),
                'second_half_ratio': round(home_timing['second_half_goals_ratio'], 3)
            },
            'away': {
                'ht_goals_ratio': round(away_timing['ht_goals_ratio'], 3),
                'second_half_ratio': round(away_timing['second_half_goals_ratio'], 3)
            }
        },
        'league_position': {
            'home': round(home_pos, 3),
            'away': round(away_pos, 3)
        },
        'recent_form_string': {
            'home': _get_form_string(home_recent, home_team),
            'away': _get_form_string(away_recent, away_team)
        }
    }
    
    return comparison


def _get_form_string(df: pd.DataFrame, team: str) -> str:
    """Generate a form string like 'WWDLW' for last 5 matches."""
    form = []
    for _, row in df.head(5).iterrows():
        if row['HomeTeam'] == team:
            if row['FTR'] == 'H': form.append('W')
            elif row['FTR'] == 'D': form.append('D')
            else: form.append('L')
        else:
            if row['FTR'] == 'A': form.append('W')
            elif row['FTR'] == 'D': form.append('D')
            else: form.append('L')
    return ''.join(form)


def print_team_comparison(comparison: Dict):
    """Pretty print team comparison."""
    home = comparison['teams']['home']
    away = comparison['teams']['away']
    
    print(f"\n{'='*70}")
    print(f"TEAM COMPARISON: {home} vs {away}")
    print(f"{'='*70}")
    
    # Form
    print(f"\n📊 FORM (Last 10 matches)")
    print(f"{'':20} {home:15} {away:15} {'Edge':>10}")
    print(f"{'-'*60}")
    
    h_form = comparison['form']['home']
    a_form = comparison['form']['away']
    
    print(f"{'Overall Form':20} {h_form['overall']:15.3f} {a_form['overall']:15.3f} {h_form['overall']-a_form['overall']:>+10.3f}")
    print(f"{'Win Rate':20} {h_form['wins']:15.3f} {a_form['wins']:15.3f} {h_form['wins']-a_form['wins']:>+10.3f}")
    print(f"{'Goals Scored/G':20} {h_form['goals_scored']:15.2f} {a_form['goals_scored']:15.2f} {h_form['goals_scored']-a_form['goals_scored']:>+10.2f}")
    print(f"{'Goals Conceded/G':20} {h_form['goals_conceded']:15.2f} {a_form['goals_conceded']:15.2f} {h_form['goals_conceded']-a_form['goals_conceded']:>+10.2f}")
    
    # Recent form string
    print(f"\n📈 RECENT RESULTS")
    print(f"{home}: {comparison['recent_form_string']['home']}")
    print(f"{away}: {comparison['recent_form_string']['away']}")
    
    # Streaks
    print(f"\n🔥 STREAKS & MOMENTUM")
    h_streak = comparison['streaks']['home']
    a_streak = comparison['streaks']['away']
    print(f"{'Current Streak':20} {h_streak['current']:+d}{a_streak['current']:15d}")
    print(f"{'Momentum':20} {h_streak['momentum']:15.2f} {a_streak['momentum']:15.2f}")
    print(f"{'Max Win Streak':20} {h_streak['max_win_streak']:15d} {a_streak['max_win_streak']:15d}")
    
    # Attacking
    print(f"\n⚽ ATTACKING STRENGTH")
    h_att = comparison['attacking']['home']
    a_att = comparison['attacking']['away']
    print(f"{'Attack Strength':20} {h_att['strength']:15.3f} {a_att['strength']:15.3f} {h_att['strength']-a_att['strength']:>+10.3f}")
    print(f"{'Avg Goals/G':20} {h_att['avg_goals']:15.2f} {a_att['avg_goals']:15.2f}")
    print(f"{'Avg Shots/G':20} {h_att['avg_shots']:15.2f} {a_att['avg_shots']:15.2f}")
    print(f"{'Avg Shots on T/G':20} {h_att['avg_shots_on_target']:15.2f} {a_att['avg_shots_on_target']:15.2f}")
    
    # Defensive
    print(f"\n🛡️  DEFENSIVE STRENGTH")
    h_def = comparison['defensive']['home']
    a_def = comparison['defensive']['away']
    print(f"{'Defense Strength':20} {h_def['strength']:15.3f} {a_def['strength']:15.3f} {h_def['strength']-a_def['strength']:>+10.3f}")
    print(f"{'Clean Sheet %':20} {h_def['clean_sheet_prob']:15.1%} {a_def['clean_sheet_prob']:15.1%}")
    
    # Head to Head
    print(f"\n🏆 HEAD-TO-HEAD")
    h2h = comparison['head_to_head']
    print(f"{home} advantage: {h2h['advantage']:.3f} ({h2h['description']})")
    
    # Goal Timing
    print(f"\n⏰ GOAL TIMING")
    h_time = comparison['goal_timing']['home']
    a_time = comparison['goal_timing']['away']
    print(f"{'HT Goals Ratio':20} {h_time['ht_goals_ratio']:15.1%} {a_time['ht_goals_ratio']:15.1%}")
    print(f"{'2nd Half Ratio':20} {h_time['second_half_ratio']:15.1%} {a_time['second_half_ratio']:15.1%}")
    
    print(f"\n{'='*70}\n")


# =============================================================================
# WALK-FORWARD BACKTESTING
# =============================================================================

def walk_forward_backtest(df: pd.DataFrame, market_name: str, league: str,
                         model_type: str = 'xgboost',
                         train_size: int = 500, test_size: int = 100,
                         step: int = 50) -> pd.DataFrame:
    """
    Walk-forward backtesting for time-series data.
    
    Args:
        df: DataFrame with features and targets
        market_name: Market identifier
        league: League identifier
        model_type: Model algorithm
        train_size: Number of training samples
        test_size: Number of test samples
        step: Step size for rolling window
    
    Returns:
        DataFrame with backtest results
    """
    from src.feature_engineering import MARKET_FEATURES, MARKET_TARGETS, MARKET_TYPES
    
    features = MARKET_FEATURES.get(market_name, [])
    target = MARKET_TARGETS.get(market_name)
    market_type = MARKET_TYPES.get(market_name, 'classification')
    
    # Filter available features
    available_features = [f for f in features if f in df.columns]
    
    if target not in df.columns:
        print(f"Target {target} not found")
        return pd.DataFrame()
    
    # Clean data
    df_clean = df.dropna(subset=available_features + [target]).copy()
    df_clean = df_clean.replace([np.inf, -np.inf], np.nan).fillna(0)
    
    results = []
    n = len(df_clean)
    
    print(f"\nWalk-Forward Backtest: {market_name} ({league})")
    print(f"{'='*50}")
    print(f"Total samples: {n}, Train: {train_size}, Test: {test_size}, Step: {step}")
    
    for start in range(0, n - train_size - test_size + 1, step):
        end = start + train_size + test_size
        
        if end > n:
            break
        
        train = df_clean.iloc[start:start + train_size]
        test = df_clean.iloc[start + train_size:end]
        
        X_train = train[available_features]
        y_train = train[target]
        X_test = test[available_features]
        y_test = test[target]
        
        if len(X_train) < 50 or len(X_test) < 10:
            continue
        
        # Handle label encoding for classification
        if market_type == 'classification':
            le = LabelEncoder()
            y_train_enc = le.fit_transform(y_train)
            y_test_enc = le.transform(y_test)
        else:
            y_train_enc = y_train.values
            y_test_enc = y_test.values
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Get model
        if model_type == 'xgboost':
            if market_type == 'classification':
                model = xgb.XGBClassifier(
                    n_estimators=200, max_depth=5, learning_rate=0.1,
                    use_label_encoder=False, eval_metric='mlogloss',
                    random_state=42
                )
            else:
                model = xgb.XGBRegressor(
                    n_estimators=200, max_depth=5, learning_rate=0.1,
                    random_state=42
                )
        elif model_type == 'random_forest':
            if market_type == 'classification':
                model = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
            else:
                model = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42)
        else:
            if market_type == 'classification':
                model = GradientBoostingClassifier(n_estimators=200, max_depth=5, random_state=42)
            else:
                model = GradientBoostingRegressor(n_estimators=200, max_depth=5, random_state=42)
        
        # Train with sample weights for classification
        if market_type == 'classification':
            sample_weights = compute_sample_weight('balanced', y_train_enc)
            model.fit(X_train_scaled, y_train_enc, sample_weight=sample_weights)
        else:
            model.fit(X_train_scaled, y_train_enc)
        
        # Predict
        y_pred = model.predict(X_test_scaled)
        
        # Calculate metrics
        if market_type == 'classification':
            accuracy = accuracy_score(y_test_enc, y_pred)
            
            # Get probabilities if available
            if hasattr(model, 'predict_proba'):
                y_prob = model.predict_proba(X_test_scaled)
                max_prob = np.max(y_prob, axis=1).mean()
            else:
                max_prob = accuracy
            
            results.append({
                'window': len(results) + 1,
                'train_start': start,
                'train_end': start + train_size,
                'test_start': start + train_size,
                'test_end': end,
                'accuracy': accuracy,
                'avg_confidence': max_prob,
                'n_train': len(X_train),
                'n_test': len(X_test)
            })
        else:
            from sklearn.metrics import mean_absolute_error, r2_score
            mae = mean_absolute_error(y_test_enc, y_pred)
            r2 = r2_score(y_test_enc, y_pred)
            
            results.append({
                'window': len(results) + 1,
                'train_start': start,
                'train_end': start + train_size,
                'test_start': start + train_size,
                'test_end': end,
                'mae': mae,
                'r2': r2,
                'n_train': len(X_train),
                'n_test': len(X_test)
            })
    
    results_df = pd.DataFrame(results)
    
    if not results_df.empty:
        print(f"\nBacktest Results:")
        print(f"Windows completed: {len(results_df)}")
        
        if market_type == 'classification':
            print(f"Mean Accuracy: {results_df['accuracy'].mean():.4f} (+/- {results_df['accuracy'].std():.4f})")
            print(f"Min Accuracy: {results_df['accuracy'].min():.4f}")
            print(f"Max Accuracy: {results_df['accuracy'].max():.4f}")
        else:
            print(f"Mean MAE: {results_df['mae'].mean():.4f}")
            print(f"Mean R²: {results_df['r2'].mean():.4f}")
    
    return results_df


def run_full_backtest(df: pd.DataFrame, league: str,
                     markets: List[str] = None,
                     model_type: str = 'xgboost') -> Dict[str, pd.DataFrame]:
    """
    Run backtesting for all markets.
    
    Returns:
        Dictionary of market -> backtest results
    """
    if markets is None:
        markets = [
            'match_result', 'over_under_25', 'btts',
            'double_chance_1x', 'double_chance_x2',
            'goals_bucket', 'ht_result'
        ]
    
    all_results = {}
    
    for market in markets:
        results = walk_forward_backtest(df, market, league, model_type)
        if not results.empty:
            all_results[market] = results
    
    return all_results
