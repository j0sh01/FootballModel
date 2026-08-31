"""
Football Prediction API
FastAPI application for match predictions and team analysis.

Run: python api/app.py
  or: uvicorn api.app:app --reload --host 0.0.0.0 --port 8000

API docs: http://localhost:8000/docs
"""

import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from contextlib import asynccontextmanager

from api.service import get_service, PredictionService


# =============================================================================
# Lifespan (startup/shutdown)
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models on startup."""
    print("Starting Football Prediction API...")
    service = get_service()
    print(f"Ready! {sum(len(m) for m in service.models.values())} models loaded.")
    yield
    print("Shutting down...")


# =============================================================================
# App
# =============================================================================

app = FastAPI(
    title="Football Prediction API",
    description="Predict EPL, Championship, Bundesliga, Serie A, La Liga, Ligue 1 and more",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =============================================================================
# Pydantic Models
# =============================================================================

class OddsInput(BaseModel):
    """Bookmaker odds for a match."""
    home: Optional[float] = Field(None, description="Home win odds", example=1.85)
    draw: Optional[float] = Field(None, description="Draw odds", example=3.60)
    away: Optional[float] = Field(None, description="Away win odds", example=4.20)
    over_25: Optional[float] = Field(None, description="Over 2.5 goals odds", example=1.75)
    under_25: Optional[float] = Field(None, description="Under 2.5 goals odds", example=2.10)
    ah_home: Optional[float] = Field(None, description="Asian Handicap home", example=1.90)
    ah_away: Optional[float] = Field(None, description="Asian Handicap away", example=1.95)


class PredictRequest(BaseModel):
    """Request to predict a match."""
    league: str = Field(..., description="League key (e.g., 'epl', 'bundesliga1')", example="epl")
    home_team: str = Field(..., description="Home team name", example="Arsenal")
    away_team: str = Field(..., description="Away team name", example="Chelsea")
    markets: Optional[List[str]] = Field(None, description="Specific markets to predict (default: all)")
    odds: Optional[OddsInput] = Field(None, description="Bookmaker odds for enhanced predictions")


class CompareRequest(BaseModel):
    """Request to compare two teams."""
    league: str = Field(..., description="League key", example="epl")
    home_team: str = Field(..., description="Home team name", example="Arsenal")
    away_team: str = Field(..., description="Away team name", example="Chelsea")


class LeagueInfo(BaseModel):
    """League information."""
    key: str
    name: str
    country: str
    tier: int
    has_models: bool
    n_models: int
    markets: List[str]


class ModelInfo(BaseModel):
    """Model information."""
    market: str
    model_type: str
    market_type: str
    accuracy: Optional[float]
    mae: Optional[float]
    n_features: int


class PredictionResult(BaseModel):
    """Single market prediction."""
    prediction: str
    confidence: Optional[float] = None
    probabilities: Optional[Dict[str, float]] = None
    predicted_value: Optional[float] = None
    model_type: Optional[str] = None
    market_type: Optional[str] = None
    error: Optional[str] = None


class MatchPrediction(BaseModel):
    """Full match prediction response."""
    league: str
    league_key: str
    home_team: str
    away_team: str
    predictions: Dict[str, PredictionResult]


class TeamComparison(BaseModel):
    """Team comparison response."""
    teams: Dict[str, str]
    form: Dict[str, Dict[str, Any]]
    head_to_head: Dict[str, Any]
    streaks: Dict[str, Dict[str, Any]]
    attacking: Dict[str, Dict[str, Any]]
    defensive: Dict[str, Dict[str, Any]]
    goal_timing: Dict[str, Dict[str, Any]]
    league_position: Dict[str, float]
    recent_form_string: Dict[str, str]


# =============================================================================
# Endpoints
# =============================================================================

@app.get("/", tags=["Root"])
def root():
    """API health check."""
    service = get_service()
    total_models = sum(len(m) for m in service.models.values())
    return {
        "status": "running",
        "service": "Football Prediction API",
        "version": "1.0.0",
        "total_models": total_models,
        "leagues_available": len(service.models),
        "docs": "/docs"
    }


@app.get("/api/leagues", response_model=List[LeagueInfo], tags=["Leagues"])
def get_leagues():
    """List all available leagues with their models."""
    service = get_service()
    return service.get_leagues()


@app.get("/api/leagues/{league}/teams", response_model=List[str], tags=["Leagues"])
def get_teams(league: str):
    """Get all teams in a league."""
    service = get_service()
    
    # Ensure league data is loaded
    try:
        service._ensure_league_data(league)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    
    teams = service.teams.get(league, [])
    if not teams:
        raise HTTPException(status_code=404, detail=f"No teams found for league '{league}'")
    
    return teams


@app.get("/api/leagues/{league}/models", response_model=List[ModelInfo], tags=["Models"])
def get_models(league: str):
    """Get trained models for a league."""
    service = get_service()
    
    models = service.get_models_info(league)
    if not models:
        raise HTTPException(status_code=404, detail=f"No models found for league '{league}'")
    
    return models


@app.post("/api/predict", response_model=MatchPrediction, tags=["Predictions"])
def predict_match(request: PredictRequest):
    """
    Predict the outcome of a football match.
    
    Returns predictions for multiple betting markets including:
    - **match_result**: Home/Draw/Away (1X2)
    - **over_under_25**: Over/Under 2.5 goals
    - **btts**: Both Teams to Score
    - **double_chance_1x**: Home or Draw
    - **double_chance_x2**: Draw or Away
    - **goals_bucket**: 0-1, 2-3, 4+ goals
    - **ht_result**: Half-Time Result
    - **home_clean_sheet**: Will home team keep a clean sheet
    - **away_clean_sheet**: Will away team keep a clean sheet
    - **home_win_to_nil**: Home win without conceding
    - **away_win_to_nil**: Away win without conceding
    - **over_under_15**: Over/Under 1.5 goals
    - **over_under_35**: Over/Under 3.5 goals
    - **ht_double_chance_1x**: Half-Time Double Chance 1X
    """
    service = get_service()
    
    # Validate league
    if request.league not in service.models:
        raise HTTPException(
            status_code=404,
            detail=f"No models for league '{request.league}'. Available: {list(service.models.keys())}"
        )
    
    # Validate teams
    service._ensure_league_data(request.league)
    teams = service.teams.get(request.league, [])
    
    if request.home_team not in teams:
        raise HTTPException(
            status_code=400,
            detail=f"Team '{request.home_team}' not found in {request.league}. "
                   f"Did you mean: {[t for t in teams if request.home_team.lower() in t.lower()][:5]}?"
        )
    
    if request.away_team not in teams:
        raise HTTPException(
            status_code=400,
            detail=f"Team '{request.away_team}' not found in {request.league}. "
                   f"Did you mean: {[t for t in teams if request.away_team.lower() in t.lower()][:5]}?"
        )
    
    if request.home_team == request.away_team:
        raise HTTPException(status_code=400, detail="Home and away teams must be different")
    
    # Convert odds
    odds = None
    if request.odds:
        odds = {k: v for k, v in request.odds.model_dump().items() if v is not None}
        if not odds:
            odds = None
    
    try:
        result = service.predict_match(
            league=request.league,
            home_team=request.home_team,
            away_team=request.away_team,
            markets=request.markets,
            odds=odds
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.post("/api/compare", response_model=TeamComparison, tags=["Analysis"])
def compare(request: CompareRequest):
    """
    Compare two teams with detailed statistics.
    
    Returns:
    - **form**: Recent form (last 10 matches) for both teams
    - **head_to_head**: H2H advantage
    - **streaks**: Current streaks and momentum
    - **attacking**: Attack strength, goals, shots
    - **defensive**: Defense strength, clean sheet probability
    - **goal_timing**: HT vs 2nd half goal ratios
    - **league_position**: Points per game proxy
    - **recent_form_string**: W/D/L string for last 5 matches
    """
    service = get_service()
    
    # Validate league
    try:
        service._ensure_league_data(request.league)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    
    teams = service.teams.get(request.league, [])
    
    if request.home_team not in teams:
        raise HTTPException(
            status_code=400,
            detail=f"Team '{request.home_team}' not found. "
                   f"Did you mean: {[t for t in teams if request.home_team.lower() in t.lower()][:5]}?"
        )
    
    if request.away_team not in teams:
        raise HTTPException(
            status_code=400,
            detail=f"Team '{request.away_team}' not found. "
                   f"Did you mean: {[t for t in teams if request.away_team.lower() in t.lower()][:5]}?"
        )
    
    try:
        result = service.compare_teams(request.league, request.home_team, request.away_team)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Comparison error: {str(e)}")


@app.get("/api/markets", tags=["Markets"])
def get_markets():
    """List all available betting markets with descriptions."""
    from src.feature_engineering import MARKET_TYPES
    
    markets = {
        'match_result': {'name': 'Match Result (1X2)', 'type': 'classification', 'risk': 'low',
                        'description': 'Predict Home win, Draw, or Away win'},
        'over_under_25': {'name': 'Over/Under 2.5 Goals', 'type': 'classification', 'risk': 'low',
                         'description': 'Will there be more than 2.5 goals?'},
        'btts': {'name': 'Both Teams to Score', 'type': 'classification', 'risk': 'low',
                'description': 'Will both teams score?'},
        'double_chance_1x': {'name': 'Double Chance 1X', 'type': 'classification', 'risk': 'low',
                            'description': 'Home win or Draw'},
        'double_chance_x2': {'name': 'Double Chance X2', 'type': 'classification', 'risk': 'low',
                            'description': 'Draw or Away win'},
        'goals_bucket': {'name': 'Total Goals Bucket', 'type': 'classification', 'risk': 'low',
                        'description': '0-1, 2-3, or 4+ goals'},
        'ht_result': {'name': 'Half-Time Result', 'type': 'classification', 'risk': 'medium',
                     'description': 'Score at half-time'},
        'home_clean_sheet': {'name': 'Home Clean Sheet', 'type': 'classification', 'risk': 'low',
                            'description': 'Will the home team keep a clean sheet?'},
        'away_clean_sheet': {'name': 'Away Clean Sheet', 'type': 'classification', 'risk': 'low',
                            'description': 'Will the away team keep a clean sheet?'},
        'home_win_to_nil': {'name': 'Home Win to Nil', 'type': 'classification', 'risk': 'medium',
                           'description': 'Home win without conceding'},
        'away_win_to_nil': {'name': 'Away Win to Nil', 'type': 'classification', 'risk': 'medium',
                           'description': 'Away win without conceding'},
        'over_under_15': {'name': 'Over/Under 1.5 Goals', 'type': 'classification', 'risk': 'low',
                         'description': 'Will there be more than 1.5 goals?'},
        'over_under_35': {'name': 'Over/Under 3.5 Goals', 'type': 'classification', 'risk': 'medium',
                         'description': 'Will there be more than 3.5 goals?'},
        'ht_double_chance_1x': {'name': 'HT Double Chance 1X', 'type': 'classification', 'risk': 'medium',
                               'description': 'Half-time: Home or Draw'},
    }
    
    return markets


@app.get("/api/health", tags=["Root"])
def health():
    """Detailed health check."""
    service = get_service()
    
    league_status = {}
    for league, models in service.models.items():
        league_status[league] = {
            'n_models': len(models),
            'markets': list(models.keys())
        }
    
    return {
        "status": "healthy",
        "total_models": sum(len(m) for m in service.models.values()),
        "leagues": league_status
    }


# =============================================================================
# Run
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api.app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
