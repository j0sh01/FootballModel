# Football Prediction System

Predict football match outcomes across 10 leagues using machine learning.

## Quick Start

```bash
# 1. Clone the repository
git clone https://github.com/yourusername/FootballModel.git
cd FootballModel

# 2. Create virtual environment
python3 -m venv venv
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run setup (downloads data, trains models)
python setup.py

# 5. Start the API
python api/app.py
```

**API docs:** http://localhost:8000/docs

## What Gets Downloaded & Trained

| Step | What | Size | Time |
|------|------|------|------|
| Download | 80 CSV files from football-data.co.uk | ~50MB | ~2 min |
| Features | 30+ engineered features per match | ~100MB cache | ~10 min |
| Training | 140 models (10 leagues × 14 markets) | ~200MB | ~15 min |

## Leagues Covered

| League | Country | Tier | Matches/Season |
|--------|---------|:----:|:--------------:|
| English Premier League | England | 1 | 380 |
| English Championship | England | 2 | 552 |
| Bundesliga 1 | Germany | 1 | 306 |
| Bundesliga 2 | Germany | 2 | 306 |
| Serie A | Italy | 1 | 380 |
| Serie B | Italy | 2 | 380 |
| La Liga | Spain | 1 | 380 |
| La Liga 2 | Spain | 2 | 462 |
| Ligue 1 | France | 1 | 180-380 |
| Ligue 2 | France | 2 | 180-380 |

## Betting Markets

### Low Risk (Recommended)
- **Match Result (1X2)** — Home/Draw/Away
- **Over/Under 2.5 Goals** — Binary
- **Both Teams to Score (BTTS)** — Binary
- **Double Chance 1X** — Home or Draw (~67% hit rate)
- **Double Chance X2** — Draw or Away
- **Goals Bucket** — 0-1, 2-3, 4+

### Medium Risk
- **Half-Time Result**
- **Over/Under 1.5 & 3.5 Goals**
- **HT Double Chance 1X**

### New Markets
- **Clean Sheet (Home/Away)**
- **Win to Nil (Home/Away)**

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/leagues` | List all leagues |
| `GET` | `/api/leagues/{league}/teams` | Get teams in a league |
| `GET` | `/api/leagues/{league}/models` | List trained models |
| `POST` | `/api/predict` | **Predict a match** |
| `POST` | `/api/compare` | **Compare two teams** |
| `GET` | `/api/markets` | List all markets |
| `GET` | `/docs` | Interactive API docs |

### Example Prediction

```bash
curl -X POST http://localhost:8000/api/predict \
  -H "Content-Type: application/json" \
  -d '{
    "league": "epl",
    "home_team": "Arsenal",
    "away_team": "Chelsea",
    "odds": {"home": 1.85, "draw": 3.60, "away": 4.20}
  }'
```

## Project Structure

```
FootballModel/
├── api/                    # FastAPI prediction API
│   ├── app.py             # API endpoints
│   └── service.py         # Prediction service
├── src/                    # Core modules
│   ├── data_loader.py     # Load CSV data
│   ├── feature_engineering.py  # 30+ features
│   ├── market_models.py   # XGBoost, RF, GB, Ensemble
│   └── evaluation.py      # Backtesting, profit sim
├── notebooks/              # Jupyter notebooks
├── data/                   # League CSVs (downloaded)
├── models/                 # Trained .pkl files
├── setup.py               # One-command setup
├── train_all.py           # Retrain all models
├── train_from_cache.py    # Fast retrain from cache
└── requirements.txt       # Python dependencies
```

## Development

```bash
# Retrain from cached features (fast)
python train_from_cache.py

# Retrain everything (slow)
python train_all.py

# Run notebooks
jupyter notebook notebooks/

# Run API with auto-reload
uvicorn api.app:app --reload
```

## Model Performance

| Market | Avg Accuracy | Best League |
|--------|:---:|:---:|
| Away Win to Nil | **80.4%** | La Liga 2 |
| Away Clean Sheet | **70.6%** | La Liga |
| Home Win to Nil | **70.4%** | La Liga 2 |
| Double Chance 1X | **66.7%** | Serie A |
| Over/Under 2.5 | **53.9%** | Bundesliga 1 |
| Match Result | **46.2%** | La Liga |

## Tech Stack

- **Python 3.12** with pandas, numpy
- **XGBoost**, Random Forest, Gradient Boosting
- **FastAPI** for REST API
- **scikit-learn** for preprocessing & metrics
- **Elo ratings** for team strength
- **Walk-forward backtesting** for validation
