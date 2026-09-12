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

**Test console:** http://localhost:8000/ui/
**API docs:** http://localhost:8000/docs

## Web Test Console

The API ships with a self-contained test console (`frontend/`) — plain HTML, CSS and
JavaScript, no build step, no framework and **no database**. FastAPI serves it, so there
is nothing extra to install or run.

| URL | What it is |
|-----|------------|
| `/` | Redirects to the console |
| `/ui/` | The test console |
| `/api/*` | JSON API used by the console |
| `/docs` | Swagger UI |

What you can do in it:

- **Predict** — pick a league, then queue as many fixtures as you want (each with its own
  optional bookmaker odds), choose which markets to run, and hit **Predict all**. Matches
  are processed as one batched request per league and every fixture gets its own result
  card with probabilities, fair odds and the edge against the odds you supplied. Failed
  fixtures (e.g. a typo in a team name) fail alone and never abort the batch.
  `⌘/Ctrl + Enter` queues the fixture you are typing, or predicts the whole queue.
  The queue survives page reloads (kept in `localStorage`).
- **Compare** — the underlying model features for two teams side by side (form, streaks,
  attack/defense strength, H2H, goal timing).
- **Models** — every model the API loaded for a league, with walk-forward accuracy or MAE,
  plus an accuracy chart, sortable and filterable. Markets served as an ensemble show
  their member algorithms.
- **Activity** — API health, the raw JSON of the last response, a request log with
  status/latency, and an exportable history of past runs (JSON or CSV).

Your selections, run history and request log are kept in the browser's `localStorage`.
The header's **API base** field lets you point the console at a different host (leave it
empty when the console is served by the API).

### Running the console on its own

It is three static files, so any static host works:

```bash
python -m http.server 5500 --directory frontend   # then set API base to http://localhost:8000
```

## How Models Are Combined

Each league has 14 markets and every market is scored by its own model. When more than
one algorithm was trained for a market, the API combines them instead of keeping only one:

| `ENSEMBLE_MODE` | Behaviour |
|-----------------|-----------|
| `soft` (default) | Averages the class probabilities of every model trained for the market |
| `hard` | Majority-votes their labels |
| `off` | Loads a single model per market (`PREFERRED_MODEL_TYPE`, default `xgboost`) |

```bash
ENSEMBLE_MODE=off python api/app.py      # the smaller, single-model path
```

Only `match_result`, `over_under_25` and `btts` currently have all three algorithms
(XGBoost, Random Forest, Gradient Boosting) on disk, so those are the markets that get
combined — every other market has one model. Combined markets are labelled `ensemble` in
the console and in the prediction cards, which list their member algorithms.

Combining is not a free win. Measured across all 30 combined markets on this repo's
holdout, soft voting moved accuracy by a mean of **+0.02 points** (median +0.16; better on
16 markets, worse on 11, tied on 3) while lowering average peak confidence by about
**4.6 points**. The lower confidence is the more useful half: averaged probabilities are
less overconfident. Treat the ensemble as a calibration improvement, not an accuracy jump
— and expect roughly **200 MB more RAM** and a couple of extra seconds of startup, since
the Random Forest members are much larger than the XGBoost ones.

A market is only scored as an ensemble when all of its members were trained on the same
features and the same split. If a member was trained earlier, from a different feature
cache, the API reports no accuracy for that market. `python retrain_ensembles.py`
rebuilds every member of every ensemble market from `data/cached_features` and reports
what changed — see DEPLOYMENT.md section 9.

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
| `GET` | `/ui/` | **Test console** (static HTML/CSS/JS) |
| `GET` | `/api/info` | Service info and model counts |
| `GET` | `/api/leagues` | List all leagues |
| `GET` | `/api/leagues/{league}/teams` | Get teams in a league |
| `GET` | `/api/leagues/{league}/models` | List trained models |
| `POST` | `/api/predict` | **Predict a match** |
| `POST` | `/api/predict/batch` | **Predict many matches of one league** (max 100) |
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
  }
```

### Example Batch Prediction

```bash
curl -X POST http://localhost:8000/api/predict/batch \
  -H "Content-Type: application/json" \
  -d '{
    "league": "epl",
    "matches": [
      {"home_team": "Arsenal", "away_team": "Chelsea",
       "odds": {"home": 1.85, "draw": 3.60, "away": 4.20}},
      {"home_team": "Liverpool", "away_team": "Everton"}
    ]
  }
```

Each entry comes back with its own `status`, `elapsed_ms` and either a full
`prediction` or an `error` — results are always in request order.

## Project Structure

```
FootballModel/
├── api/                    # FastAPI prediction API
│   ├── app.py             # API endpoints + static UI mount
│   └── service.py         # Prediction service
├── frontend/               # Test console (plain HTML/CSS/JS, no build step)
│   ├── index.html
│   ├── styles.css
│   └── app.js
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
├── retrain_ensembles.py   # Rebuild ensemble members from cached features
├── requirements.txt       # Python dependencies
└── DEPLOYMENT.md          # VPS deployment guide (nginx, systemd, HTTPS)
```

## Deployment

See **[DEPLOYMENT.md](DEPLOYMENT.md)** for a step-by-step VPS walkthrough: server
hardening, Python venv, systemd unit, nginx reverse proxy, HTTPS via certbot,
zero-downtime model updates, backups and troubleshooting.

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
