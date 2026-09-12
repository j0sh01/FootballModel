# Deploying to a VPS

A complete, copy-paste guide for running the Football Prediction System (FastAPI + the
test console UI) on a Linux VPS behind nginx with HTTPS.

**Nothing here needs a database.** The app is stateless: it loads `.pkl` models into
memory, reads cached features from disk, and serves static files. The only state the UI
keeps (your selections, run history, request log) lives in the browser's `localStorage`.

---

## 0. What you are deploying

```
        Internet
           │  https://your-domain.com
           ▼
   ┌──────────────────┐        ┌──────────────────────────────────────┐
   │  nginx  (443/80) │───────▶│  uvicorn  127.0.0.1:8000             │
   │  TLS, gzip       │        │  ├─ /            → redirect /ui/     │
   └──────────────────┘        │  ├─ /ui/         → test console      │
                               │  ├─ /api/*       → JSON API          │
                               │  └─ /docs        → Swagger UI        │
                               └──────────────────────────────────────┘
                                        │ reads
                                        ▼
                             models/*.pkl  +  data/cached_features/*.pkl
                             (loaded into RAM once at startup)
```

Measured on this repo (models loaded at startup; each league you query adds its feature
cache on top — see `ENSEMBLE_MODE` in section 9):

| Metric | `ENSEMBLE_MODE=off` | `ENSEMBLE_MODE=soft` (default) |
|--------|---------------------|-------------------------------|
| Markets served | 140 (one model each) | 140 markets, 30 combined |
| Startup (models only) | ~2.4 s | ~5.0 s |
| Resident memory after load | ~290 MB | ~490 MB |
| `models/` on disk | 280 MB | 280 MB |
| `data/` (CSV + feature cache) | 70 MB | 70 MB |

Soft voting is the default: where a market has XGBoost, Random Forest **and** Gradient
Boosting on disk, it averages their probabilities. That costs roughly **200 MB extra RAM**
and about **2.5 s more startup**, because the Random Forest members are far larger than
the XGBoost ones. On a 2 GB VPS either mode fits comfortably alongside nginx; pick `off`
for the smaller footprint.

**Recommended VPS:** 2 vCPU, 2 GB RAM minimum, 4 GB comfortable, 40 GB disk.
CPU is only busy during predictions; inference needs no GPU.

---

## 1. Prepare the server

```bash
# On your machine
ssh root@YOUR_SERVER_IP
```

```bash
# On the server — updates and basics
apt update && apt upgrade -y
apt install -y python3 python3-venv python3-pip git nginx ufw fail2ban rsync curl
```

Create a non-root user that will own the app (never run the service as root):

```bash
adduser --disabled-password --gecos "" football
usermod -aG www-data football
mkdir -p /srv/football-model
chown -R football:football /srv/football-model
```

Firewall — only SSH and HTTP(S):

```bash
ufw allow OpenSSH
ufw allow 'Nginx Full'
ufw --force enable
ufw status
```

Copy your SSH key over (`ssh-copy-id football@YOUR_SERVER_IP`) and confirm you can log in
as `football` before you disable password auth in `/etc/ssh/sshd_config`. Optional but
recommended, plus `systemctl enable --now fail2ban`.

Set the clock so timestamps in logs and the UI make sense:

```bash
timedatectl set-timezone Etc/UTC   # or your local zone
```

---

## 2. Install Python dependencies

Log in as the app user and create the virtualenv:

```bash
su - football
cd /srv/football-model
python3 -m venv venv
./venv/bin/pip install --upgrade pip wheel
```

---

## 3. Get the code, data and models onto the server

`data/` and `models/` are listed in `.gitignore` (the CSVs and `.pkl` files are large),
so **a plain `git clone` gives you code only**. Choose one of the two paths below.

### Option A — clone, then let the project build its own data and models

```bash
cd /srv/football-model
git clone https://github.com/yourusername/FootballModel.git .

./venv/bin/pip install -r requirements.txt
./venv/bin/python setup.py          # downloads CSVs, engineers features, trains models
```

Training is the slow part: expect roughly 15–30 minutes and a sustained CPU load on a
small VPS. Run it inside `tmux` or `screen` so a dropped SSH connection doesn't kill it:

```bash
tmux new -s train
./venv/bin/python setup.py
# detach with Ctrl-B then D ; reattach with: tmux attach -t train
```

If training on the VPS is too heavy, train on your laptop and copy only the artefacts.

### Option B — copy trained artefacts from your machine (recommended)

From **your local machine**, in the project root:

```bash
rsync -avz --progress \
  models/ data/cached_features/ \
  football@YOUR_SERVER_IP:/srv/football-model/
```

Then on the server:

```bash
cd /srv/football-model
./venv/bin/pip install -r requirements.txt
mkdir -p data models
ls models/            # expect one folder per league (epl, laliga, …) full of .pkl
```

> The API needs `data/<league>/**.csv` **or** `data/cached_features/<league>_features.pkl`
> to serve team lists and to compute features. Copying `data/cached_features/` alone is
> enough to run predictions; copy the league CSV folders too if you plan to retrain on
> the server.

---

## 4. Smoke test before touching systemd

```bash
cd /srv/football-model
./venv/bin/python -m uvicorn api.app:app --host 127.0.0.1 --port 8000
```

In a second SSH session:

```bash
curl -s http://127.0.0.1:8000/api/health | head -c 200; echo
curl -s -o /dev/null -w '%{http_code} %{redirect_url}\n' http://127.0.0.1:8000/
curl -s http://127.0.0.1:8000/api/leagues | head -c 200; echo

curl -s -X POST http://127.0.0.1:8000/api/predict \
  -H 'Content-Type: application/json' \
  -d '{"league":"epl","home_team":"Arsenal","away_team":"Chelsea",
       "odds":{"home":1.85,"draw":3.60,"away":4.20}}' | head -c 400; echo
```

You should see `{"status":"healthy", ...}`, a `307` redirect to `/ui/`, and a JSON
prediction. Stop the server with `Ctrl-C` when you're happy.

---

## 5. Run it as a systemd service

Create `/etc/systemd/system/football-api.service` (as root):

```bash
sudo nano /etc/systemd/system/football-api.service
```

```ini
[Unit]
Description=Football Prediction API
Documentation=http://127.0.0.1:8000/docs
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=football
Group=football
WorkingDirectory=/srv/football-model
Environment=PYTHONUNBUFFERED=1
Environment=PATH=/srv/football-model/venv/bin
# soft (default) averages xgboost + random_forest + gradient_boosting where all three
# exist; 'off' loads one model per market and saves ~200 MB RAM. See section 9.
Environment=ENSEMBLE_MODE=soft
Environment=PREFERRED_MODEL_TYPE=xgboost
# --workers 2 doubles memory usage (each worker loads every model).
# Start with 1; raise it only if you have RAM to spare.
ExecStart=/srv/football-model/venv/bin/python -m uvicorn api.app:app \
    --host 127.0.0.1 --port 8000 \
    --workers 1 \
    --proxy-headers --forwarded-allow-ips=127.0.0.1
Restart=always
RestartSec=5
TimeoutStopSec=20

# Light hardening. The app only reads its files, so the filesystem can stay read-only.
NoNewPrivileges=yes
PrivateTmp=yes
ProtectSystem=full
ProtectHome=yes
ReadWritePaths=/srv/football-model

[Install]
WantedBy=multi-user.target
```

Enable and start it:

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now football-api
systemctl status football-api --no-pager
journalctl -u football-api -n 30 --no-pager
```

Logs live in the journal, so there is no log file to rotate. Watch a live tail with
`journalctl -u football-api -f`.

Useful commands:

```bash
sudo systemctl restart football-api     # after a code or model change
sudo systemctl stop football-api
sudo systemctl is-enabled football-api
```

---

## 6. nginx reverse proxy

Create `/etc/nginx/sites-available/football-api`:

```nginx
server {
    listen 80;
    listen [::]:80;
    server_name your-domain.com www.your-domain.com;

    # Predictions are small JSON bodies; 2 MB is plenty of headroom.
    client_max_body_size 2m;

    gzip on;
    gzip_comp_level 5;
    gzip_min_length 512;
    gzip_proxied any;
    gzip_types text/plain text/css application/javascript application/json image/svg+xml;

    # Optional: long-lived static assets (CSS/JS are versionless, so keep it short)
    location ~* ^/ui/.*\.(css|js)$ {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        expires 1h;
        add_header Cache-Control "public";
    }

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_http_version 1.1;
        proxy_set_header Host              $host;
        proxy_set_header X-Real-IP         $remote_addr;
        proxy_set_header X-Forwarded-For   $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;

        # First request per league can take a few seconds while features load.
        proxy_connect_timeout 10s;
        proxy_read_timeout   120s;
        proxy_send_timeout   120s;
    }
}
```

Enable it and reload:

```bash
sudo ln -s /etc/nginx/sites-available/football-api /etc/nginx/sites-enabled/
sudo rm -f /etc/nginx/sites-enabled/default
sudo nginx -t
sudo systemctl reload nginx
```

Visit `http://YOUR_SERVER_IP/` (or your domain) — you should land on the test console at
`/ui/`. The API and Swagger stay at `/api/*` and `/docs`.

---

## 7. Domain and HTTPS

Point an `A` record for your domain at the server IP, wait for DNS to propagate, then:

```bash
sudo apt install -y certbot python3-certbot-nginx
sudo certbot --nginx -d your-domain.com -d www.your-domain.com
```

Certbot rewrites the nginx block for 443, adds the HTTP→HTTPS redirect, and installs a
renewal timer. Verify:

```bash
sudo certbot renew --dry-run
curl -I https://your-domain.com/ui/
```

Once HTTPS works, tighten the API's CORS in `api/app.py` — `allow_origins=["*"]` is fine
for local testing but should name your origin in production:

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://your-domain.com"],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)
```

Because the console ships from the same host, CORS is not needed for the normal setup —
only for the "host the UI somewhere else" case described in section 10.

---

## 8. Verify the whole stack

```bash
# service
systemctl is-active football-api

# app responds locally
curl -s http://127.0.0.1:8000/api/info

# nginx path works over TLS
curl -sI https://your-domain.com/ui/ | head -1
curl -s  https://your-domain.com/api/health | head -c 160; echo

# a real prediction through the proxy
curl -s -X POST https://your-domain.com/api/predict \
  -H 'Content-Type: application/json' \
  -d '{"league":"epl","home_team":"Liverpool","away_team":"Everton"}' | head -c 300; echo
```

Then open `https://your-domain.com/ui/`, run a prediction, and check the **Activity**
tab — the request log should show each call with status `200`.

---

## 9. Day-2 operations

**Deploy a code change**

```bash
cd /srv/football-model
git pull
./venv/bin/pip install -r requirements.txt       # only if dependencies changed
sudo systemctl restart football-api
journalctl -u football-api -n 20 --no-pager
```

**Deploy new models**

Retraining writes into `models/`, so sync and restart:

```bash
# from your machine
rsync -avz models/ football@YOUR_SERVER_IP:/srv/football-model/models/
ssh football@YOUR_SERVER_IP 'sudo systemctl restart football-api'
```

**Switch how models are combined**

`ENSEMBLE_MODE` decides this when models are loaded — no retraining and no data change:

| Value | Behaviour |
|-------|-----------|
| `soft` (default) | Average the class probabilities of every model trained for a market |
| `hard` | Majority-vote the labels of every model trained for a market |
| `off` | Load a single model per market — `PREFERRED_MODEL_TYPE`, default `xgboost` |

```bash
sudo systemctl edit football-api      # add: Environment=ENSEMBLE_MODE=off
sudo systemctl restart football-api
journalctl -u football-api -n 5 --no-pager
# expect: Loaded 140 markets across 10 leagues (30 ensembles, mode=soft)
```

Which markets get combined is decided by what is on disk: a market with a single `.pkl`
cannot be combined, and only `match_result`, `over_under_25` and `btts` have all three
algorithms. `/api/leagues/{league}/models` and the console's **Models** tab list the
members per market.

> **If a market reports no accuracy**, its members were trained on different features or
> different splits, so there is no single holdout to score them on. Predictions still work
> (the members do vote), but the reported accuracy stays blank.
>
> `retrain_ensembles.py` fixes exactly this: it retrains every member of every ensemble
> market from `data/cached_features`, overwriting the `.pkl` files in place, then prints
> the holdout size of each member before and after. Correctly refreshed members all share
> one holdout size, and the market is marked `OK` instead of `MISALIGNED`.
>
> ```bash
> ./venv/bin/python retrain_ensembles.py --dry-run         # list what would be retrained
> ./venv/bin/python retrain_ensembles.py                   # every league, ~15 min
> ./venv/bin/python retrain_ensembles.py --leagues epl serieb
> sudo systemctl restart football-api
> ```
>
> Run it in `tmux` — it is CPU-bound, and 90 models take roughly 15 minutes on a small
> VPS. Retraining overwrites the members in place, so the service must be restarted to
> pick them up.
>
> Note that `train_from_cache.py` **skips any model file that already exists**, so simply
> re-running it will not refresh a stale member — use `retrain_ensembles.py` for that.
>
> A backup of the current ensemble members is worth taking first:
>
> ```bash
> mkdir -p backups
> tar czf backups/ensemble-members-$(date +%F).tar.gz \
>     models/*/match_result_*.pkl models/*/over_under_25_*.pkl models/*/btts_*.pkl
> ```

**Retrain on the server** (heavy — use `tmux`, keep a snapshot first):

```bash
cp -r models models.bak.$(date +%F)         # keep a rollback copy
./venv/bin/python train_from_cache.py       # fast path: retrains from cached features
./venv/bin/python train_all.py              # full path: reloads CSVs, then trains
sudo systemctl restart football-api
```

**Backups** — the irreplaceable parts are the trained models and the feature cache:

```bash
tar czf /srv/backup/football-$(date +%F).tar.gz -C /srv/football-model models data/cached_features
```

Ship that to object storage or another host with `rclone`/`scp`, and check a restore at
least once. CSV folders can always be re-downloaded by `setup.py`.

**Monitoring** — for a single box, `journalctl` plus a simple uptime check is usually
enough:

```bash
watch -n 5 'systemctl --no-pager status football-api | head -5; ps -o rss= -p $(systemctl show -p MainPID --value football-api) | awk "{printf \"RSS %.0f MB\\n\", \$1/1024}"'
```

Point an external uptime monitor at `https://your-domain.com/api/health`.

---

## 10. Serving the UI from somewhere else

The console is three static files (`frontend/index.html`, `styles.css`, `app.js`) with no
build step, so it can live on GitHub Pages, Netlify, S3, or any static host. Set the
**API base** field in the console header to your API origin, for example
`https://your-domain.com`, and it will call that host instead of the page's origin. The
value is remembered in `localStorage`.

Remember to list that static host in the API's `allow_origins` (section 7).

---

## 11. Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| Console shows **API unreachable** | Service down, or nginx not proxying | `systemctl status football-api`, `journalctl -u football-api -n 50`, `sudo nginx -t` |
| `502 Bad Gateway` in the browser | uvicorn not listening on 127.0.0.1:8000 | Compare `ExecStart` port with `proxy_pass`; reload systemd and nginx |
| `404` on `/api/leagues/…/teams` | League data missing | Copy `data/<league>/*.csv` or `data/cached_features/<league>_features.pkl` |
| Prediction takes ~10 s the first time | Features are being loaded/engineered for that league | Expected once per league per worker; later calls are fast |
| Service restarts in a loop | Bad model file, or `WorkingDirectory` wrong | `journalctl -u football-api -n 80 --no-pager` — the load error names the `.pkl` |
| OOM kill (`status=9/KILL`) | Too many workers or multiple leagues loaded | Use `--workers 1`; add RAM or swap |
| Memory jumped ~200 MB after an upgrade | `ENSEMBLE_MODE=soft` also loads the large Random Forest members | Set `ENSEMBLE_MODE=off`, or add RAM |
| Ensemble market shows accuracy `—` | Its members came from different features or splits | `./venv/bin/python retrain_ensembles.py --leagues <league>` (section 9) |
| UI loads but tables stay empty | Mixed content: page on HTTPS calling `http://…` API base | Clear the API base field (empty = same origin), or use `https://` |
| `413 Request Entity Too Large` | nginx body limit | Raise `client_max_body_size` |
| Slow/blocked training on 1 vCPU | CPU saturation | Train locally and sync `models/` (section 3B), or use `tmux` and be patient |

**Where the logs are**

```bash
journalctl -u football-api -f          # app logs
sudo tail -f /var/log/nginx/error.log  # proxy logs
```

---

## 12. Optional: Docker instead of systemd + venv

`Dockerfile`:

```dockerfile
FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8000
CMD ["python", "-m", "uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000", \
     "--proxy-headers", "--forwarded-allow-ips=*"]
```

```bash
docker build -t football-model .
docker run -d --name football-api --restart unless-stopped \
  -p 127.0.0.1:8000:8000 \
  -v "$PWD/models:/app/models" \
  -v "$PWD/data:/app/data" \
  football-model
```

Keep nginx in front either way; point `proxy_pass` at the same `127.0.0.1:8000`.

---

## 13. Production checklist

- [ ] Service runs as the unprivileged `football` user, not root
- [ ] `ufw` allows only 22, 80, 443
- [ ] SSH key auth only; password login disabled; fail2ban active
- [ ] HTTPS with auto-renewal verified (`certbot renew --dry-run`)
- [ ] `allow_origins` narrowed in `api/app.py`; `allow_credentials=False`
- [ ] Models and feature cache backed up somewhere off the box
- [ ] `systemctl enable` done so the API survives a reboot (`sudo reboot` and re-check `/api/health`)
- [ ] RAM headroom confirmed with the RSS command in section 9
- [ ] DNS record documented, and a calendar note for cert expiry (auto-renew should cover it)

---

## 14. Quick reference — fresh Ubuntu 24.04, start to finish

```bash
# as root
apt update && apt upgrade -y
apt install -y python3 python3-venv python3-pip git nginx ufw
adduser --disabled-password --gecos "" football
mkdir -p /srv/football-model && chown -R football:football /srv/football-model
ufw allow OpenSSH && ufw allow 'Nginx Full' && ufw --force enable

su - football
cd /srv/football-model
git clone https://github.com/yourusername/FootballModel.git .
python3 -m venv venv
./venv/bin/pip install -r requirements.txt
# then either: ./venv/bin/python setup.py
# or rsync models/ + data/cached_features/ from your machine

./venv/bin/python -m uvicorn api.app:app --host 127.0.0.1 --port 8000   # smoke test, Ctrl-C

# as root: create the unit from section 5 and the nginx site from section 6, then
systemctl daemon-reload && systemctl enable --now football-api
ln -s /etc/nginx/sites-available/football-api /etc/nginx/sites-enabled/
rm -f /etc/nginx/sites-enabled/default && nginx -t && systemctl reload nginx

# then HTTPS
apt install -y certbot python3-certbot-nginx
certbot --nginx -d your-domain.com
```

Open `https://your-domain.com/ui/` and run a prediction.
