/* ==========================================================================
   Football Model — Test Console
   Vanilla JS single-file app. No build step, no dependencies, no database.
   Talks to the FastAPI backend (same origin by default, or any base URL).
   ========================================================================== */

(() => {
  'use strict';

  /* ------------------------------ constants ------------------------------ */

  const LS = {
    base: 'fm.apiBase',
    theme: 'fm.theme',
    league: 'fm.league',
    markets: 'fm.markets',
    odds: 'fm.odds',
    queue: 'fm.queue',
    history: 'fm.history',
  };

  const MAX_HISTORY = 40;
  const MAX_LOG = 200;

  // Human labels for model output classes that are bare codes.
  // Match result maps A->0, D->1, H->2; binary markets use 0/1
  // (see src/feature_engineering.py: create_market_targets).
  const LABEL_MAP = {
    H: 'Home win', D: 'Draw', A: 'Away win',
    '1': 'Home win', X: 'Draw', '2': 'Away win',
    HOME: 'Home win', DRAW: 'Draw', AWAY: 'Away win',
  };

  const CLASS_LABELS = {
    match_result: { 0: 'Away win', 1: 'Draw', 2: 'Home win' },
    ht_result: { 0: 'Away lead at HT', 1: 'Draw at HT', 2: 'Home lead at HT' },
    over_under_15: { 0: 'Under 1.5', 1: 'Over 1.5' },
    over_under_25: { 0: 'Under 2.5', 1: 'Over 2.5' },
    over_under_35: { 0: 'Under 3.5', 1: 'Over 3.5' },
    btts: { 0: 'No', 1: 'Yes' },
    double_chance_1x: { 0: 'No — away wins', 1: 'Yes — home or draw' },
    double_chance_12: { 0: 'No — draw', 1: 'Yes — home or away' },
    double_chance_x2: { 0: 'No — home wins', 1: 'Yes — draw or away' },
    ht_double_chance_1x: { 0: 'No — away leads', 1: 'Yes — home or draw at HT' },
    home_clean_sheet: { 0: 'No', 1: 'Yes' },
    away_clean_sheet: { 0: 'No', 1: 'Yes' },
    home_win_to_nil: { 0: 'No', 1: 'Yes' },
    away_win_to_nil: { 0: 'No', 1: 'Yes' },
    asian_handicap: { 0: 'Away covers', 1: 'Home covers' },
  };

  const state = {
    apiBase: localStorage.getItem(LS.base) || '',
    leagues: [],
    markets: {},          // key -> {name, type, risk, description}
    teams: {},            // league -> [team]
    models: {},           // league -> [modelInfo]
    trained: {},          // league -> Set(market)
    selected: new Set(),  // selected market keys
    customMarkets: false,
    log: [],
    queue: loadQueue(),
    history: loadHistory(),
    lastResponse: null,
    busy: false,
  };

  /* ------------------------------ tiny helpers ------------------------------ */

  const $ = (sel, root = document) => root.querySelector(sel);

  function h(tag, props, ...kids) {
    const el = document.createElement(tag);
    for (const [k, v] of Object.entries(props || {})) {
      if (v == null || v === false) continue;
      if (k === 'class') el.className = v;
      else if (k === 'text') el.textContent = v;
      else if (k === 'html') el.innerHTML = v; // trusted, local SVG only
      else if (k === 'style' && typeof v === 'object') Object.assign(el.style, v);
      else if (k === 'dataset') Object.assign(el.dataset, v);
      else if (k.startsWith('on') && typeof v === 'function') el.addEventListener(k.slice(2), v);
      else if (v === true) el.setAttribute(k, '');
      else el.setAttribute(k, v);
    }
    for (const kid of kids.flat(Infinity)) {
      if (kid == null || kid === false) continue;
      el.append(kid instanceof Node ? kid : document.createTextNode(String(kid)));
    }
    return el;
  }

  const prefixed = (p) => p.replace(/\/+$/, '');

  function pretty(key) {
    return String(key)
      .replace(/_/g, ' ')
      .replace(/\b\w/g, (c) => c.toUpperCase());
  }

  function classLabel(raw, market) {
    const value = String(raw).trim();
    const specific = CLASS_LABELS[market];
    if (specific && specific[value] != null) return specific[value];
    if (LABEL_MAP[value.toUpperCase()]) return LABEL_MAP[value.toUpperCase()];
    if (/^-?\d+(\.\d+)?$/.test(value)) return `class ${value}`;
    return pretty(value);
  }

  const pct = (v, digits = 1) =>
    v == null || Number.isNaN(v) ? '—' : `${(v * 100).toFixed(digits)}%`;

  const num = (v, digits = 2) =>
    v == null || Number.isNaN(v) ? '—' : Number(v).toFixed(digits);

  function debounce(fn, ms = 250) {
    let t;
    return (...a) => { clearTimeout(t); t = setTimeout(() => fn(...a), ms); };
  }

  function toast(title, message = '', kind = 'ok', ms = 4200) {
    const node = h('div', { class: `toast toast--${kind}` },
      h('strong', { text: title }),
      message ? h('span', { text: message }) : null,
    );
    $('#toasts').append(node);
    setTimeout(() => {
      node.style.opacity = '0';
      node.style.transition = 'opacity .25s';
      setTimeout(() => node.remove(), 260);
    }, ms);
  }

  /* ------------------------------ persistence ------------------------------ */

  function loadHistory() {
    try {
      const raw = JSON.parse(localStorage.getItem(LS.history) || '[]');
      return Array.isArray(raw) ? raw : [];
    } catch { return []; }
  }

  function saveHistory() {
    state.history = state.history.slice(0, MAX_HISTORY);
    try { localStorage.setItem(LS.history, JSON.stringify(state.history)); } catch { /* full */ }
  }

  function loadQueue() {
    try {
      const raw = JSON.parse(localStorage.getItem(LS.queue) || '[]');
      if (!Array.isArray(raw)) return [];
      return raw
        .filter((m) => m && m.league && m.home && m.away)
        .map((m, i) => ({
          id: i + 1,
          league: m.league,
          home: m.home,
          away: m.away,
          odds: m.odds || null,
          status: null,
        }));
    } catch { return []; }
  }

  function saveQueue() {
    try {
      const clean = state.queue.map(({ id, league, home, away, odds }) => ({ id, league, home, away, odds }));
      localStorage.setItem(LS.queue, JSON.stringify(clean));
    } catch { /* full */ }
  }

  function saveSettings() {
    try {
      localStorage.setItem(LS.base, state.apiBase);
      localStorage.setItem(LS.league, $('#league').value || '');
      localStorage.setItem(LS.markets, JSON.stringify([...state.selected]));
      localStorage.setItem(LS.odds, JSON.stringify({
        home: $('#oddsHome').value, draw: $('#oddsDraw').value, away: $('#oddsAway').value,
        over: $('#oddsOver').value, under: $('#oddsUnder').value,
        ahHome: $('#oddsAhHome').value, ahAway: $('#oddsAhAway').value,
      }));
    } catch { /* ignore */ }
  }

  function restoreInputs() {
    try {
      const o = JSON.parse(localStorage.getItem(LS.odds) || '{}');
      if (o.home) $('#oddsHome').value = o.home;
      if (o.draw) $('#oddsDraw').value = o.draw;
      if (o.away) $('#oddsAway').value = o.away;
      if (o.over) $('#oddsOver').value = o.over;
      if (o.under) $('#oddsUnder').value = o.under;
      if (o.ahHome) $('#oddsAhHome').value = o.ahHome;
      if (o.ahAway) $('#oddsAhAway').value = o.ahAway;
      const saved = JSON.parse(localStorage.getItem(LS.markets) || 'null');
      if (Array.isArray(saved) && saved.length) { state.selected = new Set(saved); state.customMarkets = true; }
    } catch { /* ignore */ }
  }

  /* ------------------------------ API layer ------------------------------ */

  class ApiError extends Error {
    constructor(message, status, payload) {
      super(message);
      this.status = status;
      this.payload = payload;
    }
  }

  const apiUrl = (path) => `${prefixed(state.apiBase)}${path}`;

  async function api(path, { method = 'GET', body = null, note = '' } = {}) {
    const started = performance.now();
    let res;
    try {
      res = await fetch(apiUrl(path), {
        method,
        headers: body ? { 'Content-Type': 'application/json' } : undefined,
        body: body ? JSON.stringify(body) : undefined,
      });
    } catch (err) {
      const ms = Math.round(performance.now() - started);
      pushLog({ method, path, status: 0, ms, note: note || 'network error' });
      throw new ApiError(`Cannot reach the API at ${apiUrl(path) || path}`, 0, { detail: String(err) });
    }

    const ms = Math.round(performance.now() - started);
    const text = await res.text();
    let data = null;
    try { data = text ? JSON.parse(text) : null; } catch { data = { raw: text }; }

    pushLog({
      method, path, status: res.status, ms,
      note: note || (res.ok ? '' : shortDetail(data)),
    });

    if (!res.ok) throw new ApiError(errorMessage(res.status, data), res.status, data);
    return data;
  }

  function shortDetail(payload) {
    const d = payload && payload.detail;
    if (!d) return '';
    return typeof d === 'string' ? d : JSON.stringify(d).slice(0, 120);
  }

  function errorMessage(status, payload) {
    const d = payload && payload.detail;
    if (typeof d === 'string') return d;
    if (Array.isArray(d)) {
      return d.map((e) => `${(e.loc || []).slice(1).join('.')}: ${e.msg}`).join('; ');
    }
    if (status === 0) return 'Network error — is the API running?';
    return `Request failed with status ${status}`;
  }

  function pushLog(row) {
    state.log.unshift({ time: new Date().toISOString(), ...row });
    state.log = state.log.slice(0, MAX_LOG);
    renderLog();
  }

  /* ------------------------------ status pill ------------------------------ */

  function setPill(kind, text) {
    const pill = $('#healthPill');
    pill.className = `pill pill--${kind}`;
    $('#healthText').textContent = text;
  }

  async function checkHealth(announce = false) {
    setPill('busy', 'Checking…');
    try {
      const data = await api('/api/health', { note: 'health check' });
      const models = data.total_models ?? 0;
      const leagues = Object.keys(data.leagues || {}).length;
      setPill('ok', `${models} models · ${leagues} leagues`);
      $('#apiVersion').textContent = data.status ? `status: ${data.status}` : '';
      $('#healthBlock').textContent = JSON.stringify(data, null, 2);
      if (announce) toast('API healthy', `${models} models across ${leagues} leagues`);
      return data;
    } catch (err) {
      setPill('err', 'API unreachable');
      $('#healthBlock').textContent = String(err.message);
      if (announce) toast('API unreachable', err.message, 'error', 6000);
      return null;
    }
  }

  /* ------------------------------ bootstrap ------------------------------ */

  async function loadLeagues() {
    const leagues = await api('/api/leagues', { note: 'league catalogue' });
    state.leagues = leagues.filter((l) => l && l.key);

    const options = (includeEmpty) => {
      const frag = document.createDocumentFragment();
      if (includeEmpty) frag.append(h('option', { value: '', text: '— select league —' }));
      for (const l of state.leagues) {
        const label = `${l.name}${l.has_models ? ` · ${l.n_models} markets` : ' · no models'}`;
        frag.append(h('option', {
          value: l.key, text: label, disabled: !l.has_models && !includeEmpty,
        }));
      }
      return frag;
    };

    const prev = $('#league').value || localStorage.getItem(LS.league) || '';
    for (const sel of [$('#league'), $('#cmpLeague'), $('#modelLeague')]) {
      const value = sel.value;
      sel.replaceChildren(options(true));
      const target = sel === $('#league') ? prev : value;
      if (target && [...sel.options].some((o) => o.value === target)) sel.value = target;
    }

    if (!$('#league').value) {
      const first = state.leagues.find((l) => l.has_models);
      if (first) $('#league').value = first.key;
    }

    for (const l of state.leagues) state.trained[l.key] = new Set(l.markets || []);

    if (!$('#cmpLeague').value) $('#cmpLeague').value = $('#league').value;
    if (!$('#modelLeague').value) $('#modelLeague').value = $('#league').value;
    return state.leagues;
  }

  async function loadMarkets() {
    const markets = await api('/api/markets', { note: 'market catalogue' });
    state.markets = markets || {};
    renderMarketChips();
  }

  async function loadTeams(league) {
    if (!league) return [];
    if (state.teams[league]) return state.teams[league];
    const teams = await api(`/api/leagues/${encodeURIComponent(league)}/teams`, { note: `teams · ${league}` });
    state.teams[league] = teams || [];
    return state.teams[league];
  }

  function teamDatalist(teams) {
    const dl = $('#teamList');
    dl.replaceChildren(...(teams || []).map((t) => h('option', { value: t })));
  }

  function leagueMeta(league) {
    return state.leagues.find((l) => l.key === league) || null;
  }

  async function syncPredictLeague() {
    const league = $('#league').value;
    const meta = leagueMeta(league);
    if (!meta) { $('#leagueHint').textContent = ''; return; }

    const parts = [meta.country, meta.tier === 1 ? 'Tier 1' : `Tier ${meta.tier}`, `${meta.n_models} markets`];
    $('#leagueHint').textContent = parts.filter(Boolean).join(' · ');
    $('#leagueHint').className = 'hint';

    if (!state.customMarkets) {
      state.selected = new Set(meta.markets || []);
      renderMarketChips();
    }
    renderMarketChips();

    try {
      const teams = await loadTeams(league);
      teamDatalist(teams);
      const known = new Set(teams);
      for (const input of [$('#homeTeam'), $('#awayTeam')]) {
        input.classList.toggle('is-invalid', !!input.value && !known.has(input.value));
      }
    } catch (err) {
      teamDatalist([]);
      $('#leagueHint').textContent = err.message;
      $('#leagueHint').className = 'hint is-error';
    }
    saveSettings();
  }

  /* ------------------------------ market chips ------------------------------ */

  function renderMarketChips() {
    const league = $('#league').value;
    const trained = state.trained[league] || new Set();
    const keys = Object.keys(state.markets);

    if (!keys.length) {
      $('#marketChips').replaceChildren(h('span', { class: 'muted small', text: 'Loading markets…' }));
      return;
    }

    $('#marketChips').replaceChildren(...keys.map((key) => {
      const info = state.markets[key] || {};
      const on = state.selected.has(key);
      const isTrained = trained.has(key);
      const chip = h('label', {
        class: `chip ${on ? 'is-on' : ''} ${isTrained ? '' : 'is-untrained'}`,
        title: `${info.description || pretty(key)}${isTrained ? '' : ' — no model trained for this league'}`,
      },
        h('input', {
          type: 'checkbox', value: key, checked: on,
          onchange: (e) => {
            state.customMarkets = true;
            if (e.target.checked) state.selected.add(key); else state.selected.delete(key);
            e.target.closest('.chip').classList.toggle('is-on', e.target.checked);
            saveSettings();
          },
        }),
        h('span', { text: info.name || pretty(key) }),
        info.risk ? h('span', { class: `badge tag tag--${info.risk}`, text: info.risk }) : null,
      );
      return chip;
    }));
  }

  /* ------------------------------ match queue ------------------------------ */

  function readNumber(input) {
    const raw = input.value.trim();
    if (!raw) return null;
    const v = Number(raw);
    return Number.isFinite(v) && v > 1 ? v : null;
  }

  function collectOdds() {
    const odds = {
      home: readNumber($('#oddsHome')),
      draw: readNumber($('#oddsDraw')),
      away: readNumber($('#oddsAway')),
      over_25: readNumber($('#oddsOver')),
      under_25: readNumber($('#oddsUnder')),
      ah_home: readNumber($('#oddsAhHome')),
      ah_away: readNumber($('#oddsAhAway')),
    };
    const cleaned = {};
    for (const [k, v] of Object.entries(odds)) if (v != null) cleaned[k] = v;
    return Object.keys(cleaned).length ? cleaned : null;
  }

  function nextQueueId() {
    return Math.max(0, ...state.queue.map((m) => m.id)) + 1;
  }

  function addMatch(event) {
    if (event) event.preventDefault();
    if (state.busy) return toast('Batch running', 'Wait for the current predictions to finish.', 'warn');

    const league = $('#league').value;
    const home = $('#homeTeam').value.trim();
    const away = $('#awayTeam').value.trim();

    if (!league) return toast('Pick a league', 'Select a league before adding matches.', 'warn');
    if (!home || !away) return toast('Missing team', 'Both home and away teams are required.', 'warn');
    if (home === away) return toast('Same team twice', 'Home and away teams must be different.', 'warn');

    const known = state.teams[league] || [];
    if (known.length) {
      const missing = [home, away].filter((t) => !known.includes(t));
      if (missing.length) {
        const guess = missing.map((m) => {
          const near = known.filter((t) => t.toLowerCase().includes(m.toLowerCase())).slice(0, 3);
          return `${m}${near.length ? ` (did you mean ${near.join(', ')}?)` : ''}`;
        }).join(' · ');
        return toast('Unknown team name', guess, 'warn', 7000);
      }
    }

    if (state.queue.some((m) => m.league === league && m.home === home && m.away === away)) {
      return toast('Already queued', `${home} vs ${away} is in the queue.`, 'warn');
    }

    state.queue.push({
      id: nextQueueId(),
      league,
      home,
      away,
      odds: collectOdds(),
      status: null,
    });
    saveQueue();
    renderQueue();

    // Reset the entry form for the next fixture
    $('#homeTeam').value = '';
    $('#awayTeam').value = '';
    for (const sel of ['#oddsHome', '#oddsDraw', '#oddsAway', '#oddsOver', '#oddsUnder', '#oddsAhHome', '#oddsAhAway']) {
      $(sel).value = '';
    }
    $('#queueOdds').hidden = true;
    $('#queueOddsToggle').textContent = '+ odds for this match (optional)';
    saveSettings();
    $('#homeTeam').focus();
  }

  function removeMatch(id) {
    if (state.busy) return;
    state.queue = state.queue.filter((m) => m.id !== id);
    saveQueue();
    renderQueue();
  }

  function editMatch(id) {
    if (state.busy) return;
    const item = state.queue.find((m) => m.id === id);
    if (!item) return;
    if ($('#league').value !== item.league) {
      $('#league').value = item.league;
      syncPredictLeague();
    }
    $('#homeTeam').value = item.home;
    $('#awayTeam').value = item.away;
    setOddsInputs(item.odds);
    removeMatch(id);
    $('#homeTeam').focus();
  }

  function setOddsInputs(odds) {
    const o = odds || {};
    const set = (sel, v) => { $(sel).value = v ?? ''; };
    set('#oddsHome', o.home); set('#oddsDraw', o.draw); set('#oddsAway', o.away);
    set('#oddsOver', o.over_25); set('#oddsUnder', o.under_25);
    set('#oddsAhHome', o.ah_home); set('#oddsAhAway', o.ah_away);
    const hasOdds = collectOdds() != null;
    $('#queueOdds').hidden = !hasOdds;
    $('#queueOddsToggle').textContent = hasOdds ? 'hide odds' : '+ odds for this match (optional)';
  }

  function toggleQueueOdds() {
    const box = $('#queueOdds');
    box.hidden = !box.hidden;
    $('#queueOddsToggle').textContent = box.hidden ? '+ odds for this match (optional)' : 'hide odds';
  }

  function clearQueue() {
    if (state.busy) return;
    if (!state.queue.length) return;
    state.queue = [];
    saveQueue();
    renderQueue();
    toast('Queue cleared');
  }

  function renderQueue() {
    const bar = $('#queueBar');
    const list = $('#queueList');
    if (!state.queue.length) {
      bar.hidden = true;
      list.replaceChildren();
      return;
    }

    bar.hidden = false;
    const withOdds = state.queue.filter((m) => m.odds).length;
    $('#queueCount').textContent = `${state.queue.length} match${state.queue.length === 1 ? '' : 'es'}`;
    $('#queueSummaryText').textContent = withOdds
      ? `${withOdds} with odds · selected markets apply to every match`
      : 'no odds supplied · selected markets apply to every match';

    list.replaceChildren(...state.queue.map((m, i) => {
      const meta = leagueMeta(m.league);
      const statusTag =
        m.status === 'queued' ? h('span', { class: 'tag tag--queued', text: 'queued…' }) :
        m.status === 'ok' ? h('span', { class: 'tag tag--low', text: 'done' }) :
        m.status === 'failed' ? h('span', { class: 'tag tag--high', text: 'failed' }) : null;
      return h('div', { class: `queue-item ${m.status === 'queued' ? 'is-queued' : ''}` },
        h('span', { class: 'queue-idx', text: String(i + 1) }),
        h('div', { class: 'queue-fixture' },
          h('strong', { text: `${m.home} vs ${m.away}` }),
          h('span', { class: 'muted small', text: meta ? meta.name : m.league }),
        ),
        h('div', { class: 'queue-tags' },
          m.odds ? h('span', { class: 'tag', text: 'odds' }) : null,
          m.league !== $('#league').value ? h('span', { class: 'tag tag--medium', text: 'other league' }) : null,
          statusTag,
        ),
        h('div', { class: 'mini-actions' },
          h('button', { class: 'btn btn--ghost btn--small', type: 'button', text: 'Edit', disabled: state.busy, onclick: () => editMatch(m.id) }),
          h('button', { class: 'btn btn--ghost btn--small', type: 'button', text: '✕', title: 'Remove from queue', disabled: state.busy, onclick: () => removeMatch(m.id) }),
        ),
      );
    }));
  }

  /* ------------------------------ batch predict flow ------------------------------ */

  function groupQueueByLeague() {
    const groups = new Map();
    for (const item of state.queue) {
      if (!groups.has(item.league)) groups.set(item.league, []);
      groups.get(item.league).push(item);
    }
    return groups;
  }

  async function predictAll() {
    if (state.busy) return;
    if (!state.queue.length) return toast('Queue is empty', 'Add at least one fixture first.', 'warn');

    const n = state.queue.length;
    state.busy = true;
    $('#predictAll').disabled = true;
    $('#predictState').textContent = `Running ${n} match${n === 1 ? '' : 'es'}…`;
    $('#predictState').className = 'hint muted small';
    setPill('busy', 'Predicting…');

    for (const m of state.queue) m.status = 'queued';
    renderQueue();
    $('#predictResults').replaceChildren(
      h('div', { class: 'card batch-status' },
        h('strong', { text: `Processing ${n} match${n === 1 ? '' : 'es'}…` }),
        h('p', { class: 'muted small', text: 'Results appear below as each league group finishes.' }),
        ...Array.from({ length: Math.min(3, n) }, () => h('div', { class: 'skeleton' })),
      ),
    );

    const groups = groupQueueByLeague();
    let okTotal = 0;
    let failTotal = 0;

    try {
      for (const [league, items] of groups) {
        // The market selection belongs to the currently selected league; for
        // other leagues send all markets so we don't filter out everything.
        const sameLeague = league === $('#league').value;
        const body = {
          league,
          matches: items.map((m) => ({ home_team: m.home, away_team: m.away, odds: m.odds })),
          markets: sameLeague && state.selected.size ? [...state.selected] : null,
        };
        try {
          const data = await api('/api/predict/batch', { method: 'POST', body, note: `batch ×${items.length} · ${league}` });
          state.lastResponse = data;
          renderLastResponse();
          renderBatchGroup(league, items, data);

          (data.results || []).forEach((r, i) => {
            const item = items[i];
            if (!item) return;
            item.status = r.status === 'ok' ? 'ok' : 'failed';
            if (r.status === 'ok') {
              okTotal += 1;
              rememberRun({ league, home: item.home, away: item.away, odds: item.odds, response: r.prediction });
            } else {
              failTotal += 1;
            }
          });
        } catch (err) {
          // The whole batch call failed (network or league-level) — fail its items.
          for (const item of items) {
            item.status = 'failed';
            failTotal += 1;
          }
          $('#predictResults').append(
            h('div', { class: 'card' },
              h('div', { class: 'result-head' },
                h('div', null,
                  h('h2', { text: leagueMeta(league)?.name || league }),
                  h('p', { class: 'muted small', text: `${items.length} match${items.length === 1 ? '' : 'es'} not predicted` })),
              ),
              h('div', { class: 'error-box', text: err.message }),
            ),
          );
        }
        renderQueue();
      }

      $('#predictState').textContent = `Done — ${okTotal} predicted, ${failTotal} failed at ${new Date().toLocaleTimeString()}`;
      $('#predictState').className = failTotal ? 'hint is-error' : 'hint is-ok';
      if (failTotal) toast('Some matches failed', `${failTotal} of ${n} failed — see the result cards.`, 'warn', 7000);
      else toast('Batch complete', `${okTotal} match${okTotal === 1 ? '' : 'es'} predicted.`);
    } finally {
      state.busy = false;
      $('#predictAll').disabled = false;
      $('#predictResults .batch-status')?.remove();
      checkHealth();
    }
  }

  function renderBatchGroup(league, items, data) {
    const results = data.results || [];
    const blocks = results.map((r, i) => {
      const item = items[i] || { home: '?', away: '?', odds: null };
      const head = h('div', { class: 'result-head' },
        h('div', null,
          h('div', { class: 'fixture' },
            h('h2', { text: item.home }),
            h('span', { class: 'vs', text: 'vs' }),
            h('h2', { text: item.away }),
          ),
          h('p', { class: 'muted small', text: `${data.league || league} · ${new Date().toLocaleString()} · ${r.elapsed_ms} ms` }),
        ),
        h('div', { class: 'meta-chips' },
          h('span', { class: `tag ${r.status === 'ok' ? 'tag--low' : 'tag--high'}`, text: r.status === 'ok' ? 'ok' : 'failed' }),
          item.odds ? h('span', { class: 'tag', text: 'odds supplied' }) : h('span', { class: 'tag', text: 'no odds (estimated)' }),
        ),
      );

      if (r.status !== 'ok' || !r.prediction) {
        return h('div', { class: 'card' }, head, h('div', { class: 'error-box', text: r.error || 'Prediction failed.' }));
      }

      const pred = r.prediction;
      const picks = pred.predictions || {};
      const markets = Object.keys(picks).sort(marketOrder);
      const odds = item.odds;

      if (!markets.length) {
        return h('div', { class: 'card' }, head, h('div', { class: 'empty' },
          h('h3', { text: 'No markets returned' }),
          h('p', { text: 'Try selecting different markets or retrain this league.' })));
      }

      const cards = markets.map((key) => renderMarketCard(key, picks[key], pred, odds));
      const summary = renderSummary(markets, picks, pred, odds);
      return h('div', { class: 'card' }, head, summary, h('div', { class: 'market-grid' }, cards));
    });

    $('#predictResults').append(...blocks);
  }

  function marketOrder(a, b) {
    const ka = Object.keys(state.markets);
    const ia = ka.indexOf(a); const ib = ka.indexOf(b);
    if (ia === -1 && ib === -1) return a.localeCompare(b);
    if (ia === -1) return 1;
    if (ib === -1) return -1;
    return ia - ib;
  }

  function fairOdds(p) {
    return p && p > 0.0001 ? (1 / p) : null;
  }

  // Edge is only meaningful where we have matching bookmaker odds:
  // match_result, over/under 2.5 and the asian handicap sides.
  function edgeFor(market, label, p, odds, data) {
    if (!odds || p == null) return null;
    const up = String(label).trim().toUpperCase();
    const isHome = up === 'HOME WIN' || up === `${data.home_team}`.toUpperCase();
    const isAway = up === 'AWAY WIN' || up === `${data.away_team}`.toUpperCase();

    let book = null;
    if (market === 'match_result') {
      if (isHome) book = odds.home;
      else if (isAway) book = odds.away;
      else if (up === 'DRAW') book = odds.draw;
    } else if (market === 'over_under_25') {
      if (up.startsWith('OVER')) book = odds.over_25;
      else if (up.startsWith('UNDER')) book = odds.under_25;
    } else if (market === 'asian_handicap') {
      if (up.startsWith('HOME')) book = odds.ah_home;
      else if (up.startsWith('AWAY')) book = odds.ah_away;
    }
    if (!book || !(book > 1)) return null;
    return { book, edge: p * book - 1 };
  }

  function renderMarketCard(key, pick, data, odds) {
    const info = state.markets[key] || {};
    const title = info.name || pretty(key);
    const isClf = (pick.market_type || info.type) === 'classification' || !!pick.probabilities;

    const top = h('div', { class: 'market-top' },
      h('div', null,
        h('div', { class: 'market-title', text: title }),
        h('div', { class: 'market-sub', text: pick.model_type ? `model: ${pretty(pick.model_type)}${pick.members?.length ? ` (${pick.members.length} learners)` : ''}` : (info.description || '') }),
      ),
      info.risk ? h('span', { class: `tag tag--${info.risk}`, text: info.risk }) : null,
    );

    if (pick.error) {
      return h('div', { class: 'market' }, top, h('div', { class: 'error-box', text: pick.error }));
    }

    let main;
    let bars = null;
    let extra = null;

    if (isClf && pick.probabilities) {
      const sorted = Object.entries(pick.probabilities).sort((a, b) => b[1] - a[1]);
      const pickLabel = String(pick.prediction);
      const pickProb = pick.probabilities[pickLabel] ?? (pick.confidence ?? sorted[0][1]);

      main = h('div', { class: 'market-pick' },
        h('span', { class: 'pick-value', text: classLabel(pick.prediction, key) }),
        h('span', { class: 'conf', text: pct(pick.confidence ?? pickProb) }),
      );

      bars = h('div', { class: 'bars' }, sorted.map(([label, p]) => {
        const isPick = label === pickLabel;
        return h('div', { class: `bar-row ${isPick ? 'is-pick' : ''}` },
          h('span', { class: 'bar-label', title: classLabel(label, key), text: classLabel(label, key) }),
          h('div', { class: 'bar-track' }, h('div', { class: 'bar-fill', style: { width: `${Math.max(2, p * 100)}%` } })),
          h('span', { class: 'bar-val', text: `${pct(p)} · ${fairOdds(p) ? fairOdds(p).toFixed(2) : '—'}` }),
        );
      }));

      const edge = edgeFor(key, classLabel(pickLabel, key), pickProb, odds, data);
      extra = h('div', { class: 'kv' },
        h('div', null, h('span', { class: 'k', text: 'Fair odds' }), h('span', { class: 'v', text: pickProb ? fairOdds(pickProb).toFixed(2) : '—' })),
        h('div', null, h('span', { class: 'k', text: 'Book odds' }), h('span', { class: 'v', text: edge ? edge.book.toFixed(2) : '—' })),
        h('div', null,
          h('span', { class: 'k', text: 'Edge' }),
          h('span', { class: `v ${edge ? (edge.edge >= 0 ? 'pos' : 'neg') : ''}`, text: edge ? `${edge.edge >= 0 ? '+' : ''}${(edge.edge * 100).toFixed(1)}%` : '—' }),
        ),
        pick.members?.length ? h('div', null,
          h('span', { class: 'k', text: 'Members' }),
          h('span', { class: 'v', text: pick.members.map(pretty).join(' + ') }),
        ) : null,
      );
    } else {
      const value = pick.predicted_value ?? pick.prediction;
      main = h('div', { class: 'market-pick' },
        h('span', { class: 'pick-value is-num', text: Number.isFinite(Number(value)) ? num(value) : String(value) }),
        pick.confidence != null ? h('span', { class: 'conf', text: pct(pick.confidence) }) : null,
      );
      extra = h('div', { class: 'kv' },
        h('div', null, h('span', { class: 'k', text: 'Type' }), h('span', { class: 'v', text: 'regression' })),
        h('div', null, h('span', { class: 'k', text: 'Model' }), h('span', { class: 'v', text: pretty(pick.model_type || 'n/a') })),
      );
    }

    return h('div', { class: 'market' }, top, main, bars, extra);
  }

  function renderSummary(markets, picks, data, odds) {
    const confidences = markets
      .map((m) => picks[m] && picks[m].confidence)
      .filter((v) => typeof v === 'number');
    const avg = confidences.length ? confidences.reduce((a, b) => a + b, 0) / confidences.length : null;
    const best = markets
      .filter((m) => picks[m] && typeof picks[m].confidence === 'number')
      .sort((a, b) => picks[b].confidence - picks[a].confidence)[0];

    const highConf = markets
      .filter((m) => picks[m] && typeof picks[m].confidence === 'number' && picks[m].confidence >= 0.6).length;

    return h('div', { class: 'card' },
      h('div', { class: 'stat-row' },
        stat('Markets', markets.length),
        stat('Avg confidence', avg != null ? pct(avg) : '—'),
        stat('Markets ≥60%', highConf),
        stat('Top pick', best ? `${state.markets[best]?.name || pretty(best)} · ${pct(picks[best].confidence)}` : '—'),
      ),
      h('p', { class: 'muted small', style: { margin: '12px 0 0' }, text: 'Confidence is the model’s probability for its own pick, not a guarantee. Regression markets output a value instead of a class.' }),
    );
  }

  function stat(label, value) {
    return h('div', { class: 'stat' },
      h('div', { class: 'k', text: label }),
      h('div', { class: 'v', text: String(value) }),
    );
  }

  /* ------------------------------ compare flow ------------------------------ */

  async function runCompare(event) {
    if (event) event.preventDefault();
    const league = $('#cmpLeague').value;
    const home = $('#cmpHome').value.trim();
    const away = $('#cmpAway').value.trim();
    if (!league || !home || !away) return toast('Missing input', 'Pick a league and two teams.', 'warn');
    if (home === away) return toast('Same team twice', 'Home and away teams must be different.', 'warn');

    teamDatalist(state.teams[league] || []);
    const box = $('#compareResults');
    box.replaceChildren(h('div', { class: 'skeleton' }), h('div', { class: 'skeleton' }));
    $('#compareState').textContent = 'Comparing…';

    try {
      const data = await api('/api/compare', {
        method: 'POST',
        body: { league, home_team: home, away_team: away },
        note: `compare ${home} vs ${away}`,
      });
      state.lastResponse = data;
      renderLastResponse();
      renderComparison(data);
      $('#compareState').textContent = `Updated ${new Date().toLocaleTimeString()}`;
      $('#compareState').className = 'hint is-ok';
    } catch (err) {
      box.replaceChildren(h('div', { class: 'error-box', text: err.message }));
      $('#compareState').textContent = err.message;
      $('#compareState').className = 'hint is-error';
      toast('Comparison failed', err.message, 'error', 6000);
    }
  }

  function formString(str) {
    if (!str) return h('span', { class: 'muted small', text: 'no recent matches' });
    return h('div', { class: 'form-string' }, String(str).split('').map((c) => {
      const cls = c === 'W' ? 'w' : c === 'D' ? 'd' : 'l';
      return h('span', { class: cls, text: c });
    }));
  }

  function metricRows(card, rows) {
    const body = rows.map((r) => {
      const hw = r.homeBetter ?? false;
      const aw = r.awayBetter ?? false;
      return h('div', { class: 'cmp-metric' },
        h('span', { class: `l ${hw ? 'win' : ''}`, text: r.home }),
        h('span', { class: 'm', text: r.label }),
        h('span', { class: `r ${aw ? 'win' : ''}`, text: r.away }),
      );
    });
    return h('div', { class: 'market' }, h('div', { class: 'market-title', text: card }), h('div', null, body));
  }

  function renderComparison(d) {
    const home = d.teams?.home || 'Home';
    const away = d.teams?.away || 'Away';
    const f = d.form || {};
    const s = d.streaks || {};
    const att = d.attacking || {};
    const def = d.defensive || {};
    const gt = d.goal_timing || {};
    const lp = d.league_position || {};
    const h2h = d.head_to_head || {};

    const head = h('div', { class: 'card' },
      h('div', { class: 'cmp-head' },
        h('div', { class: 'cmp-team' }, h('div', { class: 'name', text: home }), formString(d.recent_form_string?.home)),
        h('div', { class: 'muted small', text: 'vs' }),
        h('div', { class: 'cmp-team' }, h('div', { class: 'name', text: away }), formString(d.recent_form_string?.away)),
      ),
      h('p', { class: 'muted small', style: { margin: '12px 0 0' }, text: 'Form string is the last 5 matches, oldest → newest. Highlighted values are the stronger side for that metric.' }),
    );

    const grid = h('div', { class: 'cmp-grid' },
      metricRows('Form (last 10)', [
        { label: 'Points/game', home: num(f.home?.overall), away: num(f.away?.overall), ...pair(f.home?.overall, f.away?.overall) },
        { label: 'Win rate', home: pct(f.home?.wins, 0), away: pct(f.away?.wins, 0), ...pair(f.home?.wins, f.away?.wins) },
        { label: 'Goals/game', home: num(f.home?.goals_scored), away: num(f.away?.goals_scored), ...pair(f.home?.goals_scored, f.away?.goals_scored) },
        { label: 'Conceded/game', home: num(f.home?.goals_conceded), away: num(f.away?.goals_conceded), ...pair(f.home?.goals_conceded, f.away?.goals_conceded, false) },
        { label: 'Home-split form', home: num(f.home?.home_form), away: num(f.away?.home_form), ...pair(f.home?.home_form, f.away?.home_form) },
        { label: 'Away-split form', home: num(f.home?.away_form), away: num(f.away?.away_form), ...pair(f.home?.away_form, f.away?.away_form) },
      ]),
      metricRows('Streaks & momentum', [
        { label: 'Current streak', home: signed(s.home?.current), away: signed(s.away?.current), ...pair(s.home?.current, s.away?.current) },
        { label: 'Momentum', home: num(s.home?.momentum), away: num(s.away?.momentum), ...pair(s.home?.momentum, s.away?.momentum) },
        { label: 'Max win streak', home: str(s.home?.max_win_streak), away: str(s.away?.max_win_streak), ...pair(s.home?.max_win_streak, s.away?.max_win_streak) },
        { label: 'Max loss streak', home: str(s.home?.max_loss_streak), away: str(s.away?.max_loss_streak), ...pair(s.home?.max_loss_streak, s.away?.max_loss_streak, false) },
      ]),
      metricRows('Attacking', [
        { label: 'Attack strength', home: num(att.home?.strength), away: num(att.away?.strength), ...pair(att.home?.strength, att.away?.strength) },
        { label: 'Avg goals', home: num(att.home?.avg_goals), away: num(att.away?.avg_goals), ...pair(att.home?.avg_goals, att.away?.avg_goals) },
        { label: 'Avg shots', home: num(att.home?.avg_shots), away: num(att.away?.avg_shots), ...pair(att.home?.avg_shots, att.away?.avg_shots) },
        { label: 'Shots on target', home: num(att.home?.avg_shots_on_target), away: num(att.away?.avg_shots_on_target), ...pair(att.home?.avg_shots_on_target, att.away?.avg_shots_on_target) },
      ]),
      metricRows('Defensive', [
        { label: 'Defense strength', home: num(def.home?.strength), away: num(def.away?.strength), ...pair(def.home?.strength, def.away?.strength) },
        { label: 'Clean sheet prob', home: pct(def.home?.clean_sheet_prob, 0), away: pct(def.away?.clean_sheet_prob, 0), ...pair(def.home?.clean_sheet_prob, def.away?.clean_sheet_prob) },
        { label: 'Corners conceded', home: num(def.home?.avg_corners_conceded), away: num(def.away?.avg_corners_conceded), ...pair(def.home?.avg_corners_conceded, def.away?.avg_corners_conceded, false) },
      ]),
      metricRows('Goal timing', [
        { label: '1st-half goal ratio', home: pct(gt.home?.ht_goals_ratio, 0), away: pct(gt.away?.ht_goals_ratio, 0), ...pair(gt.home?.ht_goals_ratio, gt.away?.ht_goals_ratio) },
        { label: '2nd-half goal ratio', home: pct(gt.home?.second_half_ratio, 0), away: pct(gt.away?.second_half_ratio, 0), ...pair(gt.home?.second_half_ratio, gt.away?.second_half_ratio) },
      ]),
      metricRows('Context', [
        { label: 'League position proxy', home: num(lp.home, 3), away: num(lp.away, 3), ...pair(lp.home, lp.away) },
        { label: 'H2H advantage', home: num(h2h.advantage, 3), away: '—', homeBetter: typeof h2h.advantage === 'number' && h2h.advantage > 0.6, awayBetter: typeof h2h.advantage === 'number' && h2h.advantage < 0.4 },
      ]),
    );

    const note = h('div', { class: 'card' },
      h('div', { class: 'stat-row' },
        stat('H2H verdict', h2h.description || '—'),
        stat('H2H advantage', num(h2h.advantage, 3)),
        stat('Fixture', `${home} vs ${away}`),
      ),
    );

    $('#compareResults').replaceChildren(head, grid, note);
    $('#cmpHome').value = home;
    $('#cmpAway').value = away;
  }

  function pair(a, b, higher = true) {
    if (typeof a !== 'number' || typeof b !== 'number' || a === b) return { homeBetter: false, awayBetter: false };
    return higher ? { homeBetter: a > b, awayBetter: b > a } : { homeBetter: a < b, awayBetter: b < a };
  }
  const signed = (v) => (typeof v === 'number' && v > 0 ? `+${v}` : String(v ?? '—'));
  const str = (v) => (v == null ? '—' : String(v));

  /* ------------------------------ models tab ------------------------------ */

  let modelSort = { key: 'market', dir: 1 };

  async function loadModels() {
    const league = $('#modelLeague').value;
    const tbody = $('#modelsTable tbody');
    if (!league) { tbody.replaceChildren(h('tr', null, h('td', { colspan: 6, class: 'muted center', text: 'Select a league.' }))); return; }

    if (!state.models[league]) {
      tbody.replaceChildren(h('tr', null, h('td', { colspan: 6, class: 'muted center', text: 'Loading…' })));
      try {
        state.models[league] = await api(`/api/leagues/${encodeURIComponent(league)}/models`, { note: `models · ${league}` }) || [];
      } catch (err) {
        tbody.replaceChildren(h('tr', null, h('td', { colspan: 6, class: 'error-box', text: err.message })));
        return;
      }
    }

    const all = state.models[league];
    const filter = $('#modelSearch').value.trim().toLowerCase();
    const rows = all
      .filter((m) => !filter || `${m.market} ${m.model_type} ${m.market_type} ${(m.members || []).join(' ')}`.toLowerCase().includes(filter))
      .sort((a, b) => {
        const { key, dir } = modelSort;
        const av = a[key]; const bv = b[key];
        if (av == null) return 1;
        if (bv == null) return -1;
        if (typeof av === 'number' && typeof bv === 'number') return (av - bv) * dir;
        return String(av).localeCompare(String(bv)) * dir;
      });

    tbody.replaceChildren(...(rows.length ? rows.map((m) => h('tr', null,
      h('td', { text: state.markets[m.market]?.name || pretty(m.market) }),
      h('td', {
        text: m.members?.length ? `${pretty(m.model_type)} · ${m.members.length}` : pretty(m.model_type || '—'),
        title: m.members?.length ? `Member models: ${m.members.map(pretty).join(' + ')}` : null,
      }),
      h('td', { text: m.market_type || '—' }),
      h('td', { class: 'num', text: m.accuracy != null ? pct(m.accuracy) : '—' }),
      h('td', { class: 'num', text: m.mae != null ? num(m.mae, 3) : '—' }),
      h('td', { class: 'num', text: m.n_features ?? '—' }),
    )) : [h('tr', null, h('td', { colspan: 6, class: 'muted center', text: 'No models match this filter.' }))]));

    renderModelSummary(all);
    renderAccuracyChart(all);
  }

  function renderModelSummary(models) {
    const clf = models.filter((m) => m.accuracy != null);
    const reg = models.filter((m) => m.mae != null);
    const best = clf.slice().sort((a, b) => b.accuracy - a.accuracy)[0];
    const avg = clf.length ? clf.reduce((a, m) => a + m.accuracy, 0) / clf.length : null;

    $('#modelSummary').replaceChildren(
      stat('Models loaded', models.length),
      stat('Classification', clf.length),
      stat('Regression', reg.length),
      stat('Avg accuracy', avg != null ? pct(avg) : '—'),
      stat('Best market', best ? `${state.markets[best.market]?.name || pretty(best.market)} · ${pct(best.accuracy)}` : '—'),
    );
  }

  function renderAccuracyChart(models) {
    const rows = models
      .filter((m) => m.accuracy != null)
      .sort((a, b) => b.accuracy - a.accuracy);

    const card = $('#accuracyCard');
    if (!rows.length) { card.hidden = true; return; }
    card.hidden = false;

    $('#accuracyChart').replaceChildren(...rows.map((m) => h('div', { class: 'chart-row' },
      h('span', { class: 'bar-label', text: state.markets[m.market]?.name || pretty(m.market) }),
      h('div', { class: 'bar-track' },
        h('div', { class: 'bar-fill', style: { width: `${Math.max(2, m.accuracy * 100)}%` } })),
      h('span', { class: 'bar-val', text: `${pct(m.accuracy)} · ${pretty(m.model_type)}${m.members?.length ? ` (${m.members.length})` : ''}` }),
    )));
  }

  /* ------------------------------ history / log ------------------------------ */

  function rememberRun({ league, home, away, odds, response }) {
    const picks = {};
    for (const [market, p] of Object.entries(response.predictions || {})) {
      picks[market] = {
        prediction: p.prediction ?? p.predicted_value ?? null,
        confidence: p.confidence ?? null,
      };
    }
    state.history.unshift({
      ts: new Date().toISOString(),
      league,
      league_name: response.league || league,
      home, away, odds, picks,
    });
    saveHistory();
    renderHistory();
  }

  function renderHistory() {
    const box = $('#historyBlock');
    if (!state.history.length) {
      box.replaceChildren(h('div', { class: 'empty' }, h('h3', { text: 'No runs recorded' }), h('p', { text: 'Predictions you run are stored in this browser only.' })));
      return;
    }

    box.replaceChildren(...state.history.slice(0, 12).map((run, index) => {
      const picks = Object.entries(run.picks || {});
      const best = picks.slice().sort((a, b) => (b[1].confidence || 0) - (a[1].confidence || 0)).slice(0, 4);

      return h('div', { class: 'card', style: { marginBottom: '0' } },
        h('div', { class: 'head-row' },
          h('div', null,
            h('h3', { text: `${run.home} vs ${run.away}` }),
            h('p', { class: 'muted small', text: `${run.league_name || run.league} · ${new Date(run.ts).toLocaleString()} · ${picks.length} markets` }),
          ),
          h('div', { class: 'mini-actions' },
            h('button', {
              class: 'btn btn--ghost btn--small', type: 'button', text: 'Reload',
              onclick: () => {
                $('#league').value = run.league;
                $('#homeTeam').value = run.home;
                $('#awayTeam').value = run.away;
                const o = run.odds || {};
                const set = (sel, v) => { $(sel).value = v ?? ''; };
                set('#oddsHome', o.home); set('#oddsDraw', o.draw); set('#oddsAway', o.away);
                set('#oddsOver', o.over_25); set('#oddsUnder', o.under_25);
                set('#oddsAhHome', o.ah_home); set('#oddsAhAway', o.ah_away);
                switchTab('predict');
                syncPredictLeague();
                toast('Fixture loaded', 'Press “+ Add to queue” to re-test it.');
              },
            }),
            h('button', {
              class: 'btn btn--ghost btn--small', type: 'button', text: 'Delete',
              onclick: () => {
                state.history.splice(index, 1);
                saveHistory();
                renderHistory();
              },
            }),
          ),
        ),
        h('div', { style: { marginTop: '12px' } },
          h('div', { class: 'kv' }, best.map(([market, p]) => h('div', null,
            h('span', { class: 'k', text: `${state.markets[market]?.name || pretty(market)}` }),
            h('span', { class: 'v', text: `${classLabel(p.prediction, market)}${p.confidence != null ? ` · ${pct(p.confidence, 0)}` : ''}` }),
          ))),
        ),
      );
    }));
  }

  function renderLog() {
    const tbody = $('#logTable tbody');
    $('#logCount').textContent = state.log.length ? `· ${state.log.length}` : '';
    if (!state.log.length) {
      tbody.replaceChildren(h('tr', null, h('td', { colspan: 6, class: 'muted center', text: 'No requests logged yet.' })));
      return;
    }
    tbody.replaceChildren(...state.log.slice(0, 60).map((row) => h('tr', null,
      h('td', { text: new Date(row.time).toLocaleTimeString() }),
      h('td', null, h('span', { class: 'tag', text: row.method })),
      h('td', { class: 'wrap', text: row.path }),
      h('td', { class: `num ${row.status >= 400 || row.status === 0 ? 'v neg' : 'v pos'}`, text: row.status === 0 ? 'ERR' : String(row.status) }),
      h('td', { class: 'num', text: row.ms }),
      h('td', { class: 'wrap muted small', text: row.note || '' }),
    )));
  }

  function renderLastResponse() {
    $('#lastResponseBlock').textContent = state.lastResponse
      ? JSON.stringify(state.lastResponse, null, 2)
      : 'Nothing yet.';
  }

  function download(filename, text, type = 'application/json') {
    const blob = new Blob([text], { type });
    const url = URL.createObjectURL(blob);
    const a = h('a', { href: url, download: filename });
    document.body.append(a);
    a.click();
    a.remove();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  }

  function exportJson() {
    download(`football-model-runs-${Date.now()}.json`, JSON.stringify({
      exported_at: new Date().toISOString(),
      api_base: prefixed(state.apiBase) || location.origin,
      history: state.history,
      request_log: state.log,
    }, null, 2));
    toast('Exported', 'Runs and request log saved as JSON.');
  }

  function exportCsv() {
    const rows = [['timestamp', 'league', 'home', 'away', 'market', 'prediction', 'confidence']];
    for (const run of state.history) {
      for (const [market, p] of Object.entries(run.picks || {})) {
        rows.push([run.ts, run.league, run.home, run.away, market, String(p.prediction ?? ''), p.confidence ?? '']);
      }
    }
    if (rows.length === 1) return toast('Nothing to export', 'Run a prediction first.', 'warn');
    const csv = rows.map((r) => r.map((c) => `"${String(c).replace(/"/g, '""')}"`).join(',')).join('\n');
    download(`football-model-runs-${Date.now()}.csv`, csv, 'text/csv');
    toast('Exported', 'Runs saved as CSV.');
  }

  /* ------------------------------ tabs & wiring ------------------------------ */

  function switchTab(name) {
    for (const btn of document.querySelectorAll('.tab')) {
      const active = btn.dataset.tab === name;
      btn.classList.toggle('is-active', active);
      btn.setAttribute('aria-selected', String(active));
    }
    for (const panel of document.querySelectorAll('.panel')) {
      const active = panel.id === `tab-${name}`;
      panel.classList.toggle('is-active', active);
      panel.hidden = !active;
    }
    if (name === 'models') loadModels();
    if (name === 'activity') { renderLog(); renderHistory(); }
  }

  function applyTheme(theme) {
    document.documentElement.dataset.theme = theme;
    try { localStorage.setItem(LS.theme, theme); } catch { /* ignore */ }
  }

  function randomFixture() {
    const teams = state.teams[$('#league').value] || [];
    if (teams.length < 2) return toast('No teams loaded', 'Pick a league with team data first.', 'warn');
    const pool = [...teams];
    const a = pool.splice(Math.floor(Math.random() * pool.length), 1)[0];
    const b = pool[Math.floor(Math.random() * pool.length)];
    $('#homeTeam').value = a;
    $('#awayTeam').value = b;
    syncPredictLeague();
  }

  function wire() {
    // tabs
    for (const btn of document.querySelectorAll('.tab')) {
      btn.addEventListener('click', () => switchTab(btn.dataset.tab));
    }

    // theme
    const stored = localStorage.getItem(LS.theme);
    applyTheme(stored || (window.matchMedia('(prefers-color-scheme: light)').matches ? 'light' : 'dark'));
    $('#themeToggle').addEventListener('click', () => {
      applyTheme(document.documentElement.dataset.theme === 'dark' ? 'light' : 'dark');
    });

    // api base
    $('#apiBase').value = state.apiBase;
    const rebase = debounce(async () => {
      state.apiBase = $('#apiBase').value.trim();
      saveSettings();
      state.teams = {};
      state.models = {};
      const health = await checkHealth();
      if (health) {
        try { await loadLeagues(); await loadMarkets(); await syncPredictLeague(); } catch { /* pill shows error */ }
      }
    }, 500);
    $('#apiBase').addEventListener('input', rebase);

    $('#refreshStatus').addEventListener('click', () => checkHealth(true));

    // predict queue
    $('#predictForm').addEventListener('submit', addMatch);
    $('#addMatch').addEventListener('click', addMatch);
    $('#predictAll').addEventListener('click', predictAll);
    $('#clearQueue').addEventListener('click', clearQueue);
    $('#queueOddsToggle').addEventListener('click', toggleQueueOdds);
    $('#league').addEventListener('change', syncPredictLeague);
    $('#swapTeams').addEventListener('click', () => {
      const a = $('#homeTeam').value;
      $('#homeTeam').value = $('#awayTeam').value;
      $('#awayTeam').value = a;
    });
    $('#randomFixture').addEventListener('click', randomFixture);
    for (const sel of ['#homeTeam', '#awayTeam']) {
      $(sel).addEventListener('focus', () => { teamDatalist(state.teams[$('#league').value] || []); });
    }
    for (const sel of ['#oddsHome', '#oddsDraw', '#oddsAway', '#oddsOver', '#oddsUnder', '#oddsAhHome', '#oddsAhAway']) {
      $(sel).addEventListener('change', saveSettings);
    }

    $('#marketsAll').addEventListener('click', () => {
      state.customMarkets = true;
      state.selected = new Set(Object.keys(state.markets));
      renderMarketChips(); saveSettings();
    });
    $('#marketsNone').addEventListener('click', () => {
      state.customMarkets = true;
      state.selected = new Set();
      renderMarketChips(); saveSettings();
    });
    $('#marketsTrained').addEventListener('click', () => {
      state.customMarkets = false;
      const meta = leagueMeta($('#league').value);
      state.selected = new Set(meta?.markets || []);
      renderMarketChips(); saveSettings();
    });

    // compare
    $('#compareForm').addEventListener('submit', runCompare);
    $('#cmpLeague').addEventListener('change', async () => {
      try { await loadTeams($('#cmpLeague').value); } catch { /* handled on submit */ }
      teamDatalist(state.teams[$('#cmpLeague').value] || []);
    });
    for (const sel of ['#cmpHome', '#cmpAway']) {
      $(sel).addEventListener('focus', () => { teamDatalist(state.teams[$('#cmpLeague').value] || []); });
    }
    $('#copyCompareTeams').addEventListener('click', () => {
      $('#cmpLeague').value = $('#league').value;
      $('#cmpHome').value = $('#homeTeam').value;
      $('#cmpAway').value = $('#awayTeam').value;
      loadTeams($('#cmpLeague').value).then(() => teamDatalist(state.teams[$('#cmpLeague').value] || [])).catch(() => {});
    });

    // models
    $('#modelLeague').addEventListener('change', loadModels);
    $('#modelSearch').addEventListener('input', debounce(loadModels, 200));
    for (const th of document.querySelectorAll('#modelsTable thead th[data-sort]')) {
      th.addEventListener('click', () => {
        const key = th.dataset.sort;
        modelSort = { key, dir: modelSort.key === key ? -modelSort.dir : (key === 'accuracy' || key === 'mae' || key === 'n_features' ? -1 : 1) };
        loadModels();
      });
    }

    // activity
    $('#copyLast').addEventListener('click', async () => {
      if (!state.lastResponse) return toast('Nothing to copy', 'Run a request first.', 'warn');
      try {
        await navigator.clipboard.writeText(JSON.stringify(state.lastResponse, null, 2));
        toast('Copied', 'Last response JSON copied to clipboard.');
      } catch {
        toast('Copy failed', 'Clipboard is unavailable in this browser context.', 'warn');
      }
    });
    $('#exportJson').addEventListener('click', exportJson);
    $('#exportCsv').addEventListener('click', exportCsv);
    $('#clearLog').addEventListener('click', () => { state.log = []; renderLog(); });
    $('#clearHistory').addEventListener('click', () => {
      state.history = [];
      saveHistory();
      renderHistory();
      toast('History cleared');
    });

    // shortest path to a result: Enter in a team field queues the fixture,
    // Ctrl/⌘ + Enter predicts the whole queue.
    document.addEventListener('keydown', (e) => {
      if ((e.metaKey || e.ctrlKey) && e.key === 'Enter' && !$('#tab-predict').hidden) {
        const active = document.activeElement;
        if (active === $('#homeTeam') || active === $('#awayTeam')) addMatch(e);
        else predictAll();
      }
    });
  }

  /* ------------------------------ init ------------------------------ */

  async function init() {
    wire();
    restoreInputs();
    renderHistory();
    renderQueue();
    renderLog();
    $('#marketChips').replaceChildren(h('span', { class: 'muted small', text: 'Loading markets…' }));

    const health = await checkHealth();
    if (!health) {
      $('#league').replaceChildren(h('option', { value: '', text: 'API unreachable' }));
      $('#predictResults').replaceChildren(h('div', { class: 'error-box' },
        'Cannot reach the API. Start it with  python api/app.py  and reload — or point the “API base” field at a running instance.'));
      return;
    }

    try {
      await Promise.all([loadLeagues(), loadMarkets()]);
      await syncPredictLeague();
      $('#predictState').textContent = 'Ready. Add matches to the queue, then Predict all (Ctrl/⌘ + Enter).';
    } catch (err) {
      toast('Setup problem', err.message, 'error', 6000);
    }
  }

  document.addEventListener('DOMContentLoaded', init);
})();
