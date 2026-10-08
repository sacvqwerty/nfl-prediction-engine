import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import Optional
import numpy as np

app = FastAPI(title="NFL Quantitative Simulation Engine")

# =====================================================================
# 1. DATABASE: ALL 32 NFL TEAMS STATS & EFFICIENCY PROFILES
# =====================================================================
NFL_TEAMS = {
    "ARI": {"name": "Arizona Cardinals", "conf": "NFC", "off_pass_epa": 0.05, "off_rush_epa": 0.02, "def_pass_epa": 0.08, "def_rush_epa": 0.04, "pace": 1.01, "qb": "Kyler Murray", "hfa": 1.4},
    "ATL": {"name": "Atlanta Falcons", "conf": "NFC", "off_pass_epa": 0.08, "off_rush_epa": 0.06, "def_pass_epa": 0.04, "def_rush_epa": 0.01, "pace": 0.99, "qb": "Michael Penix Jr.", "hfa": 1.6},
    "BAL": {"name": "Baltimore Ravens", "conf": "AFC", "off_pass_epa": 0.16, "off_rush_epa": 0.12, "def_pass_epa": -0.04, "def_rush_epa": -0.08, "pace": 1.00, "qb": "Lamar Jackson", "hfa": 2.2},
    "BUF": {"name": "Buffalo Bills", "conf": "AFC", "off_pass_epa": 0.18, "off_rush_epa": 0.05, "def_pass_epa": -0.02, "def_rush_epa": 0.01, "pace": 1.04, "qb": "Josh Allen", "hfa": 2.4},
    "CAR": {"name": "Carolina Panthers", "conf": "NFC", "off_pass_epa": -0.12, "off_rush_epa": -0.05, "def_pass_epa": 0.12, "def_rush_epa": 0.07, "pace": 0.96, "qb": "Bryce Young", "hfa": 1.2},
    "CHI": {"name": "Chicago Bears", "conf": "NFC", "off_pass_epa": 0.03, "off_rush_epa": 0.01, "def_pass_epa": -0.03, "def_rush_epa": -0.01, "pace": 1.02, "qb": "Caleb Williams", "hfa": 1.8},
    "CIN": {"name": "Cincinnati Bengals", "conf": "AFC", "off_pass_epa": 0.15, "off_rush_epa": -0.02, "def_pass_epa": 0.05, "def_rush_epa": 0.02, "pace": 1.01, "qb": "Joe Burrow", "hfa": 1.7},
    "CLE": {"name": "Cleveland Browns", "conf": "AFC", "off_pass_epa": -0.08, "off_rush_epa": 0.01, "def_pass_epa": -0.07, "def_rush_epa": -0.05, "pace": 0.97, "qb": "Deshaun Watson", "hfa": 1.9},
    "DAL": {"name": "Dallas Cowboys", "conf": "NFC", "off_pass_epa": 0.14, "off_rush_epa": -0.01, "def_pass_epa": -0.03, "def_rush_epa": 0.02, "pace": 1.03, "qb": "Dak Prescott", "hfa": 1.9},
    "DEN": {"name": "Denver Broncos", "conf": "AFC", "off_pass_epa": 0.01, "off_rush_epa": -0.02, "def_pass_epa": -0.05, "def_rush_epa": -0.03, "pace": 0.98, "qb": "Bo Nix", "hfa": 2.5},
    "DET": {"name": "Detroit Lions", "conf": "NFC", "off_pass_epa": 0.17, "off_rush_epa": 0.10, "def_pass_epa": 0.01, "def_rush_epa": -0.04, "pace": 1.02, "qb": "Jared Goff", "hfa": 2.1},
    "GB":  {"name": "Green Bay Packers", "conf": "NFC", "off_pass_epa": 0.13, "off_rush_epa": 0.06, "def_pass_epa": -0.02, "def_rush_epa": -0.02, "pace": 1.00, "qb": "Jordan Love", "hfa": 2.3},
    "HOU": {"name": "Houston Texans", "conf": "AFC", "off_pass_epa": 0.14, "off_rush_epa": 0.02, "def_pass_epa": -0.04, "def_rush_epa": -0.01, "pace": 1.01, "qb": "C.J. Stroud", "hfa": 1.6},
    "IND": {"name": "Indianapolis Colts", "conf": "AFC", "off_pass_epa": 0.02, "off_rush_epa": 0.08, "def_pass_epa": 0.03, "def_rush_epa": 0.04, "pace": 1.03, "qb": "Anthony Richardson", "hfa": 1.5},
    "JAX": {"name": "Jacksonville Jaguars", "conf": "AFC", "off_pass_epa": 0.04, "off_rush_epa": -0.01, "def_pass_epa": 0.06, "def_rush_epa": 0.03, "pace": 1.00, "qb": "Trevor Lawrence", "hfa": 1.3},
    "KC":  {"name": "Kansas City Chiefs", "conf": "AFC", "off_pass_epa": 0.19, "off_rush_epa": 0.04, "def_pass_epa": -0.06, "def_rush_epa": -0.04, "pace": 1.00, "qb": "Patrick Mahomes", "hfa": 2.6},
    "LV":  {"name": "Las Vegas Raiders", "conf": "AFC", "off_pass_epa": -0.09, "off_rush_epa": -0.06, "def_pass_epa": 0.02, "def_rush_epa": 0.01, "pace": 0.98, "qb": "Gardner Minshew", "hfa": 1.3},
    "LAC": {"name": "Los Angeles Chargers", "conf": "AFC", "off_pass_epa": 0.09, "off_rush_epa": 0.07, "def_pass_epa": -0.05, "def_rush_epa": -0.02, "pace": 0.96, "qb": "Justin Herbert", "hfa": 1.4},
    "LAR": {"name": "Los Angeles Rams", "conf": "NFC", "off_pass_epa": 0.11, "off_rush_epa": 0.03, "def_pass_epa": 0.04, "def_rush_epa": 0.05, "pace": 1.01, "qb": "Matthew Stafford", "hfa": 1.5},
    "MIA": {"name": "Miami Dolphins", "conf": "AFC", "off_pass_epa": 0.12, "off_rush_epa": 0.04, "def_pass_epa": 0.01, "def_rush_epa": 0.02, "pace": 1.04, "qb": "Tua Tagovailoa", "hfa": 1.8},
    "MIN": {"name": "Minnesota Vikings", "conf": "NFC", "off_pass_epa": 0.10, "off_rush_epa": 0.01, "def_pass_epa": -0.06, "def_rush_epa": -0.03, "pace": 0.99, "qb": "J.J. McCarthy", "hfa": 2.2},
    "NE":  {"name": "New England Patriots", "conf": "AFC", "off_pass_epa": -0.10, "off_rush_epa": -0.02, "def_pass_epa": 0.01, "def_rush_epa": -0.03, "pace": 0.97, "qb": "Drake Maye", "hfa": 1.7},
    "NO":  {"name": "New Orleans Saints", "conf": "NFC", "off_pass_epa": 0.04, "off_rush_epa": 0.03, "def_pass_epa": 0.02, "def_rush_epa": 0.01, "pace": 0.99, "qb": "Derek Carr", "hfa": 2.0},
    "NYG": {"name": "New York Giants", "conf": "NFC", "off_pass_epa": -0.07, "off_rush_epa": -0.04, "def_pass_epa": 0.03, "def_rush_epa": 0.06, "pace": 1.00, "qb": "Daniel Jones", "hfa": 1.4},
    "NYJ": {"name": "New York Jets", "conf": "AFC", "off_pass_epa": 0.06, "off_rush_epa": 0.02, "def_pass_epa": -0.08, "def_rush_epa": -0.04, "pace": 0.98, "qb": "Aaron Rodgers", "hfa": 1.6},
    "PHI": {"name": "Philadelphia Eagles", "conf": "NFC", "off_pass_epa": 0.12, "off_rush_epa": 0.11, "def_pass_epa": 0.01, "def_rush_epa": -0.03, "pace": 1.02, "qb": "Jalen Hurts", "hfa": 2.2},
    "PIT": {"name": "Pittsburgh Steelers", "conf": "AFC", "off_pass_epa": -0.02, "off_rush_epa": 0.04, "def_pass_epa": -0.05, "def_rush_epa": -0.06, "pace": 0.97, "qb": "Russell Wilson", "hfa": 2.1},
    "SF":  {"name": "San Francisco 49ers", "conf": "NFC", "off_pass_epa": 0.18, "off_rush_epa": 0.09, "def_pass_epa": -0.03, "def_rush_epa": -0.05, "pace": 0.98, "qb": "Brock Purdy", "hfa": 2.0},
    "SEA": {"name": "Seattle Seahawks", "conf": "NFC", "off_pass_epa": 0.08, "off_rush_epa": 0.00, "def_pass_epa": 0.02, "def_rush_epa": 0.04, "pace": 1.03, "qb": "Geno Smith", "hfa": 2.4},
    "TB":  {"name": "Tampa Bay Buccaneers", "conf": "NFC", "off_pass_epa": -0.15, "off_rush_epa": -0.06, "def_pass_epa": 0.04, "def_rush_epa": 0.02, "pace": 0.98, "qb": "Jalon Daniels (Rookie)", "hfa": 1.5},
    "TEN": {"name": "Tennessee Titans", "conf": "AFC", "off_pass_epa": -0.06, "off_rush_epa": 0.01, "def_pass_epa": 0.00, "def_rush_epa": -0.02, "pace": 0.98, "qb": "Will Levis", "hfa": 1.4},
    "WAS": {"name": "Washington Commanders", "conf": "NFC", "off_pass_epa": 0.13, "off_rush_epa": 0.07, "def_pass_epa": 0.05, "def_rush_epa": 0.04, "pace": 1.02, "qb": "Jayden Daniels", "hfa": 1.5}
}

# =====================================================================
# 2. QUANTITATIVE SIMULATION ENGINE
# =====================================================================
class MatchupRequest(BaseModel):
    home_team: str
    away_team: str
    spread: float
    total: float
    home_qb_status: Optional[str] = "starter"
    away_qb_status: Optional[str] = "starter"
    simulations: Optional[int] = 50000

class SimulationEngine:
    @staticmethod
    def calculate_drive_probabilities(net_epa: float) -> np.ndarray:
        base_logits = np.array([0.24, 0.16, 0.595, 0.005])
        adj_logits = base_logits.copy()
        adj_logits[0] += 0.48 * net_epa
        adj_logits[1] += 0.22 * net_epa
        adj_logits[2] -= 0.70 * net_epa
        exp_logits = np.exp(adj_logits - np.max(adj_logits))
        return exp_logits / np.sum(exp_logits)

    @classmethod
    def simulate(cls, home_data: dict, away_data: dict, spread: float, total: float, n_sims: int = 50000):
        home_off_eff = (0.65 * home_data["off_pass_epa"]) + (0.35 * home_data["off_rush_epa"])
        away_def_eff = (0.65 * away_data["def_pass_epa"]) + (0.35 * away_data["def_rush_epa"])
        home_net_epa = home_off_eff - away_def_eff

        away_off_eff = (0.65 * away_data["off_pass_epa"]) + (0.35 * away_data["off_rush_epa"])
        home_def_eff = (0.65 * home_data["def_pass_epa"]) + (0.35 * home_data["def_rush_epa"])
        away_net_epa = away_off_eff - home_def_eff

        home_probs = cls.calculate_drive_probabilities(home_net_epa)
        away_probs = cls.calculate_drive_probabilities(away_net_epa)

        home_poss = np.random.poisson(11.4 * home_data["pace"], n_sims)
        away_poss = np.random.poisson(11.4 * away_data["pace"], n_sims)

        outcomes_map = np.array([7, 3, 0, -2])

        home_scores = np.zeros(n_sims)
        away_scores = np.zeros(n_sims)

        for i in range(n_sims):
            h_events = np.random.choice(outcomes_map, size=home_poss[i], p=home_probs)
            h_td_noise = np.random.choice([-1, 0, 1], size=home_poss[i], p=[0.06, 0.90, 0.04])
            home_scores[i] = np.sum(h_events + (h_events == 7) * h_td_noise)

            a_events = np.random.choice(outcomes_map, size=away_poss[i], p=away_probs)
            a_td_noise = np.random.choice([-1, 0, 1], size=away_poss[i], p=[0.06, 0.90, 0.04])
            away_scores[i] = np.sum(a_events + (a_events == 7) * a_td_noise)

        home_scores += home_data["hfa"]

        ties = (np.round(home_scores) == np.round(away_scores))
        tie_count = np.sum(ties)
        if tie_count > 0:
            home_scores[ties] += np.random.choice([3, 6], size=tie_count, p=[0.55, 0.45])

        margins = home_scores - away_scores
        totals = home_scores + away_scores

        home_wins = np.mean(margins > 0)
        away_wins = np.mean(margins < 0)

        cover_home = np.mean(margins > -spread)
        over_prob = np.mean(totals > total)

        def to_moneyline(p):
            if p >= 0.5:
                return int(-((p / (1.0 - p)) * 100))
            else:
                return int(+(((1.0 - p) / p) * 100))

        hist_bins = list(range(-35, 36))
        hist_vals, _ = np.histogram(margins, bins=range(-36, 37))
        hist_normalized = [round(float(v / n_sims) * 100, 2) for v in hist_vals]

        return {
            "home_score": round(float(np.mean(home_scores)), 2),
            "away_score": round(float(np.mean(away_scores)), 2),
            "median_margin": round(float(np.median(margins)), 2),
            "total_points": round(float(np.mean(totals)), 2),
            "home_win_pct": round(float(home_wins * 100), 2),
            "away_win_pct": round(float(away_wins * 100), 2),
            "home_fair_ml": to_moneyline(home_wins),
            "away_fair_ml": to_moneyline(away_wins),
            "home_cover_pct": round(float(cover_home * 100), 2),
            "over_pct": round(float(over_prob * 100), 2),
            "histogram": {"bins": hist_bins, "densities": hist_normalized}
        }

# =====================================================================
# 3. FASTAPI REST ENDPOINTS
# =====================================================================
@app.get("/api/teams")
def get_teams():
    return {k: {"code": k, "name": v["name"], "conf": v["conf"], "qb": v["qb"]} for k, v in sorted(NFL_TEAMS.items())}

@app.get("/api/team/{team_code}")
def get_team_stats(team_code: str):
    code = team_code.upper()
    if code not in NFL_TEAMS:
        raise HTTPException(status_code=404, detail="Team code not found")
    return NFL_TEAMS[code]

@app.post("/api/simulate")
def run_simulation(req: MatchupRequest):
    if req.home_team not in NFL_TEAMS or req.away_team not in NFL_TEAMS:
        raise HTTPException(status_code=400, detail="Invalid team selection")

    h_data = NFL_TEAMS[req.home_team].copy()
    a_data = NFL_TEAMS[req.away_team].copy()

    if req.home_qb_status == "backup":
        h_data["off_pass_epa"] -= 0.18
    if req.away_qb_status == "backup":
        a_data["off_pass_epa"] -= 0.18

    return SimulationEngine.simulate(
        home_data=h_data,
        away_data=a_data,
        spread=req.spread,
        total=req.total,
        n_sims=req.simulations or 50000
    )

# =====================================================================
# 4. FRONTEND INTERFACE
# =====================================================================
@app.get("/", response_class=HTMLResponse)
def index_page():
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
      <meta charset="UTF-8">
      <title>QuantNFL - Simulation Engine</title>
      <meta name="viewport" content="width=device-width, initial-scale=1.0">
      <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
      <style>
        :root { --bg: #0a0e17; --card: #121926; --border: #1f2a3d; --cyan: #06b6d4; --blue: #3b82f6; --green: #10b981; --red: #ef4444; --text: #f8fafc; --muted: #94a3b8; }
        * { box-sizing: border-box; margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, sans-serif; }
        body { background: var(--bg); color: var(--text); padding: 2rem 1rem; }
        .wrapper { max-width: 1100px; margin: 0 auto; }
        header { text-align: center; margin-bottom: 2rem; }
        h1 { font-size: 2.2rem; font-weight: 800; color: #fff; margin-bottom: 0.3rem; }
        p.subtitle { color: var(--muted); font-size: 0.95rem; }
        .grid-2 { display: grid; grid-template-columns: 1fr 1fr; gap: 1.5rem; margin-bottom: 1.5rem; }
        @media (max-width: 800px) { .grid-2 { grid-template-columns: 1fr; } }
        .card { background: var(--card); border: 1px solid var(--border); border-radius: 12px; padding: 1.5rem; margin-bottom: 1.5rem; }
        .card-header { font-size: 1.1rem; font-weight: 700; color: var(--cyan); margin-bottom: 1rem; display: flex; justify-content: space-between; }
        .form-row { margin-bottom: 1rem; }
        label { display: block; font-size: 0.75rem; text-transform: uppercase; color: var(--muted); font-weight: 600; margin-bottom: 0.3rem; }
        select, input { width: 100%; padding: 0.7rem; background: #080c14; border: 1px solid var(--border); border-radius: 6px; color: #fff; font-size: 0.95rem; }
        .stat-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 0.5rem; background: #080c14; padding: 0.8rem; border-radius: 8px; border: 1px solid var(--border); margin-top: 1rem; }
        .stat-box { text-align: center; }
        .stat-val { font-size: 1rem; font-weight: 700; color: #fff; }
        .stat-lbl { font-size: 0.65rem; color: var(--muted); text-transform: uppercase; }
        .btn-run { width: 100%; padding: 1.1rem; background: linear-gradient(135deg, var(--blue), var(--cyan)); border: none; border-radius: 8px; color: #fff; font-weight: 800; font-size: 1.1rem; cursor: pointer; }
        #results-dashboard { display: none; margin-top: 1.5rem; }
        .kpi-row { display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 1rem; margin-bottom: 1.5rem; }
        .kpi { background: var(--card); border: 1px solid var(--border); padding: 1.2rem; border-radius: 10px; text-align: center; }
        .kpi-title { font-size: 0.75rem; color: var(--muted); text-transform: uppercase; font-weight: 600; }
        .kpi-val { font-size: 1.8rem; font-weight: 800; margin: 0.3rem 0; color: #fff; }
        .win-bar { height: 38px; border-radius: 8px; overflow: hidden; display: flex; margin: 1rem 0; border: 1px solid var(--border); }
        .win-bar-h { background: var(--green); display: flex; align-items: center; justify-content: center; font-weight: 800; font-size: 0.85rem; }
        .win-bar-a { background: var(--red); display: flex; align-items: center; justify-content: center; font-weight: 800; font-size: 0.85rem; }
        canvas { max-height: 280px; width: 100%; }
      </style>
    </head>
    <body>
      <div class="wrapper">
        <header>
          <h1>NFL Quantitative Simulation Engine</h1>
          <p class="subtitle">Markov-Chain Drive Simulator with 50,000 Monte Carlo Iterations</p>
        </header>

        <div class="grid-2">
          <!-- Home Team -->
          <div class="card">
            <div class="card-header"><span>HOME TEAM</span><span id="h_conf" style="font-size:0.8rem; color:var(--muted)"></span></div>
            <div class="form-row">
              <label>Select Team</label>
              <select id="home_select" onchange="loadStats('home')"></select>
            </div>
            <div class="form-row">
              <label>Quarterback Status</label>
              <select id="h_qb_status">
                <option value="starter" selected>Starting QB Active</option>
                <option value="backup">Backup / Rookie (-0.18 EPA Penalty)</option>
              </select>
            </div>
            <div class="stat-grid">
              <div class="stat-box"><div class="stat-val" id="h_stat_q">-</div><div class="stat-lbl">QB</div></div>
              <div class="stat-box"><div class="stat-val" id="h_stat_pepa">-</div><div class="stat-lbl">Pass EPA</div></div>
              <div class="stat-box"><div class="stat-val" id="h_stat_repa">-</div><div class="stat-lbl">Rush EPA</div></div>
              <div class="stat-box"><div class="stat-val" id="h_stat_depa">-</div><div class="stat-lbl">Def EPA</div></div>
            </div>
          </div>

          <!-- Away Team -->
          <div class="card">
            <div class="card-header"><span>AWAY TEAM</span><span id="a_conf" style="font-size:0.8rem; color:var(--muted)"></span></div>
            <div class="form-row">
              <label>Select Team</label>
              <select id="away_select" onchange="loadStats('away')"></select>
            </div>
            <div class="form-row">
              <label>Quarterback Status</label>
              <select id="a_qb_status">
                <option value="starter" selected>Starting QB Active</option>
                <option value="backup">Backup / Rookie (-0.18 EPA Penalty)</option>
              </select>
            </div>
            <div class="stat-grid">
              <div class="stat-box"><div class="stat-val" id="a_stat_q">-</div><div class="stat-lbl">QB</div></div>
              <div class="stat-box"><div class="stat-val" id="a_stat_pepa">-</div><div class="stat-lbl">Pass EPA</div></div>
              <div class="stat-box"><div class="stat-val" id="a_stat_repa">-</div><div class="stat-lbl">Rush EPA</div></div>
              <div class="stat-box"><div class="stat-val" id="a_stat_depa">-</div><div class="stat-lbl">Def EPA</div></div>
            </div>
          </div>
        </div>

        <div class="card">
          <div class="card-header">SPORTSBOOK BENCHMARK LINES</div>
          <div style="display:grid; grid-template-columns: 1fr 1fr; gap:1rem;">
            <div class="form-row">
              <label>Home Spread</label>
              <input type="number" id="market_spread" step="0.5" value="-8.5">
            </div>
            <div class="form-row">
              <label>Over / Under Line</label>
              <input type="number" id="market_total" step="0.5" value="48.5">
            </div>
          </div>
        </div>

        <button class="btn-run" onclick="executeSimulation()">RUN 50,000 SIMULATIONS</button>

        <div id="results-dashboard">
          <div class="card">
            <div class="card-header">SIMULATED WIN PROBABILITY & FAIR MONEYLINE</div>
            <div class="win-bar">
              <div class="win-bar-h" id="bar_h">HOME 50%</div>
              <div class="win-bar-a" id="bar_a">AWAY 50%</div>
            </div>
          </div>

          <div class="kpi-row">
            <div class="kpi"><div class="kpi-title">Projected Score</div><div class="kpi-val" id="kpi_score" style="color:var(--cyan)">-</div></div>
            <div class="kpi"><div class="kpi-title">Projected Margin</div><div class="kpi-val" id="kpi_margin">-</div></div>
            <div class="kpi"><div class="kpi-title">Home Cover %</div><div class="kpi-val" id="kpi_cover">-</div></div>
            <div class="kpi"><div class="kpi-title">Over Total %</div><div class="kpi-val" id="kpi_over">-</div></div>
          </div>

          <div class="card">
            <div class="card-header">MARGIN OF VICTORY PROBABILITY DENSITY</div>
            <canvas id="marginChart"></canvas>
          </div>
        </div>
      </div>

      <script>
        let teamsData = {};
        let chartInstance = null;

        async function init() {
          const res = await fetch('/api/teams');
          teamsData = await res.json();
          const hSelect = document.getElementById('home_select');
          const aSelect = document.getElementById('away_select');

          Object.keys(teamsData).forEach(code => {
            hSelect.add(new Option(`${teamsData[code].name} (${code})`, code));
            aSelect.add(new Option(`${teamsData[code].name} (${code})`, code));
          });

          hSelect.value = "DAL";
          aSelect.value = "TB";
          document.getElementById('a_qb_status').value = "backup";

          loadStats('home');
          loadStats('away');
        }

        async function loadStats(side) {
          const code = document.getElementById(side + '_select').value;
          const res = await fetch('/api/team/' + code);
          const d = await res.json();
          const p = side === 'home' ? 'h' : 'a';

          document.getElementById(p + '_conf').textContent = d.conf + ' Conference';
          document.getElementById(p + '_stat_q').textContent = d.qb.split(' ')[0];
          document.getElementById(p + '_stat_pepa').textContent = (d.off_pass_epa > 0 ? '+' : '') + d.off_pass_epa.toFixed(2);
          document.getElementById(p + '_stat_repa').textContent = (d.off_rush_epa > 0 ? '+' : '') + d.off_rush_epa.toFixed(2);
          document.getElementById(p + '_stat_depa').textContent = (d.def_pass_epa > 0 ? '+' : '') + d.def_pass_epa.toFixed(2);
        }

        async function executeSimulation() {
          const payload = {
            home_team: document.getElementById('home_select').value,
            away_team: document.getElementById('away_select').value,
            spread: parseFloat(document.getElementById('market_spread').value),
            total: parseFloat(document.getElementById('market_total').value),
            home_qb_status: document.getElementById('h_qb_status').value,
            away_qb_status: document.getElementById('a_qb_status').value,
            simulations: 50000
          };

          const res = await fetch('/api/simulate', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify(payload)
          });
          const data = await res.json();

          document.getElementById('results-dashboard').style.display = 'block';
          const hName = document.getElementById('home_select').value;
          const aName = document.getElementById('away_select').value;

          document.getElementById('bar_h').style.width = data.home_win_pct + '%';
          document.getElementById('bar_h').textContent = `${hName} ${data.home_win_pct}% (${data.home_fair_ml > 0 ? '+'+data.home_fair_ml : data.home_fair_ml})`;
          document.getElementById('bar_a').style.width = data.away_win_pct + '%';
          document.getElementById('bar_a').textContent = `${aName} ${data.away_win_pct}% (${data.away_fair_ml > 0 ? '+'+data.away_fair_ml : data.away_fair_ml})`;

          document.getElementById('kpi_score').textContent = `${data.home_score} - ${data.away_score}`;
          document.getElementById('kpi_margin').textContent = (data.median_margin > 0 ? `-${data.median_margin}` : `+${Math.abs(data.median_margin)}`);
          document.getElementById('kpi_cover').textContent = `${data.home_cover_pct}%`;
          document.getElementById('kpi_over').textContent = `${data.over_pct}%`;

          renderChart(data.histogram);
        }

        function renderChart(hist) {
          const ctx = document.getElementById('marginChart').getContext('2d');
          if (chartInstance) chartInstance.destroy();

          chartInstance = new Chart(ctx, {
            type: 'bar',
            data: {
              labels: hist.bins,
              datasets: [{
                data: hist.densities,
                backgroundColor: hist.bins.map(b => (Math.abs(b) === 3 || Math.abs(b) === 7) ? '#06b6d4' : (b > 0 ? '#10b981' : '#ef4444'))
              }]
            },
            options: {
              responsive: true,
              plugins: { legend: { display: false } },
              scales: {
                x: { title: { display: true, text: 'Margin of Victory (+ Home | - Away)', color: '#94a3b8' }, ticks: { color: '#94a3b8' } },
                y: { title: { display: true, text: 'Probability (%)', color: '#94a3b8' }, ticks: { color: '#94a3b8' } }
              }
            }
          });
        }

        window.onload = init;
      </script>
    </body>
    </html>
    """

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
