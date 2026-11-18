# SupeRR Selector · Pressure Intelligence Lab (build 0.2)

A premium, responsive React application shell with smooth navigation, refined typography, hover elevations, mobile sidebar, dataset inspection, real descriptive player statistics, comparison, an illustrative pressure simulator, and local dataset uploads. Includes the **six supplied original hackathon CSV files**.

## What is real vs. pending

**Implemented:** React navigation and transitions, responsive UI, FastAPI dataset inventory, dataset previews, player-level descriptive totals from ball-by-ball data, comparison for up to four selected players, pressure *demonstration* sliders, local CSV/XLSX upload with column preview.

**Not yet implemented:** validated pressure model, trustworthy top-10 pressure rankings, full feature mapping, cloud persistence, methodology PDF/report, production deployment. No invented pressure rankings are displayed.

**Important:** the provisional bowler wicket total counts raw `is_wicket` delivery flags; it needs dismissal-type validation before being used for rankings. Batter runs and strike rates are descriptive and need further rules review before submission.

## Requirements
- Python 3.10+ (3.11 recommended)
- Node.js 20+
- VS Code (recommended)

## Run on Windows (two VS Code terminals)

1. Extract the ZIP. Open the `superr-selector` folder in VS Code.
2. Terminal 1 — backend:

```powershell
cd backend
py -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe -m uvicorn app.main:app --reload
```

3. Terminal 2 — frontend (from project root):

```powershell
cd frontend
npm install
npm run dev
```

4. Open **http://localhost:5173**. Backend health: **http://localhost:8000/api/health**. API docs: **http://localhost:8000/docs**.

## Architecture
- `frontend/src/main.jsx` — SPA sections, navigation, real data display, upload interaction
- `frontend/src/style.css` — responsive design system, elevations, transitions
- `backend/app/main.py` — API routes, dataset previews, descriptive player stats, uploads
- `backend/data/` — original CSV datasets (do not modify)

## Cloud deployment notes
The frontend can be deployed to Vercel, and the backend to a Python hosting provider. Configure `VITE_API_URL` to the public backend URL, and update allowed CORS origins. Uploads currently persist only on local backend disk; production requires persistent object storage and validation/security controls. Hosting has **not** been performed, so no live URL exists yet.

## Next: analytical integrity
1. Validate joins, match innings, delivery types and wicket attribution.
2. Construct match state for each delivery and define pressure separately for chase/defend contexts.
3. Establish baseline metrics, sample-size floors and consistency checks.
4. Validate batters/bowlers top-10 results and write the methodology report.
5. Connect production storage and deploy.


## Build 0.3 updates
- Removed the sidebar's bottom-left Research Workspace card; top brand and navigation remain.
- Introduced an obsidian, glacier-teal and warm-copper visual identity and word-by-word opening headline reveal (honours reduced motion).
- Pressure Lab now queries `/api/pressure/scenarios` to display observed batters, bowlers and example matches in similar historical second-innings contexts. **Important:** chase target is a proxy from first-innings batter runs, not an official target; this does not constitute validated pressure rankings.
- Restart the Python backend to activate the new endpoint. Refresh the frontend after replacing the project.

## Version 0.4 — exploratory pressure engine

New endpoints: `/api/pressure/rankings?kind=batters`, `/api/pressure/rankings?kind=bowlers`, `/api/pressure/methodology`. Open **Rankings** in the frontend to see provisional scores, baselines, sample sizes and CSV export. Start backend and frontend as before.

**Do not submit these top tens as official uncapped selections.** Player uncapped status is not provided. The supplied `extras1_runs` column encodes extras types (e.g. 6 means no extra), rather than an amount of runs; thus verified innings totals, exact chase targets and bowler economy are unavailable. The model uses first-innings batter runs as a proxy, excludes revised matches, and does not claim predictive validation. The ranking is a research shortlist pending eligibility screening and sensitivity analysis.

Pressure score: first innings late stage and wickets lost; second innings estimated required rate, wickets lost and late stage; threshold 55/100. Batting rate is strike rate; bowling rate is bowler-attributed dismissals per 100 legal deliveries. Compare each player's pressure performance to normal performance and shrink toward their baseline. Require 30 pressure balls in >=3 matches and 60 normal balls in >=3 matches. The score is a heuristic, not a calibrated probability.
