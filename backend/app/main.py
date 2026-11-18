from pathlib import Path
from functools import lru_cache
import re
import csv
import io
from fastapi.responses import StreamingResponse
import pandas as pd
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / 'data'
FILES = ['ball_by_ball','matches','competitions','playing_eleven','teams','competition_team_player_mapping']
app = FastAPI(title='SupeRR Selector · Intelligence API', version='0.2.0')
app.add_middleware(CORSMiddleware, allow_origins=['http://localhost:5173','http://127.0.0.1:5173'], allow_methods=['*'], allow_headers=['*'])

@lru_cache(maxsize=1)
def inventory():
    out=[]
    for name in FILES:
        p=DATA/f'{name}.csv'
        if not p.exists():
            out.append(dict(name=name,available=False,rows=0,columns=[]));continue
        cols=pd.read_csv(p,nrows=0).columns.tolist()
        with p.open('rb') as f: rows=sum(chunk.count(b'\n') for chunk in iter(lambda:f.read(1024*1024),b''))-1
        out.append(dict(name=name,available=True,rows=max(0,rows),columns=cols))
    return out

@lru_cache(maxsize=1)
def player_stats():
    cols=['striker_id','striker_name','bowler_id','bowler_name','runs','is_four','is_six','is_wicket','extras_type','innings_ball_num','match_id']
    df=pd.read_csv(DATA/'ball_by_ball.csv',usecols=cols,low_memory=False)
    df['runs']=pd.to_numeric(df.runs,errors='coerce').fillna(0)
    df['is_wicket']=pd.to_numeric(df.is_wicket,errors='coerce').fillna(0)
    # This is an exploratory overall-performance view, not a pressure score.
    legal=df.extras_type.fillna('').astype(str).str.lower().str.strip().map(lambda x: not any(s in x for s in ['wide','no ball','noball']))
    bat=df.assign(valid_ball=~df.extras_type.fillna('').astype(str).str.lower().str.contains('wide')).groupby(['striker_id','striker_name'],dropna=False).agg(runs=('runs','sum'),balls=('valid_ball','sum'),matches=('match_id','nunique'),fours=('is_four','sum'),sixes=('is_six','sum')).reset_index()
    bat=bat[bat.balls>=30].copy();bat['strike_rate']=(bat.runs/bat.balls*100).round(1)
    bowl=df.assign(legal_ball=legal.astype(int)).groupby(['bowler_id','bowler_name'],dropna=False).agg(wickets=('is_wicket','sum'),balls=('legal_ball','sum'),matches=('match_id','nunique')).reset_index()
    bowl=bowl[bowl.balls>=30].copy()
    return {'batters':bat.sort_values('runs',ascending=False).head(100).rename(columns={'striker_name':'name','striker_id':'id'}).fillna('').to_dict('records'), 'bowlers':bowl.sort_values('wickets',ascending=False).head(100).rename(columns={'bowler_name':'name','bowler_id':'id'}).fillna('').to_dict('records')}

@app.get('/api/health')
def health():return {'status':'ok','version':'0.2.0'}

@app.get('/api/overview')
def overview():
    datasets=inventory()
    return {'datasets':datasets,'totals':{'rows':sum(x['rows'] for x in datasets),'files':sum(x['available'] for x in datasets)},'analysis_status':'Overall player statistics available; pressure model and rankings not yet validated.'}

@app.get('/api/players')
def players(kind:str='batters',search:str=''):
    if kind not in ['batters','bowlers']:raise HTTPException(400,'kind must be batters or bowlers')
    data=player_stats()[kind]
    if search:data=[p for p in data if search.casefold() in str(p['name']).casefold()]
    return {'players':data[:80],'kind':kind,'note':'Overall descriptive statistics, NOT pressure rankings. Bowling wickets are provisional delivery flags pending dismissal validation.'}

@app.get('/api/datasets/{name}/preview')
def preview(name:str):
    if name not in FILES:raise HTTPException(404,'Unknown original dataset')
    frame=pd.read_csv(DATA/f'{name}.csv',nrows=6).fillna('')
    return {'name':name,'columns':frame.columns.tolist(),'rows':frame.astype(str).to_dict('records')}

@app.post('/api/upload')
async def upload(file:UploadFile=File(...)):
    name=Path(file.filename or '').name
    if not re.fullmatch(r'[\w.() -]+',name) or Path(name).suffix.lower() not in ('.csv','.xlsx'):
        raise HTTPException(400,'Choose a CSV or XLSX with a simple filename')
    raw=await file.read(50*1024*1024+1)
    if len(raw)>50*1024*1024:raise HTTPException(413,'File exceeds 50 MB')
    folder=DATA/'additional_uploads';folder.mkdir(exist_ok=True)
    target=folder/name
    if target.exists():raise HTTPException(409,'A dataset with this filename already exists. Rename the new file first.')
    target.write_bytes(raw)
    try:
        frame=pd.read_csv(target,nrows=5) if name.lower().endswith('.csv') else pd.read_excel(target,nrows=5)
        if len(frame.columns)>300:raise ValueError('Too many columns (maximum 300)')
        rows=frame.where(pd.notnull(frame),'').astype(str).to_dict('records')
    except Exception as e:
        target.unlink(missing_ok=True);raise HTTPException(400,f'Could not parse dataset: {str(e)[:160]}')
    return {'filename':name,'columns':list(map(str,frame.columns)),'rows':rows,'note':'Saved locally and inspected. Not merged into official rankings. Cloud persistence requires deployment storage.'}


@lru_cache(maxsize=1)
def chase_situations():
    """Historical second-innings situations; first-innings batter runs are a target proxy.
    Extras are not reliably represented as delivery totals in this source, so never
    treat proxy target or remaining runs as official scorecard values.
    """
    cols=['match_id','innings','innings_ball_num','striker_id','striker_name',
          'bowler_id','bowler_name','runs','is_wicket','extras_type']
    df=pd.read_csv(DATA/'ball_by_ball.csv',usecols=cols,low_memory=False)
    df['runs']=pd.to_numeric(df.runs,errors='coerce').fillna(0)
    df['is_wicket']=df.is_wicket.astype(str).str.lower().eq('true').astype(int)
    df['legal']=~df.extras_type.fillna('').astype(str).str.lower().str.contains('wide|overstep|no.ball')
    first=df[df.innings==1].groupby('match_id').runs.sum().add(1).rename('target_proxy')
    second=df[df.innings==2].copy().sort_values(['match_id','innings_ball_num'])
    second=second.join(first,on='match_id').dropna(subset=['target_proxy'])
    second['runs_before']=second.groupby('match_id').runs.cumsum()-second.runs
    second['wickets_before']=second.groupby('match_id').is_wicket.cumsum()-second.is_wicket
    second['legal_before']=second.groupby('match_id').legal.cumsum()-second.legal.astype(int)
    second['runs_needed_proxy']=(second.target_proxy-second.runs_before).clip(lower=0)
    second['balls_left_proxy']=(120-second.legal_before).clip(lower=0)
    second['wickets_left']=(10-second.wickets_before).clip(lower=0)
    second=second[(second.balls_left_proxy>0)&(second.runs_needed_proxy>0)&(second.wickets_left>0)]
    match=pd.read_csv(DATA/'matches.csv',usecols=['match_id','match_date','home_team','away_team'])
    second=second.merge(match,on='match_id',how='left')
    return second

@app.get('/api/pressure/scenarios')
def pressure_scenarios(runs:int=48,balls:int=24,wickets:int=5,limit:int=12):
    if not (1<=runs<=200 and 1<=balls<=120 and 1<=wickets<=10 and 1<=limit<=30):
        raise HTTPException(400,'Scenario parameters out of range')
    df=chase_situations()
    # Match by contextual similarity, not by an unvalidated pressure score.
    nearby=df[(df.runs_needed_proxy.sub(runs).abs()<=15)&
              (df.balls_left_proxy.sub(balls).abs()<=12)&
              (df.wickets_left.sub(wickets).abs()<=2)].copy()
    if nearby.empty:
        return {'matches':[], 'batters':[], 'bowlers':[], 'deliveries':0,
                'note':'No sufficiently similar historical situations were found with the current proxy. Try adjusting the sliders.',
                'method':'Approximate historical chase-context matching; NOT a validated pressure rating.'}
    nearby['distance']=(nearby.runs_needed_proxy.sub(runs).abs()/15+
                        nearby.balls_left_proxy.sub(balls).abs()/12+
                        nearby.wickets_left.sub(wickets).abs()/2)
    nearby=nearby.sort_values('distance').drop_duplicates(['match_id','innings_ball_num']).head(1500)
    def leaderboard(id_col,name_col):
        grouped=nearby.groupby([id_col,name_col],dropna=False).agg(
            deliveries=('match_id','size'),matches=('match_id','nunique'),
            runs=('runs','sum'),wicket_flags=('is_wicket','sum'),
            proximity=('distance','mean')).reset_index()
        grouped=grouped.sort_values(['deliveries','matches'],ascending=False).head(limit)
        return [{'id':str(r[id_col]),'name':str(r[name_col]),'deliveries':int(r.deliveries),
                 'matches':int(r.matches),'runs':int(r.runs),
                 'wicket_flags':int(r.wicket_flags),
                 'proximity':round(float(r.proximity),2)} for _,r in grouped.iterrows()]
    examples=nearby.drop_duplicates('match_id').head(6)
    return {'matches':[{'match_id':int(r.match_id),'date':str(r.match_date),
                        'teams':f'{r.home_team} vs {r.away_team}',
                        'runs_needed_proxy':int(r.runs_needed_proxy),
                        'balls_left_proxy':int(r.balls_left_proxy),
                        'wickets_left':int(r.wickets_left)} for _,r in examples.iterrows()],
            'batters':leaderboard('striker_id','striker_name'),
            'bowlers':leaderboard('bowler_id','bowler_name'),
            'deliveries':len(nearby),
            'note':'Players appeared in approximately similar second-innings contexts; this is not a ranking of performance under pressure. Runs needed are estimated from first-innings batter runs only, excluding unverified extras. Wicket flags are unadjusted.',
            'method':'Within ±15 estimated runs, ±12 legal balls and ±2 wickets of the selected scenario; closest 1,500 distinct deliveries.'}

from .pressure_engine import ratings, summary

@app.get('/api/pressure/rankings')
def pressure_rankings(kind:str='batters',limit:int=10):
    if kind not in ('batters','bowlers') or not 1<=limit<=1000:
        raise HTTPException(400,'Invalid ranking parameters')
    return {'kind':kind,'players':ratings(kind)[:limit],'total_eligible':len(ratings(kind)),
            'provisional':True,'eligibility_verified':False,
            'note':'Exploratory statistical rankings. Uncapped/IPL eligibility is NOT established from these files. Do not submit without independent verification.',
            'unit':'batter strike rate' if kind=='batters' else 'bowler-attributed wickets per 100 legal balls'}

@app.get('/api/pressure/rankings/export')
def export_pressure_rankings():
    """Export every player meeting the scoring thresholds, not just the top 10."""
    columns = [
        'category', 'rank', 'id', 'name', 'score', 'score_v04',
        'pressure_rate', 'baseline_rate', 'adjusted_pressure_rate',
        'uplift', 'pressure_balls', 'baseline_balls',
        'pressure_matches', 'baseline_matches', 'pressure_runs',
        'pressure_wickets', 'confidence', 'consistency',
        'consistent_matches', 'evaluated_matches', 'consistency_adjustment'
    ]
    output = io.StringIO(newline='')
    writer = csv.DictWriter(output, fieldnames=columns, extrasaction='ignore')
    writer.writeheader()
    for kind in ('batters', 'bowlers'):
        for rank, player in enumerate(ratings(kind), start=1):
            writer.writerow({'category': kind, 'rank': rank, **player})
    output.seek(0)
    return StreamingResponse(
        iter([output.getvalue()]),
        media_type='text/csv; charset=utf-8',
        headers={'Content-Disposition': 'attachment; filename="all-pressure-rankings.csv"'}
    )

@app.get('/api/pressure/methodology')
def pressure_methodology():
    return {'summary':summary(),'threshold':55,
            'batting':'Pressure strike rate vs player non-pressure strike rate; shrinkage toward player baseline (60 prior balls).',
            'bowling':'Bowler-attributed wickets per 100 legal deliveries vs non-pressure baseline; shrinkage (90 prior balls).',
            'minimums':'At least 30 pressure balls over 3 matches and 60 normal balls over 3 matches.',
            'limitations':['No reliable extras-run quantities: chase targets use batter-run proxy, and bowler economy is unavailable.',
             'Revised-overs/target matches excluded; standard 120-ball innings assumed.',
             'First innings pressure uses stage and wickets; second innings uses estimated required rate, stage and wickets.',
             'Not a validated predictive model; sensitivity and held-out evaluation are pending.',
             'Player IPL uncapped eligibility cannot be verified from provided files.']}
