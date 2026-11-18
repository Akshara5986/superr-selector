"""Reproducible exploratory pressure ratings. NOT verified scorecards or IPL uncapped eligibility."""
from functools import lru_cache
from pathlib import Path
import numpy as np
import pandas as pd

DATA=Path(__file__).resolve().parent.parent/'data'
BOWLER_DISMISSALS={'Caught','Bowled','LBW','Stump','Hit Wicket'}

@lru_cache(maxsize=1)
def prepared():
    b=pd.read_csv(DATA/'ball_by_ball.csv',low_memory=False)
    m=pd.read_csv(DATA/'matches.csv')
    b=b[b.innings.isin([1,2])].copy()
    b['runs']=pd.to_numeric(b.runs,errors='coerce').fillna(0)
    # Dataset extras1_runs is an encoded extras type, not an extras run count.
    ext=b.extras_type.fillna('').astype(str).str.lower()
    b['legal']=~ext.isin(['wide','overstep'])
    b['batter_ball']=~ext.eq('wide')
    b['wicket']=b.is_wicket.astype(str).str.lower().eq('true').astype(int)
    b['bowler_wicket']=b.dismissal_type.fillna('').isin(BOWLER_DISMISSALS).astype(int)
    b=b.sort_values(['match_id','innings','innings_ball_num','overid','over_ball_num']).reset_index(drop=True)
    grp=b.groupby(['match_id','innings'],sort=False)
    b['legal_before']=grp.legal.cumsum()-b.legal.astype(int)
    b['wickets_before']=grp.wicket.cumsum()-b.wicket
    b['batter_runs_before']=grp.runs.cumsum()-b.runs
    b['balls_left']=(120-b.legal_before).clip(lower=0)
    b['wickets_left']=(10-b.wickets_before).clip(lower=0)
    # No verified extras-run amount. Use first innings BATTER runs as a chase proxy only.
    first=b[b.innings==1].groupby('match_id').runs.sum().add(1).rename('target_proxy')
    b=b.join(first,on='match_id')
    b['runs_needed_proxy']=(b.target_proxy-b.batter_runs_before).clip(lower=0)
    b['rrr_proxy']=np.where(b.balls_left>0,b.runs_needed_proxy*6/b.balls_left,0)
    b['pressure_score']=0.0
    # First innings: late-innings and wicket-loss context. Second innings: chase difficulty.
    first_mask=b.innings.eq(1)
    b.loc[first_mask,'pressure_score']=(np.clip((b.loc[first_mask,'legal_before']-72)/48,0,1)*45 +
       np.clip((b.loc[first_mask,'wickets_before']-3)/5,0,1)*45 +
       np.where(b.loc[first_mask,'legal_before']>=102,10,0))
    chase=~first_mask
    b.loc[chase,'pressure_score']=(np.clip((b.loc[chase,'rrr_proxy']-7)/9,0,1)*60+
       np.clip((b.loc[chase,'wickets_before']-2)/6,0,1)*25+
       np.where(b.loc[chase,'legal_before']>=90,15,0))
    b['pressure']=b.pressure_score>=55
    b['phase']=np.select([b.legal_before<36,b.legal_before<90],['Powerplay','Middle'],default='Death')
    b=b.merge(m[['match_id','comp_id','match_date','first_inn_revised_overs','second_inn_revised_overs','revised_target']],on='match_id',how='left')
    # Exclude revised matches because 120-ball assumption is not appropriate.
    b=b[b.first_inn_revised_overs.isna()&b.second_inn_revised_overs.isna()&b.revised_target.isna()]
    # Chase proxy cannot be trusted when batter-run proxy already met.
    b=b[~((b.innings==2)&((b.runs_needed_proxy<=0)|(b.balls_left<=0)))]
    return b

def group_metrics(frame,kind,pressure):
    d=frame[frame.pressure==pressure].copy()
    if kind=='batters':
        d=d[d.batter_ball]
        group=d.groupby(['striker_id','striker_name'],dropna=False).agg(runs=('runs','sum'),balls=('batter_ball','sum'),matches=('match_id','nunique'),boundaries=('is_four','sum')).reset_index()
        group['rate']=100*group.runs/group.balls.clip(lower=1)
        return group.rename(columns={'striker_id':'id','striker_name':'name'})
    d=d[d.legal]
    # Without reliable extras run amounts, bowler economy is not recoverable.
    group=d.groupby(['bowler_id','bowler_name'],dropna=False).agg(runs=('runs','sum'),balls=('legal','sum'),matches=('match_id','nunique'),wickets=('bowler_wicket','sum')).reset_index()
    group['rate']=100*group.wickets/group.balls.clip(lower=1)
    return group.rename(columns={'bowler_id':'id','bowler_name':'name'})

@lru_cache(maxsize=2)
def ratings(kind):
    if kind not in ('batters','bowlers'):raise ValueError('Invalid kind')
    d=prepared()
    a=group_metrics(d,kind,True)
    z=group_metrics(d,kind,False)
    x=a.merge(z,on='id',how='inner',suffixes=('_pressure','_baseline'))
    x['name']=x.name_pressure.fillna(x.name_baseline)
    # Require enough deliveries and independent matches in both regimes.
    x=x[(x.balls_pressure>=30)&(x.matches_pressure>=3)&(x.balls_baseline>=60)&(x.matches_baseline>=3)].copy()
    if x.empty:return []
    # Shrink pressure rate toward player's own normal rate; avoid tiny samples dominating.
    prior=60 if kind=='batters' else 90
    x['shrunk_rate']=(x.rate_pressure*x.balls_pressure+x.rate_baseline*prior)/(x.balls_pressure+prior)
    x['uplift']=x.shrunk_rate-x.rate_baseline
    x['confidence']=x.balls_pressure/(x.balls_pressure+prior)
    # Rating rewards pressure output and positive change, penalises poor match coverage.
    base=x.shrunk_rate if kind=='batters' else x.shrunk_rate*20
    x['score_raw']=base+np.clip(x.uplift,-30,30)*(0.6 if kind=='batters' else 12)+np.minimum(x.matches_pressure,15)*0.6
    x=x.sort_values(['score_raw','matches_pressure','balls_pressure'],ascending=False)
    records=[]
    for _,r in x.iterrows():
        records.append({'id':int(r.id),'name':str(r['name']),'score':round(float(r.score_raw),2),
            'pressure_rate':round(float(r.rate_pressure),2),'baseline_rate':round(float(r.rate_baseline),2),
            'adjusted_pressure_rate':round(float(r.shrunk_rate),2),'uplift':round(float(r.uplift),2),
            'pressure_balls':int(r.balls_pressure),'baseline_balls':int(r.balls_baseline),
            'pressure_matches':int(r.matches_pressure),'baseline_matches':int(r.matches_baseline),
            'pressure_runs':int(r.runs_pressure),'pressure_wickets':int(r.get('wickets_pressure',0)),
            'confidence':round(float(r.confidence),3)})
    return records

def summary():
    d=prepared()
    return {'deliveries':len(d),'matches':int(d.match_id.nunique()),'pressure_deliveries':int(d.pressure.sum()),
            'pressure_matches':int(d[d.pressure].match_id.nunique()),'pressure_share':round(float(d.pressure.mean()*100),1),
            'batters_eligible_for_scoring':len(ratings('batters')),'bowlers_eligible_for_scoring':len(ratings('bowlers'))}
