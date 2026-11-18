"""SupeRR Selector v0.5: exploratory, reproducible pressure reliability ratings.

IMPORTANT: Chase targets use batter-run proxies, not official scores. Scores are
heuristic and must not be represented as validated predictions or eligibility checks.
"""
from functools import lru_cache
from pathlib import Path
import numpy as np
import pandas as pd

DATA = Path(__file__).resolve().parent.parent / 'data'
BOWLER_DISMISSALS = {'caught', 'bowled', 'lbw', 'stump', 'stumped', 'hit wicket'}


def _legal_delivery(extras):
    e = extras.fillna('').astype(str).str.strip().str.lower()
    return ~e.isin(['wide', 'overstep', 'no ball', 'noball', 'no-ball'])


@lru_cache(maxsize=1)
def prepared():
    b = pd.read_csv(DATA / 'ball_by_ball.csv', low_memory=False)
    m = pd.read_csv(DATA / 'matches.csv', low_memory=False)
    b = b[b.innings.isin([1, 2])].copy()
    b['runs'] = pd.to_numeric(b.runs, errors='coerce').fillna(0)
    b['legal'] = _legal_delivery(b.extras_type)
    ext = b.extras_type.fillna('').astype(str).str.strip().str.lower()
    b['batter_ball'] = ~ext.eq('wide')
    b['wicket'] = b.is_wicket.astype(str).str.strip().str.lower().isin(['true', '1']).astype(int)
    dismissal = b.dismissal_type.fillna('').astype(str).str.strip().str.lower()
    b['bowler_wicket'] = (b['wicket'].eq(1) & dismissal.isin(BOWLER_DISMISSALS)).astype(int)
    b = b.sort_values(['match_id', 'innings', 'innings_ball_num', 'overid', 'over_ball_num'], kind='stable').reset_index(drop=True)
    grp = b.groupby(['match_id', 'innings'], sort=False)
    b['legal_before'] = grp.legal.cumsum() - b.legal.astype(int)
    b['wickets_before'] = grp.wicket.cumsum() - b.wicket
    b['batter_runs_before'] = grp.runs.cumsum() - b.runs
    b['balls_left'] = (120 - b.legal_before).clip(lower=0)
    b['wickets_left'] = (10 - b.wickets_before).clip(lower=0)
    first = b[b.innings.eq(1)].groupby('match_id').runs.sum().add(1).rename('target_proxy')
    b = b.join(first, on='match_id')
    b['runs_needed_proxy'] = (b.target_proxy - b.batter_runs_before).clip(lower=0)
    b['rrr_proxy'] = np.where(b.balls_left.gt(0), b.runs_needed_proxy * 6 / b.balls_left.clip(lower=1), 0)
    b['pressure_score'] = 0.0
    first_mask = b.innings.eq(1)
    b.loc[first_mask, 'pressure_score'] = (
        np.clip((b.loc[first_mask, 'legal_before'] - 72) / 48, 0, 1) * 45
        + np.clip((b.loc[first_mask, 'wickets_before'] - 3) / 5, 0, 1) * 45
        + np.where(b.loc[first_mask, 'legal_before'] >= 102, 10, 0)
    )
    chase = ~first_mask
    b.loc[chase, 'pressure_score'] = (
        np.clip((b.loc[chase, 'rrr_proxy'] - 7) / 9, 0, 1) * 60
        + np.clip((b.loc[chase, 'wickets_before'] - 2) / 6, 0, 1) * 25
        + np.where(b.loc[chase, 'legal_before'] >= 90, 15, 0)
    )
    b['phase'] = np.select([b.legal_before < 36, b.legal_before < 90], ['Powerplay', 'Middle'], default='Death')
    b = b.merge(m[['match_id', 'comp_id', 'match_date', 'first_inn_revised_overs', 'second_inn_revised_overs', 'revised_target']], on='match_id', how='left', validate='many_to_one')
    b = b[b.first_inn_revised_overs.isna() & b.second_inn_revised_overs.isna() & b.revised_target.isna()]
    b = b[~(b.innings.eq(2) & (b.runs_needed_proxy.le(0) | b.balls_left.le(0)))]
    b = b[b.balls_left.gt(0) & b.wickets_left.gt(0)].copy()
    b['pressure'] = b.pressure_score.ge(55)
    return b


def group_metrics(frame, kind, pressure):
    d = frame[frame.pressure.eq(pressure)].copy()
    if kind == 'batters':
        d = d[d.batter_ball]
        group = d.groupby(['striker_id', 'striker_name'], dropna=False).agg(
            runs=('runs', 'sum'), balls=('batter_ball', 'sum'), matches=('match_id', 'nunique'), boundaries=('is_four', 'sum')
        ).reset_index()
        group['rate'] = 100 * group.runs / group.balls.clip(lower=1)
        return group.rename(columns={'striker_id': 'id', 'striker_name': 'name'})
    d = d[d.legal]
    group = d.groupby(['bowler_id', 'bowler_name'], dropna=False).agg(
        runs=('runs', 'sum'), balls=('legal', 'sum'), matches=('match_id', 'nunique'), wickets=('bowler_wicket', 'sum')
    ).reset_index()
    group['rate'] = 100 * group.wickets / group.balls.clip(lower=1)
    return group.rename(columns={'bowler_id': 'id', 'bowler_name': 'name'})


def _match_consistency(frame, kind):
    """Fraction of pressure matches where player's rate meets own non-pressure baseline.

    Match-level rates with very few balls are noisy, so matches with fewer than
    six pressure balls are excluded from the consistency fraction.
    """
    d = frame[frame.pressure].copy()
    if kind == 'batters':
        d = d[d.batter_ball]
        per = d.groupby(['striker_id', 'match_id']).agg(balls=('batter_ball', 'sum'), successes=('runs', 'sum')).reset_index().rename(columns={'striker_id': 'id'})
    else:
        d = d[d.legal]
        per = d.groupby(['bowler_id', 'match_id']).agg(balls=('legal', 'sum'), successes=('bowler_wicket', 'sum')).reset_index().rename(columns={'bowler_id': 'id'})
    per = per[per.balls >= 6].copy()
    per['match_rate'] = 100 * per.successes / per.balls
    return per


@lru_cache(maxsize=16)
def ratings(kind, min_pressure_balls=30, min_pressure_matches=3, threshold=55):
    if kind not in ('batters', 'bowlers'):
        raise ValueError('Invalid kind')
    if not 0 <= threshold <= 100 or min_pressure_balls < 1 or min_pressure_matches < 1:
        raise ValueError('Invalid rating settings')
    d = prepared()
    if threshold != 55:
        d = d.copy()
        d['pressure'] = d.pressure_score.ge(threshold)
    a = group_metrics(d, kind, True)
    z = group_metrics(d, kind, False)
    x = a.merge(z, on='id', how='inner', suffixes=('_pressure', '_baseline'), validate='one_to_one')
    x['name'] = x.name_pressure.fillna(x.name_baseline)
    x = x[(x.balls_pressure >= min_pressure_balls) & (x.matches_pressure >= min_pressure_matches) & (x.balls_baseline >= 60) & (x.matches_baseline >= 3)].copy()
    if x.empty:
        return []
    prior = 60 if kind == 'batters' else 90
    x['shrunk_rate'] = (x.rate_pressure * x.balls_pressure + x.rate_baseline * prior) / (x.balls_pressure + prior)
    x['uplift'] = x.shrunk_rate - x.rate_baseline
    x['confidence'] = x.balls_pressure / (x.balls_pressure + prior)
    # Preserve the v0.4 scoring formula as a transparent baseline.
    base = x.shrunk_rate if kind == 'batters' else x.shrunk_rate * 20
    x['score_v04'] = base + np.clip(x.uplift, -30, 30) * (0.6 if kind == 'batters' else 12) + np.minimum(x.matches_pressure, 15) * 0.6
    per = _match_consistency(d, kind)
    if not per.empty:
        per = per.merge(x[['id', 'rate_baseline']], on='id', how='inner')
        per['meets_baseline'] = per.match_rate >= per.rate_baseline
        c = per.groupby('id').agg(consistent_matches=('meets_baseline', 'sum'), evaluated_matches=('match_id', 'nunique')).reset_index()
        x = x.merge(c, on='id', how='left')
    else:
        x['consistent_matches'] = 0
        x['evaluated_matches'] = 0
    x[['consistent_matches', 'evaluated_matches']] = x[['consistent_matches', 'evaluated_matches']].fillna(0)
    # Bayesian smoothing avoids interpreting 1/1 as perfect consistency.
    x['consistency'] = (x.consistent_matches + 2) / (x.evaluated_matches + 4)
    # v0.5: a bounded, clearly heuristic adjustment, NOT a trained prediction.
    x['consistency_adjustment'] = (x.consistency - 0.5) * 12
    x['score_raw'] = x.score_v04 + x.consistency_adjustment
    x = x.sort_values(['score_raw', 'matches_pressure', 'balls_pressure'], ascending=False)
    records = []
    for _, r in x.iterrows():
        records.append({
            'id': int(r.id), 'name': str(r['name']), 'score': round(float(r.score_raw), 2),
            'score_v04': round(float(r.score_v04), 2),
            'pressure_rate': round(float(r.rate_pressure), 2), 'baseline_rate': round(float(r.rate_baseline), 2),
            'adjusted_pressure_rate': round(float(r.shrunk_rate), 2), 'uplift': round(float(r.uplift), 2),
            'pressure_balls': int(r.balls_pressure), 'baseline_balls': int(r.balls_baseline),
            'pressure_matches': int(r.matches_pressure), 'baseline_matches': int(r.matches_baseline),
            'pressure_runs': int(r.runs_pressure), 'pressure_wickets': int(r.get('wickets_pressure', 0)),
            'confidence': round(float(r.confidence), 3),
            'consistency': round(float(r.consistency), 3),
            'consistent_matches': int(r.consistent_matches),
            'evaluated_matches': int(r.evaluated_matches),
            'consistency_adjustment': round(float(r.consistency_adjustment), 2),
        })
    return records


def sensitivity(kind):
    """Compare ranking IDs across stricter sample requirements and pressure cutoffs."""
    variants = [
        ('baseline', 30, 3, 55),
        ('more_pressure_balls', 100, 3, 55),
        ('more_matches', 30, 6, 55),
        ('higher_pressure_threshold', 30, 3, 65),
    ]
    out = {}
    for label, balls, matches, threshold in variants:
        result = ratings(kind, balls, matches, threshold)
        out[label] = {'eligible': len(result), 'top_10': [p['name'] for p in result[:10]], 'minimum_pressure_balls': balls, 'minimum_pressure_matches': matches, 'pressure_threshold': threshold}
    return out


def summary():
    d = prepared()
    return {
        'deliveries': len(d), 'matches': int(d.match_id.nunique()),
        'pressure_deliveries': int(d.pressure.sum()),
        'pressure_matches': int(d[d.pressure].match_id.nunique()),
        'pressure_share': round(float(d.pressure.mean() * 100), 1),
        'batters_eligible_for_scoring': len(ratings('batters')),
        'bowlers_eligible_for_scoring': len(ratings('bowlers')),
        'model_version': '0.5-exploratory',
        'limitations': 'Batter-run chase target proxy; heuristic consistency adjustment; no held-out validation.'
    }
