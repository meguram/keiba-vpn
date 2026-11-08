"""tp-26 走法ベクトル × レース質分析 — フルパイプライン実行スクリプト"""
import ast
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
try:
    matplotlib.rcParams['font.family'] = 'IPAexGothic'
except Exception:
    pass
import warnings
warnings.filterwarnings('ignore')

ROOT    = Path(__file__).resolve().parents[1]
LAP_DIR = ROOT / 'data/local/backup/tables_before_reexport_20261014'
LS_PATH = ROOT / 'data/local/modeling/training_pipeline/lapscores/lapscores.parquet'
OUT_DIR = ROOT / 'data/local/modeling/training_pipeline/race_quality'
OUT_DIR.mkdir(parents=True, exist_ok=True)
print('ROOT:', ROOT)

# 1. データ読み込み
YEARS = ['2020','2021','2022','2023','2024','2025']
dfs = []
for y in YEARS:
    p = LAP_DIR / y / 'race_result_flat.parquet'
    if not p.exists():
        print(f'{y}: not found'); continue
    d = pd.read_parquet(p, columns=[
        'race_id','date','venue','surface','distance','grade','race_class',
        'track_condition','field_size','finish_position',
        'horse_id','horse_name','passing_order',
        'last_3f','time_sec','lap_times',
    ])
    d['year'] = y
    dfs.append(d)

raw = pd.concat(dfs, ignore_index=True)
for c in ['last_3f','time_sec','field_size','finish_position']:
    raw[c] = pd.to_numeric(raw[c], errors='coerce')

layer_a = pd.read_parquet(
    ROOT / 'data/local/modeling/training_pipeline/layer_a.parquet',
    columns=['race_id','horse_id','sire']
)
raw = raw.merge(layer_a, on=['race_id','horse_id'], how='left')
print(f'読み込み: {len(raw):,} rows, {raw["race_id"].nunique():,} races')

# 2. ラップ展開
def safe_parse(s):
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return []
    if isinstance(s, str):
        try:
            return ast.literal_eval(s)
        except Exception:
            return []
    if isinstance(s, (list, tuple)):
        return list(s)
    return []

laps_list = [safe_parse(s) for s in raw['lap_times']]
raw['n_laps']      = [len(l) for l in laps_list]
raw['race_last3f'] = [sum(l[-3:]) if len(l) >= 3 else float('nan') for l in laps_list]
raw['lap_L1']      = [float(l[-1]) if len(l) >= 1 else float('nan') for l in laps_list]
raw['lap_L2']      = [float(l[-2]) if len(l) >= 2 else float('nan') for l in laps_list]
raw['lap_L3']      = [float(l[-3]) if len(l) >= 3 else float('nan') for l in laps_list]
scale = raw['last_3f'] / raw['race_last3f'].replace(0, float('nan'))
raw['est_L3'] = raw['lap_L3'] * scale
raw['est_L2'] = raw['lap_L2'] * scale
raw['est_L1'] = raw['lap_L1'] * scale

# 3. split_ratio (真の個別前後半比)
raw['front_n_f']      = (raw['n_laps'] - 3).clip(lower=1)
raw['first_half_sec'] = raw['time_sec'] - raw['last_3f']
raw['pace_front_pf']  = raw['first_half_sec'] / raw['front_n_f']
raw['pace_back_pf']   = raw['last_3f'] / 3
raw['split_ratio']    = raw['pace_back_pf'] / raw['pace_front_pf'].replace(0, float('nan'))

# 4. コーナー通過順
def parse_passing(s, field_size):
    if not isinstance(s, str) or not s or pd.isna(field_size) or field_size <= 1:
        return (np.nan,) * 4
    try:
        corners = [int(x) for x in s.split('-')]
    except Exception:
        return (np.nan,) * 4
    if not corners:
        return (np.nan,) * 4
    fs = float(field_size)
    C1 = corners[0]; C_last = corners[-1]
    front_ratio = C1 / fs
    shift_ratio = (C_last - C1) / fs
    pos_consistency = float(np.std(corners)) / fs if len(corners) > 1 else 0.0
    return (front_ratio, shift_ratio, pos_consistency, len(corners))

parsed = [parse_passing(po, fs) for po, fs in zip(raw['passing_order'], raw['field_size'])]
p_df = pd.DataFrame(parsed, columns=['front_ratio','shift_ratio','pos_consistency','n_corners'])
raw = pd.concat([raw.reset_index(drop=True), p_df], axis=1)
raw['late_kick'] = (raw['est_L2'] - raw['est_L1']) / raw['est_L2'].replace(0, float('nan'))

ls = pd.read_parquet(LS_PATH, columns=['race_id','horse_id','ts_fastest_f','gc_raw'])
raw = raw.merge(ls, on=['race_id','horse_id'], how='left')
print('走法ベクトル列完成')

# 5. フィルタ
VEC_COLS = ['ts_fastest_f','gc_raw','split_ratio','front_ratio','shift_ratio','late_kick']
valid = (
    raw[VEC_COLS].notna().all(axis=1) &
    (raw['field_size'] >= 4) &
    raw['finish_position'].notna() &
    (raw['n_laps'] >= 5)
)
df = raw[valid].copy().reset_index(drop=True)
print(f'有効行: {len(df):,} ({valid.mean():.1%})')

# 6. レース質ベクトル (late_kick除外)
RQ_FEATS = [c for c in VEC_COLS if c != 'late_kick']
RQ_COLS  = [f'rq_{f}' for f in RQ_FEATS]

print('レース質ベクトルを計算中 (vectorized Spearman)...')

# Spearman r = Pearson r of ranks → groupby.transform で完全ベクトル化
fin_rank = df.groupby('race_id')['finish_position'].rank(method='average')
fin_mean = fin_rank.groupby(df['race_id']).transform('mean')
fin_std  = fin_rank.groupby(df['race_id']).transform('std')

rq_cols_data = {}
for feat in RQ_FEATS:
    feat_rank = df.groupby('race_id')[feat].rank(method='average')
    feat_mean = feat_rank.groupby(df['race_id']).transform('mean')
    feat_std  = feat_rank.groupby(df['race_id']).transform('std')
    cov_num   = ((feat_rank - feat_mean) * (fin_rank - fin_mean)).groupby(df['race_id']).transform('sum')
    n_grp     = df.groupby('race_id')['race_id'].transform('count')
    denom     = (n_grp - 1) * feat_std * fin_std
    rq_cols_data[f'rq_{feat}'] = (cov_num / denom.replace(0, float('nan'))).values

rq_df = pd.DataFrame(rq_cols_data)
rq_df['race_id'] = df['race_id'].values
rq = rq_df.drop_duplicates('race_id').reset_index(drop=True)

# グループサイズ < 5 は除外
n_grp_per_race = df.groupby('race_id')['race_id'].transform('count')
valid_races = df.loc[n_grp_per_race >= 5, 'race_id'].unique()
rq = rq[rq['race_id'].isin(valid_races)]

rq_valid = rq.dropna(subset=RQ_COLS, how='any')
print(f'レース質ベクトル: {len(rq_valid):,} races (有効: {len(rq_valid)/len(rq):.1%})')

# 7. レースメタ付加
race_meta = df[['race_id','date','year','venue','surface','distance',
                'grade','race_class','track_condition']].drop_duplicates('race_id')
rq_meta = rq_valid.merge(race_meta, on='race_id', how='left')

# 8. KMeans K=5
X = rq_meta[RQ_COLS].dropna().values
sc_idx = rq_meta[RQ_COLS].dropna().index
scaler = StandardScaler()
X_sc   = scaler.fit_transform(X)
K = 5
km = KMeans(n_clusters=K, random_state=42, n_init=20)
labels = km.fit_predict(X_sc)
rq_clean = rq_meta.loc[sc_idx].copy()
rq_clean['rq_type'] = labels

centers = pd.DataFrame(scaler.inverse_transform(km.cluster_centers_), columns=RQ_COLS)
centers.index.name = 'type'

def label_type(row):
    tags = []
    if row['rq_ts_fastest_f']  < -0.20: tags.append('TS要求')
    if row['rq_split_ratio']   < -0.10: tags.append('差し有利')
    if row['rq_split_ratio']   >  0.10: tags.append('前残り')
    if row['rq_front_ratio']   < -0.10: tags.append('前込み')
    if row['rq_gc_raw']        < -0.15: tags.append('GC要求')
    return ' + '.join(tags) if tags else '標準型'

centers['レース質タイプ'] = centers.apply(label_type, axis=1)
print('=== クラスタ重心 (K=5) ===')
print(centers[['レース質タイプ'] + RQ_COLS].round(3).to_string())
rq_meta = rq_meta.merge(rq_clean[['race_id','rq_type']], on='race_id', how='left')

# 9. コース別統計
cond_key   = ['venue','distance','surface','track_condition']
course_key = ['venue','distance','surface']

def build_course_stats(df_rq, key):
    grp = df_rq.groupby(key)
    mean = grp[RQ_COLS].mean().round(4)
    std  = grp[RQ_COLS].std().round(4).add_suffix('_std')
    n    = grp['race_id'].count().rename('n_races')
    dom  = grp['rq_type'].agg(lambda x: x.value_counts().idxmax()).rename('dominant_rq_type')
    return pd.concat([mean, std, n, dom], axis=1).reset_index()

cond_stats   = build_course_stats(rq_meta, cond_key)
course_stats = build_course_stats(rq_meta, course_key)
cond_stats   = cond_stats[cond_stats['n_races'] >= 3]
course_stats = course_stats[course_stats['n_races'] >= 5]
print(f'条件別: {len(cond_stats):,}件  コース別: {len(course_stats):,}件')

# 10. 種牡馬プロファイル
sire_df = df[df['sire'].notna()].copy()
sire_profile = sire_df.groupby('sire').agg(
    n_runs        =('horse_id', 'count'),
    n_horses      =('horse_id', 'nunique'),
    ts_avg        =('ts_fastest_f', 'mean'),
    split_avg     =('split_ratio', 'mean'),
    front_avg     =('front_ratio', 'mean'),
    shift_avg     =('shift_ratio', 'mean'),
    late_kick_avg =('late_kick', 'mean'),
    gc_avg        =('gc_raw', 'mean'),
    win_rate      =('finish_position', lambda x: (x==1).mean()),
    top3_rate     =('finish_position', lambda x: (x<=3).mean()),
).reset_index()
sire_profile = sire_profile[sire_profile['n_runs'] >= 50].copy()

sire_rq = sire_df.merge(rq_meta[['race_id','rq_type']], on='race_id', how='left')
sire_rq_perf = sire_rq.groupby(['sire','rq_type']).agg(
    n=('finish_position','count'),
    top3_rate=('finish_position', lambda x: (x<=3).mean())
).reset_index()

sire_best_rq = sire_rq_perf[sire_rq_perf['n'] >= 10].copy()
sire_best_rq = sire_best_rq.loc[
    sire_best_rq.groupby('sire')['top3_rate'].idxmax()
].rename(columns={'rq_type':'best_rq_type','top3_rate':'best_rq_top3','n':'best_rq_n'})
sire_best_rq['best_rq_label'] = sire_best_rq['best_rq_type'].map(
    {k: centers.loc[k,'レース質タイプ'] for k in range(K)}
)
sire_profile = sire_profile.merge(
    sire_best_rq[['sire','best_rq_type','best_rq_top3','best_rq_label']],
    on='sire', how='left'
)
print(f'種牡馬プロファイル: {len(sire_profile):,} 頭')

# 11. 馬の走法プロファイル
prof_df = df[['race_id','horse_id','date','year'] + VEC_COLS].copy()
prof_df['date'] = pd.to_datetime(prof_df['date'])
prof_df = prof_df.sort_values(['horse_id','date']).reset_index(drop=True)
for col in VEC_COLS:
    for w in [3, 5]:
        prof_df[f'{col}_avg{w}'] = (
            prof_df.groupby('horse_id')[col]
            .transform(lambda x: x.shift(1).rolling(w, min_periods=1).mean())
        )
    prof_df[f'{col}_last1'] = prof_df.groupby('horse_id')[col].transform(lambda x: x.shift(1))
prof_feat_cols = [f'{c}_{s}' for c in VEC_COLS for s in ['avg3','avg5','last1']]
print(f'プロファイル特徴量: {len(prof_feat_cols)}列')

# 12. 推定レース質 + 相性スコア
race_info = df[['race_id','venue','distance','surface','track_condition']].drop_duplicates('race_id')
exp_cols  = {c: f'exp_{c}' for c in RQ_COLS}

merged = race_info.merge(
    cond_stats[cond_key + RQ_COLS].rename(columns=exp_cols), on=cond_key, how='left'
)
miss = merged[f'exp_{RQ_COLS[0]}'].isna()
backup = race_info.merge(
    course_stats[course_key + RQ_COLS].rename(columns=exp_cols), on=course_key, how='left'
)
for c in RQ_COLS:
    ec = f'exp_{c}'
    merged.loc[miss, ec] = backup.loc[miss, ec].values
race_exp = merged
exp_rq_cols = [f'exp_{c}' for c in RQ_COLS]
print(f'推定レース質マッチ率: {race_exp[exp_rq_cols[0]].notna().mean():.1%}')

compat = prof_df[['race_id','horse_id','date','year'] + [f'{c}_avg3' for c in VEC_COLS]].copy()
compat = compat.merge(race_exp[['race_id'] + exp_rq_cols], on='race_id', how='left')
compat = compat.merge(df[['race_id','horse_id','finish_position']].drop_duplicates(), on=['race_id','horse_id'], how='left')

horse_sign = np.array([-1, 1, -1, -1, -1])
exp_sign   = np.array([-1, -1, -1, -1, -1])
hv_cols = [f'{c}_avg3' for c in RQ_FEATS]
H = compat[hv_cols].fillna(compat[hv_cols].median()).values * horse_sign
E = compat[exp_rq_cols].fillna(0).values                    * exp_sign
H_z = (H - np.nanmean(H, axis=0)) / (np.nanstd(H, axis=0) + 1e-9)
E_z = (E - np.nanmean(E, axis=0)) / (np.nanstd(E, axis=0) + 1e-9)
compat['rq_compat'] = (H_z * E_z).sum(axis=1)
mu = compat.groupby('race_id')['rq_compat'].transform('mean')
sd = compat.groupby('race_id')['rq_compat'].transform('std').fillna(1).replace(0, 1)
compat['rq_compat_dev'] = (compat['rq_compat'] - mu) / sd * 10 + 50

ev = compat.dropna(subset=['finish_position','rq_compat_dev'])
r, p = spearmanr(ev['rq_compat_dev'], ev['finish_position'])
print(f'相性スコア Spearman vs 着順: r={r:.4f}  p={p:.2e}')

# 13. 保存
rq_meta.to_parquet(OUT_DIR / 'race_quality_vec.parquet', index=False)
course_stats.to_parquet(OUT_DIR / 'course_rq.parquet', index=False)
cond_stats.to_parquet(OUT_DIR / 'cond_rq.parquet', index=False)
sire_profile.to_parquet(OUT_DIR / 'sire_profile.parquet', index=False)

compat_save = compat[['race_id','horse_id','rq_compat','rq_compat_dev']]
final = prof_df[['race_id','horse_id','date','year'] + prof_feat_cols].merge(
    compat_save, on=['race_id','horse_id'], how='left'
)
final.to_parquet(OUT_DIR / 'horse_rq_features.parquet', index=False)

print('\n=== 保存完了 ===')
for f in sorted(OUT_DIR.glob('*.parquet')):
    print(f'  {f.name}: {f.stat().st_size/1e6:.1f} MB')
print(f'\nレース質ベクトル: {len(rq_meta):,} レース  クラスタ数: {K}')
print(f'コース別統計: {len(cond_stats):,} 条件 / {len(course_stats):,} コース')
print(f'種牡馬プロファイル: {len(sire_profile):,} 頭')
print(f'馬の走法特徴量: {final.shape}')
print(f'相性スコア Spearman r={r:.4f}')
