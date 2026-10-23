"""探索: 馬ごとの通過時刻の概算が成り立つか（ゴール前3F地点のラグは実測、早い地点は位置取りから推定）。tp-77 追加2。学習には使わない。"""
import json, numpy as np, pandas as pd
T="/home/jovyan/work/keiba-vpn/data/page_reference/tables/"
cols=["race_id","date","horse_id","distance","surface","venue","field_size","finish_position","time_sec","last_3f","passing_order","lap_times"]
r=pd.concat([pd.read_parquet(f"{T}{y}/race_result_flat.parquet",columns=cols) for y in range(2020,2026)],ignore_index=True)
r=r[r.surface.isin(["芝","ダート"])]
def laps(s):
    try: v=[float(x) for x in json.loads(s)]; return v
    except Exception: return None
rl=r.drop_duplicates("race_id")[["race_id","distance","lap_times"]].copy(); rl["laps"]=rl.lap_times.map(laps)
rl=rl[[l is not None and len(l)==int(np.ceil(d/200)) and len(l)>=5 for l,d in zip(rl.laps,rl.distance)]]
rl["C_L3"]=[sum(l[:-3]) for l in rl.laps]; rl["C_fin"]=[sum(l) for l in rl.laps]
r=r.merge(rl[["race_id","C_L3","C_fin"]],on="race_id")
r["time_sec"]=pd.to_numeric(r.time_sec,errors="coerce"); r["last_3f"]=pd.to_numeric(r.last_3f,errors="coerce")
r=r[(r.time_sec>0)&r.last_3f.between(30,50)]
r["lagL3"]=(r.time_sec-r.last_3f)-r.C_L3
r["lagfin"]=r.time_sec-r.C_fin
r["fp"]=pd.to_numeric(r.finish_position,errors="coerce"); r=r[r.fp>0]
print("N",len(r)); print("lagL3 q:",r.lagL3.quantile([.01,.1,.5,.9,.99]).round(2).tolist()); print("lagfin q:",r.lagfin.quantile([.01,.1,.5,.9,.99]).round(2).tolist())
def pos(s):
    try: return [int(x) for x in str(s).split("-")]
    except Exception: return []
P=r.passing_order.map(pos); r["ncor"]=P.map(len)
r["p_first"]=[p[0] if p else np.nan for p in P]; r["p_last"]=[p[-1] if p else np.nan for p in P]
r["q_first"]=(r.p_first-1)/(r.field_size-1); r["q_last"]=(r.p_last-1)/(r.field_size-1)
r["year"]=pd.to_datetime(r.date).dt.year
r["lagL3_s"]=r.lagL3  # seconds
print("by ncor:",r.ncor.value_counts().sort_index().to_dict())
from sklearn.ensemble import HistGradientBoostingRegressor as H
def ev(sub,feats,label,name):
    sub=sub.dropna(subset=feats+[label]); tr=sub[sub.year<=2023]; te=sub[sub.year>=2024]
    m=H(max_iter=200,learning_rate=.08).fit(tr[feats],tr[label]); p=m.predict(te[feats])
    mae=np.abs(te[label]-p).mean(); base=np.abs(te[label]-tr[label].mean()).mean()
    r2=1-((te[label]-p)**2).sum()/((te[label]-te[label].mean())**2).sum()
    print(f"{name}: n={len(te)} MAE={mae:.3f}s (定数基準 {base:.3f}s) R2={r2:.3f} label_sd={te[label].std():.3f}")
base=["distance","field_size"]
for nm,f in [("last corner pos",base+["q_last","p_last"]),("first corner pos",base+["q_first","p_first"]),("first+last",base+["q_first","q_last","p_first","p_last"]),("位置なし",base)]:
    ev(r,f+[],"lagL3",f"全距離 lagL3 ← {nm}")
# 3F / 5F exact label: 1200m (L3=600m=3F), 1600m (L3=1000m=5F)
for d,nm in ((1200,"3F通過(1200m)"),(1600,"5F通過(1600m)"),(2000,"7F通過(2000m)")):
    s=r[r.distance==d]
    for fn,f in [("first corner",base+["q_first","p_first"]),("last corner",base+["q_last","p_last"]),("位置なし",base)]:
        ev(s,f,"lagL3",f"{nm} ← {fn}")
