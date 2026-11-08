"""オッズ予測: 複数の試行の単勝の log 占有率（レース内で正規化）を平均し、複勝は指定の試行のものを使う。
    python docs/operations/retrain_trials/odds_blend.py <書き出し先> <複勝の出力元> <単勝の出力元>...
"""
import os, sys, shutil
from pathlib import Path
import numpy as np, pandas as pd
DST, PL, WS = Path(sys.argv[1]), Path(sys.argv[2]), [Path(x) for x in sys.argv[3:]]
os.environ["ODDS_PRED_OUT"] = str(DST)
sys.path.insert(0, "/home/jovyan/work/keiba-vpn")
from src.scripts.data import odds_pred_eval as oe
oe.OUT = DST; DST.mkdir(parents=True, exist_ok=True)
base = pd.read_parquet(PL / "wp_raw_oof.parquet")
acc = []
for w in WS:
    r = pd.read_parquet(w / "wp_raw_oof.parquet")[oe.KEY + ["share_raw"]]
    s = r["share_raw"]; g = s.groupby(r["race_id"])
    r["ls"] = s - g.transform("max") - np.log(np.exp(s - g.transform("max")).groupby(r["race_id"]).transform("sum"))
    acc.append(r.set_index(oe.KEY)["ls"])
m = pd.concat(acc, axis=1).mean(axis=1).rename("share_new").reset_index()
out = base.merge(m, on=oe.KEY, how="left"); out["share_raw"] = out["share_new"].fillna(out["share_raw"])
out.drop(columns=["share_new"]).to_parquet(DST / "wp_raw_oof.parquet", index=False)
oe.evaluate_phase1()
