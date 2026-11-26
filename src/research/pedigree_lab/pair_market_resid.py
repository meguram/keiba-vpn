"""調べ P の残りの q を除く回帰（数を見た後に足した 3 本 × 2 券種・区間なし・記述）。

    PYTHONPATH=. KEIBA_SESSION=pedigree-lab nice -n 19 python -m src.research.pedigree_lab.pair_market_resid

升（人気薄の升 × 相手の人気の順 × 馬単の向き）の中で平均を引き、log(m/H) を帯・log q の 2 次・log p_A に回帰する。
"""
import json
import numpy as np, pandas as pd
from src.research.pedigree_lab import pair_market as pm, data
X = pm.horses(); X['band'] = pm.bands(X, X.z.to_numpy()); X['hrow'] = np.arange(len(X))
dev = set(X.race_id)
out = {}
for ty in ('umaren', 'umatan'):
    F = pd.concat([pd.read_parquet(pm.ROOT / str(y) / 'race_pair_odds_flat.parquet', columns=['race_id', 'odds_type', 'pair', 'odds'], filters=[('odds_type', '==', ty)]) for y in range(2020, 2025)])
    F['race_id'] = F.race_id.astype(str); F = F[F.race_id.isin(dev)]
    pr = F.pair.astype(str).str.strip('[]').str.split(',', expand=True)
    F = F.assign(a=pd.to_numeric(pr[0]), b=pd.to_numeric(pr[1]), o=pd.to_numeric(F.odds, errors='coerce')); F = F[F.o > 0]
    H = X[['race_id', 'hn', 'q', 'pop', 'hrow']]
    F = F.merge(H.rename(columns={'hn': 'a', 'q': 'qa', 'pop': 'pa', 'hrow': 'ra'}), on=['race_id', 'a']).merge(H.rename(columns={'hn': 'b', 'q': 'qb', 'pop': 'pb', 'hrow': 'rb'}), on=['race_id', 'b'])
    F['m'] = (1 / F.o) / (1 / F.o).groupby(F.race_id).transform('sum')
    hab = F.qa * F.qb / (1 - F.qa); hba = F.qa * F.qb / (1 - F.qb)
    F['H'] = hab if ty == 'umatan' else hab + hba
    F = F[((F.pa >= 6) ^ (F.pb >= 6)) & (np.minimum(F.pa, F.pb) <= 5)].copy()
    la = (F.pa >= 6).to_numpy()
    r = np.where(la, F.ra, F.rb); rp = np.where(la, F.rb, F.ra)
    F['lr'] = np.log(F.m / F.H); F['lql'] = np.log(X.q.to_numpy()[r]); F['lqp'] = np.log(X.q.to_numpy()[rp])
    F['dir'] = la.astype(float)  # 馬単: 人気薄が 1 着側
    F['band'] = X.band.to_numpy()[r]
    F['st'] = X.lsb.astype(str).to_numpy()[r] + '|' + np.minimum(F.pa, F.pb).astype(str) + '|' + F['dir'].astype(str)
    F['lpa'] = np.log(X.pA.to_numpy()[r])
    F['g'] = (F.band == 2).astype(float); F['b_'] = (F.band == 1).astype(float)
    F['lql2'] = F.lql ** 2; F['lqp2'] = F.lqp ** 2; F['lqlp'] = F.lql * F.lqp
    cols = ['lr', 'lql', 'lqp', 'lql2', 'lqp2', 'lqlp', 'g', 'b_', 'lpa']
    D = F[cols].sub(F.groupby('st')[cols].transform('mean'))

    def ols(xs):
        A = D[xs].to_numpy(); c = np.linalg.lstsq(A, D['lr'].to_numpy(), rcond=None)[0]
        return {k: round(float(v), 4) for k, v in zip(xs, c)}
    out[ty] = {'帯だけ（升の中）': ols(['g', 'b_']),
               '帯 + 人気薄と相手の log q の 2 次': ols(['g', 'b_', 'lql', 'lqp', 'lql2', 'lqp2', 'lqlp']),
               'log p_A + log q の 2 次（連続）': ols(['lpa', 'lql', 'lqp', 'lql2', 'lqp2', 'lqlp'])}
    print(ty, json.dumps(out[ty], ensure_ascii=False), flush=True)
(data.OUT / 'pair_market_resid.json').write_text(json.dumps(out, ensure_ascii=False, indent=1))
