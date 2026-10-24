"""凍結した馬券候補（tp-59・tp-62）の前向き追跡。毎開催週の確定後に回し、結果を追記専用の履歴に積む。

python -m src.scripts.data.frozen_tracking freeze-blend                      # 単勝の混合（Benter）係数を config/frozen_tracking.json に凍結（既にあれば上書きしない）
python -m src.scripts.data.frozen_tracking track --from 20261022 [--to ...]  # 確定したレースを判定して data/local/meta/frozen_tracking/ledger.jsonl に追記
python -m src.scripts.data.frozen_tracking report                            # 履歴からスコアカード（latest.json / latest.html / reports/<時刻>/）
python -m src.scripts.data.frozen_tracking backtest --from 20260104 --to 20261018 --pred-dir <pred2026> --out <dir>   # 過去分でコードの正しさを確認（凍結の評価ではない）

規則（どちらも 2026-10-21 に凍結。config/tp59_frozen_rule.json・config/tp62_frozen_rule.json。判定は『その時点で分かること』だけ）:
  tp59: 単勝 100 円。V4 の勝率 × 単勝オッズ ≥ 1.5 かつ 騎手・調教師の両方が凍結した条件表（ID）に入る。
  tp62: 馬連。V4 の確率 + 単勝の Benter 混合 + 割引 Harville の確率 p_mix を凍結した折れ線で較正 → 較正後 EV ≥ 1.0 の買い目を EV 降順に最大 5 点。
        点数 k（1〜5）は『最小オッズ ≥ 5×k、的中確率の和 ≥ 20%』を満たすうち 1 点あたり EV が最大のもの。等額（1 点 100 円）。
オッズ時点（odds_variant）: final=確定オッズ（race_*_flat）／pre=発走前の最後のスナップショット（race_odds / race_pair_odds の phase。ホストの保存で読める時だけ）。
モデル（model）: v4=predictions_v3（凍結した規則の入力。評価の主）／v5=predictions_v5（参考。較正・条件表は V4 用なので凍結の評価には含めない）。
履歴は追記のみ（同じキーは書かない＝冪等）。凍結ファイルの内容が変わっていたら（ハッシュが履歴の最初と違う）追記を拒否する。
市場オッズは買いの判定にだけ使い、モデルの特徴量にはしない。2026 年以降の追跡データは学習・選択に使わない。
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
CFG59 = Path(os.environ.get("FT_CFG59", str(ROOT / "config/tp59_frozen_rule.json")))
CFG62 = Path(os.environ.get("FT_CFG62", str(ROOT / "config/tp62_frozen_rule.json")))
CFG_FT = Path(os.environ.get("FT_CFG", str(ROOT / "config/frozen_tracking.json")))
OUT_DIR = Path(os.environ.get("FROZEN_TRACKING_DIR", str(ROOT / "data/local/meta/frozen_tracking")))
TABLES = Path(os.environ.get("FT_TABLES", str(ROOT / "data/page_reference/tables")))
PRED_DIRS = {"v4": Path(os.environ.get("FT_PRED_V4", str(ROOT / "data/local/inference/predictions_v3"))),
             "v5": Path(os.environ.get("FT_PRED_V5", str(ROOT / "data/local/inference/predictions_v5")))}
BENTER_SRC = Path(os.environ.get("FT_BENTER_SRC", str(ROOT / "data/local/modeling/betting/tp36/final2026.json")))
TRACK_START = "20261022"
STAKE = 100.0
EV59_MIN = 1.5
K_EV, K_P = 40, 20                       # tp50_data と同じ（組合せ券: EV 上位 / 確率上位）
SLOTS, TH62, H62, M62 = 5, 1.0, 0.20, 5.0
PRE_WIN_PHASES = ("t02", "t05", "t10", "t15", "t60")
PRE_PAIR_PHASES = ("t15", "t60")
MODELS = ("v4", "v5")
VARIANTS = ("final", "pre")
RULES = ("tp59", "tp62")
REFERENCE_MODEL = {"v4": "primary", "v5": "reference"}


# ───────────────────────── 凍結ファイルとハッシュ ─────────────────────────
def sha256_file(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def current_hashes() -> dict:
    """凍結ファイルのハッシュ。frozen_tracking.json は規則に効く混合係数（blend）だけを対象にする（参考値 reference_sd の追記で変わらないように）。"""
    out = {p.name: sha256_file(p) for p in (CFG59, CFG62) if p.exists()}
    if CFG_FT.exists():
        blend = json.loads(CFG_FT.read_text(encoding="utf-8")).get("blend")
        out[CFG_FT.name] = hashlib.sha256(json.dumps(blend, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    return out


def check_hashes(out_dir: Path, write: bool = True) -> tuple[bool, list[str]]:
    """履歴（hashes.jsonl）の最初の記録と今のハッシュを比べる。戻り値: (一致, 差のあったファイル名)。初回は記録して一致扱い。"""
    out_dir.mkdir(parents=True, exist_ok=True)
    f = out_dir / "hashes.jsonl"
    cur = current_hashes()
    first = None
    if f.exists():
        for line in f.read_text(encoding="utf-8").splitlines():
            if line.strip():
                first = json.loads(line)["hashes"]
                break
    if first is None:
        if write:
            with f.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps({"at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "hashes": cur}, ensure_ascii=False) + "\n")
        return True, []
    diff = sorted(k for k in set(first) | set(cur) if first.get(k) != cur.get(k))
    return (not diff), diff


# ───────────────────────── 確率・規則 ─────────────────────────
def blend_win(p_win: np.ndarray, q_win: np.ndarray, a: float, b: float) -> np.ndarray:
    """softmax(a·log p + b·log q)（tp36_edge.blend と同じ。1 レース分）。"""
    z = a * np.log(np.clip(p_win, 1e-6, 1)) + b * np.log(np.clip(q_win, 1e-6, 1))
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


def make_race_frame(pred: pd.DataFrame, shut: pd.DataFrame, win_odds: dict[int, float], ab: tuple[float, float]) -> tuple[pd.DataFrame | None, str]:
    """1 レースの入力表（馬番昇順）。戻り値: (表, 除外理由)。除外条件は tp-36/50 の frame と同じ（全馬のオッズ・確率の合計）。"""
    g = pred.copy()
    g["horse_id"] = g["horse_id"].astype(str)
    s = shut[["horse_id", "horse_number", "jockey_id", "trainer_id"]].copy()
    s["horse_id"] = s["horse_id"].astype(str)
    g = g.merge(s, on="horse_id", how="inner").sort_values("horse_number").reset_index(drop=True)
    if len(g) < 4:
        return None, "頭数が足りない"
    g["horse_number"] = g["horse_number"].astype(int)
    g["win_odds"] = g["horse_number"].map(win_odds)
    if not (g["win_odds"] >= 1.0).all():                        # 取消・除外でオッズの無い馬がいるレースは対象外（tp-36 の frame と同じ。凍結した規則を評価した母集団と揃える）
        return None, "単勝オッズが揃わない"
    inv = (1.0 / g["win_odds"]).sum()
    if not (1.0 <= inv <= 1.6):
        return None, "単勝オッズの逆数の和が範囲外"
    ps = float(g["p_win"].sum())
    if not (0.97 < ps < 1.03):
        return None, "勝率の合計が1から離れている"
    g["q_win"] = (1.0 / g["win_odds"]) / inv
    g["p_blend"] = blend_win(g["p_win"].to_numpy(float), g["q_win"].to_numpy(float), *ab)
    return g, ""


def tp59_decide(g: pd.DataFrame, jpos: set, tpos: set) -> pd.DataFrame:
    d = g.copy()
    d["ev"] = d["p_win"] * d["win_odds"]
    d["j_pos"] = d["jockey_id"].astype(str).isin(jpos)
    d["t_pos"] = d["trainer_id"].astype(str).isin(tpos)
    return d[(d["ev"] >= EV59_MIN) & d["j_pos"] & d["t_pos"]]


def umaren_candidates(g: pd.DataFrame, pair_odds: dict[tuple, float]) -> pd.DataFrame:
    """tp50_data.build_race の馬連部分と同じ手順（p_mix = 混合の確率の割引 Harville）。戻り値: 列 a, b, odds, p_mix, p_model。"""
    from src.pipeline.inference.exotic_probabilities import fit_rank_distribution
    from src.scripts.data import tp36_single as t36

    nums = g["horse_number"].astype(int).to_numpy()
    wm, t2m, t3m = g["p_win"].to_numpy(float), g["p_top2"].to_numpy(float), g["p_top3"].to_numpy(float)
    r = (g["p_blend"] / g["p_win"].clip(lower=1e-6)).clip(0.2, 5.0).to_numpy()
    wx, t2x, t3x = g["p_blend"].to_numpy(float), np.clip(t2m * r, 1e-6, 0.99), np.clip(t3m * r, 1e-6, 0.99)
    dist_m, _ = fit_rank_distribution(wm, t2m, t3m)
    dist_x, _ = fit_rank_distribution(wx, t2x, t3x)
    idx, pm = t36.combo_probs(dist_m)["umaren"]
    px = t36.combo_probs(dist_x)["umaren"][1]
    lab = [tuple(sorted((int(nums[i]), int(nums[j])))) for i, j in idx]
    od = np.array([pair_odds.get(c, np.nan) for c in lab], dtype=float)
    ev = np.maximum(pm * od, px * od)

    def top_idx(score: np.ndarray, k: int) -> np.ndarray:
        return t36.top_k(np.where(np.isfinite(score), score, -np.inf), min(k, len(score)))

    keep = np.unique(np.concatenate([top_idx(ev, K_EV), top_idx(np.where(np.isfinite(od), pm, np.nan), K_P)]))
    keep = keep[np.isfinite(od[keep])]
    return pd.DataFrame({"a": [lab[i][0] for i in keep], "b": [lab[i][1] for i in keep], "odds": od[keep], "p_mix": px[keep], "p_model": pm[keep]})


def calibrate(p: np.ndarray, knots: dict) -> np.ndarray:
    """単調回帰（IsotonicRegression.predict と同じ: 区間は線形補間、範囲外は端の値）。"""
    return np.interp(np.asarray(p, float), np.asarray(knots["x"], float), np.asarray(knots["y"], float))


def tp62_decide(cand: pd.DataFrame, knots: dict) -> dict | None:
    """tp62_r500.build_rules_r500 の（馬連・order=ev・th=1.0・pfloor=0・kmax=5）と同じ選び方。選ばない時は None。"""
    if cand.empty:
        return None
    c = cand.copy()
    c["p_cal"] = calibrate(c["p_mix"].to_numpy(), knots)
    c["ev_cal"] = c["p_cal"] * c["odds"]
    c = c.sort_values("ev_cal", ascending=False, kind="stable")
    c = c[c["ev_cal"] >= TH62].head(SLOTS)
    if c.empty:
        return None
    P, O = c["p_cal"].to_numpy(), c["odds"].to_numpy()
    best_ev, best_k, best_pu = -np.inf, 0, 0.0
    for k in range(1, len(c) + 1):
        feas = O[:k].min() >= M62 * k
        pu = min(1.0, float(P[:k].sum()))            # 馬連は同時に 2 つ当たらない
        evk = float((P[:k] * O[:k]).sum() / k)
        if feas and pu >= H62 and evk > best_ev:
            best_ev, best_k, best_pu = evk, k, pu
    if best_k == 0:
        return None
    s = c.head(best_k)
    return {"k": int(best_k), "pu": float(best_pu), "ev": float(best_ev),
            "combos": [[int(a), int(b)] for a, b in zip(s["a"], s["b"])], "odds": [float(x) for x in s["odds"]],
            "p_cal": [float(x) for x in s["p_cal"]], "p_mix": [float(x) for x in s["p_mix"]]}


# ───────────────────────── 払戻・入力の読み込み ─────────────────────────
def parse_payoffs(raw) -> dict:
    """race_result_flat.payoff（JSON 文字列）→ {'win': {馬番: 払戻}, 'umaren': {(a,b): 払戻}, 'broken': set}。複数的中は複数行。"""
    out = {"win": {}, "umaren": {}, "broken": set()}
    try:
        js = json.loads(raw) if isinstance(raw, str) else dict(raw)
    except (TypeError, ValueError):
        return {**out, "broken": {"win", "umaren"}}
    for jp, key, arity in (("単勝", "win", 1), ("馬連", "umaren", 2)):
        v = js.get(jp)
        if v is None:
            continue
        for e in (v if isinstance(v, list) else [v]):
            try:
                combo = tuple(int(x) for x in str(e["numbers"]).replace("→", "-").replace(" ", "").split("-"))
                pay = float(str(e["payout"]).replace(",", ""))
            except (KeyError, ValueError, TypeError):
                out["broken"].add(key)
                continue
            if len(combo) != arity or max(combo) > 18:
                out["broken"].add(key)
                continue
            out[key][combo[0] if arity == 1 else tuple(sorted(combo))] = pay
    return out


def _read(p: Path, columns=None, filters=None) -> pd.DataFrame:
    return pd.read_parquet(p, columns=columns, filters=filters) if p.exists() else pd.DataFrame(columns=columns or [])


class Inputs:
    """1 年分のフラット表（確定オッズ・出馬表・結果）。pre 用のスナップショットは storage（ホストで HybridStorage）から取る。"""

    def __init__(self, year: int, tables: Path = TABLES, storage=None, pair_file: Path | None = None):
        t = tables / str(year)
        self.shut = _read(t / "race_shutuba_flat.parquet", ["race_id", "date", "start_time", "horse_number", "horse_id", "jockey_id", "trainer_id"])
        self.shut["race_id"] = self.shut["race_id"].astype(str)
        self.odds = _read(t / "race_odds_flat.parquet", ["race_id", "horse_number", "win_odds"])
        self.odds["race_id"] = self.odds["race_id"].astype(str)
        pair = _read(t / "race_pair_odds_flat.parquet", ["race_id", "odds_type", "pair", "odds"])
        pair = pair[pair["odds_type"] == "umaren"].copy()
        pair["race_id"] = pair["race_id"].astype(str)
        if pair_file is not None and Path(pair_file).exists():
            # バックテスト用: フラット表は 2026 年の途中からしか馬連のオッズを持たないので、tp-50 が使った確定オッズの表（odds_pair）で補う
            alt = pd.read_parquet(pair_file, columns=["race_id", "bet_type", "combo", "odds"])
            alt = alt[(alt["bet_type"] == "umaren") & alt["race_id"].astype(str).str.startswith(str(year))].copy()
            alt["race_id"] = alt["race_id"].astype(str)
            alt["pair"] = [[int(x) for x in str(c).replace(" ", "").split("-")] for c in alt["combo"]]
            pair = pd.concat([alt.assign(odds_type="umaren")[["race_id", "odds_type", "pair", "odds"]], pair[~pair["race_id"].isin(set(alt["race_id"]))]], ignore_index=True)
        self.pair = pair
        self.res = _read(t / "race_result_flat.parquet", ["race_id", "date", "payoff"]).drop_duplicates("race_id")
        self.res["race_id"] = self.res["race_id"].astype(str)
        self.storage = storage
        self._by = {n: d.groupby("race_id") for n, d in (("shut", self.shut), ("odds", self.odds), ("pair", self.pair))}

    def get(self, name: str, rid: str) -> pd.DataFrame:
        try:
            return self._by[name].get_group(rid)
        except KeyError:
            return pd.DataFrame()

    def final_win_odds(self, rid: str) -> dict[int, float]:
        o = self.get("odds", rid)
        return {int(h): float(x) for h, x in zip(o.get("horse_number", []), o.get("win_odds", [])) if pd.notna(x)}

    def final_pair_odds(self, rid: str) -> dict[tuple, float]:
        p = self.get("pair", rid)
        out = {}
        for pr, o in zip(p.get("pair", []), p.get("odds", [])):
            try:
                a, b = (int(x) for x in pr)
                if pd.notna(o) and float(o) > 0:
                    out[tuple(sorted((a, b)))] = float(o)
            except (TypeError, ValueError):
                continue
        return out

    def post_time(self, rid: str):
        s = self.get("shut", rid)
        if s.empty:
            return None
        try:
            return datetime.fromisoformat(f"{str(s['date'].iloc[0])[:10]}T{str(s['start_time'].iloc[0])[:5]}:00+09:00")
        except (TypeError, ValueError):
            return None

    def _pick(self, doc: dict | None, phases, post):
        """発走時刻より前のスナップショットのうち、phases の順で最初のもの（トップ文書も phase を持つ）。戻り値: (phase, snapshot 文書)。"""
        if not isinstance(doc, dict):
            return None, None
        docs = {str(s["phase"]): s for s in (doc.get("snapshots") or []) if isinstance(s, dict) and s.get("phase")}
        if doc.get("phase"):
            docs[str(doc["phase"])] = doc
        for ph in phases:
            s = docs.get(ph)
            if not s:
                continue
            try:
                at = datetime.fromisoformat(str(s.get("snapshot_at")))
            except (TypeError, ValueError):
                continue
            if post is None or at < post:
                return ph, s
        return None, None

    def pre_win_odds(self, rid: str):
        """(phase, {馬番: 単勝オッズ})。storage が無い・該当が無いとき (None, {})。"""
        if self.storage is None:
            return None, {}
        ph, s = self._pick(self.storage.load("race_odds", rid), PRE_WIN_PHASES, self.post_time(rid))
        out = {}
        for e in (s or {}).get("entries") or []:
            try:
                o, n = float(e.get("win_odds")), int(e.get("horse_number"))
                if o > 0:
                    out[n] = o
            except (TypeError, ValueError):
                continue
        return ph, out

    def pre_pair_odds(self, rid: str):
        if self.storage is None:
            return None, {}
        ph, s = self._pick(self.storage.load("race_pair_odds", rid), PRE_PAIR_PHASES, self.post_time(rid))
        out = {}
        for e in (s or {}).get("umaren") or []:
            try:
                a, b = (int(x) for x in e["pair"])
                o = float(e["odds"])
                if o > 0:
                    out[tuple(sorted((a, b)))] = o
            except (KeyError, TypeError, ValueError):
                continue
        return ph, out


def read_pred(model: str, day: str, pred_dirs: dict | None = None) -> list[tuple[str, pd.DataFrame]]:
    """その日の予測を [(race_id, 表)]。レース別 <day>/<race_id>.parquet と、日別 <day>.parquet の両方に対応。"""
    base = (pred_dirs or PRED_DIRS)[model]
    out = []
    d = base / day
    if d.is_dir():
        for f in sorted(d.glob("*.parquet")):
            p = pd.read_parquet(f, columns=["race_id", "horse_id", "p_win", "p_top2", "p_top3"])
            if len(p):
                out.append((str(p["race_id"].iloc[0]), p))
    f = base / f"{day}.parquet"
    if f.exists():
        p = pd.read_parquet(f, columns=["race_id", "horse_id", "p_win", "p_top2", "p_top3"])
        for rid, g in p.groupby(p["race_id"].astype(str), sort=False):
            out.append((rid, g))
    return out


# ───────────────────────── 追跡本体 ─────────────────────────
def load_rules() -> dict:
    c59, c62 = json.loads(CFG59.read_text(encoding="utf-8")), json.loads(CFG62.read_text(encoding="utf-8"))
    ft = json.loads(CFG_FT.read_text(encoding="utf-8")) if CFG_FT.exists() else {}
    return {"jpos": set(map(str, c59["jockey_pos"])), "tpos": set(map(str, c59["trainer_pos"])), "knots": c62["knots"]["umaren"],
            "ab": (float(ft["blend"]["a_model"]), float(ft["blend"]["b_market"])) if ft.get("blend") else None, "c59": c59, "c62": c62, "ft": ft}


def ledger_keys(path: Path) -> set:
    keys = set()
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                keys.add(r["key"])
    return keys


def settle_race(rid: str, date: str, model: str, variant: str, g: pd.DataFrame, pair_odds: dict, pay: dict, rules: dict, odds_info: dict) -> list[dict]:
    """1 レース・1 モデル・1 オッズ時点の判定と精算。戻り値: 台帳の行（tp59・tp62 の 2 行）。"""
    rows = []
    base = {"race_id": rid, "date": date, "model": model, "odds_variant": variant, **odds_info}
    # tp59
    bets = tp59_decide(g, rules["jpos"], rules["tpos"])
    b59 = [{"horse_number": int(r.horse_number), "odds": float(r.win_odds), "p_win": float(r.p_win), "ev": float(r.ev), "ret_yen": float(pay["win"].get(int(r.horse_number), 0.0))} for r in bets.itertuples()]
    broken59 = "win" in pay["broken"]
    rows.append({**base, "rule": "tp59", "key": f"tp59|{model}|{variant}|{rid}", "n_bets": len(b59), "stake_yen": STAKE * len(b59), "return_yen": float(sum(b["ret_yen"] for b in b59)),
                 "hit": bool(any(b["ret_yen"] > 0 for b in b59)), "bets": b59, "settled": not broken59 or not b59, "pending_reason": "単勝払戻が壊れている" if (broken59 and b59) else ""})
    # tp62
    sel = None
    if pair_odds:
        sel = tp62_decide(umaren_candidates(g, pair_odds), rules["knots"])
    if sel:
        ret = sum(pay["umaren"].get(tuple(sorted(c)), 0.0) for c in sel["combos"])
        broken62 = "umaren" in pay["broken"]
        rows.append({**base, "rule": "tp62", "key": f"tp62|{model}|{variant}|{rid}", "n_bets": sel["k"], "stake_yen": STAKE * sel["k"], "return_yen": float(ret),
                     "hit": bool(ret > 0), "bets": [{"combo": c, "odds": o, "p_cal": pc, "p_mix": pm} for c, o, pc, pm in zip(sel["combos"], sel["odds"], sel["p_cal"], sel["p_mix"])],
                     "pu": sel["pu"], "ev_cal_mean": sel["ev"], "settled": not broken62, "pending_reason": "馬連払戻が壊れている" if broken62 else ""})
    else:
        rows.append({**base, "rule": "tp62", "key": f"tp62|{model}|{variant}|{rid}", "n_bets": 0, "stake_yen": 0.0, "return_yen": 0.0, "hit": False, "bets": [],
                     "settled": True, "pending_reason": "" if pair_odds else "馬連オッズなし"})
    return rows


def run_track(days: list[str], out_dir: Path, models=MODELS, variants=VARIANTS, storage=None, tables: Path = TABLES, pred_dirs: dict | None = None,
              rules: dict | None = None, verbose: bool = True, pair_file: Path | None = None) -> dict:
    rules = rules or load_rules()
    if rules["ab"] is None:
        raise SystemExit("config/frozen_tracking.json に混合係数がない。先に freeze-blend を実行する")
    out_dir.mkdir(parents=True, exist_ok=True)
    ledger = out_dir / "ledger.jsonl"
    seen = ledger_keys(ledger)
    new_rows, skipped = [], {}
    inputs: dict[int, Inputs] = {}
    for day in sorted(days):
        y = int(day[:4])
        if y not in inputs:
            inputs[y] = Inputs(y, tables, storage, pair_file)
        inp = inputs[y]
        for model in models:
            for rid, pred in read_pred(model, day, pred_dirs):
                r = inp.res[inp.res["race_id"] == rid]
                if r.empty or not isinstance(r["payoff"].iloc[0], str):
                    skipped["結果がまだ"] = skipped.get("結果がまだ", 0) + 1
                    continue
                date = f"{day[:4]}-{day[4:6]}-{day[6:]}"
                pay = parse_payoffs(r["payoff"].iloc[0])
                shut = inp.get("shut", rid)
                for variant in variants:
                    if all(f"{ru}|{model}|{variant}|{rid}" in seen for ru in RULES):
                        continue
                    if variant == "final":
                        wo, po, info = inp.final_win_odds(rid), inp.final_pair_odds(rid), {"odds_phase_win": "final", "odds_phase_pair": "final"}
                    else:
                        pw, wo = inp.pre_win_odds(rid)
                        pp, po = inp.pre_pair_odds(rid)
                        if not wo:
                            skipped["pre オッズなし"] = skipped.get("pre オッズなし", 0) + 1
                            continue
                        info = {"odds_phase_win": pw, "odds_phase_pair": pp}
                    g, why = make_race_frame(pred, shut, wo, rules["ab"])
                    if g is None:
                        skipped[why] = skipped.get(why, 0) + 1
                        continue
                    for row in settle_race(rid, date, model, variant, g, po, pay, rules, info):
                        if row["key"] not in seen and row["settled"]:
                            new_rows.append(row)
                            seen.add(row["key"])
    if new_rows:
        with ledger.open("a", encoding="utf-8") as fh:
            for r in new_rows:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    summary = {"added": len(new_rows), "skipped": skipped, "days": len(days)}
    with (out_dir / "runs.jsonl").open("a", encoding="utf-8") as fh:
        fh.write(json.dumps({"at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "days": [min(days), max(days)] if days else [], "models": list(models), "variants": list(variants), **summary}, ensure_ascii=False) + "\n")
    if verbose:
        print(json.dumps(summary, ensure_ascii=False))
    return summary


# ───────────────────────── スコアカード ─────────────────────────
def boot_roi(race: np.ndarray, stake: np.ndarray, ret: np.ndarray, n_boot: int = 2000, seed: int = 0) -> dict:
    """購入額あたりの回収率（%）とレース単位ブートストラップ区間。race は買ったレースの識別子。"""
    if len(stake) == 0 or stake.sum() <= 0:
        return {"roi": None, "lo": None, "hi": None}
    _, inv = np.unique(race, return_inverse=True)
    st = np.bincount(inv, weights=stake)
    rt = np.bincount(inv, weights=ret)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(st), size=(n_boot, len(st)))
    r = rt[idx].sum(1) / np.maximum(st[idx].sum(1), 1e-9)
    return {"roi": float(100 * rt.sum() / st.sum()), "lo": float(100 * np.quantile(r, 0.025)), "hi": float(100 * np.quantile(r, 0.975))}


def needed_races(mu: float, sd: float, z: float = 1.645) -> float | None:
    """回収率の下限（片側 95%）が 100% を超えるのに要る『買ったレース数』の目安: n ≈ (z·sd/(mu−1))²。
    mu は買ったレース 1 つあたり払戻倍率（払戻/購入額）の平均、sd はその標準偏差。"""
    if mu is None or sd is None or mu <= 1.0:
        return None
    return float((z * sd / (mu - 1.0)) ** 2)


def load_ledger(path: Path) -> pd.DataFrame:
    rows = [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()] if path.exists() else []
    return pd.DataFrame(rows)


def expectation(rule: str, rules: dict) -> dict:
    if rule == "tp59":
        bt = rules["c59"].get("backtest") or {}
        return {"凍結時の過去の回収率(%)": {k: round(v["roi"], 1) for k, v in bt.items() if isinstance(v, dict) and "roi" in v}, "条件": "単勝 EV≥1.5 かつ騎手・調教師の残差がともに正（選択年 2022-24、確認 2025・2026 は事後に見た）"}
    bt = rules["c62"].get("backtest_unselected") or {}
    return {"凍結時の過去の回収率(%)": {k: round(100 * v["roi"], 1) for k, v in bt.items() if "roi" in v}, "予測のレース的中率": {k: v.get("pred_race_hit") for k, v in bt.items()},
            "実績のレース的中率": {k: v.get("race_hit_rate") for k, v in bt.items()}, "条件": "馬連 較正後EV≥1.0・5点まで・レース的中率≥20%・払戻≥500%（事後に見つけた候補）"}


def scorecard(led: pd.DataFrame, rules: dict, ref_sd: dict | None = None) -> dict:
    out = {"generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "hashes": current_hashes(), "cards": []}
    if led.empty:
        return out
    led = led[led["settled"]]
    for (rule, model, variant), g in led.groupby(["rule", "model", "odds_variant"]):
        races = g["race_id"].nunique()
        b = g[g["n_bets"] > 0]
        stake, ret = b["stake_yen"].to_numpy(float), b["return_yen"].to_numpy(float)
        ci = boot_roi(b["race_id"].to_numpy(), stake, ret)
        n_bets = int(b["n_bets"].sum())
        mult = (b["return_yen"] / b["stake_yen"]).to_numpy(float) if len(b) else np.array([])
        mu = float(ret.sum() / stake.sum()) if stake.sum() > 0 else None
        sd_obs = float(mult.std(ddof=1)) if len(mult) > 30 else None
        sd = max(x for x in ((ref_sd or {}).get(rule), sd_obs) if x is not None) if ((ref_sd or {}).get(rule) is not None or sd_obs is not None) else None
        months = {}
        if len(b):
            mm = b.assign(m=b["date"].astype(str).str[:7]).groupby("m").agg(stake=("stake_yen", "sum"), ret=("return_yen", "sum"), races=("race_id", "nunique"))
            months = {m: {"stake": float(r.stake), "ret": float(r.ret), "roi": round(100 * r.ret / r.stake, 1), "races": int(r.races)} for m, r in mm.iterrows()}
        win_m = sum(1 for v in months.values() if v["roi"] > 100)
        nb = needed_races(mu, sd)
        out["cards"].append({"rule": rule, "model": model, "role": REFERENCE_MODEL.get(model, "?"), "odds_variant": variant, "races_settled": int(races), "bet_races": int(len(b)), "bets": n_bets,
                             "stake_yen": float(stake.sum()), "return_yen": float(ret.sum()), "hit_races": int(b["hit"].sum()), "race_hit_rate": float(b["hit"].mean()) if len(b) else None,
                             "roi_pct": ci["roi"], "roi_lo": ci["lo"], "roi_hi": ci["hi"], "months": months, "months_over_100": win_m, "months_with_bets": len(months),
                             "needed_bet_races_for_lower_bound_over_100": nb, "remaining_bet_races_estimate": (max(0.0, nb - len(b)) if nb else None), "per_race_return_sd": sd, "expectation": expectation(rule, rules)})
    return out


def render_html(card: dict) -> str:
    rows = []
    for c in card["cards"]:
        mon = "".join(f"<tr><td>{html.escape(m)}</td><td>{v['races']}</td><td>{v['stake']:.0f}</td><td>{v['ret']:.0f}</td><td>{v['roi']}</td></tr>" for m, v in sorted(c["months"].items()))
        rows.append(f"<h3>{c['rule']} / {c['model']}({c['role']}) / {c['odds_variant']}</h3><p>確定したレース {c['races_settled']}、買ったレース {c['bet_races']}、賭け数 {c['bets']}、"
                    f"購入 {c['stake_yen']:.0f} 円、払戻 {c['return_yen']:.0f} 円、回収率 {c['roi_pct'] if c['roi_pct'] is None else round(c['roi_pct'], 1)}% "
                    f"[{None if c['roi_lo'] is None else round(c['roi_lo'], 1)}, {None if c['roi_hi'] is None else round(c['roi_hi'], 1)}]、"
                    f"100%超の月 {c['months_over_100']}/{c['months_with_bets']}。確証までの残り（買ったレース数）の目安 {None if c['remaining_bet_races_estimate'] is None else round(c['remaining_bet_races_estimate'])}。</p>"
                    f"<table border=1 cellpadding=4><tr><th>月</th><th>レース</th><th>購入</th><th>払戻</th><th>回収率%</th></tr>{mon}</table>"
                    f"<pre>{html.escape(json.dumps(c['expectation'], ensure_ascii=False, indent=1))}</pre>")
    return ("<!doctype html><meta charset=utf-8><title>凍結した馬券候補の前向き追跡</title><h1>凍結した馬券候補の前向き追跡</h1>"
            "<p>主の評価は model=v4（凍結した規則の入力）。v5 は参考。確定オッズ(final)と発走前スナップショット(pre)を併記。2026 年以降の追跡データは学習・選択に使わない。</p>"
            + (("".join(rows)) or "<p>まだ確定したレースがありません。</p>") + f"<p style='color:#666'>生成 {card['generated_at']}</p>")


def cmd_report(a) -> None:
    out = Path(a.out_dir)
    rules = load_rules()
    ref = rules["ft"].get("reference_sd") or {}
    card = scorecard(load_ledger(out / "ledger.jsonl"), rules, ref)
    same, diff = check_hashes(out, write=False)
    card["rules_changed"] = {"changed": (not same), "files": diff}
    out.mkdir(parents=True, exist_ok=True)
    (out / "latest.json").write_text(json.dumps(card, ensure_ascii=False, indent=1), encoding="utf-8")
    (out / "latest.html").write_text(render_html(card), encoding="utf-8")
    rep = out / "reports" / datetime.now().strftime("%Y%m%d_%H%M%S")
    rep.mkdir(parents=True, exist_ok=True)
    (rep / "scorecard.json").write_text(json.dumps(card, ensure_ascii=False, indent=1), encoding="utf-8")
    (rep / "scorecard.html").write_text(render_html(card), encoding="utf-8")
    print(json.dumps({"cards": len(card["cards"]), "rules_changed": card["rules_changed"]}, ensure_ascii=False))
    if not same:
        print("WARNING: 凍結ファイルが変更されている:", diff, file=sys.stderr)


def week_days(back: int = 8) -> list[str]:
    from datetime import date, timedelta

    today = date.today()
    return [(today - timedelta(days=i)).strftime("%Y%m%d") for i in range(back, -1, -1)]


def cmd_freeze_blend(a) -> None:
    if CFG_FT.exists() and json.loads(CFG_FT.read_text(encoding="utf-8")).get("blend") and not a.force:
        raise SystemExit(f"{CFG_FT} に混合係数が既にある（凍結済み）。上書きは --force")
    src = json.loads(BENTER_SRC.read_text(encoding="utf-8"))
    ab = src["benter_ab"]
    cfg = json.loads(CFG_FT.read_text(encoding="utf-8")) if CFG_FT.exists() else {}
    cfg["blend"] = {"a_model": float(ab[0]), "b_market": float(ab[1]), "source": str(BENTER_SRC.relative_to(ROOT)) if BENTER_SRC.is_relative_to(ROOT) else str(BENTER_SRC),
                    "fit": "tp36_final2026: fit_benter(2022-2025 の edge_frame)。2026 年以降のレースにも同じ係数を使う（凍結）"}
    cfg["frozen_at"] = datetime.now().strftime("%Y-%m-%d")
    CFG_FT.parent.mkdir(parents=True, exist_ok=True)
    CFG_FT.write_text(json.dumps(cfg, ensure_ascii=False, indent=1), encoding="utf-8")
    print("frozen", cfg["blend"])


def cmd_track(a) -> None:
    out = Path(a.out_dir)
    same, diff = check_hashes(out)
    if not same:
        raise SystemExit(f"凍結ファイルが変更されている（{diff}）。追記を拒否する。意図した変更なら履歴を分けて（FROZEN_TRACKING_DIR を新しくして）再開する")
    frm = max(getattr(a, "from"), TRACK_START) if not a.allow_early else getattr(a, "from")
    to = a.to or datetime.now().strftime("%Y%m%d")
    days = [d for d in week_days(60) if frm <= d <= to] if not a.days else a.days.split(",")
    storage = None
    if "pre" in a.variants.split(","):
        try:
            from src.scraper.storage import HybridStorage

            storage = HybridStorage(str(ROOT))
        except Exception as exc:  # noqa: BLE001
            print("pre オッズ用の storage を開けない（final のみ）:", exc, file=sys.stderr)
    variants = tuple(v for v in a.variants.split(",") if v == "final" or storage is not None)
    run_track(days, out, tuple(a.models.split(",")), variants, storage)
    cmd_report(argparse.Namespace(out_dir=str(out)))


def cmd_backtest(a) -> None:
    """過去の期間に同じ判定コードを流す（コードの正しさの確認。結果は凍結の評価には使わない）。"""
    out = Path(a.out)
    pred_dirs = {"v4": Path(a.pred_dir), "v5": PRED_DIRS["v5"]}
    days = sorted(f.stem for f in Path(a.pred_dir).glob("*.parquet")) or sorted(p.name for p in Path(a.pred_dir).iterdir() if p.is_dir())
    days = [d for d in days if getattr(a, "from") <= d <= a.to]
    rules = load_rules()
    run_track(days, out, ("v4",), ("final",), None, TABLES, pred_dirs, rules, pair_file=Path(a.pair_file) if a.pair_file else None)
    card = scorecard(load_ledger(out / "ledger.jsonl"), rules, rules["ft"].get("reference_sd") or {})
    (out / "backtest_scorecard.json").write_text(json.dumps(card, ensure_ascii=False, indent=1), encoding="utf-8")
    if a.write_reference:
        led = load_ledger(out / "ledger.jsonl")
        led = led[led["settled"] & (led["n_bets"] > 0)]
        ref = {r: float((g["return_yen"] / g["stake_yen"]).std(ddof=1)) for r, g in led.groupby("rule")}
        cfg = json.loads(CFG_FT.read_text(encoding="utf-8"))
        cfg["reference_sd"] = ref
        cfg["reference_sd_note"] = "買ったレース 1 つあたりの払戻倍率の標準偏差（2026-01〜10 のバックテスト。確証までの件数の見積もり用の参考値）"
        CFG_FT.write_text(json.dumps(cfg, ensure_ascii=False, indent=1), encoding="utf-8")
        print("reference_sd", ref)
    for c in card["cards"]:
        print(c["rule"], c["model"], c["odds_variant"], "races", c["races_settled"], "bet_races", c["bet_races"], "bets", c["bets"], "roi", None if c["roi_pct"] is None else round(c["roi_pct"], 1),
              "months>100", f"{c['months_over_100']}/{c['months_with_bets']}", "hit_races", c["hit_races"])


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("freeze-blend")
    f.add_argument("--force", action="store_true")
    t = sub.add_parser("track")
    t.add_argument("--from", dest="from", default=TRACK_START)
    t.add_argument("--to", default="")
    t.add_argument("--days", default="", help="YYYYMMDD,YYYYMMDD（指定時は期間より優先）")
    t.add_argument("--models", default="v4,v5")
    t.add_argument("--variants", default="final,pre")
    t.add_argument("--out-dir", default=str(OUT_DIR))
    t.add_argument("--allow-early", action="store_true", help="凍結日(2026-10-22)より前も許す（確認用。本番の履歴には使わない）")
    r = sub.add_parser("report")
    r.add_argument("--out-dir", default=str(OUT_DIR))
    b = sub.add_parser("backtest")
    b.add_argument("--from", dest="from", required=True)
    b.add_argument("--to", required=True)
    b.add_argument("--pred-dir", required=True)
    b.add_argument("--out", required=True)
    b.add_argument("--pair-file", default=str(ROOT / "data/local/modeling/betting_2026/odds_pair.parquet"), help="馬連の確定オッズの補助表（tp-50 の入力。フラット表に無いレースを補う）")
    b.add_argument("--write-reference", action="store_true", help="買ったレースの払戻倍率の SD を config/frozen_tracking.json の reference_sd に書く（追跡の開始前に 1 回）")
    a = ap.parse_args()
    {"freeze-blend": cmd_freeze_blend, "track": cmd_track, "report": cmd_report, "backtest": cmd_backtest}[a.cmd](a)


if __name__ == "__main__":
    main()
