"""競走馬の血統構成から適性プロファイル (L2_fine 所属度) を計算する。

入力:
    horse_id  (str) or horse_name (str)

出力 (dict):
    {
      "horse_id": str,
      "horse_name": str,
      "sex": "牡" | "牝" | "セ" | "?",
      "n_ancestors_used": int,
      "L2_scores":   {L2_id (int) -> 0.0~1.0 score (正規化済み)},
      "L2_top":      [{L2, score, name, color, super_group}, ...上位 5 件],
      "super_group_scores": {sg_id (int) -> 0.0~1.0},
      "sire":  {"id", "name", "L2"},
      "ms":    {"id", "name", "L2"},   # 母父 (Maternal Grandsire)
      "ms_ms": {"id", "name", "L2"},   # 母母父
      "lineage_breakdown": [           # 主要位置別 L2 (上位 ~10 個)
          {"pos": "F",  "name": "ディープインパクト", "L2": 8, "weight": 1.00},
          {"pos": "MF", "name": "...",            "L2": 4, "weight": 0.50},
          ...
      ],
      "perf_signal": {                  # 実成績ベースの簡易シグナル (検証用)
          "best_venue": str | None,
          "best_distance_bin": str | None,
          "best_surface": str | None,
          "n_races": int,
      }
    }

位置別重み (`gen` と `path_fm` から計算):
    base    = 0.5 ** (gen - 1)
    father_bonus = 1 + 0.4 * (F_count / len(path_fm))
    weight  = base * father_bonus
    (父系経路を歩むほど重み +40%)

性別係数 (産駒 sex):
    各 L2 に対して固定係数 (現状は均等 1.0; 将来 Spearman 最適化で学習)

Usage:
    >>> from src.research.pedigree.horse_aptitude_profile import (
    ...     HorseAptitudeProfileCalc,
    ... )
    >>> calc = HorseAptitudeProfileCalc()
    >>> prof = calc.compute(horse_name="ダノンザキッド")
    >>> print(prof["L2_top"][:3])
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
ART = ROOT / "data/research/bloodline_meta_cluster"
HORSE_AT = ROOT / "data/research/horse_aptitude"
CATS_PATH = ROOT / "data/research/pedigree_race_index/horse_pedigree_cats.parquet"


def _position_weight(gen: int, path_fm: str) -> float:
    """位置別重みを計算。

    base = 0.5^(gen-1)
    father_bonus = 1 + 0.4 * (path 内の F の比率) [父系経路ほど重み付け強]
    """
    base = 0.5 ** (gen - 1)
    if not path_fm:
        return base
    f_ratio = path_fm.count("F") / len(path_fm)
    return base * (1.0 + 0.4 * f_ratio)


FEAT_COLS_31 = [
    "win_v_東京", "win_v_中山", "win_v_阪神", "win_v_京都", "win_v_中京",
    "win_v_新潟", "win_v_小倉", "win_v_福島", "win_v_札幌", "win_v_函館",
    "win_s_芝", "win_s_ダート",
    "win_d_短距離", "win_d_マイル", "win_d_中距離", "win_d_長距離",
    "win_pace_スロー瞬発力_短距離", "win_pace_スロー瞬発力_マイル",
    "win_pace_スロー瞬発力_中距離", "win_pace_スロー瞬発力_長距離",
    "win_pace_持続力勝負_短距離", "win_pace_持続力勝負_マイル",
    "win_pace_持続力勝負_中距離", "win_pace_持続力勝負_長距離",
    "win_pace_持久力勝負_短距離", "win_pace_持久力勝負_マイル",
    "win_pace_持久力勝負_中距離", "win_pace_持久力勝負_長距離",
    "win_steep", "win_flat", "win_heavy",
]


class HorseAptitudeProfileCalc:
    """単一の Calc インスタンスで複数馬を計算 (キャッシュ効率優先)。

    モード:
        "vector" (デフォルト): 各祖先の 31D ベクトルを位置重み加重平均 →
                              L2 centroid との cosine 類似度 + softmax で所属度算出
        "label":              各祖先の L2 ラベルを one-hot で加重 → そのまま正規化 (旧方式)
    """

    def __init__(self, mode: str = "label", softmax_beta: float = 6.0,
                  l2_prior_correction: bool = True):
        """
        l2_prior_correction:
            True にすると、L2 ラベル別の祖先数 (= 事前分布) で逆数補正する。
            データの偏り (Deep Impact 系統が L2=5 で 39% を占める等) を緩和し、
            各 L2 の判定が「平均よりどれだけ偏っているか」を見るようになる。
        """
        self.mode = mode
        self.softmax_beta = softmax_beta
        self.l2_prior_correction = l2_prior_correction
        print(f"[HorseAptitudeProfileCalc] loading (mode={mode}, prior_correction={l2_prior_correction}) ...", flush=True)
        # 1) ancestor_to_l2 (主流 + 拡張)
        with open(ART / "ancestor_to_l2.json", encoding="utf-8") as f:
            self.ancestor_l2: dict[str, dict] = json.load(f)
        # 事前分布 (L2 ごとの祖先数) を計算
        from collections import Counter
        l2_counts = Counter()
        for info in self.ancestor_l2.values():
            L2 = info.get("L2", -1)
            if L2 >= 0:
                l2_counts[L2] += 1
        total = sum(l2_counts.values()) or 1
        # 逆数事前 (smoothing 付き)
        self.l2_prior = np.ones(11, dtype=float)
        for L2, n in l2_counts.items():
            # 確率 p_L2 = n/total; 逆数で重み付け
            p = n / total
            self.l2_prior[L2] = 1.0 / (p + 0.01)
        # 正規化 (合計が L2 数 = 11 になるように)
        self.l2_prior = self.l2_prior * (11.0 / self.l2_prior.sum())
        # 2) L2 メタ (名前 / super-group 等)
        with open(ART / "l2_names.json", encoding="utf-8") as f:
            l2_meta = json.load(f)
        self.l2_info = {
            int(k): v for k, v in l2_meta.get("l2_names", {}).items()
        }
        self.sg_info = {
            int(k): v for k, v in l2_meta.get("super_groups", {}).items()
        }
        # 3) L2 -> super_group マッピング
        sg_path = ART / "l2_super_groups.json"
        if sg_path.exists():
            with open(sg_path, encoding="utf-8") as f:
                sg_data = json.load(f)
            sg_map = sg_data.get("L2_to_super") or sg_data.get("l2_to_sg") or {}
            self.l2_to_sg = {int(k): int(v) for k, v in sg_map.items()}
        else:
            self.l2_to_sg = {}
        # 4) horse_pedigree_cats (各馬の祖先)
        self.cats = pd.read_parquet(
            CATS_PATH, columns=["horse_id", "stallion_id", "gen", "path_fm"],
        )
        self.cats["horse_id"] = self.cats["horse_id"].astype(str)
        self.cats["stallion_id"] = self.cats["stallion_id"].astype(str)
        print("  grouping by horse_id ...", flush=True)
        self.cats_by_horse = {
            hid: g for hid, g in self.cats.groupby("horse_id")
        }
        # 5) 馬名インデックス
        self.idx = pd.read_parquet(HORSE_AT / "horse_name_index.parquet")
        self.idx["horse_id"] = self.idx["horse_id"].astype(str)
        self.name_to_id = (
            self.idx.sort_values("last_date", ascending=False)
            .drop_duplicates("horse_name")
            .set_index("horse_name")["horse_id"]
            .to_dict()
        )
        # 6) 実成績データ
        self.perf = pd.read_parquet(HORSE_AT / "horse_perf_by_cond.parquet")
        self.perf["horse_id"] = self.perf["horse_id"].astype(str)
        # 7) (vector mode) 祖先 31D ベクトル + L2 centroid + scaler
        self.anc_vec_map: dict[str, np.ndarray] = {}
        self.cent_full: np.ndarray | None = None
        self.cent_l2_ids: list[int] = []
        self.scaler_mean: np.ndarray | None = None
        self.scaler_scale: np.ndarray | None = None
        # pace 12 列は ancestor_vectors では集計されていない (常に 0) →
        # cosine 比較で pace 軸の貢献をゼロにするため、マスクで除外する
        self.pace_mask = np.array(
            [not c.startswith("win_pace_") for c in FEAT_COLS_31], dtype=bool,
        )
        if mode == "vector":
            vec_path = ART / "ancestor_vectors.parquet"
            if not vec_path.exists():
                print("  [warn] ancestor_vectors.parquet not found → fallback to label mode")
                self.mode = "label"
            else:
                vdf = pd.read_parquet(vec_path)
                vdf["stallion_id"] = vdf["stallion_id"].astype(str)
                for r in vdf.itertuples(index=False):
                    self.anc_vec_map[r.stallion_id] = np.array(
                        [getattr(r, c) for c in FEAT_COLS_31], dtype=np.float64,
                    )
                # L2 centroid は scaler_strength.json で z-score 化された値 →
                # raw 勝率空間に戻す (= 生の win_rate スケール)。
                cent = json.load(open(ART / "l2_centroids.json", encoding="utf-8"))
                self.cent_l2_ids = sorted([int(k) for k in cent["centroids_full"]])
                cent_z = np.array(
                    [cent["centroids_full"][str(i)] for i in self.cent_l2_ids],
                    dtype=np.float64,
                )
                scaler = json.load(open(ART / "scaler_strength.json", encoding="utf-8"))
                self.scaler_mean = np.array(scaler["mean"], dtype=np.float64)
                self.scaler_scale = np.array(scaler["scale"], dtype=np.float64)
                # 逆変換: raw = z * scale + mean
                self.cent_full = cent_z * self.scaler_scale + self.scaler_mean
        # L2 ラベル数
        self.n_L2 = len(self.l2_info) if self.l2_info else 11
        print(f"  loaded: n_horses={len(self.idx):,}, n_ancestors={len(self.ancestor_l2):,}, n_L2={self.n_L2}, mode={self.mode}", flush=True)

    def _vec_to_l2_scores(self, vec_31d: np.ndarray) -> np.ndarray:
        """31D 加重平均ベクトル → 11 個の L2 への所属度 (softmax(beta × cosine))。

        ancestor_vectors は pace 12 列が常に 0 のため、その軸を除外して
        cosine を計算する (mask: 19 dim 残)。
        さらに **z-score 化** して各軸のスケール差を吸収。
        """
        mask = self.pace_mask
        v = vec_31d[mask]
        C = self.cent_full[:, mask]
        # z-score (mask 後の特徴で再正規化)
        mean = self.scaler_mean[mask]
        scale = self.scaler_scale[mask]
        v_z = (v - mean) / scale
        C_z = (C - mean) / scale
        # cosine
        vn = np.linalg.norm(v_z) + 1e-9
        Cn = np.linalg.norm(C_z, axis=1) + 1e-9
        sims = (C_z @ v_z) / (Cn * vn)
        # softmax
        z = self.softmax_beta * sims
        z = z - z.max()
        probs = np.exp(z)
        return probs / probs.sum()

    # ---------------- public API ----------------

    def resolve(self, *, horse_id: str | None = None, horse_name: str | None = None) -> str | None:
        if horse_id:
            return str(horse_id)
        if horse_name:
            return self.name_to_id.get(horse_name.strip())
        return None

    def compute(
        self,
        *,
        horse_id: str | None = None,
        horse_name: str | None = None,
    ) -> dict[str, Any]:
        hid = self.resolve(horse_id=horse_id, horse_name=horse_name)
        if hid is None:
            return {"error": f"horse not found: id={horse_id} name={horse_name}"}
        if hid not in self.cats_by_horse:
            return {"error": f"pedigree not indexed for horse_id={hid}"}

        anc_df = self.cats_by_horse[hid]

        # 1) 各祖先について、位置重み × 祖先のソフト L2 分布 を加重和
        #    (vector モード: 祖先 31D → cosine 類似度で各 L2 確率を求め、
        #                    位置重みで加算 → softmax 後正規化)
        #    (label モード:  祖先の L2 ラベル one-hot を位置重みで加算)
        used = 0
        lineage_rows = []
        sire = ms = ms_ms = None
        use_vector = (self.mode == "vector" and self.cent_full is not None
                      and len(self.anc_vec_map) > 0)
        scores = np.zeros(self.n_L2, dtype=float)
        vec_sum = np.zeros(31, dtype=np.float64)
        w_sum = 0.0

        for r in anc_df.itertuples(index=False):
            sid = r.stallion_id
            l2_info = self.ancestor_l2.get(sid)
            if not l2_info:
                continue
            L2 = int(l2_info["L2"])
            if L2 < 0 or L2 >= self.n_L2:
                continue
            w = _position_weight(int(r.gen), str(r.path_fm))
            if use_vector:
                vec = self.anc_vec_map.get(sid)
                if vec is not None:
                    # 祖先個別の L2 ソフト分布
                    anc_probs = self._vec_to_l2_scores(vec)
                    scores += w * anc_probs
                    vec_sum += w * vec
                    w_sum += w
                else:
                    scores[L2] += w
            else:
                scores[L2] += w
            used += 1
            l2_meta_row = self.l2_info.get(L2, {}) or {}
            lineage_rows.append({
                "pos": str(r.path_fm),
                "gen": int(r.gen),
                "stallion_id": sid,
                "name": l2_info.get("name", ""),
                "L2": L2,
                "l2_name": l2_meta_row.get("name") or l2_meta_row.get("label") or f"クラスタ#{L2}",
                "weight": float(w),
                "method": l2_info.get("method", ""),
            })
            if r.path_fm == "F" and r.gen == 1:
                sire = lineage_rows[-1]
            elif r.path_fm == "MF" and r.gen == 2:
                ms = lineage_rows[-1]
            elif r.path_fm == "MMF" and r.gen == 3:
                ms_ms = lineage_rows[-1]

        # 2) 事前分布補正 (オプション) + 正規化
        if self.l2_prior_correction and scores.sum() > 0:
            scores = scores * self.l2_prior
        total = scores.sum()
        scores_norm = scores / total if total > 0 else scores
        horse_vec_31d = (vec_sum / w_sum) if (use_vector and w_sum > 0) else None

        # 2) Top L2 (上位 5 件)
        order = list(np.argsort(scores_norm)[::-1])
        top_L2 = []
        for i in order[:5]:
            i = int(i)
            sg_id = self.l2_to_sg.get(i, -1)
            sg = self.sg_info.get(sg_id, {})
            l2 = self.l2_info.get(i, {})
            top_L2.append({
                "L2": i,
                "score": float(scores_norm[i]),
                "name": l2.get("label", l2.get("name", f"L2-{i}")),
                "subtitle": l2.get("subtitle", ""),
                "color": l2.get("color", "#888"),
                "icon": l2.get("icon", ""),
                "super_group": sg_id,
                "super_group_name": sg.get("name", ""),
            })

        # 3) super_group score 集計
        sg_scores: dict[int, float] = {}
        for L2_id, s in enumerate(scores_norm):
            sg_id = self.l2_to_sg.get(L2_id, -1)
            sg_scores[sg_id] = sg_scores.get(sg_id, 0.0) + float(s)

        # 4) 主要位置のみ 10 個程度を抽出 (gen<=3 or weight>=0.25)
        lineage_rows.sort(key=lambda r: -r["weight"])
        breakdown = [r for r in lineage_rows if r["gen"] <= 3][:12]

        # 5) 実成績シグナル (検証用)
        hid_perf = self.perf[self.perf["horse_id"] == hid]
        perf_signal = {"n_races": int(self.idx[self.idx["horse_id"] == hid]["n_races"].iloc[0])
                        if hid in self.idx["horse_id"].values else 0}
        for kind, key in [
            ("venue", "best_venue"),
            ("distance_bin", "best_distance_bin"),
            ("surface", "best_surface"),
            ("track_condition", "best_track_condition"),
        ]:
            sub = hid_perf[hid_perf["cond_kind"] == kind]
            if len(sub) > 0 and sub["n_races"].sum() >= 3:
                sub_min = sub[sub["n_races"] >= 2]
                if len(sub_min) > 0:
                    best_row = sub_min.sort_values("win_rate", ascending=False).iloc[0]
                    perf_signal[key] = str(best_row["cond_value"])
                    perf_signal[f"{key}_n"] = int(best_row["n_races"])
                    perf_signal[f"{key}_wr"] = float(best_row["win_rate"])

        # 6) horse_id → 馬名/sex
        row = self.idx[self.idx["horse_id"] == hid]
        horse_name_out = row["horse_name"].iloc[0] if len(row) else (horse_name or "?")
        sex = row["sex"].iloc[0] if len(row) else "?"

        return {
            "horse_id": hid,
            "horse_name": horse_name_out,
            "sex": sex,
            "mode": self.mode if use_vector else "label",
            "n_ancestors_used": used,
            "L2_scores": {int(i): float(s) for i, s in enumerate(scores_norm)},
            "L2_top": top_L2,
            "super_group_scores": {int(k): float(v) for k, v in sg_scores.items()},
            "sire": sire, "ms": ms, "ms_ms": ms_ms,
            "lineage_breakdown": breakdown,
            "perf_signal": perf_signal,
            "horse_vec_31d": (horse_vec_31d.tolist() if horse_vec_31d is not None else None),
        }


@lru_cache(maxsize=1)
def get_calc() -> HorseAptitudeProfileCalc:
    return HorseAptitudeProfileCalc()


def main():
    """CLI 動作確認: ``python -m src.research.pedigree.horse_aptitude_profile [horse_name|horse_id]``"""
    import sys
    calc = get_calc()
    targets = sys.argv[1:] or ["ダノンザキッド", "イクイノックス", "ソダシ"]
    for t in targets:
        prof = calc.compute(horse_name=t) if not t.isdigit() else calc.compute(horse_id=t)
        if "error" in prof:
            print(f"  ! {t}: {prof['error']}")
            continue
        print()
        print(f"=== {prof['horse_name']} ({prof['horse_id']}, {prof['sex']}) ===")
        print(f"  祖先 used: {prof['n_ancestors_used']}")
        sire = prof.get("sire"); ms = prof.get("ms"); ms_ms = prof.get("ms_ms")
        if sire: print(f"  父   : {sire['name']:25} L2={sire['L2']}")
        if ms:   print(f"  母父 : {ms['name']:25} L2={ms['L2']}")
        if ms_ms:print(f"  母母父: {ms_ms['name']:25} L2={ms_ms['L2']}")
        print("  Top L2:")
        for tl in prof["L2_top"][:3]:
            print(f"    L2={tl['L2']:2} score={tl['score']:.3f}  {tl['name']}  (SG{tl['super_group']}: {tl['super_group_name']})")
        ps = prof["perf_signal"]
        if ps.get("best_venue"):
            print(f"  実成績 best: 競馬場={ps.get('best_venue')}, 距離={ps.get('best_distance_bin')}, 馬場={ps.get('best_surface')}")


if __name__ == "__main__":
    main()
