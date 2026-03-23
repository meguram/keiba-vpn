"""
血統構造ベース類似度 (Pedigree Structural Similarity)

成績統計ではなく、5世代血統表の祖先共有度から種牡馬間の
"遺伝的距離" を計算する。

アルゴリズム:
  1. 各種牡馬の5世代血統ページ (blood_table) をパースし、
     62 頭の祖先名 + 出現位置(世代)を取得。
  2. 祖先ごとに世代重み (1/2)^g を付与した「祖先ベクトル」を構築。
     同一祖先が複数回出現 (インブリード) する場合は重みが合算される。
  3. 2 種牡馬間のコサイン類似度で血統構造的な近さを測る。

Usage:
  python -m research.pedigree_similarity [--min-progeny 20]
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger("research.pedigree_similarity")

GENERATION_ROWSPAN = {16: 1, 8: 2, 4: 3, 2: 4, 1: 5}

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "data" / "research" / "pedigree_similarity"


# ═══════════════════════════════════════════════════
#  5世代血統テーブルのパース
# ═══════════════════════════════════════════════════

def parse_blood_table_5gen(html: str) -> list[dict[str, Any]]:
    """
    netkeiba の horse/ped ページ HTML から5世代 (62頭) の祖先を抽出。

    Returns:
        list of {generation, position, name, horse_id}
        generation=1 が親, 5 が5代前。position は世代内の順番 (0-indexed)。
    """
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(html, "html.parser")
    bt = soup.select_one("table.blood_table, table[class*='blood'], table[summary*='血統']")
    if not bt:
        return []

    ancestors: list[dict[str, Any]] = []
    gen_counters: dict[int, int] = defaultdict(int)

    for td in bt.select("td"):
        rs = int(td.get("rowspan", 1))
        gen = GENERATION_ROWSPAN.get(rs)
        if gen is None:
            continue

        link = td.select_one("a")
        if not link:
            continue
        name = link.get_text(strip=True)
        if not name:
            continue

        href = link.get("href", "")
        horse_id = ""
        m = re.search(r"/horse/(\w+)", href)
        if m:
            horse_id = m.group(1).rstrip("/")

        pos = gen_counters[gen]
        gen_counters[gen] += 1

        ancestors.append({
            "generation": gen,
            "position": pos,
            "name": name,
            "horse_id": horse_id,
        })

    return ancestors


def build_ancestor_vector(ancestors: list[dict]) -> dict[str, float]:
    """
    祖先リストから重み付きベクトルを構築。

    重み = (1/2)^generation。同一名が複数出現 (インブリード) なら合算。
    """
    vec: dict[str, float] = defaultdict(float)
    for anc in ancestors:
        weight = (0.5) ** anc["generation"]
        vec[anc["name"]] += weight
    return dict(vec)


def cosine_similarity(v1: dict[str, float], v2: dict[str, float]) -> float:
    """2つのスパース祖先ベクトル間のコサイン類似度。"""
    keys = set(v1) | set(v2)
    if not keys:
        return 0.0
    dot = sum(v1.get(k, 0) * v2.get(k, 0) for k in keys)
    n1 = math.sqrt(sum(x * x for x in v1.values()))
    n2 = math.sqrt(sum(x * x for x in v2.values()))
    if n1 == 0 or n2 == 0:
        return 0.0
    return dot / (n1 * n2)


def overlap_coefficient(v1: dict[str, float], v2: dict[str, float]) -> float:
    """
    重み付き重複率。共有祖先の重み合計 / 小さい方の重み合計。
    コサイン類似度より直感的に「血が近い」を反映。
    """
    shared_keys = set(v1) & set(v2)
    if not shared_keys:
        return 0.0
    shared_weight = sum(min(v1[k], v2[k]) for k in shared_keys)
    min_total = min(sum(v1.values()), sum(v2.values()))
    if min_total == 0:
        return 0.0
    return shared_weight / min_total


# ═══════════════════════════════════════════════════
#  種牡馬 ID の特定 + 血統ページ取得
# ═══════════════════════════════════════════════════

def collect_sire_ids_from_embeddings() -> dict[str, dict]:
    """sire_embeddings.json から種牡馬名と産駒数を取得。"""
    embed_path = (
        Path(__file__).resolve().parent.parent
        / "data" / "research" / "bloodline_vector" / "sire_embeddings.json"
    )
    if not embed_path.exists():
        logger.error("sire_embeddings.json が見つかりません")
        return {}

    with open(embed_path, encoding="utf-8") as f:
        embed_data = json.load(f)

    result = {}
    for rec in embed_data:
        name = rec.get("name", "")
        n = rec.get("n_runners", 0)
        if name:
            result[name] = {"progeny_count": n}
    logger.info("sire_embeddings.json: %d 種牡馬", len(result))
    return result


def load_pedigrees_from_storage(
    storage,
    min_progeny: int = 20,
    target_sires: set[str] | None = None,
) -> tuple[dict[str, dict], dict[str, list[dict]]]:
    """
    GCS の horse_pedigree_5gen から種牡馬の血統データを収集。
    Phase 1/2 で保存済みのデータを活用 (スクレイピング不要)。

    戦略:
      1. horse_pedigree_5gen の全レコードから sire 名を集計
      2. min_progeny 以上の産駒を持つ種牡馬を対象に
      3. その種牡馬自身の horse_pedigree_5gen (Phase 2 で取得) を読み込み
      4. 対象種牡馬の祖先データを返却

    Returns: (sire_info, pedigrees)
        sire_info: {sire_name: {progeny_count, horse_id}}
        pedigrees: {sire_name: [ancestor_list]}
    """
    ped_keys = storage.list_keys("horse_pedigree_5gen")
    logger.info("horse_pedigree_5gen: %d レコード", len(ped_keys))

    sire_counts: dict[str, dict] = defaultdict(lambda: {"count": 0, "horse_id": ""})

    for key in ped_keys:
        data = storage.load("horse_pedigree_5gen", key)
        if not data:
            continue
        for anc in data.get("ancestors", []):
            if anc.get("generation") == 1 and anc.get("position") == 0:
                sname = anc["name"]
                sire_counts[sname]["count"] += 1
                if not sire_counts[sname]["horse_id"] and anc.get("horse_id"):
                    sire_counts[sname]["horse_id"] = anc["horse_id"]

    if target_sires:
        qualified = {
            name: info for name, info in sire_counts.items()
            if name in target_sires
        }
    else:
        qualified = {
            name: info for name, info in sire_counts.items()
            if info["count"] >= min_progeny
        }

    logger.info("対象種牡馬: %d 頭 (min_progeny=%d)", len(qualified), min_progeny)

    sire_info: dict[str, dict] = {}
    pedigrees: dict[str, list[dict]] = {}

    for sname, info in qualified.items():
        hid = info["horse_id"]
        if not hid:
            continue

        ped_data = storage.load("horse_pedigree_5gen", hid)
        if ped_data:
            ancestors = ped_data.get("ancestors", [])
            if len(ancestors) >= 10:
                sire_info[sname] = {
                    "progeny_count": info["count"],
                    "horse_id": hid,
                }
                pedigrees[sname] = ancestors

    if not pedigrees:
        logger.info("Phase 2 未完了: 子馬データから種牡馬の血統を再構築")
        pedigrees, sire_info = _reconstruct_sire_pedigrees(
            storage, ped_keys, qualified,
        )

    logger.info("血統データ取得完了: %d 種牡馬", len(pedigrees))
    return sire_info, pedigrees


def _reconstruct_sire_pedigrees(
    storage, ped_keys: list[str], qualified: dict,
) -> tuple[dict[str, list[dict]], dict[str, dict]]:
    """
    子馬の5世代データから種牡馬の4世代分を再構築。
    子馬の gen=N, 父側 → 種牡馬の gen=N-1 に変換。
    """
    sire_children: dict[str, list[str]] = defaultdict(list)

    for key in ped_keys:
        data = storage.load("horse_pedigree_5gen", key)
        if not data:
            continue
        for anc in data.get("ancestors", []):
            if anc.get("generation") == 1 and anc.get("position") == 0:
                sname = anc["name"]
                if sname in qualified:
                    sire_children[sname].append(key)
                break

    pedigrees: dict[str, list[dict]] = {}
    sire_info: dict[str, dict] = {}

    for sname, children in sire_children.items():
        child_key = children[0]
        child_data = storage.load("horse_pedigree_5gen", child_key)
        if not child_data:
            continue

        child_ancs = child_data.get("ancestors", [])
        sire_hid = ""
        reconstructed: list[dict] = []

        for anc in child_ancs:
            gen = anc["generation"]
            pos = anc["position"]

            if gen == 1 and pos == 0:
                sire_hid = anc.get("horse_id", "")
                continue
            if gen == 1 and pos == 1:
                continue
            if gen <= 1:
                continue

            sire_side_count = 2 ** (gen - 1)
            if pos < sire_side_count:
                reconstructed.append({
                    "generation": gen - 1,
                    "position": pos,
                    "name": anc["name"],
                    "horse_id": anc.get("horse_id", ""),
                })

        if len(reconstructed) >= 5:
            pedigrees[sname] = reconstructed
            sire_info[sname] = {
                "progeny_count": qualified[sname]["count"],
                "horse_id": sire_hid,
            }

    return pedigrees, sire_info


def _collect_child_horse_ids(storage, sire_names: set[str]) -> list[str]:
    """race_result の entries から horse_id を収集。"""
    race_keys = storage.list_keys("race_result")
    all_horse_ids: list[str] = []
    for key in race_keys[:200]:
        data = storage.load("race_result", key)
        if data:
            for entry in data.get("entries", []):
                hid = entry.get("horse_id", "")
                if hid:
                    all_horse_ids.append(hid)
    logger.info("race_result から %d 頭の horse_id を収集", len(all_horse_ids))
    return all_horse_ids


def fetch_sire_pedigrees(
    sire_info: dict[str, dict],
    storage,
    client,
    interval: float = 0.8,
) -> dict[str, list[dict]]:
    """
    各種牡馬の5世代 blood_table を取得・パース (レガシー版: 直接スクレイピング)。
    load_pedigrees_from_storage が使える場合はそちらを推奨。
    """
    target_sires = set(sire_info.keys())
    sire_horse_ids: dict[str, str] = {}

    horse_ids = _collect_child_horse_ids(storage, target_sires)

    logger.info("=== Phase 1: 子馬の血統ページから種牡馬 ID を特定 ===")
    import random
    random.shuffle(horse_ids)

    for idx, hid in enumerate(horse_ids):
        if len(sire_horse_ids) >= len(target_sires):
            break

        try:
            url = f"https://db.netkeiba.com/horse/ped/{hid}/"
            html = client.fetch(url)
            ancestors = parse_blood_table_5gen(html)

            sire_entry = None
            for anc in ancestors:
                if anc["generation"] == 1 and anc["position"] == 0:
                    sire_entry = anc
                    break

            if sire_entry and sire_entry["name"] in target_sires:
                sname = sire_entry["name"]
                if sname not in sire_horse_ids:
                    sire_horse_ids[sname] = sire_entry["horse_id"]
                    sire_info[sname]["horse_id"] = sire_entry["horse_id"]
                    logger.info(
                        "  [%d/%d] %s → id=%s (子馬=%s)",
                        len(sire_horse_ids), len(target_sires),
                        sname, sire_entry["horse_id"], hid,
                    )

        except Exception as e:
            logger.debug("  子馬 %s の血統取得失敗: %s", hid, e)

        time.sleep(interval * 0.3)

        if (idx + 1) % 50 == 0:
            logger.info("  Phase 1 進捗: %d 頭スキャン, %d/%d 種牡馬 ID 特定",
                        idx + 1, len(sire_horse_ids), len(target_sires))

    logger.info("Phase 1 完了: %d/%d 種牡馬の ID 特定", len(sire_horse_ids), len(target_sires))

    logger.info("=== Phase 2: 種牡馬の5世代血統を取得 ===")
    results: dict[str, list[dict]] = {}
    total = len(sire_horse_ids)

    for i, (sire_name, sire_hid) in enumerate(sorted(sire_horse_ids.items())):
        logger.info("[%d/%d] %s (id=%s)", i + 1, total, sire_name, sire_hid)

        try:
            sire_ped_url = f"https://db.netkeiba.com/horse/ped/{sire_hid}/"
            sire_html = client.fetch(sire_ped_url)
            sire_ancestors = parse_blood_table_5gen(sire_html)

            if len(sire_ancestors) < 10:
                logger.warning("  祖先数不足: %s (%d 頭)", sire_name, len(sire_ancestors))
                continue

            results[sire_name] = sire_ancestors
            logger.info("  → %d 祖先取得", len(sire_ancestors))

        except Exception as e:
            logger.error("  取得失敗 [%s]: %s", sire_name, e)
            continue

        time.sleep(interval)

    logger.info("血統取得完了: %d / %d 種牡馬", len(results), total)
    return results


# ═══════════════════════════════════════════════════
#  類似度行列の算出
# ═══════════════════════════════════════════════════

def compute_similarity_matrix(
    pedigrees: dict[str, list[dict]],
    method: str = "cosine",
) -> tuple[list[str], np.ndarray, dict[str, dict[str, float]]]:
    """
    全種牡馬ペアの血統構造類似度を算出。

    Returns:
        (sire_names, similarity_matrix, vectors)
    """
    names = sorted(pedigrees.keys())
    n = len(names)

    vectors: dict[str, dict[str, float]] = {}
    for name in names:
        vectors[name] = build_ancestor_vector(pedigrees[name])

    sim_fn = cosine_similarity if method == "cosine" else overlap_coefficient

    matrix = np.zeros((n, n))
    for i in range(n):
        matrix[i, i] = 1.0
        for j in range(i + 1, n):
            s = sim_fn(vectors[names[i]], vectors[names[j]])
            matrix[i, j] = s
            matrix[j, i] = s

    return names, matrix, vectors


def extract_top_similar(
    names: list[str], matrix: np.ndarray, top_k: int = 8,
) -> dict[str, list[tuple[str, float]]]:
    """各種牡馬について類似度 Top-K を抽出。"""
    result: dict[str, list[tuple[str, float]]] = {}
    n = len(names)
    for i in range(n):
        scores = [(names[j], float(matrix[i, j])) for j in range(n) if j != i]
        scores.sort(key=lambda x: x[1], reverse=True)
        result[names[i]] = scores[:top_k]
    return result


def compute_inbreeding_stats(pedigrees: dict[str, list[dict]]) -> dict[str, dict]:
    """各種牡馬のインブリード統計を算出。"""
    stats: dict[str, dict] = {}
    for name, ancestors in pedigrees.items():
        name_counts: dict[str, list[int]] = defaultdict(list)
        for anc in ancestors:
            name_counts[anc["name"]].append(anc["generation"])

        inbred = {
            k: sorted(v)
            for k, v in name_counts.items()
            if len(v) >= 2
        }

        inbred_formatted = []
        for anc_name, gens in sorted(inbred.items(), key=lambda x: min(x[1])):
            cross = " × ".join(str(g) for g in gens)
            inbred_formatted.append({"ancestor": anc_name, "cross": cross})

        stats[name] = {
            "total_ancestors": len(ancestors),
            "unique_ancestors": len(name_counts),
            "inbreeding_count": len(inbred),
            "inbreeding": inbred_formatted,
        }
    return stats


# ═══════════════════════════════════════════════════
#  PCA 2D 投影
# ═══════════════════════════════════════════════════

def project_pca_2d(
    names: list[str], vectors: dict[str, dict[str, float]],
) -> dict[str, tuple[float, float]]:
    """祖先ベクトルを PCA で 2D に投影。"""
    all_keys = sorted({k for v in vectors.values() for k in v})
    X = np.zeros((len(names), len(all_keys)))
    key_idx = {k: i for i, k in enumerate(all_keys)}
    for row, name in enumerate(names):
        for k, w in vectors[name].items():
            X[row, key_idx[k]] = w

    X_centered = X - X.mean(axis=0)

    U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
    coords = U[:, :2] * S[:2]

    return {name: (float(coords[i, 0]), float(coords[i, 1])) for i, name in enumerate(names)}


# ═══════════════════════════════════════════════════
#  出力 & 可視化用JSON生成
# ═══════════════════════════════════════════════════

def save_results(
    names: list[str],
    matrix: np.ndarray,
    vectors: dict[str, dict[str, float]],
    pedigrees: dict[str, list[dict]],
    sire_info: dict[str, dict],
    output_dir: Path | None = None,
):
    out = output_dir or OUTPUT_DIR
    out.mkdir(parents=True, exist_ok=True)

    coords = project_pca_2d(names, vectors)
    top_similar = extract_top_similar(names, matrix, top_k=8)
    inbreeding = compute_inbreeding_stats(pedigrees)

    sire_lines = _load_sire_lines()

    viz_data = []
    for name in names:
        x, y = coords[name]
        info = sire_info.get(name, {})
        inb = inbreeding.get(name, {})
        viz_data.append({
            "name": name,
            "x": round(x, 4),
            "y": round(y, 4),
            "sire_line": sire_lines.get(name, "その他"),
            "n_progeny": info.get("progeny_count", 0),
            "horse_id": info.get("horse_id", ""),
            "unique_ancestors": inb.get("unique_ancestors", 0),
            "inbreeding_count": inb.get("inbreeding_count", 0),
            "inbreeding": inb.get("inbreeding", []),
        })

    sim_data = {}
    for name in names:
        sim_data[name] = [
            [other, round(score, 4)]
            for other, score in top_similar[name]
        ]

    result = {
        "data": viz_data,
        "similarity": sim_data,
        "meta": {
            "method": "pedigree_structural_cosine",
            "n_stallions": len(names),
            "description": "5世代血統構造ベースの祖先共有コサイン類似度",
        },
    }

    out_path = out / "pedigree_similarity.json"
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    logger.info("保存: %s (%d 種牡馬)", out_path, len(names))

    return result


def _load_sire_lines() -> dict[str, str]:
    """bloodline_vector.py の SIRE_LINES を再利用。"""
    try:
        from research.bloodline_vector import SIRE_LINES
        mapping: dict[str, str] = {}
        for line_name, stallions in SIRE_LINES.items():
            for s in stallions:
                mapping[s] = line_name
        return mapping
    except Exception:
        return {}


# ═══════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="血統構造ベース種牡馬類似度")
    parser.add_argument("--min-progeny", type=int, default=5)
    parser.add_argument("--interval", type=float, default=0.8)
    parser.add_argument("--method", choices=["cosine", "overlap"], default="cosine")
    parser.add_argument("--use-storage", action="store_true", default=True,
                        help="horse_pedigree_5gen から直接読み込み (Phase 1/2 完了後)")
    parser.add_argument("--scrape", action="store_true",
                        help="レガシー: 直接スクレイピングして取得")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from scraper.storage import HybridStorage

    storage = HybridStorage()

    logger.info("=== 血統構造類似度 計算開始 ===")

    if args.scrape:
        from scraper.client import NetkeibaClient
        client = NetkeibaClient()
        sire_info = collect_sire_ids_from_embeddings()
        pedigrees = fetch_sire_pedigrees(sire_info, storage, client, interval=args.interval)
        client.close()
    else:
        sire_info, pedigrees = load_pedigrees_from_storage(
            storage, min_progeny=args.min_progeny,
        )

    if len(pedigrees) < 3:
        logger.error("十分な血統データが取得できませんでした (%d 頭)", len(pedigrees))
        return

    names, matrix, vectors = compute_similarity_matrix(pedigrees, method=args.method)
    result = save_results(names, matrix, vectors, pedigrees, sire_info)

    logger.info("=== 完了: %d 種牡馬の血統構造類似度 ===", len(names))

    top_pairs = []
    n = len(names)
    for i in range(n):
        for j in range(i + 1, n):
            top_pairs.append((names[i], names[j], float(matrix[i, j])))
    top_pairs.sort(key=lambda x: x[2], reverse=True)
    logger.info("--- 血統構造 最類似ペア Top 10 ---")
    for a, b, s in top_pairs[:10]:
        logger.info("  %.4f  %s  ×  %s", s, a, b)


if __name__ == "__main__":
    main()
