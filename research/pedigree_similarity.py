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

グラフ二系統モード (--mode dual):
  - 血統表は各世代 g で 2^g スロットの完全二分木。先頭 2^(g-1) が父側（当馬の父の subtree）、
    残りが母側（母の subtree）。
  - 父側: 世代上限 paternal_max_gen（既定 5）までの祖先ノードに重みを付与。
  - 母側: maternal_max_gen（既定 10）まで。現データは 5 世代表のみのため実効は 5 世代まで。
  - 類似度 = λ × overlap(父側) + (1-λ) × overlap(母側)。overlap は重み付き min 和（グラフ上の共通祖先強度）。

Usage:
  python -m research.pedigree_similarity [--min-progeny 20]
  python -m research.pedigree_similarity --mode dual --lambda-paternal 0.35 --method overlap

GCS 負荷:
  ``load_pedigrees_from_storage`` はまず ``research.pedigree_local_store`` で
  スナップショット ``horse_pedigree_5gen.jsonl.gz`` をローカル化し、
  **list_keys + 全頭 load を避ける**。未取得時のみ従来の全件走査にフォールバック。
  オフライン整備: ``python -m research.sire_factor_stats`` の snapshot 生成、または
  手元に ``data/research/_ped_snapshot_cache.jsonl.gz`` を置く。
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

# 完全な 5 世代 2 進血統表の牡スロット数（世代 g では 2^(g-1) 頭の牡）= 1+2+4+8+16
MALE_PEDIGREE_SLOTS_5GEN = 31

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "data" / "research" / "pedigree_similarity"


# ═══════════════════════════════════════════════════
#  5世代血統テーブルのパース
# ═══════════════════════════════════════════════════

def _extract_sex_from_td(td) -> str:
    """
    血統表の <td> セルから性別を判定して返す。

    判定ロジック:
      - セル内リンクに /horse/sire/ または /sire_horse.html を含む → '牡'
      - セル内リンクに /broodmare_horse / /mare/ / broodmare_horse.html を含む → '牝'
      - どちらも見つからない場合は '' (不明)
    """
    for a in td.select("a"):
        href = a.get("href", "")
        if "/sire/" in href or "sire_horse" in href:
            return "牡"
        if "/broodmare" in href or "/mare/" in href or "broodmare_horse" in href:
            return "牝"
    return ""


def parse_horse_sex_from_ped_html(html: str, horse_id: str) -> str:
    """
    馬の blood/ped HTML ページから主役馬の性別を判定して返す。

    判定順:
      1. mare_line_box 内の現在馬（太字リンク）→ '牝'
      2. プロフィール section で 牝/牡/セン を検索
      3. 不明なら ''
    """
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(html, "html.parser")

    # 1. mare_line_box: 現在馬が <b> タグで強調されていれば牝馬
    mare_box = soup.select_one(".mare_line_box, .mare_line")
    if mare_box:
        for a in mare_box.select("a"):
            href = a.get("href", "")
            if horse_id in href and a.select_one("b"):
                return "牝"

    # 2. ページテキストから 牝/牡/セン を抽出（プロフィール欄）
    for selector in ["div.horse_title", "div.db_prof_area_02", "div.db_prof_area"]:
        el = soup.select_one(selector)
        if el:
            text = el.get_text()
            if "牝" in text:
                return "牝"
            if "牡" in text or "セン" in text:
                return "牡"

    # 3. ページ全体の最初の 牝/牡 マーカー（ヘッダ付近）
    for tag in soup.select("h1, h2, .horse_info, .db_h_title"):
        text = tag.get_text()
        if "牝" in text:
            return "牝"
        if "牡" in text:
            return "牡"

    return ""


def parse_blood_table_5gen(html: str) -> list[dict[str, Any]]:
    """
    netkeiba の horse/ped ページ HTML から5世代 (62頭) の祖先を抽出。

    Returns:
        list of {generation, position, name, horse_id, sex}
        generation=1 が親, 5 が5代前。position は世代内の順番 (0-indexed)。
        sex: '牡' (stallion) / '牝' (mare) / '' (unknown)
          - blood_table 内の progeny リンク URL パターンで判定:
            /sire/ or sire_horse.html → 牡
            /broodmare / /mare/ / broodmare_horse.html → 牝
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

        sex = _extract_sex_from_td(td)

        pos = gen_counters[gen]
        gen_counters[gen] += 1

        ancestors.append({
            "generation": gen,
            "position": pos,
            "name": name,
            "horse_id": horse_id,
            "sex": sex,
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


def is_paternal_side(generation: int, position: int) -> bool:
    """
    5世代血統表の (generation, position) が「父系側」（当馬の父の subtree）か。

    世代 g では位置 0..2^g-1 が左から並び、先頭 2^(g-1) 本が父側、残りが母側。
    generation=1: position 0=父, 1=母。
    """
    if generation < 1:
        return False
    half = 2 ** (generation - 1)
    return position < half


def is_maternal_side(generation: int, position: int) -> bool:
    """母系側（当馬の母の subtree）。generation=1 の position 1（母）を含む。"""
    if generation < 1:
        return False
    half = 2 ** (generation - 1)
    return position >= half


def is_male_pedigree_slot(generation: int, position: int) -> bool:
    """
    5 世代血統表の同一世代内では、偶数 position が牡（父スロット）、奇数が牝。
    generation=1: position 0=父, 1=母。

    完全な表では牡は ``MALE_PEDIGREE_SLOTS_5GEN`` (=31) スロット（netkeiba の 62 祖先の半分）。
    種牡馬因子の加権 6 スロット（sire_factor_aptitude.ANCESTOR_SLOTS）とは別物。
    """
    if generation < 1 or generation > 5:
        return False
    try:
        pos = int(position)
    except (TypeError, ValueError):
        return False
    return pos % 2 == 0


def male_pedigree_slot_horse_ids_from_ancestors(
    ancestors: list[dict[str, Any]] | None,
) -> list[str]:
    """
    祖先リストから牡スロットの horse_id のみ列挙（空・重複は呼び出し側で扱う）。

    正規表ならエントリ数は高々 ``MALE_PEDIGREE_SLOTS_5GEN``（同一馬が複数スロットに出ると
    horse_id は重複しうる）。
    """
    out: list[str] = []
    if not ancestors:
        return out
    for anc in ancestors:
        try:
            g = int(anc.get("generation", 0))
            pos = int(anc.get("position", 0))
        except (TypeError, ValueError):
            continue
        if not is_male_pedigree_slot(g, pos):
            continue
        hid = str(anc.get("horse_id") or "").strip()
        if hid:
            out.append(hid)
    return out


def build_side_ancestor_vectors(
    ancestors: list[dict],
    *,
    paternal_max_gen: int = 5,
    maternal_max_gen: int = 10,
) -> tuple[dict[str, float], dict[str, float]]:
    """
    父系・母系それぞれに、世代上限内の祖先だけを重み付き集合（ノード名→重み）にする。

    重み = (1/2)^generation。インブリードで同一名が複数回出る場合は合算。
    """
    vp: dict[str, float] = defaultdict(float)
    vm: dict[str, float] = defaultdict(float)
    for anc in ancestors:
        gen = int(anc.get("generation", 0))
        pos = int(anc.get("position", 0))
        name = anc.get("name") or ""
        if not name:
            continue
        w = (0.5) ** gen
        if is_paternal_side(gen, pos) and gen <= paternal_max_gen:
            vp[name] += w
        if is_maternal_side(gen, pos) and gen <= maternal_max_gen:
            vm[name] += w
    return dict(vp), dict(vm)


def dual_side_similarity(
    v1p: dict[str, float],
    v1m: dict[str, float],
    v2p: dict[str, float],
    v2m: dict[str, float],
    *,
    lambda_paternal: float = 0.35,
    method: str = "overlap",
) -> float:
    """
    父系・母系の類似度を線形結合。method は cosine または overlap。
    """
    sim_fn = cosine_similarity if method == "cosine" else overlap_coefficient
    lam = float(lambda_paternal)
    lam = max(0.0, min(1.0, lam))
    sp = sim_fn(v1p, v2p)
    sm = sim_fn(v1m, v2m)
    return lam * sp + (1.0 - lam) * sm


def compute_similarity_matrix_dual(
    pedigrees: dict[str, list[dict]],
    *,
    lambda_paternal: float = 0.35,
    method: str = "overlap",
    paternal_max_gen: int = 5,
    maternal_max_gen: int = 10,
) -> tuple[list[str], np.ndarray, dict[str, tuple[dict[str, float], dict[str, float]]]]:
    """
    父系・母系を分けた重み付き集合に対し、ペアごとに dual_side_similarity を適用。

    Returns:
        (names, similarity_matrix, side_vectors)
        side_vectors[name] = (paternal_dict, maternal_dict)
    """
    names = sorted(pedigrees.keys())
    n = len(names)
    side_vectors: dict[str, tuple[dict[str, float], dict[str, float]]] = {}
    for name in names:
        side_vectors[name] = build_side_ancestor_vectors(
            pedigrees[name],
            paternal_max_gen=paternal_max_gen,
            maternal_max_gen=maternal_max_gen,
        )

    matrix = np.zeros((n, n))
    for i in range(n):
        matrix[i, i] = 1.0
        v1p, v1m = side_vectors[names[i]]
        for j in range(i + 1, n):
            v2p, v2m = side_vectors[names[j]]
            s = dual_side_similarity(
                v1p, v1m, v2p, v2m,
                lambda_paternal=lambda_paternal,
                method=method,
            )
            matrix[i, j] = s
            matrix[j, i] = s

    return names, matrix, side_vectors


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


def _load_pedigrees_from_index(
    index: dict[str, dict],
    min_progeny: int,
    target_sires: set[str] | None,
) -> tuple[dict[str, dict], dict[str, list[dict]]]:
    """ローカル / スナップショットで構築した horse_id インデックスから種牡馬血統を収集（GCS list_keys 不要）。"""
    sire_counts: dict[str, dict] = defaultdict(lambda: {"count": 0, "horse_id": ""})

    for _hid, data in index.items():
        if not data:
            continue
        for anc in data.get("ancestors", []):
            if anc.get("generation") == 1 and anc.get("position") == 0:
                sname = anc["name"]
                sire_counts[sname]["count"] += 1
                if not sire_counts[sname]["horse_id"] and anc.get("horse_id"):
                    sire_counts[sname]["horse_id"] = anc["horse_id"]
                break

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
    ped_keys = list(index.keys())

    for sname, info in qualified.items():
        hid = info["horse_id"]
        if not hid:
            continue

        ped_data = index.get(hid)
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
        pedigrees, sire_info = _reconstruct_sire_pedigrees_from_index(
            index, ped_keys, qualified,
        )

    logger.info("血統データ取得完了: %d 種牡馬", len(pedigrees))
    return sire_info, pedigrees


def _reconstruct_sire_pedigrees_from_index(
    index: dict[str, dict],
    ped_keys: list[str],
    qualified: dict,
) -> tuple[dict[str, list[dict]], dict[str, dict]]:
    """_reconstruct_sire_pedigrees の index 版（storage.load の代わりに辞書参照）。"""
    sire_children: dict[str, list[str]] = defaultdict(list)

    for key in ped_keys:
        data = index.get(key)
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
        child_data = index.get(child_key)
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


def load_pedigrees_from_storage(
    storage,
    min_progeny: int = 20,
    target_sires: set[str] | None = None,
) -> tuple[dict[str, dict], dict[str, list[dict]]]:
    """
    GCS の horse_pedigree_5gen から種牡馬の血統データを収集。
    Phase 1/2 で保存済みのデータを活用 (スクレイピング不要)。

    戦略:
      1. **推奨**: GCS スナップショット ``horse_pedigree_5gen.jsonl.gz`` を1回取得し
         ``research.pedigree_local_store`` 経由でローカルインデース化（list_keys しない）。
      2. フォールバック: horse_pedigree_5gen の全キーを列挙し個別 load（コスト大・非推奨）

    Returns: (sire_info, pedigrees)
        sire_info: {sire_name: {progeny_count, horse_id}}
        pedigrees: {sire_name: [ancestor_list]}
    """
    try:
        from research.pedigree_local_store import load_full_pedigree_index

        idx = load_full_pedigree_index(storage)
        if idx:
            logger.info("血統: スナップショット索引経由 (%d 頭)", len(idx))
            return _load_pedigrees_from_index(idx, min_progeny, target_sires)
    except Exception as e:
        logger.warning("スナップショット索引に失敗、GCS 全件走査へ: %s", e)

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
    *,
    mode: str = "full",
    dual_extra_meta: dict[str, Any] | None = None,
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

    meta_base: dict[str, Any] = {
        "pedigree_mode": mode,
        "n_stallions": len(names),
    }
    if mode == "dual":
        meta_base["method"] = "pedigree_graph_dual_side"
        meta_base["description"] = (
            "父系・母系を二分木の左右で分離し、各側で重み付き overlap/cosine、"
            "λ で線形結合した血統類似度（5世代表では母側も最大5世代まで）"
        )
        if dual_extra_meta:
            meta_base.update(dual_extra_meta)
    else:
        meta_base["method"] = "pedigree_structural_cosine"
        meta_base["description"] = "5世代血統構造ベースの祖先共有コサイン類似度"

    result = {
        "data": viz_data,
        "similarity": sim_data,
        "meta": meta_base,
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
    parser.add_argument(
        "--mode",
        choices=["full", "dual"],
        default="full",
        help="full=従来の単一祖先集合。dual=父系・母系を分けたグラフ的重み付き類似度",
    )
    parser.add_argument(
        "--lambda-paternal",
        type=float,
        default=0.35,
        help="dual モード時、父系スコアの重み (残りは母系)。既定 0.35",
    )
    parser.add_argument("--paternal-max-gen", type=int, default=5, help="dual: 父側の最大世代")
    parser.add_argument("--maternal-max-gen", type=int, default=10, help="dual: 母側の最大世代（5世代表では実効5まで）")
    parser.add_argument("--use-storage", action="store_true", default=True,
                        help="horse_pedigree_5gen から直接読み込み (Phase 1/2 完了後)")
    parser.add_argument("--scrape", action="store_true",
                        help="レガシー: 直接スクレイピングして取得")
    args = parser.parse_args()

    _r = str(Path(__file__).resolve().parent.parent)
    if _r not in sys.path:
        sys.path.insert(0, _r)
    from utils.keiba_logging import script_basic_config  # noqa: E402

    script_basic_config()
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

    if args.mode == "dual":
        m_method = args.method if args.method in ("cosine", "overlap") else "overlap"
        names, matrix, _side_vecs = compute_similarity_matrix_dual(
            pedigrees,
            lambda_paternal=args.lambda_paternal,
            method=m_method,
            paternal_max_gen=args.paternal_max_gen,
            maternal_max_gen=args.maternal_max_gen,
        )
        vectors = {name: build_ancestor_vector(pedigrees[name]) for name in names}
        dual_meta = {
            "side_similarity": m_method,
            "lambda_paternal": args.lambda_paternal,
            "paternal_max_gen": args.paternal_max_gen,
            "maternal_max_gen": args.maternal_max_gen,
            "note": "5世代 flat のみの環境では母側も generation<=5 に制限される",
        }
        result = save_results(
            names, matrix, vectors, pedigrees, sire_info,
            mode="dual",
            dual_extra_meta=dual_meta,
        )
    else:
        names, matrix, vectors = compute_similarity_matrix(pedigrees, method=args.method)
        result = save_results(names, matrix, vectors, pedigrees, sire_info, mode="full")

    logger.info("=== 完了: %d 種牡馬の血統構造類似度 (mode=%s) ===", len(names), args.mode)

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
