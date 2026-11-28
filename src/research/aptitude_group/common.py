"""aptitude-group の共通の部品（置き場・分け・台帳）。"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

from src.research.pedigree_lab import data

OUT = Path(os.environ.get("APTG_OUT", "/home/jovyan/work/aptitude_group_out"))
TABLES = "/home/jovyan/work/keiba-vpn/data/page_reference/tables"
SEED = 20261127          # 探す 70% ／確かめ 30% の開催の分け（血統の研究の seed 20261125 とは別）


def open_token(purpose: str, version: str, reason: str, period=("2016-01-01", "2024-12-31")) -> None:
    from src.research.aptitude_group import ag_ledger
    assert period[1] <= "2024-12-31", "2025・26 年はこの族では読まない"
    data.TOKENS.append(ag_ledger.open_token(period, purpose, version, reason))


def split(meetings: pd.Series) -> np.ndarray:
    """開催（race_id の先頭 8 桁）の単位の無作為の分け。0 = 探す（70%）、1 = 確かめ（30%）。"""
    u = np.array(sorted(pd.unique(meetings.astype(str))))
    perm = np.random.default_rng(SEED).permutation(len(u))
    lab = np.zeros(len(u), dtype=np.int8)
    lab[perm[int(len(u) * 0.70):]] = 1
    return meetings.astype(str).map(dict(zip(u, lab))).to_numpy()
