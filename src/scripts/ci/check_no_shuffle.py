#!/usr/bin/env python3
"""CI: sklearn の time split で shuffle=True を検知（AREA-08）。"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "src"
PATTERN = re.compile(r"train_test_split\s*\([^)]*shuffle\s*=\s*True", re.MULTILINE | re.DOTALL)

violations: list[str] = []
for path in SRC.rglob("*.py"):
    if path.resolve() == Path(__file__).resolve():
        continue
    text = path.read_text(encoding="utf-8", errors="ignore")
    if PATTERN.search(text):
        violations.append(str(path.relative_to(ROOT)))

if violations:
    print("Forbidden shuffle=True in train_test_split:")
    for v in violations:
        print(f"  - {v}")
    sys.exit(1)

print("OK: no forbidden shuffle in train_test_split")
