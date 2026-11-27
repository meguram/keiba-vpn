"""血統予想家（めぐ分析の欄）。設計は docs/operations/pedigree_lab/persona_design.md、実装の計画は persona_impl_plan.md。

- columns: 前の夜の段の列の作り（研究の凍結の関数を呼ぶだけ・sha256 を照合）。
- rules: 規則（中心化・閾値・選び・理由の文・事実の札・禁止の語）。凍結の対象。
- eve: 前の夜の段（cron）。t60: T-60 の段（5 分おきの軽い cron）。verify: 前向きの記録との照合。
"""
