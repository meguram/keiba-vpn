# DEC-001: keiba-vpn フロントエンド・バックエンド言語選定

**Date**: 2026-07-02
**Agent**: frontend-engineer, backend-engineer
**Task**: TASK-005
**Status**: ACCEPTED

---

## Context

技術スタック選定の最終決定です。フロント・バック・DB・スケーリング方針を確認してください。

---

## Decision

# keiba-vpn 言語選定の最終決定

**フロントエンド**: TypeScript + Next.js 14 (App Router)
**バックエンド**: Python + FastAPI
**DB**: PostgreSQL (Cloud SQL / Supabase) + Redis キャッシュ
**多数ユーザ対応**: Gunicorn + gevent workers + Nginx + Redis キャッシュ

AIモデルとの統合を考慮しPythonを選定。フロントはOpenAPI自動型生成でバックエンドと型安全に連携。

---

## Conclusion

**AIモデルとの連携を考慮するとバックエンドはPython+FastAPIが最適であり、フロントはTypeScript+Next.js 14でOpenAPI自動型生成を繋ぐことで、型安全性とAI機能を両立するのが最重要の決定事項です。**

---

## Consequences

- この決定はレビュー済みで承認されました
- 実装時はこのドキュメントを参照してください

---

_Approved via Multi-Agent Console — 2026-07-02_
