# yomiyasu（外部の Skill の写し）

AI が書いた日本語を推敲する Skill。めぐコラムの文の書き直しに使う（2026-11-18 にユーザーの指示で取り込み）。

- 取得元: https://github.com/nanaism/yomiyasu
- 版: v3.2.0（タグ 757b7903…、commit 3042154ee537c1b7093eefed00a97b40aae9446d）。`gh release download v3.2.0 --archive tar.gz` で取得した `skills/yomiyasu/` を、変えずに置いた
- ライセンス: MIT（`LICENSE`）。同梱の Unicode のデータは `UNICODE-LICENSE.txt`
- 取り込みのときの確認:
  - スクリプトは標準ライブラリだけで、通信の import（socket・urllib・http など）は無い
  - 外部のプロセスを起動するのは `yomiyasu_check.py` だけ。起動するのは同梱の `yomiyasu_lint.py` と `yomiyasu_diff.py` だけ
  - ファイルを書くのも `yomiyasu_check.py` だけ。書き先は記録の置き場（既定は git の最上位の `.yomiyasu/`）と自分の一時フォルダ
- このリポジトリでの使い方:
  - 記録は `--state-dir` でリポジトリの外に置く
  - 決まりは `docs/writing_style.md`。yomiyasu の原則と、めぐコラムの決まりの食い違いの扱いもそこにある
- 更新するときは、新しい版を同じ手順で取り、差分を読んでから置き換える。版と commit をこのファイルに書く
