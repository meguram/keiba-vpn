# 速報（race_result_on_time）の馬場・天候が空だった件（2026-11-15）

## 何が起きたか
- race.netkeiba.com の速報ページ（`/race/result.html`）の RaceData01 が、2026-10 下旬から次の書き方になった。
  `10:01発走 / <span> ダ1400m</span> 右 / <span class="Icon_Weather02 Icon_W02" title="曇"></span> <span class="Item04"> 良</span> <span class="Item05">/ 直線：弱向風</span>`
  「天候:」「馬場:」の文字が無いので、RaceResultOnTimeParser の正規表現では取れず、馬場・天候が空で保存された。
- 直し: e9939d63（`src/scraper/parsers.py` の `weather_track_from_racedata`。出馬表の RaceCardParser にも同じ逃げ道）。
- 確定（race_result、db.netkeiba.com）は影響なし。10/31・11/01・11/07・11/08 はどれも確定の値が入っている。

## 空のまま保存された件数（2026-11-15 12:40 JST に GCS で数えた）
| 開催日 | レース | 速報の馬場・天候が空 | 確定（race_result） |
|---|---|---|---|
| 10/25 | 36 | 0 | あり |
| 10/31 | 24 | 24 | あり |
| 11/01 | 24 | 24 | あり |
| 11/07 | 36 | 36 | あり |
| 11/08 | 36 | 36 | あり |
| 11/14 | 36 | 36 | まだ（金曜の週次で取る） |
| 11/15 | 36 | 14（数えた時点。残り 22 はまだ発走前） | まだ |
- 11/15 の残りのレースも空になる: ホストの raceday-result-runner は 07:30 JST に起動した常駐の処理で、直す前のコードを読み込んだまま動いている
  （止めて起動し直すのはホストの常駐の処理なので、このセッションではしていない）。11/16 07:30 の起動からは直したコードで取る。
- race_result_flat は、確定がまだ無い日（11/14・11/15）は速報から作るので、その 2 日だけ馬場が空。確定がある日は確定の値で埋まっている。
  金曜（11/20）の週次で確定を取れば、11/14・11/15 も確定の値で埋まる。

## 埋める手順（推奨: 速報を取り直す。11/15 の発走がすべて終わった後、例えば 17:00 JST 以降）
1. 対象: 10/31〜11/15 の速報のうち、馬場か天候が空のもの（最大 192 レース）。確定がある日も、速報を読む処理
   （推論の補い。raceday-weight-runner のセッションが用意中）のために速報を直す。
2. 取り直し（ホストで。netkeiba へ 1 レース 1 回。間隔は既定の取得の間隔・同時 1 本の決まりに従う）:
   ```
   KEIBA_ENV=stg PYTHONPATH=. /opt/conda/bin/python3 - <<'PY'
   import json, os, time, random
   from src.utils.project_env import load_project_dotenv; load_project_dotenv()
   from src.scraper.run import ScraperRunner
   r = ScraperRunner()
   days = ["20261031", "20261101", "20261107", "20261108", "20261114", "20261115"]
   for d in days:
       for x in json.load(open(f"data/page_reference/race_lists/{d}.json"))["races"]:
           rid = x["race_id"]
           cur = r.storage.load("race_result_on_time", rid) or {}
           if cur.get("track_condition") and cur.get("weather"):
               continue
           r.scrape_race_result_on_time(rid, skip_existing=False, opening_date=d)
           time.sleep(random.uniform(3, 6))
   PY
   ```
3. 結果の表を作り直す: `export_tables` の速報（race_result_on_time）と race_result_flat の 11/14・11/15 分（いつもの日次の書き出しで入る。
   急ぐなら手で日次の書き出しを回す）。
4. 確かめる: 上の数え方で「速報の馬場・天候が空」が 0 になること。

## 費用（一度きり）
- netkeiba: 最大 192 回（取得の規約の範囲。1 回 約 5 秒の間隔で 約 16 分）。
- GCS: 読み 192 件（今の速報、約 30KB）＝ 約 6MB、書き 192 件。ホストは GCP の外なので、読みの転送は 約 0.1 円。
  書き（Class A）192 件 ≒ 0.15 円、読み（Class B）≒ 0.01 円。書き出しのやり直しは日次の分の範囲。合計 1 円未満。
