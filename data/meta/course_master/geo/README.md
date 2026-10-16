# コースの実形状データ（geo/）

- `osm/<場コード>.json`: OpenStreetMap から取得した JRA 10 競馬場のコース面（芝・ダート・障害、内回り・外回りの帯、引込み線を含む）とメインスタンド候補の建物の形状（緯度経度）。
  取得: `python -m src.scripts.data.fetch_osm_tracks`。
- `routes.json`: 上の形状を骨格化して導出した周回路（内回り・外回り・ダート）、ゴール・コーナー・引込み線の位置、全コース（距離ごと）のスタート位置と経路。
  生成: `python -m src.research.race.track_geometry`（重い依存: shapely・scikit-image・skan・networkx）。アトラスと検証テストはこの保存結果だけを使う。
- `verification_notes.json`: 検証の要約と、確信できない箇所・未解決の記録。

## 帰属（OpenStreetMap）

形状データは © OpenStreetMap contributors。Open Database License (ODbL) 1.0 の下で提供されています。
https://www.openstreetmap.org/copyright
本ディレクトリのデータ（`osm/` とそこから導出した `routes.json`）を再利用する場合は、同じ帰属表示を付け、ODbL の条件に従ってください。
なお、コースの寸法・坂・高低差などの数値は OSM ではなく、`../jra_courses.json`（JRA 公式資料等から確定したコースマスタ）が正です。
