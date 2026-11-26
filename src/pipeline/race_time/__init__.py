"""基準の走破時計の関数（コース・クラス・年齢の条件・頭数・前半のペース）と、その入力の作り。

共通の部品: レース結果の欄のレースレベル・開催日レベル・メンバーレベル、track-bias の基準線、
血統の「時計の差」、本番のモデルの前半 600m の特徴量（train-serve-skew の D-1 の学び直し）が同じものを使う。
- features.front_time_at(lap_times, distance, at_m=600): ラップから任意の距離の通過時計（最初の区間が 100m・50m の距離も正しく）
- features.race_table(years, tables_root): レースごとの入力の表
- model.RaceTimeModel: 単調の制約つきの当てはめと予測（係数は config/race_time_model/）
"""
