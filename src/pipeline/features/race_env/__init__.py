"""レース環境 ENV（場の水準・偏り・隊列）の特徴量。予想家の再現戦略の層 A・B（docs/ideas/inbox/Ninjin/predictor_profile.html 第 6 章）。

学習用の表と推論用の 1 日分を同じ関数で作る。すべて「そのレースより前に結果が確定したレース」だけを使う（asof.py）。
"""
