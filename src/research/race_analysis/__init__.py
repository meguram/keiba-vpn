"""レースごとの分析結果（めぐ分析）。出走馬に紐づく分析を『モジュール』として登録し、1 本の API で返す。

新しい分析を足すときは、base.AnalysisModule を実装したクラスを作り、registry.py の MODULES に 1 行足す。
画面（frontend/components/analysis）は、モジュールの列定義（columns / groups）から表を組み立てるので、
列だけで表せる分析は画面の変更が要らない。
"""
