"""並走期間の GCP Jobs の step が起こした子プロセス（subprocess の python）にも、本番 GCS への書き込みガードを入れる。

``src.ops.parallel_step`` がこのフォルダを ``PYTHONPATH`` の先頭に足し、``KEIBA_PARALLEL_GUARD=1`` を立てる。Python は起動時に
``sitecustomize`` を自動で読み込むので、step の孫・ひ孫のプロセスも ``shadow_storage.install(root="parallel")`` を通る。
ガードを入れられなければ（google-cloud-storage 等が無い軽いプロセス）、そのプロセスは GCS を使わないので何もしない。
"""

import os

if os.environ.get("KEIBA_PARALLEL_GUARD") == "1":
    try:
        from src.ops import shadow_storage as _ss

        _ss.install(root=_ss.PARALLEL_ROOT)
    except ImportError:
        pass
