# ConoHa VPS に prod の UI を載せる手順書（stg → prod の昇格フロー）

最終更新: 2026-10-21。対象: ConoHa VPS `<VPS_IP>`（Ubuntu 22.04.3 LTS、メモリ 2GB / CPU 3Core / SSD 100GB、**SSH はポート <SSH_PORT>・鍵のみ・管理ユーザー `<VPS_ADMIN_USER>`（sudo はパスワードが要る）**、36か月前払い・2028/05/18 まで）。ユーザーの回線は VPS への SSH でタイムアウトするため、**ホスト機（keiba-remote）を踏み台（ProxyJump）にする**。
関連: `architecture_target.md`（常駐処理を GCP へ移す将来構成。別 fork が作成中。ここでは参照のみ）、`go_live.md`（一般公開）、`maintenance_deploy.md`（メンテ窓の自動反映）、`anti_scraping.md`、`slack_notifications.md`。
スクリプト: `scripts/ops/vps/`（一覧は第 12 節）。**このリポジトリのコードは VPS へ自動では入らない。以下はユーザーが VPS 上・ホスト上で実行する手順**。

## 0. 結論（要約）

| 項目 | 決定 |
|---|---|
| 呼び名 | **prod** = ConoHa VPS の UI（ユーザーが開く）、**stg** = ホスト機の UI（現在 `app.megu-keiba.com` で開いているもの）。見られる情報（API・データ）は同じ |
| ホスト名（既定。変数で変更可） | prod: `app.megu-keiba.com` ／ stg: `stg.megu-keiba.com` ／ 上流 API: `api-internal.megu-keiba.com`（Service Auth のみ） ／ 切替前の確認用: `prod-test.megu-keiba.com` ／ SSH: `ssh.megu-keiba.com`（Tunnel 経由） |
| VPS に載せるもの | Next.js（standalone）+ nginx（127.0.0.1 のみ）+ cloudflared（新トンネル `meguai-prod`）。**状態を持たない**（データ・API はホスト機のまま） |
| API の取り方 | VPS の nginx が `api-internal.megu-keiba.com` へ中継し、**Access のサービストークン**（CF-Access-Client-Id/Secret）を **VPS のサーバー側だけ**で付ける（案 b。第 3 節） |
| ビルド | **VPS では行わない**。ホスト機か CI で `build_release.sh`、tar.gz を VPS に配布 |
| リリース | `releases/<id>` と `current` のシンボリックリンク。**失敗時は自動で直前の版に戻す**。手動でも `rollback.sh` で即時 |
| 昇格 | stg はメンテ窓の自動反映、**prod は手動の `promote.sh`**（承認フラグ付きの窓内自動昇格は任意） |
| 公開面 | VPS は受信を全部閉じ、cloudflared の外向き接続だけで公開。SSH は Tunnel 経由か許可 IP のみ。ConoHa のセキュリティグループと ufw の**二重**で管理 |
| 追加費用 | 0 円（VPS は前払い済み、Cloudflare は無料枠）。月額予算 2,000 円以内の枠を使わない |

最初の小さな一歩（**10 分、何も壊さない**）: ConoHa の『イメージ保存』で旧サーバーを保存 → `vps_00_inventory.sh` で棚卸し → 旧サイトの `nginx/apache/certbot` を停止（第 4.6 節）。

## 1. 構成図

```
ブラウザ ──https──> Cloudflare ──(Access: メール OTP)──> Tunnel meguai-prod ──> [VPS] cloudflared
                                                                              │
                                                                              ▼
                                                                  nginx 127.0.0.1:8080
                                                           ┌──────────────┴───────────────┐
                                                  ページ・静的 │                                │ /api/ /data/image/
                                                           ▼                                ▼
                                                Next.js 127.0.0.1:3000      https://api-internal.megu-keiba.com
                                                 (systemd: keiba-ui)         （CF-Access サービストークンを付ける）
                                                                                        │
                                          Cloudflare ──(Access: Service Auth のみ)──> Tunnel meguai-stg ──> [ホスト機] Next.js :23200
                                                                                                                    ├ /api/v1 → Flask :23210
                                                                                                                    └ /api    → FastAPI :23220
stg: ブラウザ ──> stg.megu-keiba.com ──(Access: メール OTP)──> Tunnel meguai-stg ──> [ホスト機] Next.js :23200
```

将来 API を GCP に移したら、**nginx の `UPSTREAM_HOST` と Access のサービストークンを差し替えるだけ**で切り替わる（UI のコードは変えない）。

## 2. ホスト名と環境バッジ

- ホスト名は `scripts/ops/vps/vars.env.example`（→ `vars.env`）の変数で持つ。別の名前にしたいときはここだけ直す（Cloudflare の route / Access も同じ名前にそろえる）。
- **stg には画面左上に小さな黄色の『STG』バッジ**が出る（prod は何も出ない）。`frontend/next.config.js` が `KEIBA_ENV=stg` のビルドで `NEXT_PUBLIC_ENV_LABEL=STG` を埋める（明示指定が優先）。反映は次の stg ビルド（メンテ窓の自動反映）で入る。
- ブックマークの扱い: 今の `app.megu-keiba.com` は **prod になる**。stg は `stg.megu-keiba.com` に変わるので、検証用のブックマークだけ作り替える。

## 3. API への接続（最重要）

VPS の UI は、データを取る API（ホスト機の FastAPI/Flask）に届く必要がある。VPS からホスト機の `localhost` には届かない。

| 案 | 内容 | 利点 | 欠点 |
|---|---|---|---|
| (a) | VPS が `stg.megu-keiba.com/api/...` を呼ぶ | 新しいホスト名が要らない | 人間用の stg に機械の経路を相乗りさせる。Access のポリシーが混ざる（人の Allow と Service Auth）。stg を直したときに prod が巻き込まれる |
| **(b) 推奨** | 機械専用のホスト名 `api-internal.megu-keiba.com`（ホスト機の Next.js :23200 へのトンネル route）に **Service Auth だけ**の Access をかける | 人用と機械用が分かれる。トークンの失効・ローテーションが独立。将来 GCP に差し替えやすい（このホスト名の route を付け替えるか、nginx の上流を替える） | Cloudflare の設定が 1 組増える |
| (c) | 将来の GCP API | 常駐の可用性 | `architecture_target.md` の判断待ち |

**サービストークンはブラウザに出さない**。nginx（VPS のサーバー側）が上流へのリクエストにだけ `CF-Access-Client-Id/Secret` を付ける。ブラウザは同一オリジン（`app.megu-keiba.com`）に `/api/...` を呼び、nginx が中継する（CORS 不要、Cookie は prod のホストだけに紐づく）。

### 3.1 フロントエンドの変更（最小）

- `next.config.js` は、`KEIBA_API_URL` / `KEIBA_FASTAPI_URL` を**ビルド時に固定**する既存の仕組み。prod のビルドでは `http://127.0.0.1:8080`（VPS の nginx）を渡す（`build_release.sh` が行う）。ページのサーバー側の API 呼び出しと、`middleware.ts` のメンテ判定も同じ宛先に向く。
- 追加したのは、`NEXT_OUTPUT=standalone` のときだけの自己完結の出力と、環境バッジだけ。**既定の挙動は変えない**。

### 3.2 クライアント IP とスクレイピング対策（壊れやすい点）

VPS 経由の要求は、上流（ホスト機）から見ると**全員 VPS の IP（<VPS_IP>）**に見える（Cloudflare が付ける `CF-Connecting-IP` は VPS の IP）。放置すると、IP 単位のレート制限が全ユーザーを 1 人として数えて巻き添えにする。

- nginx は、VPS 自身の入口で **実クライアント IP**（`real_ip_header CF-Connecting-IP`）を取り、レート制限（`limit_req`）をそこで行う。
- 上流には `X-Keiba-Client-IP: <実 IP>` を付けて渡す。ホスト機の `anti_scraping` は、**`config/anti_scraping.json` の `edge_sources` に VPS の IP を入れたときだけ**、`cf-connecting-ip` が VPS の IP の要求に限って、このヘッダーを採用する（他の接続元が同じヘッダーを付けても無視。テスト: `tests/api/test_anti_scraping_edge.py`）。
- 手順（ホスト機、反映は 30 秒以内）: `"edge_sources": ["<VPS_IP>"]` を追記する。**入れるまでは従来どおり**（VPS の IP 1 件として数える）。まず `mode` を `detect` のまま、ログを見る。

## 4. VPS の構成

### 4.0 VPS の現状（実測 2026-10-21 19:27 JST）

| 項目 | 実測 |
|---|---|
| OS | Ubuntu 22.04.3 LTS x86_64（OpenSSH 8.9）。**Docker・Node.js は入っていない** |
| 資源 | メモリ 1,962MB（使用 646 / 利用可 1,159）、CPU 3、`/` は 99GB 中 21GB 使用。直前に再起動済み |
| ユーザー | `<VPS_ADMIN_USER>`（uid 1000、groups: sudo, ssh）。**sudo はパスワードが要る**（NOPASSWD なし）。root の SSH は不可、パスワード認証は無効（公開鍵のみ） |
| 待受 | `0.0.0.0:<SSH_PORT>`（SSH）、`:80`/`:443`（nginx）、`127.0.0.1:8000`（uvicorn） |
| 稼働中 | `fail2ban`、`ntp`、**旧サイト**: `nginx`（`/etc/nginx/conf.d/myapp_nginx.conf`）、`uwsgi.service`（megu-ai.megu-keiba.com）、**`keiba-api.service`**（FastAPI/uvicorn 2 ワーカー、`/var/www/megu-keiba/keiba-vpn`）、`certbot.timer`（証明書は 10/17 期限切れ） |
| ディスク | `/var/www` 6.3GB（このリポジトリの**古いコピー**、`myapp_*.conf/ini`、`scripts/`）、`/opt/venv` 2.0GB、`/home/<VPS_ADMIN_USER>` 1.8GB |

つまり旧サイトは『このプロジェクトの初期版（FastAPI の Jinja 画面）』を VPS に置いたもの。ユーザーは**使わない**意向。旧 uvicorn（2 ワーカー）と uwsgi を止めれば、メモリは 1.5GB 以上空く（Next.js standalone に十分）。

OS は **Ubuntu 22.04**。手順は 24.04 でも同じ。

### 4.1 作業の順序と締め出し防止

1. **バックアップ**（4.2）。ConoHa の『イメージ保存』か、`backup_old_site.sh`（任意）で旧データを tar にして取り出す。
2. 別のターミナルを**開いたまま**作業する（今のセッションは、設定を変えても切れない）。
3. 新しい経路（`deploy` の鍵ログイン、Tunnel 経由の SSH）を**先に確認してから**、古い経路（公開ポート）を閉じる。
4. 困ったら **ConoHa のコンソール（VNC）**で入る（`<VPS_ADMIN_USER>` のローカルのパスワード、root のパスワードのどちらかが要る。パスワードマネージャに保管しておく。SSH でパスワード認証が無効でも、コンソールのログインには関係しない）。

**接続の仕方**（回線の事情でホスト経由）:
```bash
# 手元の PC（~/.ssh/config に ssh_config.example の rakuten-server を追記。ProxyJump keiba-remote が肝）
ssh rakuten-server                       # = ssh -p <SSH_PORT> -o ProxyJump=keiba-remote <VPS_ADMIN_USER>@<VPS_IP>
scp -r scripts/ops/vps rakuten-server:/tmp/vps      # スクリプトを VPS へ（ホスト経由）
# 以降 VPS 上。sudo はパスワード入力が要る（実行のたびに聞かれる。自動化が必要な箇所は deploy の限定 sudoers だけ。4.5）
sudo bash /tmp/vps/vps_01_base_setup.sh ...
```

### 4.2 棚卸しとバックアップ

```bash
# VPS 上（<VPS_ADMIN_USER>）。棚卸しは読み取り専用
sudo bash /tmp/vps/vps_00_inventory.sh        # 動いているサービス、待受ポート、cron、証明書の期限、SSH 設定を /root/vps_inventory_*.txt に

# バックアップ（どちらか、または両方）
# (1) ConoHa のコントロールパネルで『イメージ保存』（保存先の容量の課金を確認）
# (2) 任意: 旧データを tar にして取り出す（/var/www/megu-keiba 6.3GB と /home/<VPS_ADMIN_USER> 1.8GB。仮想環境・ログ・ssh 鍵は除く）
bash /tmp/vps/backup_old_site.sh              # → ~/old_site_backup/*.tar.gz と SHA256SUMS
# 手元/ホストに取り出す（ホスト機 keiba-remote で。ホスト機には ~/.ssh/config の rakuten-server と同じ内容が要る）
scp -P <SSH_PORT> -i <鍵> <VPS_ADMIN_USER>@<VPS_IP>:old_site_backup/* /path/to/backup/
sha256sum -c SHA256SUMS_*                      # 取り出した後に確認。確認前に VPS を削除・再構築しない
```

### 4.3 初期設定

```bash
# ① ホスト機（keiba-remote）で、配布専用の SSH 鍵を作る（promote.sh はホスト機から VPS に入るため、鍵はホスト機に置く）
ssh keiba-remote 'ssh-keygen -t ed25519 -N "" -f ~/.ssh/keiba_vps_ed25519 -C keiba-host-deploy && cat ~/.ssh/keiba_vps_ed25519.pub'
#    表示された公開鍵（ssh-ed25519 AAAA… の 1 行。秘密鍵ではない）を、手元で /tmp/deploy.pub というファイルにして VPS に渡す
scp /tmp/deploy.pub rakuten-server:/tmp/deploy.pub

# ② VPS 上（<VPS_ADMIN_USER>）。ホスト機のグローバル IP からだけ SSH を許可する（手元の PC もホスト機経由で入るので、これで足りる）
HOST_IP=$(ssh keiba-remote 'curl -s -m 5 https://api.ipify.org')       # 例: 175.177.45.24（家庭回線だと変わることがある。変わったら 4.4 の B へ）
sudo bash /tmp/vps/vps_01_base_setup.sh --ssh-mode ip:$HOST_IP --deploy-pubkey-file /tmp/deploy.pub --admin-user <VPS_ADMIN_USER>

# ③ 別ターミナルで deploy の鍵ログインを確認（ホスト機から: ssh -p <SSH_PORT> -i ~/.ssh/keiba_vps_ed25519 deploy@<VPS_IP> true）
#    OK なら、パスワード認証・root ログインを無効にし、AllowUsers を deploy と <VPS_ADMIN_USER> だけにする（実際は既に鍵のみ・root 不可。念押し）
sudo bash /tmp/vps/vps_01_base_setup.sh --ssh-mode ip:$HOST_IP --deploy-pubkey-file /tmp/deploy.pub --admin-user <VPS_ADMIN_USER> --lock-ssh
```

行うこと: OS 更新、`deploy`（鍵ログイン専用）と `keibaui`（サービス実行用・ログイン不可）の作成、swap 2GB、`ufw`（受信既定拒否、SSH は指定 IP のみ）、`fail2ban`（<SSH_PORT>）、自動セキュリティ更新。`<VPS_ADMIN_USER>` はそのまま残り、管理（sudo）は従来どおり**パスワード入力**で行う。`deploy` が行えるのは第 4.5 節の 3 スクリプトだけ。

### 4.4 SSH の公開をどうするか（比較）

| 案 | 内容 | 手間 | 安全性 |
|---|---|---|---|
| A | **ホスト機のグローバル IP だけ**に SSH(<SSH_PORT>) を開ける（ufw と ConoHa のセキュリティグループ）。手元の PC もホスト機経由（ProxyJump）で入るので、許可するのはホスト機の IP 1 つで足りる | 小 | ホスト機の IP が変わると締め出される（コンソールで救出）。ホスト機が落ちると入れない |
| **B 推奨（安定後）** | 受信を**全部閉じる**。SSH は **Cloudflare Tunnel 経由**（`ssh.megu-keiba.com` → `ssh://localhost:<SSH_PORT>`、Access で保護）。`cloudflared access ssh` を手元に入れる | 中 | 公開ポートが 0 になる。ConoHa のコンソールが保険 |
| C | <SSH_PORT> を全体に公開（鍵のみ + fail2ban） | 最小 | 常時スキャンを受ける。初期設定の間だけ |

進め方: 最初は A（ホスト機の IP）。ホスト機の IP が変わりやすいなら B に移る（Tunnel 経由の SSH が確認できたら `--ssh-mode closed`）。`~/.ssh/config` は `ssh_config.example` を参照（`rakuten-server`=手元→ホスト経由、`keiba-vps`=ホスト機→VPS）。

### 4.5 ソフトウェア

```bash
# VPS 上、root（vars.env を /etc/keiba-ui/vars.env に置いてから）
bash vps_02_install_stack.sh        # Node.js 22（公式バイナリ + SHA256 検証）、nginx、cloudflared（Cloudflare 公式 apt）、systemd、sudoers
```

- **Next.js は systemd で常駐**（`keiba-ui.service`: 専用ユーザー、`Restart=always`、`MemoryHigh=700M / MemoryMax=900M`、`NODE_OPTIONS=--max-old-space-size=384`）。standalone 出力なので実行時メモリは小さい。
- **nginx は 127.0.0.1:8080 のみ**で待ち受け、cloudflared の転送先になる。
- **cloudflared は `cloudflared-prod.service`**。トークンは `/etc/cloudflared/prod.env`（600）。
- **sudo にパスワードが要る前提**: `<VPS_ADMIN_USER>` が VPS 上でスクリプトを実行するときは、そのつど sudo のパスワードを入力する。**自動化が必要なのは配布だけ**なので、`deploy` に `install_release.sh` / `rollback.sh` / `healthcheck.sh` の 3 つだけを NOPASSWD にする（`/etc/sudoers.d/keiba-deploy`、`vps_02` が作る。`/opt/keiba-ui/bin/` は root 所有で `deploy` は書き換えられない）。`<VPS_ADMIN_USER>` 自体に NOPASSWD は付けない。
- Node.js は入っていないので、`vps_02` が公式バイナリ（SHA256 検証）を `/opt/node` に入れる。旧 `/opt/venv`（Python の仮想環境 2.0GB）は触らない（旧サービス停止後に不要なら削除を検討）。

### 4.6 旧サイトの停止

旧サイト『ML-AutoPilot Keiba』は不要（ユーザー確認済み）。実体は `keiba-api.service`（uvicorn 127.0.0.1:8000）、`uwsgi.service`、`nginx`（80/443）、`certbot.timer`。証明書は 2026-10-17 に期限切れで、更新も止まっている。

```bash
sudo bash /tmp/vps/retire_old_site.sh                                 # 止める候補（keiba-api, uwsgi, nginx, certbot.timer…）を表示（何も変えない）
sudo bash /tmp/vps/retire_old_site.sh --apply --i-have-a-backup       # バックアップの後。stop + disable（データ・設定は残す）
```

データ（`/var/www/megu-keiba`、`/opt/venv` など）は残す。数週間問題がなければ削除を検討する。nginx は新構成でも使うので、`vps_02` が旧設定（`conf.d/myapp_nginx.conf` など）を `/root/old_site_configs/` に**退避**してから入れ直す（80/443 を握り続けないため）。

### 4.7 2GB のメモリ対策

| 対策 | 内容 |
|---|---|
| ビルドしない | `next build` はメモリ 2GB 超を使うことがある。ビルドはホスト機/CI |
| standalone | 依存を絞った出力（実行時 100〜250MB 程度） |
| swap 2GB・`swappiness=10` | 一時的な山をしのぐ（`vps_01`） |
| systemd の上限 | `MemoryMax=900M` で暴走しても VPS 全体を巻き込まない |
| ヘルスチェック | 空きメモリ 150MB 未満を警告（5 分おき） |

## 5. リリース（stg → prod の昇格）

### 5.1 流れ

```
(stg) 変更をコミット → メンテ窓の自動反映で stg に入る → stg で確認 ─┐
                                                                         ▼ 承認（人間）
ホスト機: bash scripts/ops/vps/promote.sh --commit <hash>
  1. build_release.sh   git のコミットから /tmp で standalone ビルド → tar.gz + sha256 + manifest
  2. scp                VPS の /srv/keiba-ui/incoming へ
  3. install_release.sh sha256 検証 → 展開 → 別ポート(3101)で起動テスト → current を切替 → 再起動 → 確認 → 失敗なら自動で戻す
  4. 履歴              data/local/meta/vps_promote_history.jsonl に記録
```

### 5.2 stg との連携

- stg は `maintenance_deploy.md` のメンテ窓（JST 03:00–04:00）の自動反映。**prod への昇格は人が決める**（同じコミットを `promote.sh` に渡す）。
- 承認フラグ付きの窓内自動昇格（任意）: `promote_if_approved.sh`（`data/local/deploy/promote_approved.json` があり期限内で、JST 03 時台のときだけ実行）。**cron には登録していない**。使うなら承認が必要。
- 昇格のたびに、prod と stg は**同じコミット**になるのが原則。ずれの確認は `compare_versions.sh`（prod の `/version.json`、直近の昇格履歴、手元の HEAD）。prod の `https://app.megu-keiba.com/version.json`（Access を通ったあと）でも見られる。

### 5.3 ロールバック

```bash
ssh keiba-vps 'sudo /opt/keiba-ui/bin/rollback.sh'            # 直前の版へ
ssh keiba-vps 'ls -1t /srv/keiba-ui/releases'                 # 世代の一覧（5 世代残す）
ssh keiba-vps 'sudo /opt/keiba-ui/bin/rollback.sh keiba-ui-xxxxxxxx-202610220300'   # 指定した版へ
```

導入時の失敗（起動テスト・切替後の確認）は、`install_release.sh` が**自動で**直前の版に戻す。切替は `current` のシンボリックリンクの付け替え + `systemctl restart`（数秒）。

## 6. 監視・運用・バックアップ

- **死活**: `keiba-ui-health.timer`（5 分おき）が、`keiba-ui` / `nginx` / `cloudflared-prod` の稼働、UI の応答、上流 API への到達、ディスク（85%）、空きメモリ（150MB）を確認し、落ちていれば 1 回再起動する。結果は `/var/lib/keiba-ui/health.json`。
- **通知**: `/etc/keiba-ui/slack.env` に `SLACK_WEBHOOK_URL=` を置くと、異常を Slack に送る（同じ内容は 1 時間に 1 回まで）。置かなければ通知しない。値は `set_secret.sh` で渡す。外部の稼働監視は Access の背後なので使えない（Cloudflare の通知機能か、ホスト機の `scrape_status_notify` から VPS の `health.json` を SSH で読む案）。
- **ログ**: `journalctl -u keiba-ui`、`/var/log/keiba-ui/deploy.log`（導入・戻しの記録）、logrotate 8 週。
- **バックアップ**: VPS は**状態を持たない**。バックアップするのは (1) `/etc/keiba-ui/`（秘密以外の設定）、(2) リリースの tar.gz（ホスト機に残る）。**秘密（トンネルのトークン、サービストークン）は Cloudflare の画面で再発行できる**ので、VPS には残すがバックアップはしない。
- **VPS を一から作り直す（目標 30 分）**: ①ConoHa でサーバー再構築（Ubuntu 22.04）3 分 → ②`vps_01`（5 分）→ ③`vps_02`（5 分）→ ④トークン 2 種を渡す（2 分）→ ⑤`promote.sh` で現行の版を配布（5 分）→ ⑥Cloudflare の route を確認（2 分）。
- **cloudflared / nginx の自動再起動**: systemd の `Restart=always`。

## 7. 認証・Cookie・Access のサービストークン

- ブラウザ → prod は、Cloudflare Access の**メール認証**（stg と同じ owner ポリシーを `app` のアプリに設定）。人を増やすときは Access の許可リストにメールを足す（`go_live.md`）。
- アプリ内の認証（`keiba_dev_session` などの Cookie）は、上流の API が発行する。nginx が `Set-Cookie` をそのまま通し、`proxy_redirect` で `Location` の上流ホスト名を相対に直す。Cookie の `Domain` は付かないので、**prod のホストだけ**に紐づく（stg のセッションとは別）。
- サービストークンは `api-internal.megu-keiba.com` の Access アプリ（Service Auth のみ）で作る。**1 トークン = 1 用途**（VPS の UI 専用）。失効・ローテーション（Cloudflare の画面で再発行 → `set_secret.sh` で VPS に渡す → `render_nginx_access.sh`）。
- 人間用の stg（`stg.megu-keiba.com`）と機械用（`api-internal`）を**同じ Access アプリに混ぜない**。

## 8. ConoHa のセキュリティグループ（コントロールパネル側のファイアウォール）と ufw

現状（実測）: セキュリティグループ **`SSH-<SSH_PORT>`**（SSH ポート <SSH_PORT>）と **`IPv4v6-Web`**（80/443）が付いている。<SSH_PORT> も 80/443 もインターネットから到達できる。

| 段階 | ConoHa のセキュリティグループ | ufw（VPS 内） |
|---|---|---|
| 今 | SSH-<SSH_PORT> + IPv4v6-Web | （未設定） |
| 初期設定中 | そのまま | `--ssh-mode ip:<自分の IP>` または `keep`。80/443 は deny（旧サイトは外から見えなくなる） |
| 新構成が動いた後 | **IPv4v6-Web を外す**（80/443 を閉じる。cloudflared は外向きなので不要） | そのまま |
| Tunnel 経由の SSH が確認できた後 | **SSH-<SSH_PORT> を外す**（または自分の IP だけを許可する新しいグループに付け替える） | `--ssh-mode closed` |

- 付け替えの場所: ConoHa のサーバー詳細画面の『ネットワーク』または『セキュリティグループ』の項目（パネルの版で名称が違う。見当たらない場合は画面を送ってもらえれば場所を案内する）。**グループを外す前に、新しい経路が使えることを確認する**（二重に閉じると、ConoHa のコンソール以外では入れなくなる）。
- **二重設定の注意**: セキュリティグループで開けても ufw が拒否していれば届かない（逆も同じ）。切り分けは `ss -ltn`（待受）→ `ufw status verbose` → セキュリティグループの順。
- **救出手順**: SSH で入れなくなったら、ConoHa のコンソール（VNC）で root のパスワードで入り、`ufw allow <SSH_PORT>/tcp` か `ufw disable` → SSH で入り直す。

## 9. ユーザー作業と Claude の代行

秘密（トークン、サービストークン、SSH 鍵）は**チャットに貼らない**。渡し方は『標準入力で `set_secret.sh` に流す』（値は画面・ログ・`ps` に出ない。`.env` 型のファイルの末尾改行の落とし穴も回避する）。

| # | 作業 | 誰が |
|---|---|---|
| 1 | ConoHa で『イメージ保存』（旧サーバーのバックアップ） | **ユーザー** |
| 2 | ホスト機での SSH 鍵作成と公開鍵の受け渡し、`vps_00`〜`vps_02` の実行（**sudo のパスワード入力が要るためユーザーが実行**） | **ユーザー**（コマンドは第 4 節）。出力を貼ってもらえれば Claude が確認。読み取りは Claude が `ssh rakuten-server` 経由で実行できる |
| 3 | Cloudflare: トンネル `meguai-prod` の作成 → トークンを `set_secret.sh` で VPS に | **ユーザー**（画面の操作は第 10 節） |
| 4 | Cloudflare: ホスト機のトンネル（`meguai-stg`）に route `stg.` / `api-internal.` を追加、Access のアプリ作成、**サービストークン作成**（Client ID/Secret は一度しか見えない） | **ユーザー** |
| 5 | サービストークンを VPS に渡す、`render_nginx_access.sh` | **ユーザー**（コマンドは `vps_02` の最後に表示） |
| 6 | ホスト機の `config/anti_scraping.json` に `edge_sources`（VPS の IP）を追記 | **Claude 代行可**（ホスト機の設定。承認後） |
| 7 | `promote.sh` の実行、`build.env` の作成（stg と NEXT_PUBLIC_* をそろえる） | **Claude 代行可**（ホスト機で実行） |
| 8 | ConoHa のセキュリティグループの付け替え | **ユーザー**（画面） |
| 9 | 動作確認（シークレットウィンドウでメール認証 → 画面、スマホ） | **ユーザー** |
| 10 | stg の `app` → `stg.` への付け替えの後始末、ブックマーク | **ユーザー** |

## 10. 段階的な切替計画

### ステップ 0: 事前準備（約 30 分）
- [ ] ConoHa の『イメージ保存』を取る（ストレージ課金の有無を確認）。必要なら `backup_old_site.sh` で旧データを tar にしてホストへ取り出す。
- [ ] `vps_00_inventory.sh` で棚卸し、`retire_old_site.sh --apply --i-have-a-backup` で旧サイト（keiba-api / uwsgi / nginx / certbot）を止める。
- [ ] 戻し方: ConoHa で『イメージ』から復元、`systemctl start nginx apache2 certbot.timer`。

### ステップ 1: VPS の初期設定（約 30 分）
- [ ] 鍵ログイン（`deploy`）、`vps_01`、`vps_02`。
- [ ] 確認: `ss -ltn`（外から見える待受が SSH だけ）、`ufw status`、`systemctl status nginx`、`node -v`。
- [ ] 戻し方: `ufw disable`、`/etc/ssh/sshd_config.d/99-keiba.conf` を削除して `systemctl reload ssh`。

### ステップ 2: UI を載せる（prod-test で確認）
Cloudflare（ユーザー）:
1. Zero Trust → Networks → Tunnels → **Create a tunnel**（Cloudflared、名前 `meguai-prod`）。表示されるコマンドのトークンを、VPS で `set_secret.sh` に流す（`printf '%s' '<コマンド全体>' | sudo bash /opt/keiba-ui/bin/set_secret.sh /etc/cloudflared/prod.env TUNNEL_TOKEN` → `sudo systemctl restart cloudflared-prod`）。Connection Status が Connected になる。
2. ホスト機の `meguai-stg` に route を追加: `api-internal.megu-keiba.com` → `http://localhost:23200`、`stg.megu-keiba.com` → `http://localhost:23200`。
3. Access: (i) 既存アプリ `app` の Destinations に `stg.megu-keiba.com` を追加（ポリシー `owner` はそのまま）、(ii) 新規アプリ `api-internal`（Self-hosted、`api-internal.megu-keiba.com`）にポリシー **Action = Service Auth** のみ（Include = 作成したサービストークン）、(iii) アプリ `prod-test`（`prod-test.megu-keiba.com`、ポリシー `owner`）。
4. サービストークン作成: Access → Service credentials → Create Service Token（名前 `vps-ui`、期間 1 年）。**Client ID / Secret をコピーして VPS に渡す**（第 7 節）。
5. `meguai-prod` に route: `prod-test.megu-keiba.com` → `http://localhost:8080`。

ホスト機（Claude 代行）:
6. `build.env` を stg にそろえて `promote.sh --commit <hash>`。
7. 確認: `https://prod-test.megu-keiba.com`（メール認証 → 画面が出る、ログインが通る、予測 API が返る、`/version.json`）。`curl http://127.0.0.1:8080/api/maintenance/status`（VPS 上）が 200。
8. 戻し方: `meguai-prod` の route と `prod-test` を消すだけ（本番に影響なし）。

### ステップ 3: app を VPS に付け替える（数分の切替）
1. stg の動作確認（`https://stg.megu-keiba.com`）が済んでいること。
2. Cloudflare: `meguai-stg` の Routes から `app.megu-keiba.com` を**削除**（DNS の CNAME も消える）。
3. すぐ `meguai-prod` に route `app.megu-keiba.com` → `http://localhost:8080` を追加（既存の DNS レコードを上書きするか聞かれたら承認）。**この間（1〜2 分）`app` は見られない**ので、メンテ窓（JST 03:00–04:00）を推奨。
4. Access のアプリ `app`（owner ポリシー）はそのまま有効（ホスト名が同じ）。
5. 確認: `https://app.megu-keiba.com/version.json` が prod の版。画面左上に STG バッジが**出ない**。`https://stg.megu-keiba.com` には STG バッジ（次の stg ビルド後）。
6. 戻し方: `meguai-prod` から `app` の route を消し、`meguai-stg` に `app.megu-keiba.com` → `http://localhost:23200` を戻す（1〜2 分）。

### ステップ 4: 安定確認（1 週間）
- [ ] `health.json` が毎日 ok、Slack に異常通知が出ていない。
- [ ] `deploy.log` に失敗なし。メンテ窓（JST 03:00–04:00）に prod も『メンテナンス中』表示になり、終了後に復帰する。
- [ ] anti_scraping のログ（`data/local/meta/anti_scraping/`）で、ユーザー単位の制限になっている（VPS の 1 IP に集約されていない）。`edge_sources` 設定後。
- [ ] ConoHa の `IPv4v6-Web`（80/443）を外す。Tunnel 経由の SSH を確認して `SSH-<SSH_PORT>` を外し、`--ssh-mode closed`。
- [ ] 戻し方: ステップ 3 の戻し方。

## 11. ユーザーに決めてもらうこと

1. **ホスト名**: prod=`app.megu-keiba.com`、stg=`stg.megu-keiba.com` でよいか（別の名前なら `vars.env` を変えるだけ）。
2. **SSH の公開方法**: A（ホスト機の IP だけ許可）／ B（Tunnel 経由）／ C（公開、暫定）。ホスト機のグローバル IP は固定か（変わるなら B）。
3. **ログイン用ユーザー**: 確認済み（`<VPS_ADMIN_USER>`、鍵のみ、sudo はパスワード要）。VNC（ConoHa のコンソール）の救出用に、`<VPS_ADMIN_USER>` か root のローカルのパスワードを保管しているか。
4. **VPS の OS**: Ubuntu 22.04 で確認済み（再構築しない場合は不要）。
5. **自動更新**: VPS の自動更新（2028/05/18）は OFF のままでよいか（使い続けるなら期限前に判断）。
6. **昇格の承認**: prod への昇格は常に手動か、承認フラグ付きの窓内自動昇格を許すか。
7. **ホスト機の常時稼働**: UI は VPS に載っても、API（上流）はホスト機。ホスト機が止まれば prod は API エラーになる（UI は出るがデータが出ない）。`architecture_target.md` の GCP 化までの許容時間。

## 12. スクリプトと設定の一覧

| ファイル | 実行場所 | 役割 |
|---|---|---|
| `scripts/ops/vps/vars.env.example` | — | ホスト名・ポート・保持世代の変数 |
| `vps_00_inventory.sh` | VPS（root） | 棚卸し（読み取り専用） |
| `retire_old_site.sh` | VPS（root） | 旧サイト（keiba-api / uwsgi / nginx / certbot）の停止（`--apply --i-have-a-backup` で実行） |
| `backup_old_site.sh` | VPS（<VPS_ADMIN_USER>） | （任意）旧データを tar + SHA256 で退避 |
| `vps_01_base_setup.sh` | VPS（root） | OS・ユーザー・swap・ufw・fail2ban・SSH |
| `vps_02_install_stack.sh` | VPS（root） | Node.js・nginx・cloudflared・systemd・sudoers |
| `set_secret.sh` | VPS | 秘密を標準入力から安全に書く（値を出さない・末尾改行を保証） |
| `render_nginx_access.sh` | VPS（root） | サービストークンから nginx の include を作って reload |
| `install_release.sh` / `rollback.sh` | VPS（sudo） | リリースの導入（自動ロールバック付き）／即時の戻し |
| `healthcheck.sh` | VPS（systemd timer） | 死活・資源の確認、Slack 通知 |
| `build_release.sh` | ホスト機/CI | standalone ビルドと tar.gz |
| `promote.sh` / `promote_if_approved.sh` / `compare_versions.sh` | ホスト機/開発 PC | 昇格・（任意）承認付き昇格・版の比較 |
| `ssh_config.example` / `build.env.example` / `templates/` | — | SSH 設定の例、stg とそろえる NEXT_PUBLIC_*、systemd/nginx の雛形 |
| `frontend/next.config.js` / `components/EnvBadge.tsx` / `lib/envLabel.ts` | リポジトリ | standalone 出力（`NEXT_OUTPUT=standalone`）、STG バッジ |
| `src/utils/anti_scraping.py` / `config/anti_scraping.json` | リポジトリ | `edge_sources` / `edge_ip_header`（既定は空=従来どおり） |

## 13. リスクと限界

- **ホスト機が止まれば prod のデータは出ない**（UI は VPS で動くが、API はホスト機）。可用性の根本解決は GCP 化（`architecture_target.md`）。
- 2GB の VPS は、Next.js 1 本と nginx、cloudflared には十分だが、**ほかのものを同居させない**。
- 画面の内容（予測・期待値など）を不特定多数に見せる前に、`go_live.md` の公開用画面仕様とデータ利用の確認が要る。この文書の prod は、**Access で保護された限定公開**を前提にしている。
- 本文書の画面操作（Cloudflare・ConoHa）は、パネルの版で名称が変わる。違う場合は画面を送ってもらえれば案内する。
