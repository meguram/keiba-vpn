/** 推奨馬券 API の失敗を原因別に分ける（文言だけでは原因が分からない問題の解消）。画面に依存しない純関数。 */

export type BetFailureKind = "offline" | "network" | "session" | "server" | "client" | "render";

export type BetFailure = {
  kind: BetFailureKind;
  message: string;
  /** ページの再読み込み（再ログイン）を勧めるか */
  reload: boolean;
};

/** fetch 自体が例外を投げた（リダイレクト先が別オリジンで失敗、接続断、オフラインなど） */
export function classifyFetchThrow(online: boolean | undefined): BetFailure {
  if (online === false) {
    return { kind: "offline", message: "ネットワークに接続できません（オフライン）。接続を確認してから再試行してください。", reload: false };
  }
  return {
    kind: "network",
    message: "推奨馬券 API に接続できませんでした。ログインの有効期限切れやネットワークの切断が考えられます。ページを再読み込みして、ログインし直してください。",
    reload: true,
  };
}

/** redirect: "manual" で受けた応答がリダイレクト＝Cloudflare Access などのログイン画面へ飛ばされた（セッション切れ） */
export function isSessionRedirect(res: { type?: string; status: number }): boolean {
  return res.type === "opaqueredirect" || (res.status >= 300 && res.status < 400);
}

export function sessionExpired(): BetFailure {
  return { kind: "session", message: "ログインの有効期限が切れました。ページを再読み込みして、ログインし直してください。", reload: true };
}

/** 応答が成功でないとき（401 は呼び出し側で『ログインが必要』に回す） */
export function classifyHttpFailure(status: number, bodyError?: string | null): BetFailure {
  if (status >= 500) {
    return {
      kind: "server",
      message: `サーバーでエラーが発生しました（HTTP ${status}）${bodyError ? `：${bodyError}` : ""}。しばらくしてから再試行してください。`,
      reload: false,
    };
  }
  return { kind: "client", message: bodyError || `推奨馬券の計算に失敗しました（HTTP ${status}）`, reload: false };
}

/** 応答の取り込み・描画の準備で想定外の例外が出た（API には届いている） */
export function classifyRenderError(): BetFailure {
  return {
    kind: "render",
    message: "推奨馬券の表示の準備でエラーが発生しました（API には接続できています）。再試行しても直らない場合は、開発者ツールのコンソールの内容を管理者に伝えてください。",
    reload: false,
  };
}
