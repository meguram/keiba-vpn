/** 予測の先読みの順番待ち（週の予想の画面）。
 *  - 先読みは全体で同時 1 本。送信した先読みは止めずに済むまで待つ（ブラウザで止めても Cloud Run の計算は続き、
 *    次を出すとサーバの同時の計算が増えるだけのため）。マウスが乗ったレースは列の先頭に入れ、新しい HOVER_KEEP 件だけ残す
 *    （待っている間になぞった、もう見ていないレースまで計算させない）。
 *  - 選んだレースが変わったら（focus）、まだ出していない列を捨て、選んだレースの取得（select に渡す）が済むまで次を出さない。
 *  2026-11-26 の stg で、ボタンの上をなぞった先読みと前後のレースの先読みが積み重なり、過去のレースの重い計算が 1 台（1 vCPU）に
 *  6 本同時に乗って各 20 秒かかった（単独なら 1.5 秒）。 */
const HOVER_KEEP = 2;

export class PredPrefetcher {
  private cur: string | null = null;
  private hovered: string[] = [];   // マウスが乗ったレース（新しい順・HOVER_KEEP 件まで）。先に出す
  private queue: string[] = [];     // 前後のレース
  private gateOpen = true;
  private seq = 0;
  private focused: string | null = null;

  constructor(
    private readonly start: (id: string) => Promise<unknown>,
    private readonly isFresh: (id: string) => boolean,
  ) {}

  /** 選んだレースが変わった: 出していない列を捨て、select まで次を出さない。同じレースなら何もしない（一覧の作り直しで列を捨てない） */
  focus(id: string): void {
    if (this.focused === id) return;
    this.focused = id;
    this.seq++;
    this.gateOpen = false;
    this.queue = [];
    this.hovered = [];
  }

  /** 選んだレースの取得。済んだら次の先読みを出す。もっと新しい focus があれば何もしない */
  select(selected: Promise<unknown>): void {
    const seq = this.seq;
    selected.then(() => undefined, () => undefined).then(() => {
      if (seq !== this.seq) return;
      this.gateOpen = true;
      this.pump();
    });
  }

  /** 選んだレースが無い（一覧が空・読み込み中）: 門を開けたままにする（閉じたまま先読みが止まらないように） */
  release(): void {
    this.focused = null;
    this.seq++;
    this.gateOpen = true;
    this.pump();
  }

  /** 先読みの希望を足す。front=true（マウスが乗った・触れた）は列の先頭に入れる */
  want(ids: string[], front = false): void {
    const add = ids.filter((id) => id !== this.cur && !this.isFresh(id));
    if (front) this.hovered = [...add, ...this.hovered.filter((x) => !add.includes(x))].slice(0, HOVER_KEEP);
    else for (const id of add) if (!this.queue.includes(id)) this.queue.push(id);
    this.pump();
  }

  /** 画面を離れるとき: 列を捨て、新しく出さない（送信中の 1 本は済むまで任せる） */
  stop(): void {
    this.seq++;
    this.gateOpen = false;
    this.queue = [];
    this.hovered = [];
  }

  private next(): string | undefined {
    return this.hovered.length ? this.hovered.shift() : this.queue.shift();
  }

  private pump(): void {
    if (!this.gateOpen || this.cur !== null) return;
    let id: string | undefined;
    while ((id = this.next()) !== undefined && this.isFresh(id)) { /* 取得済み・取得中は飛ばす */ }
    if (id === undefined) return;
    this.cur = id;
    this.start(id).then(() => undefined, () => undefined).then(() => {
      this.cur = null;
      this.pump();
    });
  }
}
