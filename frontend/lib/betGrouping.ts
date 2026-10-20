/** 推奨買い目を「同じ軸から流す」形にまとめる（表示用の純関数。購入金額・確率は変えない）。 */

export type BetItem = {
  bet_type: string;
  pair?: number[];
  pair_label?: string;
  horse_names?: string[];
  odds?: number;
  prob?: number;
  ev?: number;
  /** 較正前の期待値と、その券種の較正が基準を満たしているか（BetRecommend が較正後の値を ev に入れ替えたとき） */
  ev_raw?: number;
  ev_calibration_validated?: boolean;
  bet_amount: number;
  expected_return?: number;
  predicted_odds?: number;
  odds_gap?: number;
};

export type GroupKind = "single" | "axis1" | "axis2" | "axis1_first" | "axis1_second";

export type BetGroup = {
  key: string;
  bet_type: string;
  kind: GroupKind;
  /** 軸の馬番（流し以外は空） */
  axis: number[];
  /** 相手の馬番（昇順）と、その買い目の金額 */
  partners: { num: number; amount: number; item: BetItem }[];
  items: BetItem[];
  total: number;
  /** 金額加重の期待値 */
  evAvg: number;
  oddsMin: number;
  oddsMax: number;
};

export const BET_ORDER = ["tansho", "fukusho", "umaren", "wide", "umatan", "sanrenpuku"];

const nums = (c: BetItem): number[] =>
  c.pair && c.pair.length ? c.pair : String(c.pair_label ?? "").split(/[-→]/).map(Number).filter((n) => Number.isFinite(n));

/** 馬番 → 馬名（候補の pair と horse_names を突き合わせる） */
export function nameMap(items: BetItem[]): Map<number, string> {
  const m = new Map<number, string>();
  for (const c of items) {
    const ns = nums(c);
    (c.horse_names ?? []).forEach((n, i) => {
      if (n && ns[i] != null && !m.has(ns[i])) m.set(ns[i], n);
    });
  }
  return m;
}

function build(bet_type: string, kind: GroupKind, axis: number[], items: BetItem[], partnerOf: (c: BetItem) => number | null): BetGroup {
  const total = items.reduce((s, c) => s + c.bet_amount, 0);
  const evAvg = total > 0 ? items.reduce((s, c) => s + (c.ev ?? 0) * c.bet_amount, 0) / total : (items[0]?.ev ?? 0);
  const odds = items.map((c) => c.odds ?? 0).filter((o) => o > 0);
  const partners = kind === "single"
    ? []
    : items
        .map((c) => ({ num: partnerOf(c), amount: c.bet_amount, item: c }))
        .filter((p): p is { num: number; amount: number; item: BetItem } => p.num != null)
        .sort((a, b) => a.num - b.num);
  return {
    key: `${bet_type}:${kind}:${axis.join("-")}:${items.map((c) => c.pair_label).join(",")}`,
    bet_type, kind, axis, partners, total, evAvg,
    items: [...items].sort((a, b) => (b.ev ?? 0) - (a.ev ?? 0)),
    oddsMin: odds.length ? Math.min(...odds) : 0,
    oddsMax: odds.length ? Math.max(...odds) : 0,
  };
}

/** 出現回数が最も多い要素（同数なら期待値の合計が大きい方）。回数が min 未満なら null。 */
function mostCommon<T extends string | number>(pool: BetItem[], keysOf: (c: BetItem) => T[], min = 2): T | null {
  const cnt = new Map<T, { n: number; ev: number }>();
  for (const c of pool) for (const k of keysOf(c)) {
    const e = cnt.get(k) ?? { n: 0, ev: 0 };
    e.n += 1;
    e.ev += c.ev ?? 0;
    cnt.set(k, e);
  }
  let best: T | null = null;
  let bv = { n: 0, ev: 0 };
  for (const [k, v] of cnt) if (v.n > bv.n || (v.n === bv.n && v.ev > bv.ev)) { best = k; bv = v; }
  return bv.n >= min ? best : null;
}

function groupUnordered(bet_type: string, items: BetItem[]): BetGroup[] {
  const out: BetGroup[] = [];
  let pool = [...items];
  for (;;) {
    const axis = mostCommon(pool, (c) => nums(c));
    if (axis == null) break;
    const hit = pool.filter((c) => nums(c).includes(axis));
    out.push(build(bet_type, "axis1", [axis], hit, (c) => nums(c).find((n) => n !== axis) ?? null));
    pool = pool.filter((c) => !hit.includes(c));
  }
  return [...out, ...pool.map((c) => build(bet_type, "single", [], [c], () => null))];
}

function groupUmatan(bet_type: string, items: BetItem[]): BetGroup[] {
  const out: BetGroup[] = [];
  let pool = [...items];
  for (const [kind, pos, other] of [["axis1_first", 0, 1], ["axis1_second", 1, 0]] as const) {
    for (;;) {
      const axis = mostCommon(pool, (c) => [nums(c)[pos]]);
      if (axis == null) break;
      const hit = pool.filter((c) => nums(c)[pos] === axis);
      out.push(build(bet_type, kind, [axis], hit, (c) => nums(c)[other]));
      pool = pool.filter((c) => !hit.includes(c));
    }
  }
  return [...out, ...pool.map((c) => build(bet_type, "single", [], [c], () => null))];
}

const comb2 = (n: number) => (n * (n - 1)) / 2;

function groupTrio(bet_type: string, items: BetItem[]): BetGroup[] {
  const out: BetGroup[] = [];
  let pool = [...items];
  // ① 軸1頭流し: その馬を含む買い目が、相手 N 頭から作れる 2 頭の組合せ（C(N,2)）をちょうど全て満たすとき
  for (;;) {
    let best: { axis: number; hit: BetItem[] } | null = null;
    const horses = new Set(pool.flatMap((c) => nums(c)));
    for (const h of horses) {
      const hit = pool.filter((c) => nums(c).includes(h));
      const partners = new Set(hit.flatMap((c) => nums(c).filter((n) => n !== h)));
      const pairs = new Set(hit.map((c) => nums(c).filter((n) => n !== h).sort((a, b) => a - b).join("-")));
      if (partners.size >= 3 && pairs.size === hit.length && hit.length === comb2(partners.size) && (!best || hit.length > best.hit.length)) {
        best = { axis: h, hit };
      }
    }
    if (!best) break;
    const { axis, hit } = best;
    out.push(build(bet_type, "axis1", [axis], hit, () => null));
    pool = pool.filter((c) => !hit.includes(c));
  }
  // ② 軸2頭流し: 同じ 2 頭を含む買い目が 2 点以上
  for (;;) {
    const pair = mostCommon(pool, (c) => {
      const ns = [...nums(c)].sort((a, b) => a - b);
      return [`${ns[0]}-${ns[1]}`, `${ns[0]}-${ns[2]}`, `${ns[1]}-${ns[2]}`];
    });
    if (pair == null) break;
    const axis = String(pair).split("-").map(Number);
    const hit = pool.filter((c) => axis.every((a) => nums(c).includes(a)));
    out.push(build(bet_type, "axis2", axis, hit, (c) => nums(c).find((n) => !axis.includes(n)) ?? null));
    pool = pool.filter((c) => !hit.includes(c));
  }
  return [...out, ...pool.map((c) => build(bet_type, "single", [], [c], () => null))];
}

/** 券種順に、流しでまとめた表示用グループを返す。全ての買い目がちょうど 1 つのグループに入る。 */
export function groupBets(cands: BetItem[]): BetGroup[] {
  const out: BetGroup[] = [];
  for (const bt of BET_ORDER) {
    const items = cands.filter((c) => c.bet_type === bt);
    if (!items.length) continue;
    const groups =
      bt === "umaren" || bt === "wide" ? groupUnordered(bt, items)
      : bt === "umatan" ? groupUmatan(bt, items)
      : bt === "sanrenpuku" ? groupTrio(bt, items)
      : items.map((c) => build(bt, "single", [], [c], () => null));
    // 流し → 単独の順、それぞれ期待値（金額加重）の高い順
    const rank = (g: BetGroup) => (g.kind === "single" ? 1 : 0);
    out.push(...groups.sort((a, b) => rank(a) - rank(b) || b.evAvg - a.evAvg));
  }
  return out;
}

/** 3連複の軸1頭流しは相手が「N 頭のうちどの 2 頭でも」。表示用に相手の馬番を集める。 */
export function trioPartnerNums(g: BetGroup): number[] {
  const s = new Set<number>();
  for (const c of g.items) for (const n of nums(c)) if (!g.axis.includes(n)) s.add(n);
  return [...s].sort((a, b) => a - b);
}
