/**
 * Size check, page side. The same rule as server/app/sizing.py (the server recomputes it for the
 * try-on and never trusts the page). Sizes sold: XS–XL (0–4); 5, 6… are "XL + 1", "XL + 2" (never
 * sold, only to say how far off the biggest size is).
 */
import type { SizeEstimate } from './api';

export const SIZES = ['XS', 'S', 'M', 'L', 'XL'] as const;

export const sizeLabel = (i: number) => (i >= 0 && i < SIZES.length ? SIZES[i] : `XL + ${i - SIZES.length + 1}`);

export type Verdict = 'fits' | 'close' | 'snug' | 'loose' | 'too_small' | 'too_big' | 'unknown';

export type Length = { sleeves: number; body: number; legs: number; off: number };

export type Fit = { verdict: Verdict; needed: number | null; worn: number | null; off: number; length?: Length };

// Length (same as server/app/sizing.py): each size is cut for a body of a certain height.
export const MADE_FOR_HEIGHT = [160, 166, 172, 178, 184];
const ARM = 0.335;
const BODY = 0.29;
const LEG = 0.47;

/** cm the shopper is longer (+) than the garment is cut for -> steps; garment too short = negative. */
function steps(delta: number): number {
  const a = Math.abs(delta);
  const n = a < 4 ? 0 : a < 8 ? 1 : a < 12 ? 2 : 3;
  return delta > 0 ? -n : n;
}

/** How the size worn fits in length, per part (tops: sleeves, body; bottoms: legs). */
export function lengthFit(est: SizeEstimate, worn: number | null, parts: Set<string>): Length {
  const none = { sleeves: 0, body: 0, legs: 0, off: 0 };
  if (worn == null || worn < 0 || worn >= SIZES.length || est.arm_cm == null) return none;
  const h = MADE_FOR_HEIGHT[worn];
  const top = parts.has('upper') || parts.has('dress');
  const bottom = parts.has('lower') || parts.has('dress');
  const out = {
    sleeves: top ? steps(est.arm_cm - ARM * h) : 0,
    body: top && est.body_cm != null ? steps(est.body_cm - BODY * h) : 0,
    legs: bottom && est.leg_cm != null ? steps(est.leg_cm - LEG * h) : 0,
    off: 0,
  };
  out.off = [out.sleeves, out.body, out.legs].reduce((a, b) => (Math.abs(b) > Math.abs(a) ? b : a), 0);
  return out;
}

/** "Too short: sleeves and trousers" / "Too long: sleeves" / "" when the length is fine. */
export function lengthText(l: Length | null | undefined): string {
  if (!l || !l.off) return '';
  const name: Record<string, string> = { sleeves: 'sleeves', body: 'top', legs: 'trousers' };
  const list = (sign: number) =>
    (['sleeves', 'body', 'legs'] as const).filter((p) => Math.sign(l[p]) === sign).map((p) => name[p]);
  const join = (xs: string[]) => (xs.length > 1 ? `${xs.slice(0, -1).join(', ')} and ${xs[xs.length - 1]}` : xs[0]);
  const short = list(-1);
  const long = list(1);
  const out: string[] = [];
  if (short.length) out.push(`Too short for your height: ${join(short)}.`);
  if (long.length) out.push(`Too long for your height: ${join(long)}.`);
  return out.join(' ');
}

/** The size this shopper needs for an outfit made of these parts (upper / lower / dress). */
export function neededSize(est: SizeEstimate, parts: Set<string>): number | null {
  const wants: number[] = [];
  if ((parts.has('upper') || parts.has('dress')) && est.top_size != null) wants.push(est.top_size);
  if ((parts.has('lower') || parts.has('dress')) && est.bottom_size != null) wants.push(est.bottom_size);
  if (!wants.length) for (const s of [est.top_size, est.bottom_size]) if (s != null) wants.push(s);
  return wants.length ? Math.max(...wants) : null;
}

/** Which size to show them in, and how it fits. off < 0 = tight, > 0 = loose (in sizes). */
export function verdict(needed: number | null, stockIn: number[]): Fit {
  const stock = [...new Set(stockIn)].filter((s) => s >= 0 && s < SIZES.length).sort((a, b) => a - b);
  if (needed == null || !stock.length) return { verdict: 'unknown', needed: null, worn: null, off: 0 };
  let worn: number;
  if (stock.includes(needed)) worn = needed;
  else {
    const smaller = stock.filter((s) => s < needed);
    const bigger = stock.filter((s) => s > needed);
    if (smaller.length && bigger.length) {
      const up = Math.min(...bigger);
      const down = Math.max(...smaller);
      worn = up - needed <= needed - down ? up : down;
    } else worn = bigger.length ? Math.min(...bigger) : Math.max(...smaller);
  }
  const off = Math.max(-3, Math.min(3, worn - needed));
  let v: Verdict;
  if (off === 0) v = 'fits';
  else if (stock.every((s) => s < needed)) v = off === -1 ? 'snug' : 'too_small';
  else if (stock.every((s) => s > needed)) v = off === 1 ? 'loose' : 'too_big';
  else v = 'close';
  return { verdict: v, needed, worn, off };
}

/** What the shopper reads. */
export function fitText(f: Fit, stock: number[]): { title: string; detail: string } {
  if (f.verdict === 'unknown' || f.needed == null || f.worn == null) return { title: '', detail: '' };
  const need = sizeLabel(f.needed);
  const worn = sizeLabel(f.worn);
  const s = [...stock].sort((a, b) => a - b);
  const range = s.length ? (s.length === 1 ? sizeLabel(s[0]) : `${sizeLabel(s[0])}–${sizeLabel(s[s.length - 1])}`) : '';
  const n = Math.abs(f.off);
  const sizes = `${n} size${n === 1 ? '' : 's'}`;
  switch (f.verdict) {
    case 'fits':
      return { title: 'Fits', detail: `Your size, ${need}, is in stock.` };
    case 'close':
      return { title: f.off > 0 ? 'A little loose' : 'A little tight', detail: `${need} is out of stock. ${worn} is the closest.` };
    case 'snug':
      return { title: 'Snug', detail: `${worn} is the biggest (${range}). You're about ${need}, so it may be tight.` };
    case 'too_small':
      return { title: 'Too small', detail: `It only goes up to ${worn} (${range}). You're about ${need}: ${sizes} too small.` };
    case 'loose':
      return { title: 'A bit big', detail: `${worn} is the smallest (${range}). You're about ${need}, so it'll be loose.` };
    case 'too_big':
      return { title: 'Too big', detail: `The smallest is ${worn} (${range}). You're about ${need}: ${sizes} too big.` };
  }
}

export function fitTone(v: Verdict): 'good' | 'warn' | 'bad' | 'none' {
  if (v === 'fits') return 'good';
  if (v === 'close' || v === 'snug' || v === 'loose') return 'warn';
  if (v === 'too_small' || v === 'too_big') return 'bad';
  return 'none';
}
