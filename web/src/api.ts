export type View = 'front' | 'side' | 'back';
export type Category = 'upper' | 'lower' | 'dress';

export type EngineInfo = {
  id: string;
  label: string;
  licence: string;
  commercial_use: boolean;
  note: string;
  loaded: boolean;
  /** False while the engine's model files are still downloading. */
  available: boolean;
};

export type JobStatus = 'queued' | 'loading' | 'running' | 'done' | 'error';

export type Job = {
  id: string;
  engine: string;
  view: View;
  status: JobStatus;
  position: number | null;
  error: string | null;
  timings: { queued_s: number; model_load_s: number; run_s: number | null };
  /** Each piece of the outfit and whether it made it into the result. */
  checklist: { name: string; ok: boolean }[];
  /** Size check: which size the look is drawn in and how it fits (size letters only). */
  fit?: {
    verdict: string;
    needed: string | null;
    worn: string | null;
    off: number;
    drawn_in_own_size?: boolean;
    /** Length per part: < 0 too short, > 0 too long (steps 1-3). */
    length?: { sleeves: number; body: number; legs: number; off: number } | null;
  } | null;
};

/** The shopper's estimated measurements and size (sizes as indexes: XS=0 … XL=4, 5 = XL + 1 …). */
export type SizeEstimate = {
  confidence: 'high' | 'medium' | 'rough' | 'low' | 'none';
  method?: string;
  reason?: string | null;
  chest_cm?: number | null;
  waist_cm?: number | null;
  top_size: number | null;
  bottom_size: number | null;
  top_label?: string | null;
  bottom_label?: string | null;
  arm_cm?: number;
  body_cm?: number;
  leg_cm?: number;
  /** "photo": measured on the skeleton; "height": standard proportions of the height. */
  lengths_from?: 'photo' | 'height';
};

/** A photo the server's safety check refused (nudity, underwear or swimwear, a child). */
export type Blocked = { ok: false; code: string; message: string };
/** A photo that is not clear enough to use (blurry, dark, too small...): ask for another one. */
export type Retake = { code: string; message: string };

export type PhotoCheck = {
  blocked?: Blocked | null;
  retake?: Retake | null;
  warnings: { code: string; text: string }[];
  visible: { arms: boolean; legs: boolean };
};

export type OutfitAnalysis = {
  blocked?: Blocked | null;
  retake?: Retake | null;
  person: boolean;
  suggested: Category;
  layers: { role: string; name: string; part: 'upper' | 'lower' | 'dress' }[];
  shoes: string | null;
};

/** Live filter: an outfit's clothes cut out, and the pose (33 landmarks [x, y, visibility]) they were worn in. */
export type OutfitLayer = {
  blocked?: Blocked | null;
  retake?: Retake | null;
  width?: number;
  height?: number;
  landmarks?: [number, number, number][];
  image?: string;
};

export type Health = {
  ok: boolean;
  gpu: { name: string; vram_total_gb: number; vram_free_gb: number } | null;
  active_engine: string | null;
};

async function parse<T>(res: Response): Promise<T> {
  if (res.ok) return (await res.json()) as T;
  let message = `The server answered ${res.status}.`;
  try {
    const body = await res.json();
    if (typeof body?.detail === 'string') message = body.detail;
  } catch {
    /* not JSON */
  }
  throw Object.assign(new Error(message), { status: res.status });
}

/**
 * Where the try-on server is.
 * - Development: same origin; Vite forwards /api to the server on this computer.
 * - GitHub Pages: the server runs on the laptop with the GPU, reached through a tunnel. Its
 *   address comes from a ?server=https://... link (remembered on this device), or from
 *   VITE_API_URL at build time.
 */
const SERVER_KEY = 'tryon.server';
function apiBase(): string {
  let url = '';
  try {
    const fromLink = new URLSearchParams(window.location.search).get('server');
    if (fromLink) localStorage.setItem(SERVER_KEY, fromLink);
    url = fromLink || localStorage.getItem(SERVER_KEY) || '';
  } catch {
    /* storage blocked: fall back to the build setting */
  }
  url = url || (import.meta.env.VITE_API_URL as string | undefined) || '';
  return url.replace(/\/+$/, '');
}
export const API = apiBase();
export function setServer(url: string) {
  try {
    if (url.trim()) localStorage.setItem(SERVER_KEY, url.trim());
    else localStorage.removeItem(SERVER_KEY);
  } catch {
    /* ignore */
  }
  const u = new URL(window.location.href);
  u.searchParams.delete('server');
  window.location.replace(u.toString());
}

export const api = {
  health: () => fetch(`${API}/api/health`).then((r) => parse<Health>(r)),
  engines: () =>
    fetch(`${API}/api/engines`).then((r) => parse<{ default: string; engines: EngineInfo[] }>(r)),
  checkPhoto(file: File) {
    const form = new FormData();
    form.append('person', file);
    return fetch(`${API}/api/check-photo`, { method: 'POST', body: form }).then((r) => parse<PhotoCheck>(r));
  },
  /** Live mode: one camera frame, dressed realistically (category chosen by the server from what's in view). */
  liveFrame(frame: Blob, garment: File) {
    const form = new FormData();
    form.append('person', frame, 'frame.jpg');
    form.append('garment', garment);
    form.append('category', 'auto');
    form.append('live', 'true');
    return fetch(`${API}/api/tryon`, { method: 'POST', body: form }).then((r) => parse<Job>(r));
  },
  outfitLayer(file: File) {
    const form = new FormData();
    form.append('garment', file);
    return fetch(`${API}/api/outfit-layer`, { method: 'POST', body: form }).then((r) => parse<OutfitLayer>(r));
  },
  /** Height (cm) and optional weight (kg): used in memory for this request only, never stored. */
  size(person: File, heightCm: number, weightKg?: number) {
    const form = new FormData();
    form.append('person', person);
    form.append('height_cm', String(heightCm));
    if (weightKg) form.append('weight_kg', String(weightKg));
    return fetch(`${API}/api/size`, { method: 'POST', body: form }).then((r) => parse<SizeEstimate>(r));
  },
  analyseOutfit(file: File) {
    const form = new FormData();
    form.append('garment', file);
    return fetch(`${API}/api/analyse-outfit`, { method: 'POST', body: form }).then((r) => parse<OutfitAnalysis>(r));
  },
  submit: (form: FormData) =>
    fetch(`${API}/api/tryon`, { method: 'POST', body: form }).then((r) => parse<Job>(r)),
  job: (id: string) => fetch(`${API}/api/jobs/${id}`).then((r) => parse<Job>(r)),
  async result(id: string): Promise<Blob> {
    const res = await fetch(`${API}/api/jobs/${id}/result`, { cache: 'no-store' });
    if (!res.ok) throw new Error('The result could not be downloaded.');
    return res.blob();
  },
  forget: (id: string) => fetch(`${API}/api/jobs/${id}`, { method: 'DELETE' }).catch(() => undefined),
};
