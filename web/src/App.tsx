import { useCallback, useEffect, useRef, useState } from 'react';
import { api, setServer, API, type Category, type EngineInfo, type Health, type OutfitAnalysis, type PhotoCheck, type SizeEstimate, type View } from './api';
import { SIZES, lengthFit, neededSize, verdict as fitVerdict, type Fit } from './fit';
import { FitBadge } from './components/FitBadge';
import { SizeChips } from './components/SizeChips';
import { PhotoSlot } from './components/PhotoSlot';
import { LiveCamera } from './components/LiveCamera';
import { Mirror } from './components/Mirror';
import { OutfitStrip } from './components/OutfitStrip';
import { GestureControl, type Swipe } from './components/GestureControl';

const VIEWS: { id: View; label: string; hint: string }[] = [
  { id: 'front', label: 'Front', hint: 'Face the camera, arms relaxed' },
  { id: 'side', label: 'Side', hint: 'Turn 90°, optional' },
  { id: 'back', label: 'Back', hint: 'Turn around, optional' },
];

const CATEGORIES: { id: Category; label: string }[] = [
  { id: 'upper', label: 'Top' },
  { id: 'lower', label: 'Bottoms' },
  { id: 'dress', label: 'Full outfit' },
];


type RunStatus = 'queued' | 'loading' | 'running' | 'done' | 'error';
export type Run = {
  key: string;
  outfitId: string;
  view: View;
  engine: string;
  status: RunStatus;
  jobId?: string;
  url?: string;
  error?: string;
  position?: number | null;
  runSeconds?: number | null;
  checklist?: { name: string; ok: boolean }[];
  /** Size check for this look, from the server (size letters only). */
  fit?: {
    verdict: string;
    needed: string | null;
    worn: string | null;
    off: number;
    drawn_in_own_size?: boolean;
    length?: { sleeves: number; body: number; legs: number; off: number } | null;
  } | null;
  loadSeconds?: number;
  startedAt: number;
};

type Photo = { file: File; url: string };
type Outfit = {
  id: string;
  front: Photo;
  back: Photo | null;
  category: Category;
  description: string;
  /** The pieces the server read in the photo (coat, top under it, trousers...). */
  analysis?: OutfitAnalysis | null;
  /** The user picked the type themselves: the server's suggestion no longer overrides it. */
  typeChosen?: boolean;
  /** Sizes in stock (XS=0 … XL=4). Empty: no size check for this outfit. */
  sizes: number[];
};

/** One GPU makes one look at a time; more outfits than this would mean minutes of waiting. */
const MAX_OUTFITS = 6;
let outfitSeq = 0;

export default function App() {
  const [photos, setPhotos] = useState<Partial<Record<View, Photo>>>({});
  const [checks, setChecks] = useState<Partial<Record<View, PhotoCheck>>>({});
  // Size check. Height and weight stay in this page's memory only: never stored, never logged.
  const [heightCm, setHeightCm] = useState('');
  const [weightKg, setWeightKg] = useState('');
  const [sizeEst, setSizeEst] = useState<SizeEstimate | null>(null);
  const [sizeError, setSizeError] = useState('');
  const [outfits, setOutfits] = useState<Outfit[]>([]);
  const [editing, setEditing] = useState<string | null>(null);
  const [lockOutside, setLockOutside] = useState(false);
  const [seed, setSeed] = useState(42);

  const [serverEngines, setServerEngines] = useState<EngineInfo[]>([]);
  const [health, setHealth] = useState<Health | null>(null);
  const [serverDown, setServerDown] = useState(false);
  const [selected, setSelected] = useState<string[]>(['flux-klein']);

  const [runs, setRuns] = useState<Run[]>([]);
  const runsRef = useRef<Run[]>([]);
  runsRef.current = runs;
  const fetching = useRef(new Set<string>());
  // Server requests not sent yet. They go out one at a time, in outfit order ("in turns").
  const forms = useRef(new Map<string, FormData>());
  const submitting = useRef(false);
  const [tick, setTick] = useState(0);
  const [activeOutfit, setActiveOutfit] = useState<string | null>(null);
  const [gestures, setGestures] = useState(false);
  // Photo: upload photos, the model makes the look. Live: a real-time filter on the camera.
  const [mode, setMode] = useState<'photo' | 'live'>('photo');
  // Live mode waits for the real-time model (research/README.md). Until then it is shown only
  // with ?live=1 (testing) or a build made with VITE_LIVE=1.
  const liveEnabled = import.meta.env.VITE_LIVE === '1' || new URLSearchParams(window.location.search).get('live') === '1';
  const [activeView, setActiveView] = useState<View>('front');
  const [activeEngine, setActiveEngine] = useState<string>('flux-klein');
  const [formError, setFormError] = useState<string | null>(null);

  const engines = serverEngines;

  // ---- server status ----
  const pickedDefaults = useRef(false);
  const refreshServer = useCallback(async () => {
    try {
      const [e, h] = await Promise.all([api.engines(), api.health()]);
      setServerEngines(e.engines);
      if (!pickedDefaults.current) {
        // First contact: preselect the default engine if it's ready, else the first ready one.
        pickedDefaults.current = true;
        const ready = e.engines.filter((x) => x.available).map((x) => x.id);
        const first = ready.includes(e.default) ? e.default : ready[0];
        setSelected(first ? [first] : []);
      }
      setHealth(h);
      setServerDown(false);
    } catch {
      setServerDown(true);
    }
  }, []);
  useEffect(() => {
    refreshServer();
    const t = setInterval(refreshServer, 15000);
    return () => clearInterval(t);
  }, [refreshServer]);

  const patch = useCallback((key: string, p: Partial<Run>) => {
    setRuns((rs) => rs.map((r) => (r.key === key ? { ...r, ...p } : r)));
  }, []);

  // ---- poll server jobs ----
  useEffect(() => {
    const t = setInterval(() => {
      for (const r of runsRef.current) {
        if (!r.jobId || r.status === 'done' || r.status === 'error' || fetching.current.has(r.key)) continue;
        fetching.current.add(r.key);
        api
          .job(r.jobId)
          .then(async (job) => {
            const base = {
              position: job.position,
              runSeconds: job.timings.run_s,
              loadSeconds: job.timings.model_load_s,
            };
            if (job.status === 'done') {
              const blob = await api.result(job.id);
              forms.current.delete(r.key);
              patch(r.key, { ...base, status: 'done', url: URL.createObjectURL(blob), checklist: job.checklist, fit: job.fit ?? null });
              api.forget(job.id); // the server drops the photos and result now
            } else if (job.status === 'error') {
              forms.current.delete(r.key);
              patch(r.key, { ...base, status: 'error', error: job.error ?? 'The try-on failed.' });
            } else {
              patch(r.key, { ...base, status: job.status });
            }
          })
          .catch((err: Error & { status?: number }) => {
            // The server lost the job (it was restarted): send the look again, it waits its turn.
            if (err.status === 404 && forms.current.has(r.key)) {
              patch(r.key, { status: 'queued', jobId: undefined, position: undefined });
              setTick((t) => t + 1);
            } else if (err.status === undefined) {
              // Network error: the server is restarting. Keep polling.
            } else {
              patch(r.key, { status: 'error', error: err.message });
            }
          })
          .finally(() => fetching.current.delete(r.key));
      }
    }, 1000);
    return () => clearInterval(t);
  }, [patch]);

  // ---- send the next look once the previous one is finished ----
  useEffect(() => {
    if (submitting.current) return;
    if (runs.some((r) => r.jobId && r.status !== 'done' && r.status !== 'error')) return;
    const next = runs.find((r) => r.status === 'queued' && !r.jobId && forms.current.has(r.key));
    if (!next) return;
    // Kept until the look is finished, so it can be sent again if the server loses it.
    const form = forms.current.get(next.key)!;
    submitting.current = true;
    api
      .submit(form)
      .then((job) => patch(next.key, { jobId: job.id, status: job.status, position: job.position, startedAt: Date.now() }))
      .catch((err: Error) => patch(next.key, { status: 'error', error: err.message }))
      .finally(() => {
        submitting.current = false;
        setTick((t) => t + 1);
      });
  }, [runs, tick, patch]);

  // ---- size check ----
  const h = Number(heightCm);
  const w = Number(weightKg);
  const bodyOk = !!heightCm && h >= 120 && h <= 220;
  const weightOk = !weightKg || (w >= 30 && w <= 250);
  useEffect(() => {
    setSizeEst(null);
    setSizeError('');
    const file = photos.front?.file;
    if (!file || !heightCm) return;
    if (!bodyOk) return setSizeError('Enter a height between 120 and 220 cm.');
    if (!weightOk) return setSizeError('Enter a weight between 30 and 250 kg, or leave it empty.');
    const t = setTimeout(() => {
      api
        .size(file, h, weightKg ? w : undefined)
        .then(setSizeEst)
        .catch((e: Error) => setSizeError(e.message));
    }, 600);
    return () => clearTimeout(t);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [photos.front?.file, heightCm, weightKg]);
  const fitFor = (o: Outfit): Fit | null => {
    if (!sizeEst || sizeEst.confidence === 'none' || !o.sizes.length) return null;
    const parts = new Set((o.analysis?.layers ?? []).map((l) => l.part));
    if (!parts.size) parts.add(o.category === 'lower' ? 'lower' : o.category === 'dress' ? 'dress' : 'upper');
    const f = fitVerdict(neededSize(sizeEst, parts), o.sizes);
    return { ...f, length: lengthFit(sizeEst, f.worn, parts) };
  };

  // ---- form helpers ----
  const setPhoto = (view: View, file: File | null) => {
    setChecks((c) => {
      const next = { ...c };
      delete next[view];
      return next;
    });
    if (file) {
      // Ask the server what in this photo will limit the result (hidden hands, no legs).
      api
        .checkPhoto(file)
        .then((check) => setChecks((c) => ({ ...c, [view]: check })))
        // The safety check is part of this call: a photo that could not be checked is not used.
        .catch(() => undefined);
    }
    setPhotoFile(view, file);
  };
  const setPhotoFile = (view: View, file: File | null) =>
    setPhotos((p) => {
      if (p[view]) URL.revokeObjectURL(p[view]!.url);
      const next = { ...p };
      if (file) next[view] = { file, url: URL.createObjectURL(file) };
      else delete next[view];
      return next;
    });
  const addOutfits = (files: File[]) => {
    const room = MAX_OUTFITS - outfits.length;
    if (room <= 0) return;
    // New outfits start with the type of the one being edited: a batch is usually one kind.
    const category = outfits.find((o) => o.id === editing)?.category ?? 'upper';
    const added: Outfit[] = files.slice(0, room).map((file) => ({
      id: `o${++outfitSeq}`,
      front: { file, url: URL.createObjectURL(file) },
      back: null,
      category,
      description: '',
      sizes: [],
    }));
    setOutfits((os) => [...os, ...added]);
    setEditing(added[0].id);
    setFormError(files.length > room ? `Up to ${MAX_OUTFITS} outfits at a time; the extra photos were left out.` : null);
  };
  // Read every outfit photo as a list of pieces (also outfits already on the page before an update);
  // a photo of a whole look defaults to "Full outfit" unless the type was picked by hand.
  const analysing = useRef(new Set<string>());
  useEffect(() => {
    for (const o of outfits) {
      if (o.analysis !== undefined || analysing.current.has(o.id)) continue;
      analysing.current.add(o.id);
      api
        .analyseOutfit(o.front.file)
        .then((analysis) =>
          setOutfits((os) =>
            os.map((x) =>
              x.id === o.id
                ? { ...x, analysis, category: analysis.person && !x.typeChosen ? analysis.suggested : x.category }
                : x,
            ),
          ),
        )
        .catch(() => setOutfits((os) => os.map((x) => (x.id === o.id ? { ...x, analysis: null } : x))))
        .finally(() => analysing.current.delete(o.id));
    }
  }, [outfits]);
  const updateOutfit = (id: string, p: Partial<Outfit>) =>
    setOutfits((os) => os.map((o) => (o.id === id ? { ...o, ...p } : o)));
  const setOutfitBack = (o: Outfit, file: File | null) => {
    if (o.back) URL.revokeObjectURL(o.back.url);
    updateOutfit(o.id, { back: file ? { file, url: URL.createObjectURL(file) } : null });
  };
  const removeOutfit = (id: string) => {
    const i = outfits.findIndex((x) => x.id === id);
    if (i < 0) return;
    const o = outfits[i];
    // The pager shows this outfit's thumbnail while its looks are on screen; keep the URL until then.
    if (!runsRef.current.some((r) => r.outfitId === id)) {
      URL.revokeObjectURL(o.front.url);
      if (o.back) URL.revokeObjectURL(o.back.url);
    }
    const rest = outfits.filter((x) => x.id !== id);
    setOutfits(rest);
    setEditing(rest.length ? rest[Math.max(0, i - 1)].id : null);
  };

  const clearResults = () => {
    forms.current.clear();
    runsRef.current.forEach((r) => {
      if (r.url) URL.revokeObjectURL(r.url);
      // A look still on the server: cancel it so the GPU moves on.
      if (r.jobId && r.status !== 'done' && r.status !== 'error') api.forget(r.jobId);
    });
    setRuns([]);
  };

  // ---- run ----
  const views = VIEWS.filter((v) => photos[v.id]).map((v) => v.id);
  const busy = runs.some((r) => r.status !== 'done' && r.status !== 'error');

  const start = () => {
    setFormError(null);
    if (!views.length) return setFormError('Add at least one photo of yourself.');
    if (!outfits.length) return setFormError('Add at least one outfit.');
    if (!selected.length) return setFormError('Choose at least one method.');
    // Safety and quality, before anything runs: every photo must have been checked by the server
    // and passed (safe, and clear enough to give a good result).
    for (const v of views) {
      const c = checks[v];
      if (!c) return setFormError(serverDown ? 'The try-on server is not running, so photos can’t be checked and nothing can run.' : 'Still checking your photo. Try again in a moment.');
      if (c.blocked) return setFormError(c.blocked.message);
      if (c.retake) return setFormError(c.retake.message);
    }
    for (const [i, o] of outfits.entries()) {
      if (o.analysis === undefined) return setFormError(`Still checking outfit ${i + 1}. Try again in a moment.`);
      if (o.analysis === null) return setFormError(`Outfit ${i + 1} couldn’t be checked, so it can’t be used. Remove it or add it again.`);
      if (o.analysis.blocked) return setFormError(`Outfit ${i + 1}: ${o.analysis.blocked.message}`);
      if (o.analysis.retake) return setFormError(`Outfit ${i + 1}: ${o.analysis.retake.message}`);
    }
    const noLegs = views.every((v) => checks[v] && !checks[v]!.visible.legs);
    const needLegs = outfits.findIndex((o) => o.category === 'lower');
    if (noLegs && needLegs >= 0)
      return setFormError(`Outfit ${needLegs + 1} is bottoms, but your legs aren’t in any of your photos. Add a full-length photo or set it to Top.`);
    const usable = selected.filter((id) => !serverDown && engines.find((e) => e.id === id)?.available);
    if (!usable.length) return setFormError('The try-on server is not running. Try again in a moment.');

    clearResults();
    const now = Date.now();
    const fresh: Run[] = [];
    // Outfit by outfit, so the first outfit's looks are ready first.
    for (const outfit of outfits)
      for (const view of views)
        for (const engine of usable) {
          const key = `${outfit.id}:${view}:${engine}`;
          fresh.push({ key, outfitId: outfit.id, view, engine, status: 'queued', startedAt: now });
          const form = new FormData();
          form.append('person', photos[view]!.file);
          form.append('garment', outfit.front.file);
          if (outfit.back && view === 'back') form.append('garment_back', outfit.back.file);
          form.append('engine', engine);
          form.append('category', outfit.category);
          form.append('view', view);
          form.append('seed', String(seed));
          form.append('lock_outside', String(lockOutside));
          // Size check: the server works out the size to draw (it doesn't trust the page's verdict).
          if (bodyOk && outfit.sizes.length) {
            form.append('height_cm', heightCm);
            if (weightOk && weightKg) form.append('weight_kg', weightKg);
            form.append('sizes', outfit.sizes.map((i) => SIZES[i]).join(','));
          }
          forms.current.set(key, form);
        }
    setRuns(fresh);
    setActiveOutfit(outfits[0].id);
    setActiveView(views[0]);
    setActiveEngine(usable[0]);
  };

  // ---- live mode ----
  const switchMode = (m: 'photo' | 'live') => {
    if (m === mode) return;
    setMode(m);
    setFormError(null);
  };
  // Live mode needs no run: every outfit added can be shown at once.
  const runOutfits = mode === 'live' ? outfits : outfits.filter((o) => runs.some((r) => r.outfitId === o.id));
  const outfitIdx = Math.max(0, runOutfits.findIndex((o) => o.id === activeOutfit));
  const shownOutfit = runOutfits[outfitIdx];
  const viewRuns = runs.filter((r) => r.outfitId === shownOutfit?.id && r.view === activeView);
  const current = viewRuns.find((r) => r.engine === activeEngine) ?? viewRuns[0];
  const statusFor = (outfitId: string) =>
    (runs.find((r) => r.outfitId === outfitId && r.view === activeView && r.engine === activeEngine) ??
      runs.find((r) => r.outfitId === outfitId))?.status;
  // Server looks still to be made before the one on screen.
  const ahead =
    current && !current.jobId
      ? runs
          .slice(0, runs.indexOf(current))
          .filter((r) => r.status !== 'done' && r.status !== 'error').length
      : 0;
  const goOutfit = (i: number) => {
    const o = runOutfits[i];
    if (o) setActiveOutfit(o.id);
  };
  const canSwipe = (dir: Swipe) => (dir === 'next' ? outfitIdx < runOutfits.length - 1 : outfitIdx > 0);
  const onSwipe = (dir: Swipe) => goOutfit(outfitIdx + (dir === 'next' ? 1 : -1));
  const editingOutfit = outfits.find((o) => o.id === editing) ?? null;
  const engineLabel = (id: string) => engines.find((e) => e.id === id)?.label ?? id;

  return (
    <div className="app">
      <header className="topbar">
        <div className="brand">
          <img className="brand-mark" src={`${import.meta.env.BASE_URL}logo.svg`} alt="" width={30} height={30} />
          <h1>Velvet Mirror</h1>
        </div>
        <ServerPill health={health} down={serverDown} />
      </header>

      <main className="layout">
        <section className="steps" aria-label="Set up a try-on">
          <ol className="step-list">
            <li className="step">
              <div className="step-head">
                <h2>You</h2>
                <p>
                  {mode === 'photo'
                    ? 'One photo is enough. Add your side and back to see the outfit from every angle.'
                    : 'A live filter: the outfit appears on you in the camera and follows you. No photo is taken.'}
                </p>
              </div>
              {liveEnabled && (
              <div className="segmented" role="radiogroup" aria-label="How to add yourself">
                {(['photo', 'live'] as const).map((m) => (
                  <button
                    key={m}
                    type="button"
                    role="radio"
                    aria-checked={mode === m}
                    className={mode === m ? 'on' : ''}
                    onClick={() => switchMode(m)}
                  >
                    {m === 'photo' ? 'Photo' : 'Live camera'}
                  </button>
                ))}
              </div>
              )}
              {mode === 'live' ? (
                <p className="live-hint">The camera is in the mirror. Whatever part of you is in view gets dressed.</p>
              ) : (
              <div className="slots">
                {VIEWS.map((v) => (
                  <PhotoSlot
                    key={v.id}
                    id={`photo-${v.id}`}
                    label={v.label}
                    hint={v.hint}
                    url={photos[v.id]?.url}
                    onFile={(f) => setPhoto(v.id, f)}
                  />
                ))}
              </div>
              )}
              {mode === 'photo' && (
                <div className="body-fields">
                  <label className="field" htmlFor="height">
                    <span className="field-label">Height (cm)</span>
                    <input id="height" type="number" inputMode="numeric" min={120} max={220} placeholder="e.g. 170" value={heightCm} onChange={(e) => setHeightCm(e.target.value)} />
                  </label>
                  <label className="field" htmlFor="weight">
                    <span className="field-label">Weight (kg, optional)</span>
                    <input id="weight" type="number" inputMode="numeric" min={30} max={250} placeholder="e.g. 65" value={weightKg} onChange={(e) => setWeightKg(e.target.value)} />
                  </label>
                  <p className="pieces-note body-note">
                    Used only to check sizes, never stored.{' '}
                    {sizeError
                      ? sizeError
                      : sizeEst && sizeEst.confidence !== 'none'
                        ? `You: about ${sizeEst.top_label ?? '?'} on top${sizeEst.bottom_label ? `, ${sizeEst.bottom_label} for bottoms` : ''}${sizeEst.chest_cm ? ` (chest ~${sizeEst.chest_cm} cm${sizeEst.waist_cm ? `, waist ~${sizeEst.waist_cm} cm` : ''}${sizeEst.lengths_from === 'photo' ? `, arms ~${sizeEst.arm_cm} cm, legs ~${sizeEst.leg_cm} cm` : ''})` : ''}.${!weightKg ? ' Add your weight for a more reliable size.' : ''}`
                        : sizeEst?.reason
                          ? sizeReason(sizeEst.reason)
                          : ''}
                  </p>
                </div>
              )}
              {mode === 'photo' && <PhotoAdvice checks={checks} needLegs={outfits.some((o) => o.category !== 'upper')} />}
            </li>

            <li className="step">
              <div className="step-head">
                <h2>Outfits</h2>
                <p>Add one or several. They are made one after another, and you flip through them in the mirror.</p>
              </div>
              <OutfitStrip
                outfits={outfits.map((o, i) => ({ id: o.id, url: o.front.url, label: `Outfit ${i + 1}` }))}
                selected={editing}
                max={MAX_OUTFITS}
                onSelect={setEditing}
                onAdd={addOutfits}
              />
              {editingOutfit && (
                <div className="outfit-editor">
                  <div className="outfit-editor-head">
                    <span className="field-label">Outfit {outfits.indexOf(editingOutfit) + 1}</span>
                    <button type="button" className="link" onClick={() => removeOutfit(editingOutfit.id)} disabled={busy}>
                      Remove outfit
                    </button>
                  </div>
                  {mode === 'photo' && (
                  <div className="field">
                    <span className="field-label" id="cat-label">Type</span>
                    <div className="segmented" role="radiogroup" aria-labelledby="cat-label">
                      {CATEGORIES.map((c) => (
                        <button
                          key={c.id}
                          type="button"
                          role="radio"
                          aria-checked={editingOutfit.category === c.id}
                          className={editingOutfit.category === c.id ? 'on' : ''}
                          onClick={() => updateOutfit(editingOutfit.id, { category: c.id, typeChosen: true })}
                        >
                          {c.label}
                        </button>
                      ))}
                    </div>
                  </div>
                  )}
                  {mode === 'photo' && <Pieces outfit={editingOutfit} />}
                  {mode === 'photo' && (
                    <>
                      <SizeChips value={editingOutfit.sizes} onChange={(v) => updateOutfit(editingOutfit.id, { sizes: v })} />
                      {(() => {
                        const f = fitFor(editingOutfit);
                        return f ? <FitBadge fit={f} stock={editingOutfit.sizes} /> : null;
                      })()}
                    </>
                  )}
                  <div className="back-slot">
                    <PhotoSlot
                      key={editingOutfit.id}
                      id="garment-back"
                      label="Its back"
                      hint="Optional, makes back views exact"
                      url={editingOutfit.back?.url}
                      contain
                      onFile={(f) => setOutfitBack(editingOutfit, f)}
                    />
                  </div>
                </div>
              )}
            </li>

            <li className="step">
              <div className="step-head">
                <h2>Options</h2>
              </div>
              {serverDown && (
                <div className="server-box">
                  <p className="notice">
                    {API
                      ? `Can’t reach the try-on server at ${API}. It may be off, or its address changed.`
                      : 'The try-on server isn’t connected. Paste its address (the https link the server owner shares).'}
                  </p>
                  <form
                    className="server-form"
                    onSubmit={(e) => {
                      e.preventDefault();
                      setServer(new FormData(e.currentTarget).get('server') as string);
                    }}
                  >
                    <input name="server" type="url" placeholder="https://….trycloudflare.com" defaultValue={API} aria-label="Server address" />
                    <button type="submit" className="ghost">Connect</button>
                  </form>
                </div>
              )}
              <label className="toggle" htmlFor="lock">
                <input id="lock" type="checkbox" checked={lockOutside} onChange={(e) => setLockOutside(e.target.checked)} />
                <span>Keep face, hair and background exactly as in my photo</span>
              </label>
              <details className="advanced">
                <summary>Advanced</summary>
                <div className="adv-grid">
                  <label className="field" htmlFor="seed">
                    <span className="field-label">Seed</span>
                    <input id="seed" type="number" value={seed} onChange={(e) => setSeed(Number(e.target.value) || 0)} />
                  </label>
                </div>
              </details>
            </li>
          </ol>

          {mode === 'photo' && (
            <div className="run-bar">
              {formError && <p className="form-error" role="alert">{formError}</p>}
              <button type="button" className="primary" onClick={start} disabled={busy}>
                {busy ? 'Trying it on…' : 'Try it on'}
              </button>
            </div>
          )}
        </section>

        <section className="stage" aria-label="Result">
          <div className="gesture-bar">
            <button
              type="button"
              className={`gesture-toggle ${gestures ? 'on' : ''}`}
              role="switch"
              aria-checked={gestures}
              onClick={() => setGestures((g) => !g)}
              title="Swipe a hand in front of the camera to change outfit. The video stays on this device."
            >
              <span aria-hidden="true">✋</span> Hand gestures {gestures ? 'on' : 'off'}
            </button>
            {gestures && runOutfits.length < 2 && (
              <p className="gesture-hint">Add two or more outfits and press Try it on, then swipe to switch between them.</p>
            )}
          </div>
          {gestures && <GestureControl onSwipe={onSwipe} canGo={canSwipe} />}
          {runOutfits.length > 1 && (
            <nav className="outfit-pager" aria-label="Outfits">
              <button type="button" className="ghost" onClick={() => goOutfit(outfitIdx - 1)} disabled={outfitIdx === 0}>
                ‹ Previous
              </button>
              <div className="pager-strip">
                {runOutfits.map((o, i) => (
                  <button
                    key={o.id}
                    type="button"
                    className={`pager-thumb ${i === outfitIdx ? 'on' : ''}`}
                    aria-label={`Outfit ${i + 1}`}
                    aria-current={i === outfitIdx}
                    onClick={() => goOutfit(i)}
                  >
                    <img src={o.front.url} alt="" />
                    <span className={`dot ${statusFor(o.id) ?? ''}`} aria-hidden="true" />
                  </button>
                ))}
              </div>
              <button
                type="button"
                className="ghost"
                onClick={() => goOutfit(outfitIdx + 1)}
                disabled={outfitIdx === runOutfits.length - 1}
              >
                Next outfit ›
              </button>
            </nav>
          )}
          {runs.length > 0 && (
            <div className="tabs-row">
              <div className="tabs" role="tablist" aria-label="Angle">
                {views.filter((v) => runs.some((r) => r.view === v && r.outfitId === shownOutfit?.id)).map((v) => (
                  <button key={v} role="tab" aria-selected={activeView === v} className={activeView === v ? 'on' : ''} onClick={() => setActiveView(v)}>
                    {VIEWS.find((x) => x.id === v)!.label}
                  </button>
                ))}
              </div>
              <div className="tabs" role="tablist" aria-label="Method">
                {viewRuns.map((r) => (
                  <button key={r.key} role="tab" aria-selected={current?.key === r.key} className={current?.key === r.key ? 'on' : ''} onClick={() => setActiveEngine(r.engine)}>
                    <span className={`dot ${r.status}`} aria-hidden="true" />
                    {engineLabel(r.engine)}
                  </button>
                ))}
              </div>
            </div>
          )}

          {mode === 'live' ? (
            <LiveCamera
              outfit={shownOutfit ? { id: shownOutfit.id, file: shownOutfit.front.file } : undefined}
              title={shownOutfit && runOutfits.length > 1 ? `Outfit ${outfitIdx + 1} of ${runOutfits.length}` : undefined}
            />
          ) : (
            <Mirror
              before={photos[current?.view ?? activeView]?.url ?? photos.front?.url}
              run={current}
              engineLabel={current ? engineLabel(current.engine) : ''}
              ahead={ahead}
              title={shownOutfit && runOutfits.length > 1 ? `Outfit ${outfitIdx + 1} of ${runOutfits.length}` : undefined}
            />
          )}

          {runs.length > 0 && (
            <div className="stage-actions">
              {current?.url && (
                <a className="ghost" href={current.url} download={`tryon-outfit${outfitIdx + 1}-${current.view}-${current.engine}.png`}>
                  Download
                </a>
              )}
              <button type="button" className="ghost" onClick={clearResults} disabled={busy}>
                Clear results
              </button>
            </div>
          )}
          <p className="privacy">Photos are processed in memory and deleted from the server as soon as your result is ready.</p>
        </section>
      </main>
    </div>
  );
}

const PART_FOR: Record<Category, string[]> = { upper: ['upper'], lower: ['lower'], dress: ['upper', 'lower', 'dress'] };

/** What we read in the outfit photo, as a checklist; pieces the chosen type leaves alone are struck out. */
function Pieces({ outfit }: { outfit: Outfit }) {
  if (outfit.analysis === undefined) return <p className="pieces-note">Reading the outfit…</p>;
  if (outfit.analysis?.retake)
    return (
      <p className="form-error" role="alert">
        {outfit.analysis.retake.message}
      </p>
    );
  if (outfit.analysis?.blocked)
    return (
      <p className="form-error" role="alert">
        {outfit.analysis.blocked.message}
      </p>
    );
  if (outfit.analysis === null)
    return (
      <p className="form-error" role="alert">
        This outfit couldn’t be checked, so it can’t be used. Remove it and add it again.
      </p>
    );
  if (!outfit.analysis || !outfit.analysis.person || !outfit.analysis.layers.length) return null;
  const used = PART_FOR[outfit.category];
  return (
    <div className="field">
      <span className="field-label">What we see in the photo</span>
      <ul className="pieces">
        {outfit.analysis.layers.map((l) => {
          const on = used.includes(l.part);
          return (
            <li key={l.name} className={on ? '' : 'off'}>
              <span aria-hidden="true">{on ? '✓' : '–'}</span>
              {l.name}
              {!on && <span className="visually-hidden"> (not changed with this type)</span>}
            </li>
          );
        })}
      </ul>
      {used.length === 1 && outfit.analysis.layers.some((l) => !used.includes(l.part)) && (
        <p className="pieces-note">Only the ticked pieces change. Choose Full outfit to wear the whole look.</p>
      )}
    </div>
  );
}

const VIEW_LABEL: Record<View, string> = { front: 'Front', side: 'Side', back: 'Back' };

function PhotoAdvice({ checks, needLegs }: { checks: Partial<Record<View, PhotoCheck>>; needLegs: boolean }) {
  const items: { key: string; view: View; text: string; strong: boolean }[] = [];
  (Object.keys(checks) as View[]).forEach((view) => {
    for (const w of checks[view]!.warnings) {
      // "No legs" only matters when bottoms or a full outfit are being tried on.
      if (w.code === 'no_legs' && !needLegs) continue;
      items.push({ key: `${view}-${w.code}`, view, text: w.text, strong: w.code !== 'hands_hidden' });
    }
  });
  if (!items.length) return null;
  return (
    <ul className="advice" aria-live="polite">
      {items.map((i) => (
        <li key={i.key} className={i.strong ? 'strong' : ''}>
          <span className="advice-view">{VIEW_LABEL[i.view]}</span>
          {i.text}
        </li>
      ))}
    </ul>
  );
}

function ServerPill({ health, down }: { health: Health | null; down: boolean }) {
  if (down) return <span className="pill bad">Server offline</span>;
  if (!health) return <span className="pill">Checking server…</span>;
  if (!health.gpu) return <span className="pill warn">Server on CPU (slow)</span>;
  return (
    <span className="pill ok" title={`${health.gpu.vram_free_gb} GB of ${health.gpu.vram_total_gb} GB free`}>
      {health.gpu.name.replace('NVIDIA GeForce ', '')} · {health.gpu.vram_total_gb} GB
    </span>
  );
}

/** Why the photo couldn't be measured, in words. */
function sizeReason(reason: string): string {
  switch (reason) {
    case 'turned':
      return 'Face the camera in your photo to check the size.';
    case 'no_scale':
    case 'no_head':
      return 'Use a photo with your head and body in view to check the size.';
    default:
      return 'Your size couldn’t be measured from this photo. Try a front-facing, head-to-feet photo.';
  }
}
