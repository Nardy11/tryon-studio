import { useEffect, useRef, useState } from 'react';
import { FilesetResolver, PoseLandmarker } from '@mediapipe/tasks-vision';
import { api } from '../api';

/**
 * Live mode is a real-time filter. No photo of you is ever taken.
 *
 * 1. For each outfit, the server sends the clothes alone (cut out of the outfit photo, after the
 *    same safety checks as photo mode) and the pose the model wore them in.
 * 2. MediaPipe Pose tracks you in the camera, about 30 times a second, in this browser.
 * 3. Each part of the clothes (torso, hips, upper arms, forearms, thighs, shins) is moved from the
 *    model's pose to yours and drawn over the camera image.
 *
 * It dresses whatever it can see: only shoulders and chest in view -> the top; legs in view ->
 * the trousers too. Your video never leaves this device.
 */
type Props = {
  outfit?: { id: string; file: File };
  title?: string;
};

const WASM = 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm';
const MODEL =
  'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task';
const MAX_SIDE = 960; // drawing canvas: plenty for a phone, light enough for 30 fps
const SEEN = 0.55; // a landmark counts as visible above this

type P = { x: number; y: number; v: number };
type Pose = P[];
type Rigged = { ref: Pose; bones: Bone[]; parts: Map<string, HTMLCanvasElement> };

let landmarker: Promise<PoseLandmarker> | null = null;
function getLandmarker(): Promise<PoseLandmarker> {
  if (!landmarker) {
    landmarker = (async () => {
      const fileset = await FilesetResolver.forVisionTasks(WASM);
      return PoseLandmarker.createFromOptions(fileset, {
        baseOptions: { modelAssetPath: MODEL, delegate: 'GPU' },
        runningMode: 'VIDEO',
        numPoses: 1,
      });
    })();
    landmarker.catch(() => {
      landmarker = null;
    });
  }
  return landmarker;
}

// ---------- geometry ----------
type M = [number, number, number, number, number, number]; // canvas setTransform(a, b, c, d, e, f)
const sub = (a: P, b: P): P => ({ x: a.x - b.x, y: a.y - b.y, v: 1 });
const add = (a: P, b: P): P => ({ x: a.x + b.x, y: a.y + b.y, v: 1 });
const mid = (a: P, b: P): P => ({ x: (a.x + b.x) / 2, y: (a.y + b.y) / 2, v: Math.min(a.v, b.v) });
const len = (a: P) => Math.hypot(a.x, a.y) || 1e-6;
const dist = (a: P, b: P) => len(sub(a, b));

/** The affine map taking three source points to three destination points. */
function affine3(s: P[], d: P[]): M | null {
  const [s0, s1, s2] = s;
  const det = (s1.x - s0.x) * (s2.y - s0.y) - (s2.x - s0.x) * (s1.y - s0.y);
  if (Math.abs(det) < 1e-6) return null;
  const solve = (d0: number, d1: number, d2: number) => {
    const p = ((d1 - d0) * (s2.y - s0.y) - (d2 - d0) * (s1.y - s0.y)) / det;
    const q = ((s1.x - s0.x) * (d2 - d0) - (s2.x - s0.x) * (d1 - d0)) / det;
    return [p, q, d0 - p * s0.x - q * s0.y];
  };
  const [a, c, e] = solve(d[0].x, d[1].x, d[2].x);
  const [b, dd, f] = solve(d[0].y, d[1].y, d[2].y);
  return [a, b, c, dd, e, f];
}

/** Rotation + uniform scale + shift taking segment s0->s1 to d0->d1 (a limb). */
function similar(s0: P, s1: P, d0: P, d1: P): M {
  const vs = sub(s1, s0);
  const vd = sub(d1, d0);
  const k = len(vd) / len(vs);
  const ang = Math.atan2(vd.y, vd.x) - Math.atan2(vs.y, vs.x);
  const a = k * Math.cos(ang);
  const b = k * Math.sin(ang);
  return [a, b, -b, a, d0.x - (a * s0.x - b * s0.y), d0.y - (b * s0.x + a * s0.y)];
}

const [LS, RS, LE, RE, LW, RW, LH, RH, LK, RK, LA, RA] = [11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28];

/**
 * The model's skeleton as bones, each with a width. Every pixel of the clothes belongs to exactly
 * one bone (the nearest, relative to its width), like rigging a 3D model: a sleeve moves with its
 * arm, the torso with the torso, and nothing is drawn twice.
 */
type Bone = { name: string; a: P; b: P; r: number; ends?: [number, number] };

function frame(r: Pose) {
  const has = (i: number) => !!r[i] && r[i].v > 0.3;
  const sw = dist(r[LS], r[RS]);
  const hips = has(LH) && has(RH);
  const mS = mid(r[LS], r[RS]);
  const mH = hips ? mid(r[LH], r[RH]) : add(mS, { x: 0, y: sw * 1.4, v: 1 });
  const T = dist(mS, mH);
  // The shoulders as wide as they'd be facing the camera: a model photographed turned sideways
  // has narrow-looking shoulders, and sizing by them blew the clothes up.
  const W = Math.max(sw, T / 1.3);
  const hw = hips ? Math.max(dist(r[LH], r[RH]), sw * 0.55) : sw * 0.8;
  return { has, sw, hips, mS, mH, T, W, hw };
}

function bones(r: Pose): Bone[] {
  const f = frame(r);
  if (!f.has(LS) || !f.has(RS)) return [];
  const out: Bone[] = [{ name: 'torso', a: f.mS, b: f.mH, r: f.W * 0.55 }];
  if (f.hips && f.has(LK) && f.has(RK)) out.push({ name: 'pelvis', a: f.mH, b: mid(r[LK], r[RK]), r: f.hw * 0.55 });
  const limbs: [string, number, number, number][] = [
    ['thighL', LH, LK, f.hw * 0.3],
    ['thighR', RH, RK, f.hw * 0.3],
    ['shinL', LK, LA, f.hw * 0.22],
    ['shinR', RK, RA, f.hw * 0.22],
    ['armL', LS, LE, f.W * 0.17],
    ['armR', RS, RE, f.W * 0.17],
    ['foreL', LE, LW, f.W * 0.14],
    ['foreR', RE, RW, f.W * 0.14],
  ];
  for (const [name, i, j, rad] of limbs) if (f.has(i) && f.has(j)) out.push({ name, a: r[i], b: r[j], r: rad, ends: [i, j] });
  return out;
}

/** Cut the clothes image into one canvas per bone (each pixel to its nearest bone). */
function rig(img: HTMLImageElement, bs: Bone[]): Map<string, HTMLCanvasElement> {
  const w = img.naturalWidth;
  const h = img.naturalHeight;
  const src = document.createElement('canvas');
  src.width = w;
  src.height = h;
  const sctx = src.getContext('2d', { willReadFrequently: true })!;
  sctx.drawImage(img, 0, 0);
  const data = sctx.getImageData(0, 0, w, h);
  const owner = new Int16Array(w * h).fill(-1);
  const owner2 = new Int16Array(w * h).fill(-1);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const i = y * w + x;
      if (data.data[i * 4 + 3] === 0) continue;
      let best = Infinity;
      let second = Infinity;
      let kb = -1;
      let ks = -1;
      for (let k = 0; k < bs.length; k++) {
        const { a, b, r } = bs[k];
        const abx = b.x - a.x;
        const aby = b.y - a.y;
        const L2 = abx * abx + aby * aby || 1e-6;
        const t = Math.max(0, Math.min(1, ((x - a.x) * abx + (y - a.y) * aby) / L2));
        const d = Math.hypot(x - (a.x + t * abx), y - (a.y + t * aby)) / r;
        if (d < best) {
          second = best;
          ks = kb;
          best = d;
          kb = k;
        } else if (d < second) {
          second = d;
          ks = k;
        }
      }
      owner[i] = kb;
      if (ks >= 0 && second < best * 1.12 + 0.05) owner2[i] = ks;
    }
  }
  const out = new Map<string, HTMLCanvasElement>();
  bs.forEach((bone, k) => {
    const c = document.createElement('canvas');
    c.width = w;
    c.height = h;
    const ctx = c.getContext('2d')!;
    const part = ctx.createImageData(w, h);
    for (let i = 0; i < w * h; i++) {
      if (owner[i] !== k && owner2[i] !== k) continue;
      part.data[i * 4] = data.data[i * 4];
      part.data[i * 4 + 1] = data.data[i * 4 + 1];
      part.data[i * 4 + 2] = data.data[i * 4 + 2];
      part.data[i * 4 + 3] = data.data[i * 4 + 3];
    }
    ctx.putImageData(part, 0, 0);
    out.set(bone.name, c);
  });
  return out;
}

/**
 * For each bone whose body part is visible right now: the map from the model's pose to yours.
 * Bones out of view get no map and aren't drawn: whatever is in frame gets dressed.
 */
function maps(r: Pose, live: Pose, bs: Bone[]): Map<string, M> {
  const out = new Map<string, M>();
  const seen = (i: number) => !!live[i] && live[i].v > SEEN;
  if (!seen(LS) || !seen(RS)) return out;
  const f = frame(r);
  const lS = mid(live[LS], live[RS]);
  let torso: M;
  if (f.hips && seen(LH) && seen(RH)) {
    const lH = mid(live[LH], live[RH]);
    // A model facing the camera: shoulders and hips. Turned sideways: along the torso, so the
    // clothes aren't stretched sideways.
    torso = (f.sw >= f.W * 0.85 && affine3([r[LS], r[RS], f.mH], [live[LS], live[RS], lH])) || similar(f.mS, f.mH, lS, lH);
  } else {
    // Only the shoulders in view (close to the camera, at a desk): sized by the frontal width.
    const m = similar(r[LS], r[RS], live[LS], live[RS]);
    const k = f.sw / f.W;
    const [a, b] = [m[0] * k, m[1] * k];
    torso = [a, b, -b, a, lS.x - (a * f.mS.x - b * f.mS.y), lS.y - (b * f.mS.x + a * f.mS.y)];
  }
  out.set('torso', torso);
  for (const bone of bs) {
    if (bone.name === 'torso') continue;
    if (bone.name === 'pelvis') {
      if (seen(LH) && seen(RH)) {
        const m = seen(LK) && seen(RK) ? affine3([r[LH], r[RH], bone.b], [live[LH], live[RH], mid(live[LK], live[RK])]) : null;
        out.set('pelvis', m || torso);
      }
      continue;
    }
    const [i, j] = bone.ends!;
    if (seen(i) && seen(j)) out.set(bone.name, similar(r[i], r[j], live[i], live[j]));
  }
  return out;
}

// Back to front: legs under the torso, sleeves on top.
const ORDER = ['pelvis', 'thighL', 'thighR', 'shinL', 'shinR', 'torso', 'armL', 'armR', 'foreL', 'foreR'];

/** How far you've moved since a frame was sent: mean shift of the main joints, in shoulder widths. */
function moved(a: Pose, b: Pose): number {
  const ids = [LS, RS, LE, RE, LW, RW, LH, RH];
  const sw = dist(a[LS], a[RS]) || 1;
  let sum = 0;
  let n = 0;
  for (const i of ids) {
    if (a[i].v > SEEN && b[i].v > SEEN) {
      sum += dist(a[i], b[i]);
      n++;
    }
  }
  // A body part coming into view or leaving it (stepping back, sitting down) counts as a big move.
  const changed = [LH, RH, LK, RK].some((i) => a[i].v > SEEN !== b[i].v > SEEN);
  return changed ? 9 : n ? sum / n / sw : 9;
}

const REFRESH_MOVE = 0.18; // shoulder widths moved before a new realistic frame is worth making
const REFRESH_MS = 20000; // and at least this often while you're in view

export function LiveCamera({ outfit, title }: Props) {
  const canvas = useRef<HTMLCanvasElement>(null);
  const video = useRef<HTMLVideoElement>(null);
  const [camera, setCamera] = useState<'starting' | 'ready' | 'error'>('starting');
  const [error, setError] = useState('');
  // waiting: nothing on you yet for this outfit; dressed: the realistic clothes are on you.
  const [look, setLook] = useState<'waiting' | 'dressed'>('waiting');
  const [notice, setNotice] = useState('');
  const [seenBody, setSeenBody] = useState<'none' | 'face' | 'body'>('none');
  const [firstRun, setFirstRun] = useState(true);
  const outfitRef = useRef(outfit);
  outfitRef.current = outfit;
  const state = useRef({
    layer: null as (Rigged & { outfitId: string }) | null,
    sentPose: null as Pose | null, // your pose (shares of the frame) in the last frame sent
    sentAt: 0,
    jobId: '' as string,
    busy: false,
    pauseUntil: 0,
  });

  // A new outfit: drop the old clothes, and stop the frame that was being made for it.
  useEffect(() => {
    const st = state.current;
    st.layer = null;
    st.sentPose = null;
    if (st.jobId) api.forget(st.jobId);
    st.jobId = '';
    setLook('waiting');
    setNotice('');
  }, [outfit?.id]);

  useEffect(() => {
    let stopped = false;
    let stream: MediaStream | null = null;
    let raf = 0;
    let lastTime = -1;
    let live: Pose | null = null; // canvas pixels
    let liveN: Pose | null = null; // shares of the frame
    let prevGrey: Uint8ClampedArray | null = null;
    let stillness = 99;
    const small = document.createElement('canvas');
    small.width = 64;
    small.height = 48;
    const sctx = small.getContext('2d', { willReadFrequently: true })!;

    const motion = (v: HTMLVideoElement) => {
      sctx.drawImage(v, 0, 0, 64, 48);
      const d = sctx.getImageData(0, 0, 64, 48).data;
      const g = new Uint8ClampedArray(64 * 48);
      for (let i = 0; i < g.length; i++) g[i] = (d[i * 4] * 3 + d[i * 4 + 1] * 6 + d[i * 4 + 2]) / 10;
      const p = prevGrey;
      prevGrey = g;
      if (!p) return 99;
      let s = 0;
      for (let i = 0; i < g.length; i++) s += Math.abs(g[i] - p[i]);
      return s / g.length;
    };

    /** Send the current camera frame for a realistic look; put the result on you when it's back. */
    const send = async (v: HTMLVideoElement) => {
      const st = state.current;
      const o = outfitRef.current;
      if (!o || !liveN) return;
      st.busy = true;
      const pose = liveN.map((p) => ({ ...p }));
      const c = document.createElement('canvas');
      const k = Math.min(1, 1024 / Math.max(v.videoWidth, v.videoHeight));
      c.width = Math.round(v.videoWidth * k);
      c.height = Math.round(v.videoHeight * k);
      c.getContext('2d')!.drawImage(v, 0, 0, c.width, c.height);
      const blob = await new Promise<Blob | null>((r) => c.toBlob(r, 'image/jpeg', 0.9));
      st.sentPose = pose;
      st.sentAt = performance.now();
      try {
        if (!blob) throw new Error('The camera frame could not be read.');
        const job = await api.liveFrame(blob, o.file);
        st.jobId = job.id;
        let j = job;
        while (!stopped && j.status !== 'done' && j.status !== 'error') {
          await new Promise((r) => setTimeout(r, 300));
          if (st.jobId !== job.id) return; // outfit changed meanwhile
          j = await api.job(job.id);
        }
        if (stopped || st.jobId !== job.id) return;
        st.jobId = '';
        setFirstRun(false);
        if (j.status === 'error') {
          // Refused or failed: say so quietly; the next frame tries again.
          setNotice(j.error ?? '');
          return;
        }
        const url = URL.createObjectURL(await api.result(job.id));
        const img = await new Promise<HTMLImageElement>((resolve, reject) => {
          const i = new Image();
          i.onload = () => resolve(i);
          i.onerror = reject;
          i.src = url;
        });
        if (outfitRef.current?.id !== o.id) return;
        const ref = pose.map((p) => ({ x: p.x * img.naturalWidth, y: p.y * img.naturalHeight, v: p.v }));
        const bs = bones(ref);
        if (!bs.length) return;
        const parts = rig(img, bs);
        URL.revokeObjectURL(url);
        st.layer = { ref, bones: bs, parts, outfitId: o.id };
        setLook('dressed');
        setNotice('');
      } catch (e) {
        // A photo the server turned down (blurry, too dark...): wait a moment, then try again.
        setNotice((e as Error).message || 'The try-on server could not be reached.');
        st.pauseUntil = performance.now() + 3000;
      } finally {
        st.busy = false;
      }
    };

    (async () => {
      try {
        if (!window.isSecureContext) throw new Error('insecure');
        if (!navigator.mediaDevices?.getUserMedia) throw new Error('This browser has no camera access.');
        const [lm, s] = await Promise.all([
          getLandmarker(),
          navigator.mediaDevices.getUserMedia({
            video: { facingMode: 'user', width: { ideal: 1280 }, height: { ideal: 720 } },
            audio: false,
          }),
        ]);
        stream = s;
        if (stopped) return;
        const v = video.current!;
        v.srcObject = s;
        await v.play();
        const cv = canvas.current!;
        const k = Math.min(1, MAX_SIDE / Math.max(v.videoWidth, v.videoHeight));
        cv.width = Math.round(v.videoWidth * k);
        cv.height = Math.round(v.videoHeight * k);
        const ctx = cv.getContext('2d')!;
        setCamera('ready');

        const loop = () => {
          if (stopped) return;
          raf = requestAnimationFrame(loop);
          if (v.readyState < 2 || v.currentTime === lastTime) return;
          lastTime = v.currentTime;
          const W = cv.width;
          const H = cv.height;
          const now = performance.now();
          const res = lm.detectForVideo(v, now);
          const pts = res.landmarks[0];
          if (pts) {
            const next = pts.map((p) => ({ x: p.x, y: p.y, v: p.visibility ?? 1 }));
            // Smooth the jitter, keep the motion.
            liveN = liveN ? next.map((p, i) => ({ x: liveN![i].x * 0.4 + p.x * 0.6, y: liveN![i].y * 0.4 + p.y * 0.6, v: p.v })) : next;
            live = liveN.map((p) => ({ x: p.x * W, y: p.y * H, v: p.v }));
          } else {
            liveN = null;
            live = null;
          }
          const body = !live ? 'none' : live[LS].v > SEEN && live[RS].v > SEEN ? 'body' : 'face';
          setSeenBody(body);
          stillness = stillness * 0.7 + motion(v) * 0.3;

          ctx.setTransform(1, 0, 0, 1, 0, 0);
          ctx.drawImage(v, 0, 0, W, H);
          const st = state.current;
          const g = st.layer;
          if (g && live && g.outfitId === outfitRef.current?.id) {
            // The realistic clothes, made on you a moment ago, moved to where you are now.
            const ms = maps(g.ref, live, g.bones);
            for (const name of ORDER) {
              const m = ms.get(name);
              const part = g.parts.get(name);
              if (!m || !part) continue;
              ctx.setTransform(...m);
              ctx.drawImage(part, 0, 0);
            }
            ctx.setTransform(1, 0, 0, 1, 0, 0);
          }

          // A new realistic frame when the GPU is free, you're in view and fairly still, and you
          // have moved (or it's been a while). No photo step: it just keeps adapting.
          if (
            outfitRef.current &&
            !st.busy &&
            body === 'body' &&
            now > st.pauseUntil &&
            stillness < 6 &&
            (!st.sentPose || !liveN || moved(st.sentPose, liveN) > REFRESH_MOVE || now - st.sentAt > REFRESH_MS)
          ) {
            void send(v);
          }
        };
        loop();
      } catch (err) {
        if (stopped) return;
        const e = err as DOMException;
        setError(
          e?.message === 'insecure'
            ? 'insecure'
            : e?.name === 'NotAllowedError'
              ? 'Camera permission was refused. Allow the camera for this site to use Live mode.'
              : e?.name === 'NotFoundError'
                ? 'No camera was found.'
                : e?.message || 'The camera could not start.',
        );
        setCamera('error');
      }
    })();

    return () => {
      stopped = true;
      cancelAnimationFrame(raf);
      stream?.getTracks().forEach((t) => t.stop());
      const st = state.current;
      if (st.jobId) api.forget(st.jobId);
    };
  }, []);

  if (camera === 'error') {
    if (error === 'insecure') {
      const secureUrl = `https://${window.location.host}${window.location.pathname}`;
      return (
        <p className="form-error" role="alert">
          Live mode needs the camera, and browsers only allow it on secure (https) pages.{' '}
          <a href={secureUrl}>Open the secure version of this page</a>.
        </p>
      );
    }
    return (
      <p className="form-error" role="alert">
        {error}
      </p>
    );
  }

  const hint =
    camera === 'starting'
      ? 'Starting the camera…'
      : !outfit
        ? 'Add an outfit and it appears on you.'
        : seenBody === 'none'
          ? 'Step in front of the camera.'
          : seenBody === 'face'
            ? 'Show your shoulders to the camera and the outfit appears.'
            : look === 'waiting'
              ? firstRun
                ? 'Dressing you… the first look takes longer while the model starts.'
                : 'Dressing you…'
              : 'The outfit follows you and refreshes as you move. Swipe or use Next to change outfit.';

  return (
    <figure className="mirror-wrap">
      <div className={`mirror live-mirror ${look === 'dressed' || !outfit || firstRun ? 'open' : 'closed'}`}>
        <video ref={video} muted playsInline className="live-video" aria-hidden="true" />
        <canvas ref={canvas} className="live-canvas" aria-label="You, live, wearing the outfit" />
        <div className="stage-curtains" aria-hidden="true">
          <span className="drape left" />
          <span className="drape right" />
          <span className="lift" />
          <span className="valance" />
        </div>
        {outfit && look === 'waiting' && seenBody === 'body' && (
          <div className={`plaque ${firstRun ? 'low' : ''}`} role="status">
            {title && <span className="plaque-title">{title}</span>}
            <span className="m-progress" aria-hidden="true" />
            <span>{notice || (firstRun ? 'Dressing you… (the first look takes about a minute)' : 'Dressing you…')}</span>
          </div>
        )}
      </div>
      {hint && (
        <figcaption className="live-caption">
          <span className="live-hint" aria-live="polite">
            {look === 'dressed' && notice ? notice : hint}
          </span>
        </figcaption>
      )}
    </figure>
  );
}
