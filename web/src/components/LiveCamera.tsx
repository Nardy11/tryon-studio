import { useEffect, useRef, useState } from 'react';
import { FilesetResolver, PoseLandmarker, type NormalizedLandmark } from '@mediapipe/tasks-vision';
import type { Run } from '../App';

/**
 * Live mode: the outfit stays on you while you move.
 *
 * 1. One photo, once. The first time you hold still, a 3-2-1 countdown takes one frame. That
 *    frame goes through the same server checks and try-on as an uploaded photo. No more photos
 *    are taken unless you press "New photo".
 * 2. The server sends back only the new clothes (transparent elsewhere).
 * 3. MediaPipe Pose tracks your body in the camera. Each part of the clothes (torso, hips, upper
 *    arms, forearms, thighs, shins) is moved from where that body part was in the photo to where
 *    it is now, about 30 times a second, and drawn over the live camera image.
 *
 * Small and medium moves follow well. Turning sideways or around can't: the clothes were made
 * for the front. Then it says so, and "New photo" makes them again from the new pose.
 * Nothing leaves this browser except the one photo sent for the try-on.
 */
type Props = {
  onCapture: (file: File) => void;
  /** Looks are being made: no new photo meanwhile. */
  busy: boolean;
  /** The current outfit's look for this photo: the clothes only, as a transparent PNG. */
  run?: Run;
  title?: string;
  /** Changes when the server turned the photo down: get ready to take another one. */
  retryKey?: number;
};

const WASM = 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm';
const MODEL =
  'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task';

const STILL = 3.2; // mean frame change below this = holding still
const SETTLE_MS = 700;
const COUNT_MS = 3000;
const MAX_SIDE = 960; // the drawing canvas; plenty for a phone screen, light enough for 30 fps

type P = { x: number; y: number; v: number };
type Pose = P[];

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
const sub = (a: P, b: P) => ({ x: a.x - b.x, y: a.y - b.y, v: 1 });
const add = (a: P, b: P) => ({ x: a.x + b.x, y: a.y + b.y, v: 1 });
const mul = (a: P, k: number) => ({ x: a.x * k, y: a.y * k, v: 1 });
const mid = (a: P, b: P) => ({ x: (a.x + b.x) / 2, y: (a.y + b.y) / 2, v: Math.min(a.v, b.v) });
const len = (a: P) => Math.hypot(a.x, a.y) || 1e-6;
const unit = (a: P) => mul(a, 1 / len(a));
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

/** A rounded band around segment A->B, extended past either end. */
function capsule(a: P, b: P, r: number, extA = 0, extB = 0): P[] {
  const u = unit(sub(b, a));
  const n = { x: -u.y, y: u.x, v: 1 };
  const A = sub(a, mul(u, extA));
  const B = add(b, mul(u, extB));
  const pts: P[] = [];
  // Half circle bulging past B, then half circle bulging back past A.
  for (let i = 0; i <= 8; i++) {
    const t = -Math.PI / 2 + (Math.PI * i) / 8;
    pts.push(add(B, add(mul(u, r * Math.cos(t)), mul(n, r * Math.sin(t)))));
  }
  for (let i = 0; i <= 8; i++) {
    const t = Math.PI / 2 + (Math.PI * i) / 8;
    pts.push(add(A, add(mul(u, r * Math.cos(t)), mul(n, r * Math.sin(t)))));
  }
  return pts;
}

type Part = { clip: P[]; map: M | null };

/** The clothes, cut into body parts, each with the map from the photo's pose to the live pose. */
function parts(ref: Pose, live: Pose): Part[] {
  const ok = (i: number) => live[i] && live[i].v > 0.5 && ref[i] && ref[i].v > 0.3;
  const [LS, RS, LE, RE, LW, RW, LH, RH, LK, RK, LA, RA] = [11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28];
  if (!ok(LS) || !ok(RS) || !ref[LH] || !ref[RH]) return [];
  const r = ref;
  const sw = dist(r[LS], r[RS]);
  const hw = Math.max(dist(r[LH], r[RH]), sw * 0.55);
  const mS = mid(r[LS], r[RS]);
  const mH = mid(r[LH], r[RH]);
  const down = unit(sub(mH, mS));
  const T = dist(mS, mH);
  // The hips are often hidden (a coat, a desk): keep their shape relative to the shoulders.
  const liveHip = (i: number) => (ok(i) ? live[i] : null);
  const torsoMap =
    liveHip(LH) && liveHip(RH)
      ? affine3([r[LS], r[RS], mH], [live[LS], live[RS], mid(live[LH], live[RH])])
      : similar(r[LS], r[RS], live[LS], live[RS]);
  const out: Part[] = [];
  const outward = (p: P, q: P, k: number) => add(p, mul(unit(sub(p, q)), k));

  // Hips and skirt, between the hips and the knees.
  if (r[LK] && r[RK]) {
    const mK = mid(r[LK], r[RK]);
    const kw = Math.max(dist(r[LK], r[RK]), hw * 0.8);
    const clip = [
      outward(sub(r[LH], mul(down, T * 0.1)), r[RH], hw * 0.5),
      outward(sub(r[RH], mul(down, T * 0.1)), r[LH], hw * 0.5),
      outward(r[RK], r[LK], kw * 0.7),
      outward(r[LK], r[RK], kw * 0.7),
    ];
    const map =
      ok(LH) && ok(RH) && ok(LK) && ok(RK)
        ? affine3([r[LH], r[RH], mK], [live[LH], live[RH], mid(live[LK], live[RK])])
        : torsoMap;
    out.push({ clip, map });
  }
  // Thighs and shins (trousers, long skirts).
  for (const [H, K, A] of [
    [LH, LK, LA],
    [RH, RK, RA],
  ]) {
    if (r[H] && r[K]) out.push({ clip: capsule(r[H], r[K], hw * 0.45, hw * 0.1, 0), map: ok(H) && ok(K) ? similar(r[H], r[K], live[H], live[K]) : torsoMap });
    if (r[K] && r[A]) out.push({ clip: capsule(r[K], r[A], hw * 0.32, 0, dist(r[K], r[A]) * 0.12), map: ok(K) && ok(A) ? similar(r[K], r[A], live[K], live[A]) : torsoMap });
  }
  // Torso, from above the shoulders to just below the hips.
  out.push({
    clip: [
      outward(sub(r[LS], mul(down, sw * 0.25)), r[RS], sw * 0.25),
      outward(sub(r[RS], mul(down, sw * 0.25)), r[LS], sw * 0.25),
      outward(add(r[RH], mul(down, T * 0.18)), r[LH], hw * 0.45),
      outward(add(r[LH], mul(down, T * 0.18)), r[RH], hw * 0.45),
    ],
    map: torsoMap,
  });
  // Sleeves on top.
  for (const [S, E, W] of [
    [LS, LE, LW],
    [RS, RE, RW],
  ]) {
    if (r[S] && r[E]) out.push({ clip: capsule(r[S], r[E], sw * 0.3, sw * 0.12, 0), map: ok(S) && ok(E) ? similar(r[S], r[E], live[S], live[E]) : torsoMap });
    if (r[E] && r[W]) out.push({ clip: capsule(r[E], r[W], sw * 0.24, 0, sw * 0.05), map: ok(E) && ok(W) ? similar(r[E], r[W], live[E], live[W]) : torsoMap });
  }
  return out;
}

type Phase = 'starting' | 'waiting' | 'counting' | 'sent' | 'error';

export function LiveCamera({ onCapture, busy, run, title, retryKey = 0 }: Props) {
  const canvas = useRef<HTMLCanvasElement>(null);
  const video = useRef<HTMLVideoElement>(null);
  const [phase, setPhase] = useState<Phase>('starting');
  const [error, setError] = useState('');
  const [count, setCount] = useState(0);
  const [turned, setTurned] = useState(false);
  const [flash, setFlash] = useState(0);
  const state = useRef({
    ref: null as Pose | null, // pose in the photo the clothes were made for (canvas pixels)
    garment: null as HTMLImageElement | null,
    captured: false,
    countFrom: 0,
    wantCount: false, // "New photo" pressed: count down whether or not you are still
  });
  const cb = useRef({ onCapture });
  cb.current = { onCapture };
  const newPhotoRef = useRef<() => void>(() => undefined);

  // The photo was turned down (blurry, too dark...): take another when you hold still.
  useEffect(() => {
    if (!retryKey) return;
    const st = state.current;
    st.captured = false;
    st.wantCount = false;
    st.countFrom = 0;
    setPhase('waiting');
  }, [retryKey]);

  // Load the clothes for the current outfit.
  const garmentUrl = run?.status === 'done' ? run.url : undefined;
  useEffect(() => {
    if (!garmentUrl) {
      state.current.garment = null;
      return;
    }
    const img = new Image();
    img.onload = () => {
      state.current.garment = img;
    };
    img.src = garmentUrl;
  }, [garmentUrl]);

  useEffect(() => {
    let stopped = false;
    let stream: MediaStream | null = null;
    let raf = 0;
    let lastTime = -1;
    let prevGrey: Uint8ClampedArray | null = null;
    let stillSince = 0;
    let live: Pose | null = null;
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

    const capture = (v: HTMLVideoElement, W: number, H: number) => {
      const c = document.createElement('canvas');
      c.width = v.videoWidth;
      c.height = v.videoHeight;
      c.getContext('2d')!.drawImage(v, 0, 0);
      state.current.ref = live ? live.map((p) => ({ ...p })) : null;
      void W;
      void H;
      state.current.captured = true;
      state.current.countFrom = 0;
      state.current.wantCount = false;
      state.current.garment = null;
      setFlash((f) => f + 1);
      setPhase('sent');
      setCount(0);
      c.toBlob((blob) => blob && cb.current.onCapture(new File([blob], `live-${Date.now()}.jpg`, { type: 'image/jpeg' })), 'image/jpeg', 0.92);
    };
    newPhotoRef.current = () => {
      state.current.captured = false;
      state.current.wantCount = true;
      state.current.countFrom = performance.now();
      state.current.garment = null;
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
        setPhase('waiting');

        const loop = () => {
          if (stopped) return;
          raf = requestAnimationFrame(loop);
          if (v.readyState < 2 || v.currentTime === lastTime) return;
          lastTime = v.currentTime;
          const W = cv.width;
          const H = cv.height;
          const now = performance.now();
          const res = lm.detectForVideo(v, now);
          const pts: NormalizedLandmark[] | undefined = res.landmarks[0];
          if (pts) {
            const next = pts.map((p) => ({ x: p.x * W, y: p.y * H, v: p.visibility ?? 1 }));
            // Smooth the jitter, keep the motion.
            live = live ? next.map((p, i) => ({ x: live![i].x * 0.45 + p.x * 0.55, y: live![i].y * 0.45 + p.y * 0.55, v: p.v })) : next;
          } else {
            live = null;
          }

          ctx.setTransform(1, 0, 0, 1, 0, 0);
          ctx.drawImage(v, 0, 0, W, H);

          const st = state.current;
          // Draw the clothes, part by part, where the body is now.
          if (st.garment && st.ref && live) {
            const ps = parts(st.ref, live);
            for (const p of ps) {
              if (!p.map) continue;
              ctx.save();
              ctx.setTransform(...p.map);
              ctx.beginPath();
              p.clip.forEach((q, i) => (i ? ctx.lineTo(q.x, q.y) : ctx.moveTo(q.x, q.y)));
              ctx.closePath();
              ctx.clip();
              ctx.drawImage(st.garment, 0, 0, W, H);
              ctx.restore();
            }
            // Turned sideways: the front of the clothes can't follow.
            const swRef = dist(st.ref[11], st.ref[12]);
            const swNow = dist(live[11], live[12]);
            setTurned(swNow < swRef * 0.6);
          } else {
            setTurned(false);
          }

          // The one photo: when you first hold still (or after "New photo").
          if (!st.captured) {
            const m = motion(v);
            if (st.wantCount) {
              const left = COUNT_MS - (now - st.countFrom);
              setPhase('counting');
              setCount(Math.max(1, Math.ceil(left / 1000)));
              if (left <= 0) capture(v, W, H);
            } else if (m < STILL && live) {
              if (!stillSince) stillSince = now;
              if (!st.countFrom && now - stillSince > SETTLE_MS) st.countFrom = now;
              if (st.countFrom) {
                const left = COUNT_MS - (now - st.countFrom);
                setPhase('counting');
                setCount(Math.max(1, Math.ceil(left / 1000)));
                if (left <= 0) capture(v, W, H);
              }
            } else if (m > STILL * 1.6) {
              stillSince = 0;
              st.countFrom = 0;
              setPhase('waiting');
              setCount(0);
            }
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
        setPhase('error');
      }
    })();

    return () => {
      stopped = true;
      cancelAnimationFrame(raf);
      stream?.getTracks().forEach((t) => t.stop());
    };
  }, []);

  if (phase === 'error') {
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

  const pending = !!run && run.status !== 'done' && run.status !== 'error';
  const dressed = run?.status === 'done';
  // Curtains: closed while the clothes are being made, open when they're on you (or before the photo).
  const open = !pending;
  const hint =
    phase === 'starting'
      ? 'Starting the camera…'
      : phase === 'counting'
        ? 'Hold still…'
        : !state.current.captured
          ? 'Step back so you fit from head to feet, then hold still. One photo is taken, once.'
          : pending
            ? 'Making your outfit… it then stays on you while you move.'
            : turned
              ? 'Face the camera to keep the outfit on, or take a new photo in this pose.'
              : dressed
                ? 'Move around: the outfit follows you. Swipe or use Next to change outfit.'
                : '';

  return (
    <figure className="mirror-wrap">
      <div className={`mirror live-mirror ${open ? 'open' : 'closed'}`}>
        <video ref={video} muted playsInline className="live-video" aria-hidden="true" />
        <canvas ref={canvas} className="live-canvas" aria-label="You, live, wearing the outfit" />
        {phase === 'counting' && count > 0 && (
          <span key={count} className="live-count" aria-hidden="true">
            {count}
          </span>
        )}
        {flash > 0 && <span key={flash} className="live-flash" aria-hidden="true" />}
        <div className="stage-curtains" aria-hidden="true">
          <span className="drape left" />
          <span className="drape right" />
          <span className="lift" />
          <span className="valance" />
        </div>
        {pending && (
          <div className="plaque" role="status">
            {title && <span className="plaque-title">{title}</span>}
            <span className="m-progress" aria-hidden="true" />
            <span>Making your outfit…</span>
          </div>
        )}
        {run?.status === 'error' && (
          <div className="plaque error" role="alert">
            {title && <span className="plaque-title">{title}</span>}
            <span>{run.error}</span>
          </div>
        )}
      </div>
      <figcaption className="live-caption">
        <span className="live-hint" aria-live="polite">
          {hint}
        </span>
        {state.current.captured && (
          <button type="button" className="ghost" onClick={() => newPhotoRef.current()} disabled={busy}>
            New photo
          </button>
        )}
      </figcaption>
    </figure>
  );
}
