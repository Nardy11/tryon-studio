import { useEffect, useRef, useState } from 'react';
import { FilesetResolver, HandLandmarker } from '@mediapipe/tasks-vision';

/**
 * Swipe a hand in front of the camera to change outfit. Runs entirely in the browser
 * (MediaPipe Hand Landmarker): no camera frame ever leaves the device.
 *
 * Directions are as the user sees themselves in the (mirrored) preview:
 *   right -> left = "next", left -> right = "previous".
 */
export type Swipe = 'next' | 'previous';

const WASM = 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm';
const MODEL =
  'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task';

/**
 * Tuning. Coordinates are shares of the frame (0..1).
 * - A swipe is a mostly-sideways move of at least MIN_TRAVEL within WINDOW_MS.
 * - Fast hands blur and the tracker drops them for a few frames: the trail survives gaps up to
 *   GAP_MS, and a hand that leaves the frame at a side after moving toward it also counts.
 * - While a hand was seen recently, a coarse motion tracker (frame difference) fills in when the
 *   landmarker loses a blurred hand mid-swipe.
 */
const MIN_TRAVEL = 0.2;
const WINDOW_MS = 900;
const GAP_MS = 300;
const EDGE = 0.14;
const ARMED_MS = 1500;
const COOLDOWN_MS = 1300; // about one curtain close + open

let landmarker: Promise<HandLandmarker> | null = null;
function getLandmarker(): Promise<HandLandmarker> {
  if (!landmarker) {
    landmarker = (async () => {
      const fileset = await FilesetResolver.forVisionTasks(WASM);
      return HandLandmarker.createFromOptions(fileset, {
        baseOptions: { modelAssetPath: MODEL, delegate: 'GPU' },
        runningMode: 'VIDEO',
        numHands: 1,
        // Lower than the defaults (0.5): a moving hand is blurry and is otherwise dropped.
        minHandDetectionConfidence: 0.35,
        minHandPresenceConfidence: 0.35,
        minTrackingConfidence: 0.3,
      });
    })();
    landmarker.catch(() => {
      landmarker = null;
    });
  }
  return landmarker;
}

type State = 'starting' | 'ready' | 'error';

type Props = {
  onSwipe: (dir: Swipe) => void;
  /** Shown briefly when a swipe has nowhere to go ("This is the last outfit"). */
  canGo: (dir: Swipe) => boolean;
};

export function GestureControl({ onSwipe, canGo }: Props) {
  const video = useRef<HTMLVideoElement>(null);
  const [state, setState] = useState<State>('starting');
  const [error, setError] = useState('');
  const [hand, setHand] = useState(false);
  const [flash, setFlash] = useState<{ dir: Swipe; ok: boolean; n: number } | null>(null);
  // Latest callbacks without restarting the camera on every render.
  const cb = useRef({ onSwipe, canGo });
  cb.current = { onSwipe, canGo };

  useEffect(() => {
    let stopped = false;
    let stream: MediaStream | null = null;
    let raf = 0;
    type P = { x: number; y: number; t: number };
    let trail: P[] = []; // palm centre from the hand landmarker
    let motion: P[] = []; // centre of moving pixels, used while the landmarker misses
    let quietUntil = 0;
    let lastSeen = -1e9;
    let lastHand = false;
    let lastVideoTime = -1;
    let prevGrey: Uint8ClampedArray | null = null;
    const small = document.createElement('canvas');
    small.width = 64;
    small.height = 48;
    const sctx = small.getContext('2d', { willReadFrequently: true })!;

    const fire = (dir: Swipe, now: number) => {
      const ok = cb.current.canGo(dir);
      if (ok) cb.current.onSwipe(dir);
      setFlash((f) => ({ dir, ok, n: (f?.n ?? 0) + 1 }));
      trail = [];
      motion = [];
      quietUntil = now + COOLDOWN_MS;
    };

    /** Sideways travel inside the window, measured from the far end (so a wind-up does not cancel it). */
    const swipeIn = (pts: P[], minTravel: number): Swipe | null => {
      if (pts.length < 3) return null;
      const last = pts[pts.length - 1];
      let best: Swipe | null = null;
      for (const p of pts) {
        const dx = last.x - p.x;
        const dy = last.y - p.y;
        if (Math.abs(dx) >= minTravel && Math.abs(dy) < Math.abs(dx) * 0.8) best = dx < 0 ? 'next' : 'previous';
      }
      return best;
    };

    /** Centre of the pixels that changed since the last frame (mirrored x), or null if too few. */
    const motionCentre = (v: HTMLVideoElement): { x: number; y: number } | null => {
      sctx.drawImage(v, 0, 0, 64, 48);
      const d = sctx.getImageData(0, 0, 64, 48).data;
      const grey = new Uint8ClampedArray(64 * 48);
      for (let i = 0; i < grey.length; i++) grey[i] = (d[i * 4] * 3 + d[i * 4 + 1] * 6 + d[i * 4 + 2]) / 10;
      const prev = prevGrey;
      prevGrey = grey;
      if (!prev) return null;
      let n = 0, sx = 0, sy = 0;
      for (let i = 0; i < grey.length; i++) {
        if (Math.abs(grey[i] - prev[i]) > 28) {
          n++;
          sx += i % 64;
          sy += Math.floor(i / 64);
        }
      }
      // A hand is a few % of the frame; the whole body moving (or the camera adjusting) is much more.
      if (n < 40 || n > 64 * 48 * 0.35) return null;
      return { x: 1 - sx / n / 64, y: sy / n / 48 };
    };

    (async () => {
      try {
        // Browsers only allow the camera on https (or on this computer, localhost).
        if (!window.isSecureContext) throw new Error('insecure');
        if (!navigator.mediaDevices?.getUserMedia) throw new Error('This browser has no camera access.');
        const [lm, s] = await Promise.all([
          getLandmarker(),
          navigator.mediaDevices.getUserMedia({
            video: { facingMode: 'user', width: { ideal: 640 }, height: { ideal: 480 }, frameRate: { ideal: 30 } },
            audio: false,
          }),
        ]);
        stream = s;
        if (stopped) return;
        const v = video.current!;
        v.srcObject = s;
        await v.play();
        setState('ready');

        const loop = () => {
          if (stopped) return;
          raf = requestAnimationFrame(loop);
          if (v.readyState < 2 || v.currentTime === lastVideoTime) return;
          lastVideoTime = v.currentTime;
          const now = performance.now();
          const res = lm.detectForVideo(v, now);
          const pts = res.landmarks[0];
          const m = motionCentre(v);
          const seen = !!pts;
          if (seen !== lastHand) {
            lastHand = seen;
            setHand(seen);
          }
          trail = trail.filter((p) => now - p.t <= WINDOW_MS);
          motion = motion.filter((p) => now - p.t <= WINDOW_MS);
          if (now < quietUntil) return;

          if (pts) {
            lastSeen = now;
            // Palm centre: wrist + the four knuckles, mirrored to match what the user sees.
            const ids = [0, 5, 9, 13, 17];
            const x = 1 - ids.reduce((a, i) => a + pts[i].x, 0) / ids.length;
            const y = ids.reduce((a, i) => a + pts[i].y, 0) / ids.length;
            trail.push({ x, y, t: now });
            const dir = swipeIn(trail, MIN_TRAVEL);
            if (dir) return fire(dir, now);
          } else if (trail.length && now - trail[trail.length - 1].t > GAP_MS) {
            // Lost for good. If it was heading out of the frame at a side, that was a swipe.
            const last = trail[trail.length - 1];
            const dir = swipeIn(trail, MIN_TRAVEL / 2);
            const outLeft = last.x < EDGE && dir === 'next';
            const outRight = last.x > 1 - EDGE && dir === 'previous';
            trail = [];
            if (outLeft || outRight) return fire(dir!, now);
          }

          // Backup: the landmarker drops a blurred hand; the moving pixels still show the swipe.
          if (m && now - lastSeen < ARMED_MS) {
            motion.push({ ...m, t: now });
            const dir = swipeIn(motion, MIN_TRAVEL * 1.4);
            if (dir) return fire(dir, now);
          } else if (!m) {
            motion = [];
          }
        };
        loop();
      } catch (err) {
        if (stopped) return;
        const e = err as DOMException;
        setError(
          e?.name === 'NotAllowedError'
            ? 'Camera permission was refused. Allow the camera for this site to use hand gestures.'
            : e?.name === 'NotFoundError'
              ? 'No camera was found.'
              : e?.message || 'Hand tracking could not start.',
        );
        setState('error');
      }
    })();

    return () => {
      stopped = true;
      cancelAnimationFrame(raf);
      stream?.getTracks().forEach((t) => t.stop());
    };
  }, []);

  useEffect(() => {
    if (!flash) return;
    const t = setTimeout(() => setFlash(null), 1100);
    return () => clearTimeout(t);
  }, [flash]);

  if (state === 'error') {
    // On plain http the browser never allows the camera (only https, or localhost on this
    // computer). Offer the same page over https; everything else on the page works on both.
    const insecure = typeof window !== 'undefined' && !window.isSecureContext;
    const secureUrl = insecure ? `https://${window.location.host}${window.location.pathname}` : '';
    return (
      <p className="gesture-error" role="alert">
        {insecure ? (
          <>
            Hand gestures need the camera, and browsers only allow it on secure (https) pages.{' '}
            <a href={secureUrl}>Open the secure version of this page</a>.
          </>
        ) : (
          error
        )}
      </p>
    );
  }

  return (
    <div className={`gesture ${hand ? 'hand' : ''}`}>
      <div className="gesture-cam">
        <video ref={video} muted playsInline aria-hidden="true" />
        {flash && (
          <span key={flash.n} className={`gesture-flash ${flash.ok ? '' : 'blocked'}`} aria-hidden="true">
            {flash.dir === 'next' ? '‹' : '›'}
          </span>
        )}
      </div>
      <p className="gesture-hint" aria-live="polite">
        {state === 'starting'
          ? 'Starting the camera…'
          : flash && !flash.ok
            ? flash.dir === 'next'
              ? 'This is the last outfit'
              : 'This is the first outfit'
            : flash
              ? flash.dir === 'next'
                ? 'Next outfit'
                : 'Previous outfit'
              : hand
                ? 'Hand found. Swipe ← next · → back'
                : 'Raise a hand, then swipe ← next · → back'}
      </p>
    </div>
  );
}
