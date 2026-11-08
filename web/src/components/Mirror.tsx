import { useEffect, useState } from 'react';
import type { Run } from '../App';

type Props = {
  before?: string;
  run?: Run;
  engineLabel: string;
  /** Looks still to be generated before this one (it waits its turn on the GPU). */
  ahead: number;
  title?: string;
};

/** The close takes about this long (side curtains, then the front panel drops); the picture is swapped behind it. */
const CLOSE_MS = 950;

const reduceMotion = () =>
  typeof window !== 'undefined' && window.matchMedia('(prefers-reduced-motion: reduce)').matches;

/**
 * The mirror is a small stage. The red curtains close while a look is being made and open on
 * the result: a front panel lifts from the bottom to the top, then the two curtains behind it part
 * from the middle to the sides. Switching to another look closes them first, so the swap happens
 * behind them.
 * Once open, a gold rope splits the mirror: drag it (or use the arrow keys) to compare the
 * look, on the left, with your own photo, on the right.
 */
export function Mirror({ before, run, engineLabel, ahead, title }: Props) {
  const [pos, setPos] = useState(60);
  const [elapsed, setElapsed] = useState(0);
  const [open, setOpen] = useState(true);
  const [shown, setShown] = useState<{ url: string; label: string } | null>(null);

  const ready = run?.status === 'done' && !!run.url;
  const pending = !!run && !ready && run.status !== 'error';

  // Close, swap the picture while closed, then open. With no run at all, the curtains stay
  // tied back and the mirror simply shows your photo.
  useEffect(() => {
    if (!run) {
      setShown(null);
      setOpen(true);
      return;
    }
    setOpen(false);
    if (!ready) return;
    const t = setTimeout(
      () => {
        setShown({ url: run.url!, label: engineLabel });
        setOpen(true);
      },
      reduceMotion() ? 0 : CLOSE_MS,
    );
    return () => clearTimeout(t);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [run?.key, run?.url, ready]);

  useEffect(() => {
    if (!pending || !run?.jobId) return;
    setElapsed((Date.now() - run.startedAt) / 1000);
    const t = setInterval(() => setElapsed((Date.now() - run.startedAt) / 1000), 250);
    return () => clearInterval(t);
  }, [pending, run?.jobId, run?.startedAt]);

  if (!before) {
    return (
      <div className="mirror empty">
        <div className="mirror-empty">
          <p className="mirror-empty-title">Your reflection appears here</p>
          <p>Add a photo of yourself and one or more outfits, then choose Try it on.</p>
        </div>
      </div>
    );
  }

  const showCompare = open && !!shown && !!run;

  return (
    <figure className="mirror-wrap">
      <div className={`mirror ${open ? 'open' : 'closed'}`}>
        <img src={before} alt="Your photo" className="m-img" />
        {shown && run && (
          <img
            src={shown.url}
            alt={`You wearing the outfit (${shown.label})`}
            className="m-img m-after"
            style={{ clipPath: `inset(0 ${100 - pos}% 0 0)` }}
          />
        )}
        {showCompare && (
          <>
            <div className="rope" style={{ left: `${pos}%` }} aria-hidden="true">
              <span className="tassel" />
            </div>
            <span className="m-tag left">Wearing it</span>
            <span className="m-tag right">Your photo</span>
            <input
              className="compare"
              type="range"
              min={0}
              max={100}
              step={0.5}
              value={pos}
              onChange={(e) => setPos(Number(e.target.value))}
              aria-label="Pull the rope to compare the look with your photo"
            />
          </>
        )}

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
            <span>{statusText(run!, elapsed, ahead)}</span>
          </div>
        )}
        {run?.status === 'error' && (
          <div className="plaque error" role="alert">
            {title && <span className="plaque-title">{title}</span>}
            <span>{run.error}</span>
          </div>
        )}
      </div>
      {showCompare && run?.status === 'done' && !!run.checklist?.length && (
        <ul className="m-checklist" aria-label="Pieces in this look">
          {run.checklist.map((c) => (
            <li key={c.name} className={c.ok ? 'ok' : 'miss'}>
              <span aria-hidden="true">{c.ok ? '✓' : '✗'}</span> {c.name}
              {!c.ok && <span className="visually-hidden"> (not found in the result)</span>}
            </li>
          ))}
        </ul>
      )}
      {showCompare && run?.status === 'done' && (
        <figcaption className="m-caption">
          {title && <span>{title}</span>}
          <span>{engineLabel}</span>
          <span className="mono">
            {run.runSeconds != null ? `${run.runSeconds.toFixed(1)} s` : ''}
            {run.loadSeconds ? ` · model load ${run.loadSeconds.toFixed(0)} s` : ''}
          </span>
        </figcaption>
      )}
    </figure>
  );
}

function statusText(run: Run, elapsed: number, ahead: number): string {
  if (!run.jobId) {
    return ahead > 0 ? `Waiting its turn · ${ahead} look${ahead === 1 ? '' : 's'} before this one` : 'Up next';
  }
  const t = `${Math.floor(elapsed)} s`;
  switch (run.status) {
    case 'queued':
      return run.position ? `In the server queue, ${run.position} ahead · ${t}` : `Waiting for the GPU · ${t}`;
    case 'loading':
      return `Loading the model onto the GPU (the first run takes a while) · ${t}`;
    default:
      return `Dressing you · ${t}`;
  }
}
